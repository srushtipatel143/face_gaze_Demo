"""
Vision Engine for Face Verification and Eye-Gaze Estimation.
Powered by MediaPipe Face Mesh and OpenCV.
"""

import base64
import io
import re
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Optional, Dict, Any, List

# Landmark indices for canonical facial feature points
# Selected 68 anchor landmarks across eyes, brows, nose, mouth contour, and jaw
ANCHOR_LANDMARKS = [
    # Left eye & brow
    33, 133, 159, 145, 70, 63, 105, 66, 107,
    # Right eye & brow
    263, 362, 386, 374, 336, 296, 334, 293, 300,
    # Nose
    1, 2, 98, 327, 168, 197, 5, 4, 129, 358,
    # Mouth & lips
    61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317,
    # Face contour & jawline
    10, 152, 234, 454, 58, 288, 172, 397, 136, 365, 149, 378,
    # Temples & cheeks
    127, 356, 116, 345, 123, 352, 50, 280, 205, 425
]

# 3D generic facial model points for head pose estimation (SolvePnP)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left eye corner (index 33)
    (225.0, 170.0, -135.0),      # Right eye corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
], dtype=np.float64)


class VisionEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # FaceMesh with iris refinement enabled (provides landmarks 468-477 for iris)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Dedicated static face mesh for reference photo validation
        self.reference_face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.6
        )

    @staticmethod
    def decode_base64_image(image_str: str) -> np.ndarray:
        """Decode base64 encoded image or Data URL into BGR numpy array."""
        if not image_str:
            raise ValueError("Image data is empty")
        
        # Strip Data URL prefix if present (e.g. data:image/jpeg;base64,...)
        if "," in image_str:
            image_str = image_str.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_str)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 string: {str(e)}")

        if len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError("Image file size exceeds maximum allowed limit (10MB)")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image buffer could not be rendered as a valid image")

        return img

    @staticmethod
    def assess_quality(image: np.ndarray, face_box: Optional[Dict[str, int]] = None) -> Tuple[float, List[str]]:
        """
        Evaluate image brightness, sharpness/blur, and face area ratio.
        Returns a quality score (0-100) and any warning/failure reasons.
        """
        issues = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Sharpness / Blur check using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 45.0:
            issues.append(f"Image appears blurry (sharpness: {laplacian_var:.1f}, min: 45.0)")

        # 2. Brightness check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 40.0:
            issues.append(f"Lighting is too dark (brightness: {mean_brightness:.1f}/255, min: 40)")
        elif mean_brightness > 225.0:
            issues.append(f"Lighting is overexposed (brightness: {mean_brightness:.1f}/255, max: 225)")

        # 3. Face size ratio check
        face_ratio = 0.0
        if face_box:
            face_area = face_box["width"] * face_box["height"]
            img_area = w * h
            face_ratio = face_area / float(img_area)
            if face_ratio < 0.05:
                issues.append("Face is too far from the camera. Please move closer.")

        # Compute composite quality score (0 - 100)
        sharpness_score = min(100.0, (laplacian_var / 120.0) * 50.0)
        brightness_score = 50.0 - abs(mean_brightness - 128.0) * (50.0 / 128.0)
        quality_score = max(5.0, min(100.0, round(sharpness_score + brightness_score, 1)))

        return quality_score, issues

    @staticmethod
    def extract_face_box(landmarks, img_w: int, img_h: int) -> Dict[str, int]:
        """Compute pixel bounding box from normalized face landmarks."""
        xs = [int(p.x * img_w) for p in landmarks.landmark]
        ys = [int(p.y * img_h) for p in landmarks.landmark]

        min_x = max(0, min(xs))
        max_x = min(img_w, max(xs))
        min_y = max(0, min(ys))
        max_y = min(img_h, max(ys))

        # Add modest padding around the face
        pad_x = int((max_x - min_x) * 0.08)
        pad_y = int((max_y - min_y) * 0.12)

        x = max(0, min_x - pad_x)
        y = max(0, min_y - pad_y)
        width = min(img_w - x, (max_x - min_x) + 2 * pad_x)
        height = min(img_h - y, (max_y - min_y) + 2 * pad_y)

        return {"x": x, "y": y, "width": width, "height": height}

    @staticmethod
    def compute_embedding(landmarks) -> np.ndarray:
        """
        Construct a normalized 3D facial feature embedding vector from 468 face landmarks.
        Uses scale-normalized and center-aligned canonical anchor landmarks to yield
        an invariant biometric descriptor.
        """
        pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float64)

        # Reference anchor points
        nose = pts[1]
        left_eye = pts[33]
        right_eye = pts[263]

        # Inter-ocular distance for scale invariance
        iod = np.linalg.norm(right_eye - left_eye) + 1e-6

        # Center on nose and scale
        norm_pts = (pts - nose) / iod

        # Select anchor subset
        anchors = norm_pts[ANCHOR_LANDMARKS].flatten()

        # Additional relative inter-landmark distances for structural rigidity
        pairs = [
            (33, 263), (1, 152), (61, 291), (10, 152), (33, 1), (263, 1),
            (70, 300), (105, 334), (61, 1), (291, 1), (133, 362), (234, 454)
        ]
        pair_dists = [np.linalg.norm(norm_pts[i] - norm_pts[j]) for i, j in pairs]

        feature_vector = np.concatenate([anchors, pair_dists])
        unit_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-6)
        return unit_vector

    @staticmethod
    def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Calculate cosine similarity between two unit biometric embeddings (range 0.0 - 1.0)."""
        dot = float(np.dot(embedding_a, embedding_b))
        return max(0.0, min(1.0, dot))

    @staticmethod
    def estimate_head_pose(landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
        """
        Estimate pitch, yaw, and roll in degrees using SolvePnP on key facial landmarks.
        Pitch: positive = looking up, negative = looking down.
        Yaw: positive = turned right, negative = turned left.
        """
        pts = landmarks.landmark
        image_points = np.array([
            (pts[1].x * img_w, pts[1].y * img_h),       # Nose tip
            (pts[152].x * img_w, pts[152].y * img_h),   # Chin
            (pts[33].x * img_w, pts[33].y * img_h),     # Left eye corner
            (pts[263].x * img_w, pts[263].y * img_h),   # Right eye corner
            (pts[61].x * img_w, pts[61].y * img_h),     # Left mouth corner
            (pts[291].x * img_w, pts[291].y * img_h)    # Right mouth corner
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize angles to standard degrees (-180 to 180)
        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180

        return round(pitch, 2), round(yaw, 2), round(roll, 2)

    @staticmethod
    def estimate_iris_gaze(landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
        """
        Calculate iris pupil horizontal and vertical position ratios within the eye boundaries.
        0.5 = dead center gaze.
        < 0.40 = gaze shifted left.
        > 0.60 = gaze shifted right.
        """
        pts = landmarks.landmark
        if len(pts) < 478:
            return 0.5, 0.5

        # Left eye (indices 33 outer, 133 inner, 468 iris center)
        left_outer = pts[33].x
        left_inner = pts[133].x
        left_iris = pts[468].x
        left_eye_w = abs(left_inner - left_outer) + 1e-6
        left_h_ratio = (left_iris - min(left_outer, left_inner)) / left_eye_w

        # Right eye (indices 362 inner, 263 outer, 473 iris center)
        right_inner = pts[362].x
        right_outer = pts[263].x
        right_iris = pts[473].x
        right_eye_w = abs(right_outer - right_inner) + 1e-6
        right_h_ratio = (right_iris - min(right_inner, right_outer)) / right_eye_w

        avg_h_ratio = float((left_h_ratio + right_h_ratio) / 2.0)

        # Vertical ratio (eyelid top 159, bottom 145)
        top_lid = pts[159].y
        bot_lid = pts[145].y
        iris_y = pts[468].y
        eye_h = abs(bot_lid - top_lid) + 1e-6
        v_ratio = float((iris_y - top_lid) / eye_h)

        return round(avg_h_ratio, 3), round(v_ratio, 3)

    def analyze_gaze(
        self, landmarks, img_w: int, img_h: int, yaw_thresh: float = 14.0, pitch_thresh: float = 12.0
    ) -> Dict[str, Any]:
        """
        Classify gaze direction and looking-away status by combining head pose (yaw/pitch)
        and iris position.
        """
        pitch, yaw, roll = self.estimate_head_pose(landmarks, img_w, img_h)
        iris_h, iris_v = self.estimate_iris_gaze(landmarks, img_w, img_h)

        direction = "CENTER"
        is_looking_away = False

        # Check Horizontal displacement (Yaw or Iris)
        if yaw < -yaw_thresh or iris_h < 0.38:
            direction = "LOOKING_LEFT"
            is_looking_away = True
        elif yaw > yaw_thresh or iris_h > 0.62:
            direction = "LOOKING_RIGHT"
            is_looking_away = True
        # Check Vertical displacement (Pitch or Iris)
        elif pitch > pitch_thresh or iris_v < 0.30:
            direction = "LOOKING_UP"
            is_looking_away = True
        elif pitch < -pitch_thresh or iris_v > 0.70:
            direction = "LOOKING_DOWN"
            is_looking_away = True

        return {
            "gaze_direction": direction,
            "is_looking_away": is_looking_away,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "iris_h": iris_h,
            "iris_v": iris_v
        }

    def process_reference_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze reference photo for Phase 2:
        - Detect faces (reject if != 1)
        - Validate quality
        - Generate reference embedding
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.reference_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "success": False,
                "faces_detected": 0,
                "message": "No face detected. Please ensure your face is well-lit and directly facing the webcam."
            }

        face_count = len(results.multi_face_landmarks)
        if face_count > 1:
            return {
                "success": False,
                "faces_detected": face_count,
                "message": f"Multiple faces detected ({face_count}). Only one person should be present in the photo."
            }

        landmarks = results.multi_face_landmarks[0]
        face_box = self.extract_face_box(landmarks, w, h)
        quality_score, issues = self.assess_quality(image, face_box)

        if quality_score < 30.0 and len(issues) > 0:
            return {
                "success": False,
                "faces_detected": 1,
                "quality_score": quality_score,
                "message": f"Photo quality insufficient: {'; '.join(issues)}"
            }

        embedding = self.compute_embedding(landmarks)

        return {
            "success": True,
            "faces_detected": 1,
            "quality_score": quality_score,
            "embedding": embedding,
            "face_box": face_box,
            "metadata": {
                "width": w,
                "height": h,
                "quality_score": quality_score,
                "issues": issues
            },
            "message": "Reference face successfully validated and embedding generated."
        }

    def process_live_frame(self, image: np.ndarray, reference_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Analyze real-time frame for Phase 3 & Phase 4:
        - Count faces
        - Compute live face embedding & match similarity
        - Compute gaze direction & head pose
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "faces_detected": 0,
                "verification_status": "NO_FACE",
                "similarity": None,
                "is_match": False,
                "gaze_direction": "UNKNOWN",
                "is_looking_away": True,
                "gaze_yaw": None,
                "gaze_pitch": None,
                "face_box": None
            }

        face_count = len(results.multi_face_landmarks)
        if face_count > 1:
            first_box = self.extract_face_box(results.multi_face_landmarks[0], w, h)
            return {
                "faces_detected": face_count,
                "verification_status": "MULTIPLE_FACES",
                "similarity": None,
                "is_match": False,
                "gaze_direction": "UNKNOWN",
                "is_looking_away": True,
                "gaze_yaw": None,
                "gaze_pitch": None,
                "face_box": first_box
            }

        # Exactly 1 face detected
        landmarks = results.multi_face_landmarks[0]
        face_box = self.extract_face_box(landmarks, w, h)
        live_embedding = self.compute_embedding(landmarks)
        similarity = self.compute_similarity(reference_embedding, live_embedding)

        # Verification threshold: similarity >= 0.82 indicates same face identity
        is_match = similarity >= 0.82
        verification_status = "MATCH" if is_match else "NO_MATCH"

        gaze_res = self.analyze_gaze(landmarks, w, h)

        return {
            "faces_detected": 1,
            "verification_status": verification_status,
            "similarity": round(similarity, 4),
            "is_match": is_match,
            "gaze_direction": gaze_res["gaze_direction"],
            "is_looking_away": gaze_res["is_looking_away"],
            "gaze_yaw": gaze_res["yaw"],
            "gaze_pitch": gaze_res["pitch"],
            "face_box": face_box
        }


# Singleton instance
engine = VisionEngine()
