"""
Vision Engine for Face Verification and Eye-Gaze Estimation.
Powered by OpenCV YuNet, OpenCV SFace, and MediaPipe Face Mesh.
"""

import base64
import io
import re
from pathlib import Path
from threading import Lock
import urllib.request
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2
import mediapipe as mp

WEIGHTS = Path(__file__).parent / "weights"
SFACE_MATCH_THRESHOLD = 0.363  # Cosine similarity threshold for OpenCV SFace

def ensure_weights():
    """Ensure ONNX model weights exist locally in backend/weights/."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    models = {
        "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    }
    for filename, url in models.items():
        dest = WEIGHTS / filename
        if not dest.exists() or dest.stat().st_size < 1000:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                while chunk := resp.read(1024 * 1024):
                    f.write(chunk)

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
        ensure_weights()
        self._lock = Lock()
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
        # OpenCV YuNet Face Detector
        yunet_file = str(WEIGHTS / "face_detection_yunet_2023mar.onnx")
        self.detector = cv2.FaceDetectorYN.create(
            yunet_file, "", (320, 320),
            score_threshold=0.8,
        )
        # OpenCV SFace Face Recognizer
        sface_file = str(WEIGHTS / "face_recognition_sface_2021dec.onnx")
        self.recognizer = cv2.FaceRecognizerSF.create(
            sface_file, ""
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

    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        """Detect faces in image using OpenCV YuNet. Returns array of shape (N, 15)."""
        h, w = image.shape[:2]
        with self._lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def compute_embedding(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Align and crop face using 5 landmarks from YuNet, then compute 128-d SFace feature embedding."""
        with self._lock:
            aligned = self.recognizer.alignCrop(image, face_row)
            return self.recognizer.feature(aligned)  # shape (1, 128)

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two SFace embeddings."""
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        with self._lock:
            return float(self.recognizer.match(a_arr, b_arr, cv2.FaceRecognizerSF_FR_COSINE))

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
        - Detect faces with YuNet (reject if != 1)
        - Validate quality
        - Generate reference SFace embedding
        """
        h, w = image.shape[:2]
        faces = self.detect_faces(image)

        if len(faces) == 0:
            return {
                "success": False,
                "faces_detected": 0,
                "message": "No face detected. Please ensure your face is well-lit and directly facing the webcam."
            }

        face_count = len(faces)
        if face_count > 1:
            return {
                "success": False,
                "faces_detected": face_count,
                "message": f"Multiple faces detected ({face_count}). Only one person should be present in the photo."
            }

        face_row = faces[0]
        face_box = {
            "x": max(0, int(face_row[0])),
            "y": max(0, int(face_row[1])),
            "width": int(face_row[2]),
            "height": int(face_row[3])
        }
        quality_score, issues = self.assess_quality(image, face_box)

        if quality_score < 30.0 and len(issues) > 0:
            return {
                "success": False,
                "faces_detected": 1,
                "quality_score": quality_score,
                "message": f"Photo quality insufficient: {'; '.join(issues)}"
            }

        embedding = self.compute_embedding(image, face_row)

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
        - Count faces using YuNet
        - Compute live face embedding & match similarity using SFace
        - Compute gaze direction & head pose using MediaPipe Face Mesh
        """
        h, w = image.shape[:2]
        faces = self.detect_faces(image)
        face_count = len(faces)

        if face_count == 0:
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

        if face_count > 1:
            first_face = faces[0]
            first_box = {
                "x": max(0, int(first_face[0])),
                "y": max(0, int(first_face[1])),
                "width": int(first_face[2]),
                "height": int(first_face[3])
            }
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

        # Exactly 1 face detected by YuNet
        face_row = faces[0]
        face_box = {
            "x": max(0, int(face_row[0])),
            "y": max(0, int(face_row[1])),
            "width": int(face_row[2]),
            "height": int(face_row[3])
        }

        # Compute SFace embedding and cosine similarity
        live_embedding = self.compute_embedding(image, face_row)
        similarity = self.compute_similarity(reference_embedding, live_embedding)

        # Head pose & eye-gaze tracking via MediaPipe Face Mesh
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self._lock:
            mesh_results = self.face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            gaze_res = self.analyze_gaze(landmarks, w, h)
        else:
            gaze_res = {
                "gaze_direction": "CENTER",
                "is_looking_away": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "iris_h": 0.5,
                "iris_v": 0.5
            }

        yaw = gaze_res.get("yaw", 0.0)
        pitch = gaze_res.get("pitch", 0.0)

        # Verification threshold: similarity >= 0.363 indicates same face identity
        # Extreme head angles (>35° yaw or pitch) degrade facial recognition accuracy;
        # mark as UNKNOWN rather than NO_MATCH to prevent false alarms.
        is_match = similarity >= SFACE_MATCH_THRESHOLD
        if is_match:
            verification_status = "MATCH"
        elif (yaw is not None and abs(yaw) > 35.0) or (pitch is not None and abs(pitch) > 35.0):
            verification_status = "UNKNOWN"
        else:
            verification_status = "NO_MATCH"

        return {
            "faces_detected": 1,
            "verification_status": verification_status,
            "similarity": round(float(similarity), 4),
            "is_match": is_match,
            "gaze_direction": gaze_res["gaze_direction"],
            "is_looking_away": gaze_res["is_looking_away"],
            "gaze_yaw": gaze_res["yaw"],
            "gaze_pitch": gaze_res["pitch"],
            "face_box": face_box
        }


# Singleton instance
engine = VisionEngine()
