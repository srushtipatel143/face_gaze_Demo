import base64
import threading
from typing import Any, Dict, Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np

import config
from records import FrameAnalysis, MATCH, NO_MATCH, UNKNOWN, NO_FACE, MULTIPLE_FACES
from vision.face import FaceMatcher, row_to_box, select_faces
from vision.gaze import estimate_head_pose, estimate_eyes


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


def make_thumbnail(image: np.ndarray) -> Optional[str]:
    """Resize image preserving aspect ratio and encode to base64 JPEG data URL."""
    try:
        h, w = image.shape[:2]
        max_dim = config.THUMBNAIL_MAX_DIM
        quality = config.THUMBNAIL_QUALITY
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            resized = image
        success, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return None
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def pick_landmarks(mesh_results, box: Dict[str, int], img_w: int, img_h: int):
    if not mesh_results.multi_face_landmarks:
        return None
    cx = box["x"] + box["width"] / 2.0
    cy = box["y"] + box["height"] / 2.0
    best, best_d = None, None
    for lm in mesh_results.multi_face_landmarks:
        xs = [p.x * img_w for p in lm.landmark]
        ys = [p.y * img_h for p in lm.landmark]
        mx = (min(xs) + max(xs)) / 2.0
        my = (min(ys) + max(ys)) / 2.0
        d = (mx - cx) ** 2 + (my - cy) ** 2
        if best_d is None or d < best_d:
            best, best_d = lm, d
    # Reject if the closest mesh face is far away (more than one box width)
    if best_d is not None and best_d ** 0.5 > box["width"]:
        return None
    return best


def assess_quality(image: np.ndarray, face_box: Optional[Dict[str, int]] = None) -> Tuple[float, List[str]]:
    """Evaluate image brightness, sharpness/blur, and face area ratio on face crop."""
    issues = []
    h, w = image.shape[:2]

    if face_box:
        fx = max(0, face_box["x"])
        fy = max(0, face_box["y"])
        fw = max(1, min(w - fx, face_box["width"]))
        fh = max(1, min(h - fy, face_box["height"]))
        face_crop = image[fy:fy + fh, fx:fx + fw]
        face_area = fw * fh
        img_area = w * h
        face_ratio = face_area / float(img_area)
        if face_ratio < config.REF_MIN_FACE_RATIO or fw < config.REF_MIN_FACE_PX or fh < config.REF_MIN_FACE_PX:
            issues.append("Face is too small or too far from the camera. Please move closer.")
    else:
        face_crop = image

    if face_crop.size == 0:
        face_crop = image

    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    laplacian_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    if laplacian_var < config.REF_MIN_SHARPNESS:
        issues.append(f"Face is blurry (sharpness: {laplacian_var:.1f}, min: {config.REF_MIN_SHARPNESS}). Please hold still and refocus.")

    mean_brightness = float(np.mean(face_gray))
    if mean_brightness < config.REF_MIN_BRIGHTNESS:
        issues.append(f"Face lighting is too dark (brightness: {mean_brightness:.1f}/255, min: {config.REF_MIN_BRIGHTNESS}). Please face a light source.")
    elif mean_brightness > config.REF_MAX_BRIGHTNESS:
        issues.append(f"Face lighting is overexposed (brightness: {mean_brightness:.1f}/255, max: {config.REF_MAX_BRIGHTNESS}). Please reduce direct glare.")

    sharpness_score = min(60.0, (laplacian_var / 120) * 60.0)
    brightness_score = 40.0 - abs(mean_brightness - 128.0) * (40.0 / 128.0)
    quality_score = max(5.0, min(100.0, round(sharpness_score + brightness_score, 1)))

    return quality_score, issues


class VisionEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.face = FaceMatcher()
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=4,
            refine_landmarks=True, min_detection_confidence=0.5,
        )

    def analyze_frame(self, image: np.ndarray, reference_embedding: Any) -> FrameAnalysis:
        h, w = image.shape[:2]
        thumb = make_thumbnail(image)
        with self._lock:
            faces = self.face.detect(image)
            main, count = select_faces(faces)
            if count == 0:
                return FrameAnalysis(faces_detected=0, identity=NO_FACE, thumbnail=thumb)
            box = row_to_box(main, w, h)
            if count > 1:
                return FrameAnalysis(faces_detected=count, identity=MULTIPLE_FACES,
                                     face_box=box, thumbnail=thumb)
            live = self.face.embed(image, main)
            similarity = self.face.similarity(reference_embedding, live)
            mesh_results = self.mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks = pick_landmarks(mesh_results, box, w, h)
        yaw = pitch = roll = iris_h = iris_v = openness = eyes_open = None
        if landmarks is not None:
            pitch, yaw, roll = estimate_head_pose(landmarks, w, h)
            iris_h, iris_v, openness, eyes_open = estimate_eyes(landmarks)

        if similarity >= config.SFACE_MATCH_THRESHOLD:
            identity = MATCH
        elif yaw is not None and (abs(yaw) > config.UNCERTAIN_POSE_DEG
                                  or abs(pitch) > config.UNCERTAIN_POSE_DEG):
            identity = UNKNOWN
        else:
            identity = NO_MATCH

        return FrameAnalysis(
            faces_detected=1, identity=identity, similarity=round(similarity, 4),
            face_box=box, yaw=yaw, pitch=pitch, roll=roll, iris_h=iris_h, iris_v=iris_v,
            openness=openness, eyes_open=eyes_open, thumbnail=thumb,
        )

    def analyze_reference(self, image: np.ndarray) -> Dict[str, Any]:
        """Port process_reference_image() here unchanged for now (section 14 rewrites it).
        Use self.face / self.mesh / self._lock instead of the old attributes."""
        h, w = image.shape[:2]
        with self._lock:
            faces = self.face.detect(image)
            main, count = select_faces(faces)

            if count == 0:
                return {
                    "success": False,
                    "faces_detected": 0,
                    "issues": ["No face detected"],
                    "message": "No face detected. Please ensure your face is well-lit and directly facing the webcam."
                }

            if count > 1:
                return {
                    "success": False,
                    "faces_detected": count,
                    "issues": [f"Multiple faces detected ({count})"],
                    "message": f"Multiple faces detected ({count}). Only one person should be present in the photo."
                }

            face_box = row_to_box(main, w, h)
            quality_score, issues = assess_quality(image, face_box)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mesh_results = self.mesh.process(rgb)
            landmarks = pick_landmarks(mesh_results, face_box, w, h)
            if landmarks is None:
                issues.append("Could not detect facial landmarks. Ensure face is unobstructed and facing camera.")
            else:
                pitch, yaw, _ = estimate_head_pose(landmarks, w, h)
                if abs(yaw) > config.REF_MAX_POSE_DEG or abs(pitch) > config.REF_MAX_POSE_DEG:
                    issues.append(f"Face is turned away (yaw: {yaw:.1f}°, pitch: {pitch:.1f}°, maximum ±{config.REF_MAX_POSE_DEG}°). Please look straight at the camera.")
                _, _, openness, eyes_open = estimate_eyes(landmarks)
                if not eyes_open:
                    issues.append(f"Eyes appear closed or blinking (openness: {openness:.2f}, min: {config.EYES_OPEN_MIN}). Please keep eyes open.")

            if len(issues) > 0:
                return {
                    "success": False,
                    "faces_detected": 1,
                    "quality_score": quality_score,
                    "issues": issues,
                    "message": f"Reference photo rejected: {'; '.join(issues)}"
                }

            embedding = self.face.embed(image, main)

            return {
                "success": True,
                "faces_detected": 1,
                "quality_score": quality_score,
                "embedding": embedding,
                "face_box": face_box,
                "issues": [],
                "metadata": {
                    "width": w,
                    "height": h,
                    "quality_score": quality_score,
                    "issues": issues
                },
                "message": "Reference face successfully validated and embedding generated."
            }


engine = VisionEngine()
