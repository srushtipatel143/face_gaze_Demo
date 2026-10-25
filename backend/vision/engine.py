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
            issues.append("Face is too small or too far from the camera. Move closer.")
    else:
        face_crop = image

    if face_crop.size == 0:
        face_crop = image

    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    laplacian_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    if laplacian_var < config.REF_MIN_SHARPNESS:
        issues.append("Face is blurry. Hold still and try again.")

    mean_brightness = float(np.mean(face_gray))
    if mean_brightness < config.REF_MIN_BRIGHTNESS:
        issues.append("Face lighting is too dark. Face a light source.")
    elif mean_brightness > config.REF_MAX_BRIGHTNESS:
        issues.append("Face lighting is too bright. Reduce direct glare.")

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

    def analyze_frame(self, image: np.ndarray, reference_embedding: Any,
                      session_reference: Optional[Any] = None) -> FrameAnalysis:
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
            if session_reference is not None:
                sim_session = self.face.similarity(session_reference, live)
                similarity = max(similarity, sim_session)
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
            openness=openness, eyes_open=eyes_open, thumbnail=thumb, embedding=live,
        )

    def analyze_reference(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze reference photo with structured 7-item checklist."""
        h, w = image.shape[:2]
        with self._lock:
            faces = self.face.detect(image)
            main, count = select_faces(faces)

            # Checks: always 7 items in this exact order
            checks: List[Dict[str, Any]] = []

            # 1. face_detected
            face_detected_passed = count >= 1
            checks.append({
                "id": "face_detected",
                "label": "Face detected",
                "passed": face_detected_passed,
                "detail": None if face_detected_passed else "No face found. Face the camera in good light."
            })

            # 2. single_face
            if not face_detected_passed:
                single_face_passed = None
                single_face_detail = None
            else:
                single_face_passed = count == 1
                single_face_detail = None if single_face_passed else "Only you should be in the photo."

            checks.append({
                "id": "single_face",
                "label": "Only one person",
                "passed": single_face_passed,
                "detail": single_face_detail
            })

            # If face_detected or single_face fails, all later checks are null
            if not (face_detected_passed and single_face_passed):
                for check_id, check_label in [
                    ("face_size", "Close enough to the camera"),
                    ("sharpness", "Image is sharp"),
                    ("lighting", "Good lighting"),
                    ("facing_camera", "Facing the camera"),
                    ("eyes_open", "Eyes open"),
                ]:
                    checks.append({
                        "id": check_id,
                        "label": check_label,
                        "passed": None,
                        "detail": None
                    })

                issues = [c["detail"] for c in checks if c["passed"] is False and c["detail"]]
                return {
                    "success": False,
                    "faces_detected": count,
                    "quality_score": None,
                    "checks": checks,
                    "issues": issues,
                    "embedding": None,
                    "metadata": {
                        "width": w,
                        "height": h,
                        "quality_score": None,
                        "checks": checks,
                        "issues": issues,
                    }
                }

            # At this point, count == 1 and main face is found
            face_box = row_to_box(main, w, h)
            fx = max(0, face_box["x"])
            fy = max(0, face_box["y"])
            fw = max(1, min(w - fx, face_box["width"]))
            fh = max(1, min(h - fy, face_box["height"]))
            face_crop = image[fy:fy + fh, fx:fx + fw]
            if face_crop.size == 0:
                face_crop = image

            # 3. face_size
            face_area = fw * fh
            img_area = w * h
            face_ratio = face_area / float(img_area)
            face_size_passed = bool(
                face_ratio >= config.REF_MIN_FACE_RATIO
                and fw >= config.REF_MIN_FACE_PX
                and fh >= config.REF_MIN_FACE_PX
            )
            checks.append({
                "id": "face_size",
                "label": "Close enough to the camera",
                "passed": face_size_passed,
                "detail": None if face_size_passed else "Move closer to the camera."
            })

            # 4. sharpness
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
            sharpness_passed = bool(laplacian_var >= config.REF_MIN_SHARPNESS)
            checks.append({
                "id": "sharpness",
                "label": "Image is sharp",
                "passed": sharpness_passed,
                "detail": None if sharpness_passed else "The photo is blurry. Hold still."
            })

            # 5. lighting
            mean_brightness = float(np.mean(face_gray))
            lighting_passed = bool(config.REF_MIN_BRIGHTNESS <= mean_brightness <= config.REF_MAX_BRIGHTNESS)
            if lighting_passed:
                lighting_detail = None
            elif mean_brightness < config.REF_MIN_BRIGHTNESS:
                lighting_detail = "Too dark: face a light source."
            else:
                lighting_detail = "Too bright: reduce glare."
            checks.append({
                "id": "lighting",
                "label": "Good lighting",
                "passed": lighting_passed,
                "detail": lighting_detail
            })

            # Quality score
            sharpness_score = min(60.0, (laplacian_var / 120) * 60.0)
            brightness_score = 40.0 - abs(mean_brightness - 128.0) * (40.0 / 128.0)
            quality_score = max(5.0, min(100.0, round(sharpness_score + brightness_score, 1)))

            # 6. facing_camera & 7. eyes_open
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mesh_results = self.mesh.process(rgb)
            landmarks = pick_landmarks(mesh_results, face_box, w, h)

            if landmarks is None:
                facing_camera_passed = False
                facing_camera_detail = "Look straight at the camera."
                eyes_open_passed = False
                eyes_open_detail = "Keep your eyes open."
            else:
                pitch, yaw, _ = estimate_head_pose(landmarks, w, h)
                facing_camera_passed = bool(
                    abs(yaw) <= config.REF_MAX_POSE_DEG
                    and abs(pitch) <= config.REF_MAX_POSE_DEG
                )
                facing_camera_detail = None if facing_camera_passed else "Look straight at the camera."

                _, _, openness, eyes_open = estimate_eyes(landmarks)
                eyes_open_passed = bool(eyes_open)
                eyes_open_detail = None if eyes_open_passed else "Keep your eyes open."

            checks.append({
                "id": "facing_camera",
                "label": "Facing the camera",
                "passed": facing_camera_passed,
                "detail": facing_camera_detail
            })

            checks.append({
                "id": "eyes_open",
                "label": "Eyes open",
                "passed": eyes_open_passed,
                "detail": eyes_open_detail
            })

            success = all(c["passed"] is True for c in checks)
            issues = [c["detail"] for c in checks if c["passed"] is False and c["detail"]]

            embedding = self.face.embed(image, main) if success else None

            return {
                "success": success,
                "faces_detected": 1,
                "quality_score": quality_score,
                "checks": checks,
                "issues": issues,
                "embedding": embedding,
                "face_box": face_box,
                "metadata": {
                    "width": w,
                    "height": h,
                    "quality_score": quality_score,
                    "checks": checks,
                    "issues": issues,
                }
            }


engine = VisionEngine()
