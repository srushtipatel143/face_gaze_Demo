import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
WEIGHT_URLS = {
    config.MODEL_FILES["detector"]:
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    config.MODEL_FILES["recognizer"]:
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure_weights() -> None:
    """Fallback download for local runs. In Docker the files are baked in (section 10)."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for filename, url in WEIGHT_URLS.items():
        dest = WEIGHTS / filename
        if not dest.exists() or dest.stat().st_size < 100_000:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                while chunk := resp.read(1024 * 1024):
                    f.write(chunk)


def row_to_box(row: np.ndarray, img_w: int, img_h: int) -> Dict[str, int]:
    """YuNet row (x, y, w, h, ...) -> clamped integer box."""
    x = max(0, int(row[0]))
    y = max(0, int(row[1]))
    w = max(1, min(img_w - x, int(row[2])))
    h = max(1, min(img_h - y, int(row[3])))
    return {"x": x, "y": y, "width": w, "height": h}


class FaceMatcher:
    """Not thread-safe. VisionEngine guards every call with its lock."""

    def __init__(self) -> None:
        ensure_weights()
        self.detector = cv2.FaceDetectorYN.create(
            str(WEIGHTS / config.MODEL_FILES["detector"]), "", (320, 320),
            score_threshold=config.YUNET_SCORE_THRESHOLD,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(WEIGHTS / config.MODEL_FILES["recognizer"]), ""
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def embed(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(image, face_row)
        return self.recognizer.feature(aligned)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(1, -1)
        b = np.asarray(b, dtype=np.float32).reshape(1, -1)
        return float(self.recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def select_faces(faces: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Section 4 version: first face, raw count. Replaced in section 5."""
    if len(faces) == 0:
        return None, 0
    return faces[0], len(faces)
