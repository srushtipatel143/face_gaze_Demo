from collections import deque
import statistics
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config
from records import (CENTER, LOOKING_LEFT, LOOKING_RIGHT, LOOKING_UP, LOOKING_DOWN)

# 3D generic facial model points for head pose estimation (SolvePnP)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135),       # Left eye corner (index 33)
    (225.0, 170.0, -135),        # Right eye corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
], dtype=np.float64)


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
    """
    Estimate pitch, yaw, and roll in degrees using SolvePnP on key facial landmarks.
    Pitch: positive = looking down, negative = looking up.
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

    if yaw > 90:
        yaw = yaw - 180
    elif yaw < -90:
        yaw = yaw + 180

    if roll > 90:
        roll = roll - 180
    elif roll < -90:
        roll = roll + 180

    return round(pitch, 2), round(yaw, 2), round(roll, 2)


def eye_openness(pts, top: int, bottom: int, outer: int, inner: int) -> float:
    """Compute eyelid vertical gap divided by eye horizontal width."""
    gap = abs(pts[bottom].y - pts[top].y)
    width = abs(pts[inner].x - pts[outer].x) + 1e-6
    return float(gap / width)


def estimate_eyes(landmarks) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Returns (iris_h, iris_v, openness, eyes_open). Copy body of estimate_iris_gaze.
    Change only the early return: if fewer than 478 landmarks, return (None, None, None, None)."""
    pts = landmarks.landmark
    if len(pts) < 478:
        return None, None, None, None

    # Openness calculation for both eyes
    # Left eye: top lid 159, bottom lid 145, outer corner 33, inner corner 133
    left_open = eye_openness(pts, 159, 145, 33, 133)
    # Right eye: top lid 386, bottom lid 374, outer corner 263, inner corner 362
    right_open = eye_openness(pts, 386, 374, 263, 362)
    avg_openness = float((left_open + right_open) / 2.0)
    eyes_open = avg_openness >= config.EYES_OPEN_MIN

    # Horizontal ratio (Left eye: 33 outer, 133 inner, 468 iris; Right eye: 362 inner, 263 outer, 473 iris)
    left_outer = pts[33].x
    left_inner = pts[133].x
    left_iris = pts[468].x
    left_eye_w = abs(left_inner - left_outer) + 1e-6
    left_h_ratio = (left_iris - min(left_outer, left_inner)) / left_eye_w

    right_inner = pts[362].x
    right_outer = pts[263].x
    right_iris = pts[473].x
    right_eye_w = abs(right_outer - right_inner) + 1e-6
    right_h_ratio = (right_iris - min(right_inner, right_outer)) / right_eye_w

    avg_h_ratio = float((left_h_ratio + right_h_ratio) / 2.0)

    # Vertical ratio using both eyes (Left iris 468, lids 159/145; Right iris 473, lids 386/374)
    left_eye_h = abs(pts[145].y - pts[159].y) + 1e-6
    left_v_ratio = (pts[468].y - min(pts[159].y, pts[145].y)) / left_eye_h

    right_eye_h = abs(pts[374].y - pts[386].y) + 1e-6
    right_v_ratio = (pts[473].y - min(pts[386].y, pts[374].y)) / right_eye_h

    avg_v_ratio = float((left_v_ratio + right_v_ratio) / 2.0)

    return round(avg_h_ratio, 3), round(avg_v_ratio, 3), round(avg_openness, 3), eyes_open


def classify_gaze(yaw: float, pitch: float, iris_h: float, iris_v: float,
                  baseline: Dict[str, float]) -> Tuple[str, bool]:
    """Pure. Directions are from the USER's point of view on an UNMIRRORED camera image."""
    d_yaw = yaw - baseline["yaw"]
    d_pitch = pitch - baseline["pitch"]
    d_ih = iris_h - baseline["iris_h"]
    d_iv = iris_v - baseline["iris_v"]

    if d_yaw < -config.GAZE_YAW_AWAY_DEG or d_ih > config.GAZE_IRIS_H_AWAY:
        return LOOKING_LEFT, True
    if d_yaw > config.GAZE_YAW_AWAY_DEG or d_ih < -config.GAZE_IRIS_H_AWAY:
        return LOOKING_RIGHT, True
    if d_pitch > config.GAZE_PITCH_DOWN_DEG or d_iv > config.GAZE_IRIS_V_DOWN or (
            d_pitch > config.GAZE_PITCH_DOWN_SOFT_DEG and d_iv > (config.GAZE_IRIS_V_DOWN * 0.5)):
        return LOOKING_DOWN, True
    if d_pitch < -config.GAZE_PITCH_UP_DEG or d_iv < -config.GAZE_IRIS_V_UP or (
            d_pitch < -config.GAZE_PITCH_UP_SOFT_DEG and d_iv < -(config.GAZE_IRIS_V_UP * 0.5)):
        return LOOKING_UP, True
    return CENTER, False


def _median_values(samples) -> Dict[str, float]:
    return {
        "yaw": round(statistics.median(s[0] for s in samples), 2),
        "pitch": round(statistics.median(s[1] for s in samples), 2),
        "iris_h": round(statistics.median(s[2] for s in samples), 3),
        "iris_v": round(statistics.median(s[3] for s in samples), 3),
    }


class GazeBaseline:
    def __init__(self) -> None:
        self.samples = []                     # (yaw, pitch, iris_h, iris_v) during calibration
        self.first_t: Optional[float] = None
        self.values: Optional[Dict[str, float]] = None
        self.unstable = False
        self.trailing = deque(maxlen=config.TRAILING_BASELINE_WINDOW)

    @property
    def ready(self) -> bool:
        return self.values is not None

    def progress(self, t_s: float) -> float:
        if self.ready:
            return 1.0
        if self.first_t is None:
            return 0.0
        time_part = min(1.0, (t_s - self.first_t) / config.CALIBRATION_SECONDS)
        count_part = min(1.0, len(self.samples) / config.CALIBRATION_MIN_SAMPLES)
        return round(min(time_part, count_part), 2)

    def add_calibration_sample(self, t_s, yaw, pitch, iris_h, iris_v) -> None:
        if self.ready:
            return
        if self.first_t is None:
            self.first_t = t_s
        self.samples.append((yaw, pitch, iris_h, iris_v))
        if (t_s - self.first_t >= config.CALIBRATION_SECONDS
                and len(self.samples) >= config.CALIBRATION_MIN_SAMPLES):
            self.values = _median_values(self.samples)
            yaw_std = float(np.std([s[0] for s in self.samples]))
            pitch_std = float(np.std([s[1] for s in self.samples]))
            self.unstable = (yaw_std > config.CALIBRATION_MAX_STD_DEG
                             or pitch_std > config.CALIBRATION_MAX_STD_DEG)

    def add_on_screen_sample(self, yaw, pitch, iris_h, iris_v) -> None:
        if not self.ready:
            return
        self.trailing.append((yaw, pitch, iris_h, iris_v))
        if len(self.trailing) == self.trailing.maxlen:
            self.values = _median_values(self.trailing)

    def summary(self) -> Dict[str, bool]:
        return {"completed": self.ready, "unstable": self.unstable}

