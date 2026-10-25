"""
Central configuration. Every tunable number lives here.
Each value can be overridden with an environment variable of the same name.
"""
import os


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


RULES_VERSION = "2026.10.1"          # bump when any verdict rule or threshold changes

# ---- Frames -------------------------------------------------------------------
FRAME_PERIOD_S = _f("FRAME_PERIOD_S", 0.2)            # frontend sends one frame every 200 ms
THUMBNAIL_MAX_DIM = _i("THUMBNAIL_MAX_DIM", 240)
THUMBNAIL_QUALITY = _i("THUMBNAIL_QUALITY", 65)

# ---- Face detection and matching ---------------------------------------------
YUNET_SCORE_THRESHOLD = _f("YUNET_SCORE_THRESHOLD", 0.8)        # current
SFACE_MATCH_THRESHOLD = _f("SFACE_MATCH_THRESHOLD", 0.363)      # current
SECOND_FACE_MIN_AREA_RATIO = _f("SECOND_FACE_MIN_AREA_RATIO", 0.30)  # new (section 5)
UNCERTAIN_POSE_DEG = _f("UNCERTAIN_POSE_DEG", 35.0)              # current
SIMILARITY_NEAR_MARGIN = _f("SIMILARITY_NEAR_MARGIN", 0.05)      # new (section 8)
SESSION_REFERENCE_ENABLED = os.getenv("SESSION_REFERENCE_ENABLED", "false").lower() in ("true", "1", "yes")
SESSION_REFERENCE_FRAMES = _i("SESSION_REFERENCE_FRAMES", 5)
SESSION_REFERENCE_MIN_SIMILARITY = _f("SESSION_REFERENCE_MIN_SIMILARITY", 0.5)

# ---- Reference photo checks ---------------------------------------------------
REF_MIN_SHARPNESS = _f("REF_MIN_SHARPNESS", 40.0)       # current
REF_MIN_BRIGHTNESS = _f("REF_MIN_BRIGHTNESS", 45.0)     # current
REF_MAX_BRIGHTNESS = _f("REF_MAX_BRIGHTNESS", 220.0)    # current
REF_MIN_FACE_RATIO = _f("REF_MIN_FACE_RATIO", 0.05)     # current
REF_MIN_FACE_PX = _i("REF_MIN_FACE_PX", 80)             # current
REF_MAX_POSE_DEG = _f("REF_MAX_POSE_DEG", 15.0)         # current

# ---- Eyes and gaze ------------------------------------------------------------
EYES_OPEN_MIN = _f("EYES_OPEN_MIN", 0.18)               # current
GAZE_YAW_AWAY_DEG = _f("GAZE_YAW_AWAY_DEG", 20.0)       # current
GAZE_IRIS_H_AWAY = _f("GAZE_IRIS_H_AWAY", 0.15)         # current
GAZE_PITCH_DOWN_DEG = _f("GAZE_PITCH_DOWN_DEG", 15.0)   # current
GAZE_PITCH_DOWN_SOFT_DEG = _f("GAZE_PITCH_DOWN_SOFT_DEG", 8.0)    # current
GAZE_IRIS_V_DOWN = _f("GAZE_IRIS_V_DOWN", 0.16)         # current
GAZE_PITCH_UP_DEG = _f("GAZE_PITCH_UP_DEG", 18.0)       # current
GAZE_PITCH_UP_SOFT_DEG = _f("GAZE_PITCH_UP_SOFT_DEG", 10.0)       # current
GAZE_IRIS_V_UP = _f("GAZE_IRIS_V_UP", 0.16)             # current

# ---- Calibration / baseline ---------------------------------------------------
CALIBRATION_SECONDS = _f("CALIBRATION_SECONDS", 5.0)    # new (section 6)
CALIBRATION_MIN_SAMPLES = _i("CALIBRATION_MIN_SAMPLES", 20)   # new (section 6)
CALIBRATION_MAX_STD_DEG = _f("CALIBRATION_MAX_STD_DEG", 8.0)  # current
TRAILING_BASELINE_WINDOW = _i("TRAILING_BASELINE_WINDOW", 100)  # current
POST_CALIBRATION_CHECK_FRAMES = _i("POST_CALIBRATION_CHECK_FRAMES", 30)  # current
POST_CALIBRATION_MIN_FRAMES = _i("POST_CALIBRATION_MIN_FRAMES", 15)      # current
POST_CALIBRATION_AWAY_RATIO = _f("POST_CALIBRATION_AWAY_RATIO", 0.5)     # current

# ---- Intervals ----------------------------------------------------------------
GAZE_OPEN_AFTER = _i("GAZE_OPEN_AFTER", 5)              # current
GAZE_CLOSE_AFTER = _i("GAZE_CLOSE_AFTER", 3)            # current
IDENTITY_OPEN_AFTER = _i("IDENTITY_OPEN_AFTER", 3)      # current
IDENTITY_CLOSE_AFTER = _i("IDENTITY_CLOSE_AFTER", 2)    # current
MAX_UNKNOWN_GAP_S = _f("MAX_UNKNOWN_GAP_S", 1.0)        # current

# ---- Verdicts -----------------------------------------------------------------
FACE_DIFFERENT_PERSON_FAIL_S = _f("FACE_DIFFERENT_PERSON_FAIL_S", 2.0)  # current
FACE_MULTIPLE_FACES_FAIL_S = _f("FACE_MULTIPLE_FACES_FAIL_S", 5.0)      # current
FACE_NO_FACE_FAIL_PCT = _f("FACE_NO_FACE_FAIL_PCT", 30.0)               # current
FACE_NO_FACE_REVIEW_PCT = _f("FACE_NO_FACE_REVIEW_PCT", 15.0)           # current
FACE_IDENTITY_UNKNOWN_REVIEW_PCT = _f("FACE_IDENTITY_UNKNOWN_REVIEW_PCT", 25.0)  # current
GAZE_AWAY_FAIL_PCT = _f("GAZE_AWAY_FAIL_PCT", 35.0)                     # current
GAZE_AWAY_REVIEW_PCT = _f("GAZE_AWAY_REVIEW_PCT", 20.0)                 # current
GAZE_LONG_AWAY_S = _f("GAZE_LONG_AWAY_S", 10.0)                         # current
GAZE_UNKNOWN_REVIEW_PCT = _f("GAZE_UNKNOWN_REVIEW_PCT", 25.0)           # current

# ---- Static face (possible photo) heuristic -------------------------------------
STATIC_FACE_MIN_DURATION_S = _f("STATIC_FACE_MIN_DURATION_S", 30.0)     # current
STATIC_FACE_MIN_POSES = _i("STATIC_FACE_MIN_POSES", 100)                # current
STATIC_FACE_WINDOW = _i("STATIC_FACE_WINDOW", 150)                      # current
STATIC_FACE_MAX_POSE_STD = _f("STATIC_FACE_MAX_POSE_STD", 0.6)          # current

# ---- Sessions and database ----------------------------------------------------
SESSION_MAX_AGE_S = _i("SESSION_MAX_AGE_S", 3600)
DB_CONNECT_ATTEMPTS = _i("DB_CONNECT_ATTEMPTS", 10)
DB_CONNECT_RETRY_S = _f("DB_CONNECT_RETRY_S", 2.0)

MODEL_FILES = {
    "detector": "face_detection_yunet_2023mar.onnx",
    "recognizer": "face_recognition_sface_2021dec.onnx",
    "landmarks": "mediapipe-0.10.14 face_mesh (refine_landmarks)",
}


def snapshot() -> dict:
    """All upper-case settings as a plain dict. Stored with every report."""
    return {k: v for k, v in globals().items() if k.isupper()}
