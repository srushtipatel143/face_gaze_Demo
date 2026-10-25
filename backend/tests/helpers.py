import os
import sys

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from records import FrameRecord, FrameAnalysis

CAL_OK = {"completed": True, "unstable": False}   # pass as evaluate(..., calibration=CAL_OK)


def rec(i, identity="MATCH", away=False, direction=None, faces=1, calibrating=False,
        sim=0.6, eyes_open=True, yaw=None, pitch=None, period=0.2):
    """Build a FrameRecord at t = i * period.
    Default yaw/pitch vary slightly per frame so the static-face rule does NOT trigger."""
    if faces != 1:
        away = None
    if yaw is None:
        yaw = (i % 5) * 0.8
    if pitch is None:
        pitch = (i % 7) * 0.6
    gaze = "CALIBRATING" if calibrating else (direction or ("CENTER" if away is False else "UNKNOWN"))
    return FrameRecord(
        t_s=round(i * period, 3),
        faces_detected=faces,
        identity=identity,
        similarity=sim if faces == 1 else None,
        yaw=yaw if faces == 1 else None,
        pitch=pitch if faces == 1 else None,
        iris_h=0.5,
        iris_v=0.5,
        eyes_open=eyes_open,
        gaze_direction=gaze,
        is_looking_away=None if calibrating else away,
        calibrating=calibrating,
    )
