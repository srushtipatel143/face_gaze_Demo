import os
import sys
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from records import CENTER, LOOKING_LEFT, LOOKING_RIGHT, LOOKING_UP, LOOKING_DOWN
from vision.gaze import classify_gaze, GazeBaseline


def test_classify_gaze_directions():
    """Baseline zeros/0.5; yaw -25; yaw +25; pitch -20; pitch +20; small deltas -> LEFT, RIGHT, DOWN, UP, CENTER."""
    base = {"yaw": 0.0, "pitch": 0.0, "iris_h": 0.5, "iris_v": 0.5}

    # 1. Yaw -25 degrees -> LEFT
    dir_left, away_left = classify_gaze(yaw=-25.0, pitch=0.0, iris_h=0.5, iris_v=0.5, baseline=base)
    assert dir_left == LOOKING_LEFT
    assert away_left is True

    # 2. Yaw +25 degrees -> RIGHT
    dir_right, away_right = classify_gaze(yaw=25.0, pitch=0.0, iris_h=0.5, iris_v=0.5, baseline=base)
    assert dir_right == LOOKING_RIGHT
    assert away_right is True

    # 3. Pitch +20 degrees -> DOWN
    dir_down, away_down = classify_gaze(yaw=0.0, pitch=20.0, iris_h=0.5, iris_v=0.5, baseline=base)
    assert dir_down == LOOKING_DOWN
    assert away_down is True

    # 4. Pitch -20 degrees -> UP
    dir_up, away_up = classify_gaze(yaw=0.0, pitch=-20.0, iris_h=0.5, iris_v=0.5, baseline=base)
    assert dir_up == LOOKING_UP
    assert away_up is True

    # 5. Small deltas (natural jitter) -> CENTER
    dir_center, away_center = classify_gaze(yaw=2.0, pitch=1.5, iris_h=0.52, iris_v=0.48, baseline=base)
    assert dir_center == CENTER
    assert away_center is False


def test_classify_gaze_soft_thresholds():
    """Soft pitch thresholds combined with iris vertical deviation, and pure iris gaze."""
    base = {"yaw": 0.0, "pitch": 0.0, "iris_h": 0.5, "iris_v": 0.5}

    # Soft down: pitch 9.0 (> 8.0) and iris_v 0.60 (delta 0.10 > 0.08)
    dir_s_down, away_s_down = classify_gaze(yaw=0.0, pitch=9.0, iris_h=0.5, iris_v=0.60, baseline=base)
    assert dir_s_down == LOOKING_DOWN
    assert away_s_down is True

    # Soft up: pitch -11.0 (< -10.0) and iris_v 0.40 (delta -0.10 < -0.08)
    dir_s_up, away_s_up = classify_gaze(yaw=0.0, pitch=-11.0, iris_h=0.5, iris_v=0.40, baseline=base)
    assert dir_s_up == LOOKING_UP
    assert away_s_up is True

    # Pure iris down (no head pitch movement) -> DOWN
    dir_i_down, away_i_down = classify_gaze(yaw=0.0, pitch=0.0, iris_h=0.5, iris_v=0.68, baseline=base)
    assert dir_i_down == LOOKING_DOWN
    assert away_i_down is True

    # Pure iris up (no head pitch movement) -> UP
    dir_i_up, away_i_up = classify_gaze(yaw=0.0, pitch=0.0, iris_h=0.5, iris_v=0.32, baseline=base)
    assert dir_i_up == LOOKING_UP
    assert away_i_up is True


def test_baseline_completes_after_time_and_samples():
    """25 samples over 4 s; then more until 5 s -> not ready at 4 s; ready after 5 s; progress 1.0."""
    baseline = GazeBaseline()
    assert not baseline.ready
    assert baseline.progress(0.0) == 0.0

    # Feed 25 samples over 4.0 seconds (e.g. at intervals of 0.16s)
    # Samples count is 25 (>= 20), but elapsed time is only 4.0s (< 5.0s)
    for i in range(25):
        t = i * (4.0 / 24.0)
        baseline.add_calibration_sample(t, yaw=0.5, pitch=-1.0, iris_h=0.5, iris_v=0.5)

    assert not baseline.ready
    assert baseline.progress(4.0) < 1.0

    # Feed additional samples past 5.0s (e.g. t = 5.1s)
    baseline.add_calibration_sample(5.1, yaw=0.5, pitch=-1.0, iris_h=0.5, iris_v=0.5)

    assert baseline.ready
    assert baseline.progress(5.1) == 1.0
    assert baseline.values == {"yaw": 0.5, "pitch": -1.0, "iris_h": 0.5, "iris_v": 0.5}
    assert baseline.unstable is False
    assert baseline.summary() == {"completed": True, "unstable": False}


def test_baseline_unstable_detection():
    """High variance during calibration marks baseline as unstable."""
    baseline = GazeBaseline()
    for i in range(30):
        t = i * 0.2
        # Severe yaw swinging between -20 and +20 (std > 8.0 deg)
        yaw_val = 20.0 if i % 2 == 0 else -20.0
        baseline.add_calibration_sample(t, yaw=yaw_val, pitch=0.0, iris_h=0.5, iris_v=0.5)

    assert baseline.ready
    assert baseline.unstable is True
    assert baseline.summary()["unstable"] is True


def test_baseline_trailing_ignores_away():
    """Ready baseline; feed only away samples via classify; window must not change -> values unchanged."""
    baseline = GazeBaseline()
    # Initialize baseline to ready state with yaw=0.0, pitch=0.0
    for i in range(25):
        t = i * 0.25
        baseline.add_calibration_sample(t, yaw=0.0, pitch=0.0, iris_h=0.5, iris_v=0.5)

    assert baseline.ready
    initial_values = dict(baseline.values)
    assert initial_values["yaw"] == 0.0

    # Simulate feeding 100 away frames (e.g. user looking away: yaw = 30.0)
    for _ in range(100):
        direction, is_away = classify_gaze(yaw=30.0, pitch=0.0, iris_h=0.5, iris_v=0.5, baseline=baseline.values)
        assert is_away is True
        # As in session_store, away frames must not be fed to add_on_screen_sample
        if not is_away:
            baseline.add_on_screen_sample(yaw=30.0, pitch=0.0, iris_h=0.5, iris_v=0.5)

    # Values must remain strictly unchanged and trailing deque must remain empty
    assert len(baseline.trailing) == 0
    assert baseline.values == initial_values
