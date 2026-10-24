"""
Pytest Suite for Rule Evaluation and Interval Tracking (L10 & L7).
Tests run pure evaluation functions over synthetic in-memory record lists in milliseconds without camera hardware.
"""

import pytest
import sys
import os

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluator import (
    IntervalTracker,
    evaluate_session_records,
    evaluate_face_verdict,
    evaluate_gaze_verdict,
)


def test_interval_tracker_away_then_center():
    """6 away frames then 3 center frames gives one interval of 1.2 s (L10 requirement)."""
    tracker = IntervalTracker("looking_away", open_after=5, close_after=3, period_s=0.2)

    # 6 away frames: t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    for i in range(6):
        tracker.update(t_s=round(i * 0.2, 2), is_anomaly=True, details="Looked away")

    # 3 center frames: t = 1.2, 1.4, 1.6
    for i in range(6, 9):
        tracker.update(t_s=round(i * 0.2, 2), is_anomaly=False)

    assert len(tracker.intervals) == 1, f"Expected exactly 1 interval, got {len(tracker.intervals)}"
    iv = tracker.intervals[0]
    assert iv["type"] == "looking_away"
    assert iv["start_s"] == 0.0
    assert iv["end_s"] == 1.2
    assert iv["duration_s"] == 1.2


def test_face_fail_different_person():
    """20 s of one non-matching face gives face FAIL with reason different_person (L10 requirement)."""
    records = []
    # 20 seconds at 5 FPS = 100 frames
    for i in range(100):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "NO_MATCH",
            "similarity": 0.25,
            "is_match": False,
            "yaw": 0.0,
            "pitch": 0.0,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": True,
            "gaze_direction": "CENTER",
            "is_looking_away": False,
        })

    summary = evaluate_session_records(records, duration_seconds=20.0)
    assert summary["face"]["verdict"] == "FAIL"
    assert "different_person" in summary["face"]["reasons"]
    assert summary["metrics"]["different_person_s"] >= 19.0


def test_face_fail_multiple_faces():
    """Multiple faces present for >= 5s triggers face FAIL with reason multiple_faces."""
    records = []
    # 30 frames of 2 faces = 6.0 seconds
    for i in range(30):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 2,
            "verification_status": "MULTIPLE_FACES",
            "similarity": None,
            "is_match": False,
            "yaw": None,
            "pitch": None,
            "eyes_open": True,
            "gaze_direction": "UNKNOWN",
            "is_looking_away": None,
        })

    summary = evaluate_session_records(records, duration_seconds=6.0)
    assert summary["face"]["verdict"] == "FAIL"
    assert "multiple_faces" in summary["face"]["reasons"]
    assert summary["metrics"]["multiple_faces_s"] == 6.0


def test_gaze_fail_long_away_interval():
    """A single sustained gaze deviation >= 10s triggers gaze FAIL with reason long_away_intervals."""
    records = []
    # 60 frames of looking away (12 seconds)
    for i in range(60):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.88,
            "is_match": True,
            "yaw": -30.0,
            "pitch": 0.0,
            "iris_h": 0.7,
            "iris_v": 0.5,
            "eyes_open": True,
            "gaze_direction": "LOOKING_LEFT",
            "is_looking_away": True,
        })

    summary = evaluate_session_records(records, duration_seconds=12.0)
    assert summary["gaze"]["verdict"] == "FAIL"
    assert "long_away_intervals" in summary["gaze"]["reasons"]
    assert summary["metrics"]["long_away_intervals"] >= 1


def test_no_face_not_counted_as_looking_away():
    """Frames without a face should not count into looking_away_frames (M1)."""
    records = []
    # 50 center frames (10s)
    for i in range(50):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.90,
            "is_match": True,
            "yaw": 0.0,
            "pitch": 0.0,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": True,
            "gaze_direction": "CENTER",
            "is_looking_away": False,
        })
    # 50 no-face frames (10s)
    for i in range(50, 100):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 0,
            "verification_status": "NO_FACE",
            "similarity": None,
            "is_match": False,
            "yaw": None,
            "pitch": None,
            "eyes_open": True,
            "gaze_direction": "UNKNOWN",
            "is_looking_away": None,
        })

    summary = evaluate_session_records(records, duration_seconds=20.0)
    # Gaze valid frames is only the 50 frames where a single face was present
    assert summary["metrics"]["gaze_valid_frames"] == 50
    assert summary["metrics"]["away_pct"] == 0.0
    assert summary["gaze"]["verdict"] == "PASS"


def test_bad_baseline_verdict():
    """Unstable initial gaze samples trigger bad_baseline and REVIEW verdict with no_stable_baseline."""
    records = []
    # 50 frames with fluctuating yaw between -25 and +25
    for i in range(50):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.90,
            "is_match": True,
            "yaw": 25.0 if i % 2 == 0 else -25.0,
            "pitch": 0.0,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": True,
            "gaze_direction": "CENTER",
            "is_looking_away": False,
        })

    summary = evaluate_session_records(records, duration_seconds=10.0)
    assert summary["metrics"]["bad_baseline"] is True
    assert summary["gaze"]["verdict"] == "REVIEW"
    assert "no_stable_baseline" in summary["gaze"]["reasons"]


def test_interval_relative_times():
    """Relative start_s, end_s, and duration_s are recorded properly (L7)."""
    tracker = IntervalTracker("different_person", open_after=3, close_after=2, period_s=0.2)
    # Anomaly from 4.0s to 5.0s
    for t in [4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:
        tracker.update(t_s=t, is_anomaly=True, details="Different person", thumbnail="data:image/jpeg;base64,testthumb")
    # Return to match from 5.2s
    for t in [5.2, 5.4, 5.6]:
        tracker.update(t_s=t, is_anomaly=False)

    assert len(tracker.intervals) == 1
    iv = tracker.intervals[0]
    assert iv["start_s"] == 4.0
    assert iv["end_s"] == 5.2
    assert iv["duration_s"] == 1.2
    assert iv["thumbnail"] == "data:image/jpeg;base64,testthumb"


def test_session_verdict_all_pass():
    """Clean session with consistent match and center gaze yields PASS for both face and gaze."""
    records = []
    for i in range(50):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.92,
            "is_match": True,
            "yaw": 1.0,
            "pitch": -2.0,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": True,
            "gaze_direction": "CENTER",
            "is_looking_away": False,
        })
    summary = evaluate_session_records(records, duration_seconds=10.0)
    assert summary["face"]["verdict"] == "PASS"
    assert summary["gaze"]["verdict"] == "PASS"
    assert summary["intervals"] == []


def test_gaze_review_high_gaze_unknown():
    """When single face has >25% gaze unknown (e.g. eyes closed/steep turns), gaze verdict is REVIEW."""
    records = []
    # 30 center frames
    for i in range(30):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.90,
            "is_match": True,
            "yaw": 0.0,
            "pitch": 0.0,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": True,
            "gaze_direction": "CENTER",
            "is_looking_away": False,
        })
    # 20 unknown gaze frames (eyes closed / blink / obscured)
    for i in range(30, 50):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.90,
            "is_match": True,
            "yaw": 0.0,
            "pitch": 0.0,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": False,
            "gaze_direction": "UNKNOWN",
            "is_looking_away": None,
        })
    summary = evaluate_session_records(records, duration_seconds=10.0)
    assert summary["face"]["verdict"] == "PASS"
    assert summary["gaze"]["verdict"] == "REVIEW"
    assert "gaze_unknown" in summary["gaze"]["reasons"]


def test_static_face_heuristic_triggers_review():
    """Static face heuristic (M8): >= 30s session, >= 100 poses with zero blinks and pose_std < 0.6."""
    records = []
    for i in range(150):
        records.append({
            "t_ms": i * 200,
            "t_s": round(i * 0.2, 2),
            "faces_detected": 1,
            "verification_status": "MATCH",
            "similarity": 0.95,
            "is_match": True,
            "yaw": 0.1,  # completely rigid pose
            "pitch": -0.1,
            "iris_h": 0.5,
            "iris_v": 0.5,
            "eyes_open": True,  # never blinks
            "gaze_direction": "CENTER",
            "is_looking_away": False,
        })
    summary = evaluate_session_records(records, duration_seconds=30.0)
    assert summary["metrics"]["static_face"] is True
    assert summary["face"]["verdict"] == "REVIEW"
    assert "static_face_detected" in summary["face"]["reasons"]

