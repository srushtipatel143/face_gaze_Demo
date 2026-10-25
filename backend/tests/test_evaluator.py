import os
import sys
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluator import (
    evaluate,
    evaluate_session_records,
    evaluate_face_verdict,
    evaluate_gaze_verdict,
    face_verdict,
    gaze_verdict,
)
from records import FrameRecord, MATCH, NO_MATCH, NO_FACE, MULTIPLE_FACES, UNKNOWN
from tests.helpers import rec, CAL_OK


def test_scattered_no_match_does_not_fail():
    """300 frames, every 30th is NO_MATCH, rest MATCH, first 25 calibrating -> face PASS, different_person_s == 0."""
    records = []
    for i in range(300):
        calibrating = (i < 25)
        ident = NO_MATCH if (i % 30 == 0 and not calibrating) else MATCH
        sim = 0.2 if ident == NO_MATCH else 0.85
        records.append(rec(i, identity=ident, calibrating=calibrating, sim=sim))

    rep = evaluate(records, duration_s=60.0, calibration=CAL_OK)
    assert rep["face"]["verdict"] == "PASS"
    assert rep["metrics"]["different_person_s"] == 0.0


def test_sustained_no_match_fails():
    """20 consecutive NO_MATCH frames inside 200 MATCH frames -> face FAIL with different_person, one interval ~= 4.0 s."""
    records = []
    for i in range(200):
        is_diff = (50 <= i < 70)
        ident = NO_MATCH if is_diff else MATCH
        sim = 0.2 if is_diff else 0.85
        records.append(rec(i, identity=ident, sim=sim))

    rep = evaluate(records, duration_s=40.0, calibration=CAL_OK)
    assert rep["face"]["verdict"] == "FAIL"
    assert "different_person" in rep["face"]["reasons"]
    dp_intervals = [iv for iv in rep["intervals"] if iv["type"] == "different_person"]
    assert len(dp_intervals) == 1
    assert rep["metrics"]["different_person_s"] == pytest.approx(4.0, abs=0.2)


def test_calibration_frames_excluded():
    """150 frames (30 s), first 25 calibrating, rest centre -> gaze PASS, gaze_unknown_pct == 0."""
    records = []
    for i in range(150):
        records.append(rec(i, calibrating=(i < 25), away=False, direction="CENTER"))

    rep = evaluate(records, duration_s=30.0, calibration=CAL_OK)
    assert rep["gaze"]["verdict"] == "PASS"
    assert rep["metrics"]["gaze_unknown_pct"] == 0.0
    assert rep["metrics"]["away_s"] == 0.0


def test_away_pct_matches_intervals():
    """Mix of away runs (>= 5 frames) and 2-frame blips -> away_s == sum of looking_away interval durations; blips not counted."""
    records = []
    # 20 frames center
    for i in range(20):
        records.append(rec(len(records), away=False, direction="CENTER"))
    # 10 frames away (>= 5 frames -> forms interval of 10 * 0.2 = 2.0s)
    for i in range(10):
        records.append(rec(len(records), away=True, direction="LOOKING_LEFT"))
    # 5 frames center (closes interval)
    for i in range(5):
        records.append(rec(len(records), away=False, direction="CENTER"))
    # 2 frames away blip (< 5 frames -> ignored, no interval)
    for i in range(2):
        records.append(rec(len(records), away=True, direction="LOOKING_RIGHT"))
    # 10 frames center
    for i in range(10):
        records.append(rec(len(records), away=False, direction="CENTER"))

    rep = evaluate(records, duration_s=len(records) * 0.2, calibration=CAL_OK)
    away_intervals = [iv for iv in rep["intervals"] if iv["type"] == "looking_away"]
    assert len(away_intervals) == 1
    sum_dur = sum(iv["duration_s"] for iv in away_intervals)
    assert rep["metrics"]["away_s"] == sum_dur
    assert sum_dur == 2.0


def test_direction_breakdown():
    """10 s left run, 5 s down run -> away_by_direction_s == {'LEFT': ~=10, 'RIGHT': 0, 'UP': 0, 'DOWN': ~=5}."""
    records = []
    # 50 frames left (10s)
    for i in range(50):
        records.append(rec(len(records), away=True, direction="LOOKING_LEFT"))
    # 5 frames center (closes left interval)
    for i in range(5):
        records.append(rec(len(records), away=False, direction="CENTER"))
    # 25 frames down (5s)
    for i in range(25):
        records.append(rec(len(records), away=True, direction="LOOKING_DOWN"))
    # 5 frames center (closes down interval)
    for i in range(5):
        records.append(rec(len(records), away=False, direction="CENTER"))

    rep = evaluate(records, duration_s=len(records) * 0.2, calibration=CAL_OK)
    d = rep["metrics"]["away_by_direction_s"]
    assert d["LEFT"] == pytest.approx(10.0, abs=0.2)
    assert d["RIGHT"] == 0.0
    assert d["UP"] == 0.0
    assert d["DOWN"] == pytest.approx(5.0, abs=0.2)


def test_long_away_fails():
    """One 12 s away run -> gaze FAIL with long_away."""
    records = []
    # 60 frames = 12s away
    for i in range(60):
        records.append(rec(i, away=True, direction="LOOKING_LEFT"))

    rep = evaluate(records, duration_s=12.0, calibration=CAL_OK)
    assert rep["gaze"]["verdict"] == "FAIL"
    assert "long_away" in rep["gaze"]["reasons"]
    assert rep["metrics"]["long_away_intervals"] >= 1


def test_calibration_incomplete():
    """All frames calibrating, calibration completed False -> gaze REVIEW calibration_incomplete."""
    records = [rec(i, calibrating=True) for i in range(25)]
    rep = evaluate(records, duration_s=5.0, calibration={"completed": False, "unstable": False})
    assert rep["gaze"]["verdict"] == "REVIEW"
    assert "calibration_incomplete" in rep["gaze"]["reasons"]


def test_zero_frames_review():
    """No records -> face REVIEW no_data, gaze REVIEW no_data, overall SUSPICIOUS."""
    rep = evaluate([], duration_s=0.0, calibration=CAL_OK)
    assert rep["total_frames"] == 0
    assert rep["face"]["verdict"] == "REVIEW"
    assert rep["face"]["reasons"] == ["no_data"]
    assert rep["gaze"]["verdict"] == "REVIEW"
    assert rep["gaze"]["reasons"] == ["no_data"]
    assert rep["overall_verdict"] == "SUSPICIOUS"


def test_similarity_stats():
    """Similarities 0.30, 0.35, 0.40, 0.70 -> count 4, min 0.30, max 0.70, near_threshold_frames 2 (0.35 and 0.40 within 0.05 of 0.363)."""
    sims = [0.30, 0.35, 0.40, 0.70]
    records = [rec(i, sim=s) for i, s in enumerate(sims)]
    rep = evaluate(records, duration_s=0.8, calibration=CAL_OK)
    sim = rep["metrics"]["similarity"]
    assert sim["count"] == 4
    assert sim["min"] == 0.30
    assert sim["max"] == 0.70
    assert sim["near_threshold_frames"] == 2


def test_multiple_faces_fail():
    """Multiple faces present for >= 5s triggers face FAIL with reason multiple_faces."""
    records = [rec(i, faces=2, identity=MULTIPLE_FACES) for i in range(30)]
    rep = evaluate(records, duration_s=6.0, calibration=CAL_OK)
    assert rep["face"]["verdict"] == "FAIL"
    assert "multiple_faces" in rep["face"]["reasons"]
    assert rep["metrics"]["multiple_faces_s"] == 6.0


def test_no_face_pct_fail_and_review():
    """no_face > 30% duration triggers FAIL; > 15% triggers REVIEW."""
    # 20s session: 8s no_face (40% > 30%) -> FAIL
    records_fail = []
    for i in range(60):
        records_fail.append(rec(len(records_fail), identity=MATCH))
    for i in range(40):
        records_fail.append(rec(len(records_fail), faces=0, identity=NO_FACE))
    rep_fail = evaluate(records_fail, duration_s=20.0, calibration=CAL_OK)
    assert rep_fail["face"]["verdict"] == "FAIL"
    assert "no_face" in rep_fail["face"]["reasons"]

    # 20s session: 4s no_face (20% > 15% and <= 30%) -> REVIEW
    records_rev = []
    for i in range(80):
        records_rev.append(rec(len(records_rev), identity=MATCH))
    for i in range(20):
        records_rev.append(rec(len(records_rev), faces=0, identity=NO_FACE))
    rep_rev = evaluate(records_rev, duration_s=20.0, calibration=CAL_OK)
    assert rep_rev["face"]["verdict"] == "REVIEW"
    assert "no_face" in rep_rev["face"]["reasons"]


def test_static_face_heuristic():
    """Static face heuristic: duration >= 30s, >= 100 poses with 0 blinks and pose_std < 0.6."""
    records = []
    for i in range(160):
        # Yaw and pitch completely frozen (zero variance) and eyes always open
        records.append(rec(i, yaw=1.0, pitch=1.0, eyes_open=True))
    rep = evaluate(records, duration_s=32.0, calibration=CAL_OK)
    assert rep["metrics"]["static_face"] is True
    assert rep["face"]["verdict"] == "REVIEW"
    assert "static_face" in rep["face"]["reasons"]


def test_identity_unknown_pct_review():
    """Single face with head turned too far (> 25% UNKNOWN identity) triggers REVIEW."""
    records = []
    for i in range(70):
        records.append(rec(len(records), identity=MATCH))
    for i in range(30):
        records.append(rec(len(records), identity=UNKNOWN))
    rep = evaluate(records, duration_s=20.0, calibration=CAL_OK)
    assert rep["face"]["verdict"] == "REVIEW"
    assert "identity_unknown" in rep["face"]["reasons"]


def test_gaze_unknown_pct_review():
    """Gaze unknown > 25% triggers gaze REVIEW."""
    records = []
    for i in range(70):
        records.append(rec(len(records), eyes_open=True, away=False))
    for i in range(30):
        # Eyes closed / gaze unknown
        records.append(rec(len(records), eyes_open=False, away=None, direction="UNKNOWN"))
    rep = evaluate(records, duration_s=20.0, calibration=CAL_OK)
    assert rep["gaze"]["verdict"] == "REVIEW"
    assert "gaze_unknown" in rep["gaze"]["reasons"]


def test_evaluate_session_records_backward_compatibility():
    """evaluate_session_records supports list of dicts and wraps legacy verdicts."""
    dict_records = [
        {"t_s": 0.2 * i, "faces_detected": 1, "verification_status": "MATCH",
         "similarity": 0.9, "yaw": 0.5 * (i % 3), "pitch": 0.5 * (i % 4), "eyes_open": True,
         "gaze_direction": "CENTER", "is_looking_away": False}
        for i in range(50)
    ]
    summary = evaluate_session_records(dict_records, duration_seconds=10.0)
    assert summary["face"]["verdict"] == "PASS"
    assert summary["gaze"]["verdict"] == "PASS"
    assert summary["overall_verdict"] == "PASSED"
    assert "status" in summary
