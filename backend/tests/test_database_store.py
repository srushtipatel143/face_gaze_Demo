import sys
import os
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db_models
from db_models import (
    Base,
    VerificationSession,
    VerificationFailure,
    save_reference_session,
    save_failure_proof,
    save_report,
    list_sessions,
    get_report,
    get_session_from_db,
    get_session_failures
)


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Setup an in-memory SQLite database for testing database persistence functions."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(db_models, "engine", test_engine)
    monkeypatch.setattr(db_models, "SessionLocal", TestingSessionLocal)
    yield


def test_save_reference_session_and_retrieval():
    session_id = "sess_test_123"
    ref_image = "data:image/jpeg;base64,mockrefimage"
    embedding = [0.123, -0.456, 0.789]
    metadata = {"width": 640, "height": 480, "quality_score": 92.5}

    saved = save_reference_session(
        session_id=session_id,
        reference_image=ref_image,
        reference_embedding=embedding,
        quality_score=92.5,
        metadata=metadata
    )
    assert saved is True

    data = get_session_from_db(session_id)
    assert data is not None
    assert data["session_id"] == session_id
    assert data["quality_score"] == 92.5
    assert data["has_reference_image"] is True
    assert data["reference_image"] == ref_image


def test_save_failure_proof_images():
    session_id = "sess_test_fail_456"
    save_reference_session(session_id=session_id, reference_image="data:image/jpeg;base64,ref")

    proof_img_1 = "data:image/jpeg;base64,failproofimage_diff_person"
    saved_1 = save_failure_proof(
        session_id=session_id,
        failure_type="DIFFERENT_PERSON",
        reason="Face mismatch (similarity: 0.12)",
        proof_image=proof_img_1,
        similarity_score=0.12,
        t_s=3.4
    )
    assert saved_1 is True

    proof_img_2 = "data:image/jpeg;base64,failproofimage_look_away"
    saved_2 = save_failure_proof(
        session_id=session_id,
        failure_type="LOOKING_AWAY",
        reason="User looked away (looking left)",
        proof_image=proof_img_2,
        t_s=7.8
    )
    assert saved_2 is True

    failures = get_session_failures(session_id)
    assert len(failures) == 2
    assert failures[0]["failure_type"] == "DIFFERENT_PERSON"
    assert failures[0]["proof_image"] == proof_img_1
    assert failures[0]["similarity_score"] == 0.12
    assert failures[0]["t_s"] == 3.4

    assert failures[1]["failure_type"] == "LOOKING_AWAY"
    assert failures[1]["proof_image"] == proof_img_2


def test_save_report_and_history():
    session_id = "sess_test_complete_789"
    save_reference_session(session_id=session_id, reference_image="data:image/jpeg;base64,ref")

    report = {
        "session_id": session_id,
        "status": "completed",
        "rules_version": "2026.10.1",
        "duration_s": 15.5,
        "total_frames": 75,
        "face": {"verdict": "FAIL", "reasons": ["excessive_different_person"]},
        "gaze": {"verdict": "PASS", "reasons": []},
        "overall_verdict": "FAILED",
        "metrics": {
            "fps": 5.0,
            "period_s": 0.2,
            "matched_frames": 60,
            "unmatched_frames": 10,
            "identity_unknown_frames": 0,
            "no_face_frames": 5,
            "multiple_faces_frames": 0,
            "match_pct": 80.0,
            "different_person_s": 2.0,
            "multiple_faces_s": 0.0,
            "no_face_s": 1.0,
            "no_face_pct": 6.5,
            "identity_unknown_pct": 0.0,
            "gaze_valid_frames": 60,
            "gaze_unknown_frames": 0,
            "gaze_valid_s": 12.0,
            "away_s": 2.0,
            "away_pct": 13.3,
            "time_on_screen_pct": 86.7,
            "away_by_direction_s": {"LEFT": 2.0, "RIGHT": 0.0, "UP": 0.0, "DOWN": 0.0},
            "long_away_intervals": 0,
            "long_away_threshold_s": 10.0,
            "gaze_unknown_pct": 0.0,
            "calibration": {"completed": True, "unstable": False, "post_calibration_away": False},
            "blink_count": 2,
            "static_face": False,
            "similarity": {"threshold": 0.363, "count": 70, "min": 0.1, "p05": 0.2, "median": 0.8, "max": 0.95, "near_threshold_frames": 2},
        },
        "intervals": [
            {
                "id": "different_person-1",
                "type": "different_person",
                "start_s": 5.0,
                "end_s": 7.0,
                "duration_s": 2.0,
                "direction": None,
                "message": "Different person in view",
                "thumbnail": "data:image/jpeg;base64,intervalthumb",
            }
        ],
    }

    records = [
        {"t_s": 0.2, "faces_detected": 1, "identity": "MATCH", "similarity": 0.85, "thumbnail": "data:image/jpeg;base64,thumb"}
    ]

    saved = save_report(session_id, report, records)
    assert saved is True

    data = get_session_from_db(session_id)
    assert data is not None
    assert data["status"] == "completed"
    assert data["face_verdict"] == "FAIL"
    assert data["gaze_verdict"] == "PASS"
    assert data["match_percentage"] == 80.0
    assert "excessive_different_person" in data["verdict_reasons"]
    assert data["rules_version"] == "2026.10.1"

    # Verify interval thumbnail was persisted as a failure proof with t_s == 5.0
    failures = get_session_failures(session_id)
    assert len(failures) == 1
    assert failures[0]["proof_image"] == "data:image/jpeg;base64,intervalthumb"
    assert failures[0]["failure_type"] == "DIFFERENT_PERSON"
    assert failures[0]["t_s"] == 5.0

    # Verify get_report
    fetched_report = get_report(session_id)
    assert fetched_report is not None
    assert fetched_report["session_id"] == session_id

    # Verify list_sessions
    sessions = list_sessions(limit=10)
    assert len(sessions) >= 1
    assert any(s["session_id"] == session_id for s in sessions)

