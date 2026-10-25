import os
import sys
import time
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from records import FrameAnalysis, MATCH, NO_MATCH, NO_FACE
from session_store import SessionStore, SessionNotFound, SessionNotActive, SessionCompleted


def _make_analysis(identity=MATCH, faces=1, eyes_open=True, yaw=0.5, pitch=0.5, sim=0.8):
    return FrameAnalysis(
        faces_detected=faces,
        identity=identity,
        similarity=sim if faces == 1 else None,
        yaw=yaw if faces == 1 else None,
        pitch=pitch if faces == 1 else None,
        iris_h=0.5,
        iris_v=0.5,
        eyes_open=eyes_open,
        thumbnail="data:image/jpeg;base64,mockthumb" if identity != MATCH else None,
    )


def test_restart_clears_records():
    """store: create, start, add 3 frames, start again -> len(session.records) == 0."""
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1, 0.2], metadata={"width": 640})
    store.start(sid)

    for _ in range(3):
        store.add_frame(sid, _make_analysis())

    session = store.get(sid)
    assert len(session.records) == 3

    # Restart session
    store.start(sid)
    session_after_restart = store.get(sid)
    assert len(session_after_restart.records) == 0
    assert session_after_restart.status == "active"


def test_frame_after_stop_rejected():
    """store: start, stop, add_frame -> raises SessionNotActive."""
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1, 0.2], metadata={})
    store.start(sid)
    store.stop(sid)

    with pytest.raises(SessionNotActive):
        store.add_frame(sid, _make_analysis())


def test_stop_twice_raises():
    """store: start, stop, stop -> second raises SessionCompleted."""
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1, 0.2], metadata={})
    store.start(sid)
    data1 = store.stop(sid)
    assert data1 is not None

    with pytest.raises(SessionCompleted):
        store.stop(sid)


def test_session_not_found():
    store = SessionStore()
    with pytest.raises(SessionNotFound):
        store.get("non_existent_session_id")


def test_start_completed_session_raises():
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1], metadata={})
    store.start(sid)
    store.stop(sid)
    with pytest.raises(SessionCompleted):
        store.start(sid)


def test_get_reference_requires_active():
    store = SessionStore()
    sid = store.create(reference_embedding=[0.5, 0.6], metadata={})
    with pytest.raises(SessionNotActive):
        store.get_reference(sid)

    store.start(sid)
    ref = store.get_reference(sid)
    assert ref == [0.5, 0.6]


def test_add_frame_calibration_and_events():
    """Frames during calibration advance progress and generate records without vision models."""
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1], metadata={})
    store.start(sid)

    update = store.add_frame(sid, _make_analysis())
    assert update.record.calibrating is True
    assert update.record.identity == MATCH
    assert update.live_status["attention"] == "CALIBRATING"
    assert update.live_status["identity"] == "VERIFIED"


def test_cleanup_old_sessions():
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1], metadata={})
    # Artificially age the session
    session = store.get(sid)
    session.created_at = time.time() - 4000

    store.cleanup_old()
    with pytest.raises(SessionNotFound):
        store.get(sid)
