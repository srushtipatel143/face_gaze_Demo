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


def test_session_reference_embedding_build():
    """Unit test that the session embedding is built after 5 MATCH frames and not from NO_MATCH frames."""
    import numpy as np
    store = SessionStore()
    sid = store.create(reference_embedding=[0.1, 0.2], metadata={})
    store.start(sid)

    # 1. NO_MATCH frames with embedding -> should not build session embedding
    no_match_emb = np.array([1.0, 0.0], dtype=np.float32)
    for _ in range(5):
        a = _make_analysis(identity=NO_MATCH, sim=0.2)
        a.embedding = no_match_emb
        store.add_frame(sid, a)
    assert store.get(sid).session_embedding is None
    assert len(store.get(sid).session_embedding_candidates) == 0

    # 2. MATCH frame with low similarity (< 0.5) -> should not build
    low_sim_emb = np.array([0.5, 0.5], dtype=np.float32)
    a = _make_analysis(identity=MATCH, sim=0.45)
    a.embedding = low_sim_emb
    store.add_frame(sid, a)
    assert store.get(sid).session_embedding is None
    assert len(store.get(sid).session_embedding_candidates) == 0

    # 3. 4 MATCH frames with sim >= 0.5 -> candidates reach 4, session_embedding still None
    match_emb = np.array([0.6, 0.8], dtype=np.float32)
    for _ in range(4):
        a = _make_analysis(identity=MATCH, sim=0.7)
        a.embedding = match_emb
        store.add_frame(sid, a)
    assert store.get(sid).session_embedding is None
    assert len(store.get(sid).session_embedding_candidates) == 4

    # 4. 5th MATCH frame -> session_embedding is built and normalized
    a = _make_analysis(identity=MATCH, sim=0.8)
    a.embedding = match_emb
    store.add_frame(sid, a)
    session = store.get(sid)
    assert session.session_embedding is not None
    assert np.isclose(np.linalg.norm(session.session_embedding), 1.0)

    # 5. Further frames do not alter the frozen session embedding
    initial_frozen = np.copy(session.session_embedding)
    a = _make_analysis(identity=MATCH, sim=0.9)
    a.embedding = np.array([0.0, 1.0], dtype=np.float32)
    store.add_frame(sid, a)
    assert np.allclose(session.session_embedding, initial_frozen)
