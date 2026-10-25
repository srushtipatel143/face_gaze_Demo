import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from intervals import make_trackers, feed_trackers, interval_to_event
from records import FrameAnalysis, FrameRecord, MATCH, NO_MATCH, NO_FACE, MULTIPLE_FACES, UNKNOWN, \
    CALIBRATING, GAZE_UNKNOWN
from vision.gaze import GazeBaseline, classify_gaze


class SessionNotFound(Exception):
    pass


class SessionNotActive(Exception):
    pass


class SessionCompleted(Exception):
    pass


@dataclass
class Session:
    session_id: str
    created_at: float
    reference_embedding: Any
    reference_metadata: Dict[str, Any]
    status: str = "initialized"                  # initialized | active | completed
    start_monotonic: Optional[float] = None
    records: List[FrameRecord] = field(default_factory=list)
    baseline: GazeBaseline = field(default_factory=GazeBaseline)
    live_trackers: Dict[str, Any] = field(default_factory=lambda: make_trackers(config.FRAME_PERIOD_S))
    report: Optional[Dict[str, Any]] = None      # set after /stop
    session_embedding: Optional[Any] = None      # frozen once built
    session_embedding_candidates: List[Any] = field(default_factory=list)

    def reset_live_state(self) -> None:
        self.records = []
        self.baseline = GazeBaseline()
        self.live_trackers = make_trackers(config.FRAME_PERIOD_S)
        self.report = None
        self.session_embedding = None
        self.session_embedding_candidates = []


@dataclass
class LiveUpdate:
    record: FrameRecord
    live_status: Dict[str, str]
    events: List[Dict[str, Any]]
    calibration_progress: float
    elapsed_s: float


@dataclass
class StopData:
    records: List[FrameRecord]
    duration_s: float
    calibration: Dict[str, bool]
    session_reference_used: bool = False


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}

    def _get(self, session_id: str) -> Session:
        s = self._sessions.get(session_id)
        if s is None:
            raise SessionNotFound(session_id)
        return s

    def create(self, reference_embedding: Any, metadata: Dict[str, Any]) -> str:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._sessions[session_id] = Session(session_id, time.time(), reference_embedding, metadata)
        return session_id

    def get(self, session_id: str) -> Session:
        with self._lock:
            return self._get(session_id)

    def get_reference(self, session_id: str) -> Any:
        with self._lock:
            s = self._get(session_id)
            if s.status != "active":
                raise SessionNotActive(s.status)
            return s.reference_embedding

    def get_references(self, session_id: str) -> Tuple[Any, Optional[Any]]:
        with self._lock:
            s = self._get(session_id)
            if s.status != "active":
                raise SessionNotActive(s.status)
            session_ref = s.session_embedding if (config.SESSION_REFERENCE_ENABLED and s.session_embedding is not None) else None
            return s.reference_embedding, session_ref

    def start(self, session_id: str) -> None:
        with self._lock:
            s = self._get(session_id)
            if s.status == "completed":
                raise SessionCompleted(session_id)
            s.reset_live_state()                     # restart = clean slate (fixes mixed runs)
            s.status = "active"
            s.start_monotonic = time.monotonic()

    def add_frame(self, session_id: str, a: FrameAnalysis) -> LiveUpdate:
        with self._lock:
            s = self._get(session_id)
            if s.status != "active":
                raise SessionNotActive(s.status)     # frame arrived after /stop: rejected
            t_s = round(time.monotonic() - s.start_monotonic, 3)

            # Session reference embedding: accumulate candidate frames and freeze once ready
            if s.session_embedding is None and a.embedding is not None:
                sim = a.similarity if a.similarity is not None else 0.0
                if (a.identity == MATCH and a.faces_detected == 1
                        and sim >= config.SESSION_REFERENCE_MIN_SIMILARITY):
                    s.session_embedding_candidates.append(a.embedding)
                    if len(s.session_embedding_candidates) >= config.SESSION_REFERENCE_FRAMES:
                        normalized = []
                        for emb in s.session_embedding_candidates:
                            arr = np.asarray(emb, dtype=np.float32).flatten()
                            norm = float(np.linalg.norm(arr))
                            normalized.append(arr / norm if norm > 1e-6 else arr)
                        mean_emb = np.mean(normalized, axis=0)
                        mean_norm = float(np.linalg.norm(mean_emb))
                        s.session_embedding = mean_emb / mean_norm if mean_norm > 1e-6 else mean_emb

            gaze_valid = (a.faces_detected == 1 and a.eyes_open is True
                          and a.yaw is not None and a.pitch is not None)
            calibrating = not s.baseline.ready
            if calibrating:
                if gaze_valid:
                    s.baseline.add_calibration_sample(t_s, a.yaw, a.pitch, a.iris_h, a.iris_v)
                direction, is_away = (CALIBRATING if a.faces_detected == 1 else GAZE_UNKNOWN), None
            elif gaze_valid:
                direction, is_away = classify_gaze(a.yaw, a.pitch, a.iris_h, a.iris_v, s.baseline.values)
                if not is_away:
                    s.baseline.add_on_screen_sample(a.yaw, a.pitch, a.iris_h, a.iris_v)
            else:
                direction, is_away = GAZE_UNKNOWN, None

            keep_thumb = a.identity in (NO_FACE, MULTIPLE_FACES, NO_MATCH) or is_away is True
            rec = FrameRecord.from_analysis(
                a, t_s=t_s, gaze_direction=direction, is_looking_away=is_away,
                calibrating=calibrating, thumbnail=a.thumbnail if keep_thumb else None)
            s.records.append(rec)

            changes = feed_trackers(s.live_trackers, rec)
            return LiveUpdate(
                record=rec,
                live_status=self._live_status(s, rec),
                events=[interval_to_event(iv) for _, iv in changes],
                calibration_progress=s.baseline.progress(t_s),
                elapsed_s=t_s,
            )

    @staticmethod
    def _live_status(s: Session, rec: FrameRecord) -> Dict[str, str]:
        t = s.live_trackers
        if t["multiple_faces"].current:
            identity = "MULTIPLE_FACES"
        elif t["no_face"].current:
            identity = "NO_FACE"
        elif t["different_person"].current:
            identity = "NOT_MATCHED"
        elif rec.identity == UNKNOWN:
            identity = "UNCERTAIN"
        elif rec.identity == MATCH:
            identity = "VERIFIED"
        else:
            identity = "CHECKING"        # a single bad frame that is not (yet) an interval

        if rec.calibrating:
            attention = "CALIBRATING"
        elif t["looking_away"].current:
            attention = "AWAY"
        elif rec.is_looking_away is None:
            attention = "UNKNOWN"
        else:
            attention = "ON_SCREEN"
        return {"identity": identity, "attention": attention}

    def stop(self, session_id: str) -> StopData:
        with self._lock:
            s = self._get(session_id)
            if s.status == "completed":
                raise SessionCompleted(session_id)
            duration = (time.monotonic() - s.start_monotonic) if s.start_monotonic else 0.0
            s.status = "completed"
            ref_used = bool(config.SESSION_REFERENCE_ENABLED and s.session_embedding is not None)
            return StopData(records=list(s.records), duration_s=round(duration, 2),
                            calibration=s.baseline.summary(),
                            session_reference_used=ref_used)

    def set_report(self, session_id: str, report: Dict[str, Any]) -> None:
        with self._lock:
            self._get(session_id).report = report

    def cleanup_old(self) -> None:
        cutoff = time.time() - config.SESSION_MAX_AGE_S
        with self._lock:
            for sid in [k for k, v in self._sessions.items() if v.created_at < cutoff]:
                del self._sessions[sid]


store = SessionStore()
