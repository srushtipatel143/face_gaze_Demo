"""
In-Memory Session Database for Face & Eye-Gaze Verification
Provides thread-safe temporary session storage for reference embeddings,
live verification events, and aggregate session metrics.
"""

import uuid
import time
from typing import Dict, Any, Optional, List, Tuple
from threading import Lock

# Delegate rules, interval tracking, and verdict evaluation to pure evaluator module (L10 & L7)
from evaluator import (
    IntervalTracker,
    evaluate_face_verdict,
    evaluate_gaze_verdict,
    evaluate_session_records,
)
import config
try:
    from db_models import (
        init_db,
        save_reference_session,
        save_failure_proof,
        save_session_results,
        get_session_from_db,
        get_session_failures
    )
except ImportError:
    init_db = lambda: False
    save_reference_session = lambda *a, **kw: False
    save_failure_proof = lambda *a, **kw: False
    save_session_results = lambda *a, **kw: False
    get_session_from_db = lambda *a, **kw: None
    get_session_failures = lambda *a, **kw: []

class SessionDatabase:
    def __init__(self):
        self._lock = Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        # Try initializing database tables
        try:
            init_db()
        except Exception:
            pass

    def create_session(self, reference_embedding: Any, image_metadata: Dict[str, Any], reference_image: Optional[str] = None) -> str:
        """Create a new verification session with the reference face embedding."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": timestamp,
                "status": "initialized",  # initialized | active | completed | terminated
                "reference_image": reference_image,
                "reference_embedding": reference_embedding,
                "reference_metadata": image_metadata,
                "start_time": None,
                "end_time": None,
                "total_frames": 0,
                "matched_frames": 0,
                "unmatched_frames": 0,
                "no_face_frames": 0,
                "multiple_faces_frames": 0,
                "identity_unknown_frames": 0,
                "looking_away_frames": 0,
                "center_gaze_frames": 0,
                "gaze_unknown_frames": 0,
                "gaze_tracker": IntervalTracker("looking_away", open_after=config.GAZE_OPEN_AFTER, close_after=config.GAZE_CLOSE_AFTER, max_gap_s=config.MAX_UNKNOWN_GAP_S, period_s=config.FRAME_PERIOD_S),
                "consecutive_no_match": 0,
                "consecutive_multiple_faces": 0,
                "consecutive_no_face": 0,
                "consecutive_match": 0,
                "active_identity_alert": None,
                "baseline_samples": [],
                "baseline": None,
                "baseline_frames_required": 50,
                "post_calib_total": 0,
                "post_calib_away": 0,
                "bad_baseline": False,
                "trailing_on_screen_samples": [],
                "blink_count": 0,
                "recent_poses": [],
                "static_face_detected": False,
                "events": [],
                "recent_similarities": [],
                "records": [],
            }

        # Persist reference face data & session in PostgreSQL
        try:
            save_reference_session(
                session_id=session_id,
                reference_image=reference_image,
                reference_embedding=reference_embedding,
                quality_score=image_metadata.get("quality_score"),
                metadata=image_metadata
            )
        except Exception:
            pass

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def start_verification(self, session_id: str) -> bool:
        """Mark a session as active for live verification, resetting all counters if restarting (M6)."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            if sess["status"] == "completed":
                return False  # Refuse /start on completed session (M6)
            sess["status"] = "active"
            sess["start_time"] = time.time()
            sess["end_time"] = None
            sess["total_frames"] = 0
            sess["matched_frames"] = 0
            sess["unmatched_frames"] = 0
            sess["no_face_frames"] = 0
            sess["multiple_faces_frames"] = 0
            sess["identity_unknown_frames"] = 0
            sess["looking_away_frames"] = 0
            sess["center_gaze_frames"] = 0
            sess["gaze_unknown_frames"] = 0
            sess["gaze_tracker"] = IntervalTracker(open_after=config.GAZE_OPEN_AFTER, close_after=config.GAZE_CLOSE_AFTER, max_gap_s=config.MAX_UNKNOWN_GAP_S, period_s=config.FRAME_PERIOD_S)
            sess["baseline_samples"] = []
            sess["baseline"] = None
            sess["post_calib_total"] = 0
            sess["post_calib_away"] = 0
            sess["bad_baseline"] = False
            sess["trailing_on_screen_samples"] = []
            sess["blink_count"] = 0
            sess["recent_poses"] = []
            sess["static_face_detected"] = False
            sess["consecutive_no_match"] = 0
            sess["consecutive_multiple_faces"] = 0
            sess["consecutive_no_face"] = 0
            sess["consecutive_match"] = 0
            sess["active_identity_alert"] = None
            sess["events"] = []
            sess["recent_similarities"] = []
            return True

    def add_frame_result(self, session_id: str, result: Dict[str, Any]) -> bool:
        """Record live frame analysis results, trigger state transition events, and update statistics."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            
            sess["total_frames"] += 1

            # Extract per-frame record for pure evaluation at stop (L10)
            now = time.time()
            start_t = sess.get("start_time") or now
            elapsed = now - start_t
            idx = len(sess.get("records", []))
            t_s = round(elapsed if elapsed >= (idx * (config.FRAME_PERIOD_S / 2)) else idx * config.FRAME_PERIOD_S, 3)
            t_ms = int(t_s * 1000)

            rec = {
                "t_ms": t_ms,
                "t_s": t_s,
                "faces_detected": result.get("faces_detected", 0),
                "verification_status": result.get("verification_status", "UNKNOWN"),
                "similarity": result.get("similarity"),
                "is_match": result.get("is_match"),
                "yaw": result.get("gaze_yaw") if result.get("gaze_yaw") is not None else result.get("telemetry", {}).get("yaw"),
                "pitch": result.get("gaze_pitch") if result.get("gaze_pitch") is not None else result.get("telemetry", {}).get("pitch"),
                "iris_h": result.get("iris_h") if result.get("iris_h") is not None else result.get("telemetry", {}).get("iris_h", 0.5),
                "iris_v": result.get("iris_v") if result.get("iris_v") is not None else result.get("telemetry", {}).get("iris_v", 0.5),
                "eyes_open": result.get("eyes_open", True) if "eyes_open" in result else result.get("telemetry", {}).get("eyes_open", True),
                "gaze_direction": result.get("gaze_direction", "UNKNOWN"),
                "is_looking_away": result.get("is_looking_away"),
                "thumbnail": result.get("thumbnail"),
            }
            sess.setdefault("records", []).append(rec)

            status = result.get("verification_status")
            if status == "MATCH":
                sess["matched_frames"] += 1
            elif status == "NO_MATCH":
                sess["unmatched_frames"] += 1
            elif status == "NO_FACE":
                sess["no_face_frames"] += 1
            elif status == "MULTIPLE_FACES":
                sess["multiple_faces_frames"] += 1
            elif status == "UNKNOWN":
                sess["identity_unknown_frames"] += 1

            # Gaze counting: valid readings only count into looking_away_frames or center_gaze_frames.
            # No-face, multiple-face, and eyes-closed (blinks) frames are counted as gaze_unknown_frames.
            is_looking_away = result.get("is_looking_away")
            gaze_dir = result.get("gaze_direction", "UNKNOWN")

            # Baseline calibration and stability tracking (M4)
            if sess.get("baseline") is None:
                if result.get("calibrating", False) and result.get("gaze_yaw") is not None and result.get("gaze_pitch") is not None:
                    sess["baseline_samples"].append((
                        result["gaze_yaw"],
                        result["gaze_pitch"],
                        result.get("iris_h", 0.5),
                        result.get("iris_v", 0.5)
                    ))
                    if len(sess["baseline_samples"]) >= sess["baseline_frames_required"]:
                        import statistics
                        yaws = [s[0] for s in sess["baseline_samples"]]
                        pitches = [s[1] for s in sess["baseline_samples"]]
                        ihs = [s[2] for s in sess["baseline_samples"]]
                        ivs = [s[3] for s in sess["baseline_samples"]]
                        sess["baseline"] = {
                            "yaw": round(float(statistics.median(yaws)), 2),
                            "pitch": round(float(statistics.median(pitches)), 2),
                            "iris_h": round(float(statistics.median(ihs)), 3),
                            "iris_v": round(float(statistics.median(ivs)), 3)
                        }
            else:
                if is_looking_away is not None:
                    if sess["post_calib_total"] < 50:
                        sess["post_calib_total"] += 1
                        if is_looking_away is True:
                            sess["post_calib_away"] += 1
                        if sess["post_calib_total"] >= 25 and (sess["post_calib_away"] / sess["post_calib_total"]) > 0.50:
                            sess["bad_baseline"] = True

                    # Trailing window for on-screen updates only (never allow away frames to shift baseline)
                    if is_looking_away is False and result.get("gaze_yaw") is not None:
                        sess["trailing_on_screen_samples"].append((
                            result["gaze_yaw"],
                            result["gaze_pitch"],
                            result.get("iris_h", 0.5),
                            result.get("iris_v", 0.5)
                        ))
                        if len(sess["trailing_on_screen_samples"]) > config.TRAILING_BASELINE_WINDOW:
                            sess["trailing_on_screen_samples"].pop(0)
                        if len(sess["trailing_on_screen_samples"]) == config.TRAILING_BASELINE_WINDOW:
                            import statistics
                            yaws = [s[0] for s in sess["trailing_on_screen_samples"]]
                            pitches = [s[1] for s in sess["trailing_on_screen_samples"]]
                            ihs = [s[2] for s in sess["trailing_on_screen_samples"]]
                            ivs = [s[3] for s in sess["trailing_on_screen_samples"]]
                            sess["baseline"] = {
                                "yaw": round(float(statistics.median(yaws)), 2),
                                "pitch": round(float(statistics.median(pitches)), 2),
                                "iris_h": round(float(statistics.median(ihs)), 3),
                                "iris_v": round(float(statistics.median(ivs)), 3)
                            }

            if is_looking_away is True:
                sess["looking_away_frames"] += 1
                gaze_state = "away"
            elif is_looking_away is False:
                sess["center_gaze_frames"] += 1
                gaze_state = "ok"
            else:
                sess["gaze_unknown_frames"] += 1
                gaze_state = "unknown"

            current_time = time.time()

            # Interval tracker for gaze look-away episodes (M3)
            tracker_event = sess["gaze_tracker"].update(current_time, gaze_state)
            if tracker_event == "OPENED":
                self._internal_add_event(sess, "LOOKING_AWAY", {
                    "direction": gaze_dir if gaze_dir != "UNKNOWN" else "LOOKING_AWAY",
                    "yaw": result.get("gaze_yaw"),
                    "pitch": result.get("gaze_pitch"),
                    "message": f"User looked away ({gaze_dir.replace('_', ' ').lower() if gaze_dir != 'UNKNOWN' else 'sustained'})."
                })
                if rec.get("thumbnail"):
                    try:
                        save_failure_proof(
                            session_id=session_id,
                            failure_type="LOOKING_AWAY",
                            reason=f"User looked away ({gaze_dir})",
                            proof_image=rec["thumbnail"],
                            t_s=t_s
                        )
                    except Exception:
                        pass
            elif tracker_event == "CLOSED_OK":
                self._internal_add_event(sess, "GAZE_RETURNED", {
                    "message": "User eye-gaze returned to center focus."
                })

            if "similarity" in result and result["similarity"] is not None:
                sess["recent_similarities"].append(result["similarity"])
                if len(sess["recent_similarities"]) > 60:
                    sess["recent_similarities"].pop(0)

            # Identity hysteresis smoothing (M3)
            # Require 3 consecutive frames to raise an identity alert, preventing single-frame jitters
            if status == "NO_MATCH":
                sess["consecutive_no_match"] += 1
                sess["consecutive_match"] = 0
                sess["consecutive_multiple_faces"] = 0
                sess["consecutive_no_face"] = 0
                if sess["consecutive_no_match"] >= config.IDENTITY_OPEN_AFTER and sess["active_identity_alert"] != "NO_MATCH":
                    sess["active_identity_alert"] = "NO_MATCH"
                    self._internal_add_event(sess, "IDENTITY_MISMATCH", {
                        "message": f"Face does not match reference identity (similarity: {result.get('similarity')})."
                    })
                    if rec.get("thumbnail"):
                        try:
                            save_failure_proof(
                                session_id=session_id,
                                failure_type="DIFFERENT_PERSON",
                                reason=f"Face mismatch (similarity: {result.get('similarity')})",
                                proof_image=rec["thumbnail"],
                                similarity_score=result.get("similarity"),
                                t_s=t_s
                            )
                        except Exception:
                            pass
            elif status == "MULTIPLE_FACES":
                sess["consecutive_multiple_faces"] += 1
                sess["consecutive_match"] = 0
                sess["consecutive_no_match"] = 0
                sess["consecutive_no_face"] = 0
                if sess["consecutive_multiple_faces"] >= config.IDENTITY_OPEN_AFTER and sess["active_identity_alert"] != "MULTIPLE_FACES":
                    sess["active_identity_alert"] = "MULTIPLE_FACES"
                    self._internal_add_event(sess, "MULTIPLE_FACES", {
                        "message": f"Second face detected in camera view ({result.get('faces_detected')} faces visible)."
                    })
                    if rec.get("thumbnail"):
                        try:
                            save_failure_proof(
                                session_id=session_id,
                                failure_type="MULTIPLE_FACES",
                                reason=f"Multiple faces detected ({result.get('faces_detected')})",
                                proof_image=rec["thumbnail"],
                                t_s=t_s
                            )
                        except Exception:
                            pass
            elif status == "NO_FACE":
                sess["consecutive_no_face"] += 1
                sess["consecutive_match"] = 0
                sess["consecutive_no_match"] = 0
                sess["consecutive_multiple_faces"] = 0
                if sess["consecutive_no_face"] >= config.IDENTITY_OPEN_AFTER and sess["active_identity_alert"] != "NO_FACE":
                    sess["active_identity_alert"] = "NO_FACE"
                    self._internal_add_event(sess, "FACE_LOST", {
                        "message": "User face disappeared from camera view."
                    })
                    if rec.get("thumbnail"):
                        try:
                            save_failure_proof(
                                session_id=session_id,
                                failure_type="NO_FACE",
                                reason="Face disappeared from camera view",
                                proof_image=rec["thumbnail"],
                                t_s=t_s
                            )
                        except Exception:
                            pass
            elif status == "MATCH":
                sess["consecutive_match"] += 1
                sess["consecutive_no_match"] = 0
                sess["consecutive_multiple_faces"] = 0
                sess["consecutive_no_face"] = 0
                if sess["consecutive_match"] >= config.IDENTITY_OPEN_AFTER and sess["active_identity_alert"] is not None:
                    self._internal_add_event(sess, "FACE_RESTORED", {
                        "message": "Verified user face re-established in camera view."
                    })
                    sess["active_identity_alert"] = None
            else:
                sess["consecutive_match"] = 0
                sess["consecutive_no_match"] = 0
                sess["consecutive_multiple_faces"] = 0
                sess["consecutive_no_face"] = 0

            # Liveness / static photo detection (M8 cheap signal)
            # Living humans blink 15-20 times per minute and exhibit involuntary micro-movements
            if result.get("eyes_open") is False or (result.get("openness") is not None and result["openness"] < config.EYES_OPEN_MIN):
                sess["blink_count"] += 1

            if status == "MATCH" and result.get("gaze_yaw") is not None and result.get("gaze_pitch") is not None:
                sess["recent_poses"].append((result["gaze_yaw"], result["gaze_pitch"]))
                if len(sess["recent_poses"]) > config.STATIC_FACE_WINDOW:
                    sess["recent_poses"].pop(0)

                # After sustained matching (~30s, 150 frames) with 0 blinks and near-zero pose variance
                if sess["matched_frames"] >= config.STATIC_FACE_WINDOW and len(sess["recent_poses"]) >= config.STATIC_FACE_MIN_POSES and sess["blink_count"] == 0:
                    yaws = [p[0] for p in sess["recent_poses"]]
                    pitches = [p[1] for p in sess["recent_poses"]]
                    mean_yaw = sum(yaws) / len(yaws)
                    mean_pitch = sum(pitches) / len(pitches)
                    yaw_var = sum((y - mean_yaw) ** 2 for y in yaws) / len(yaws)
                    pitch_var = sum((p - mean_pitch) ** 2 for p in pitches) / len(pitches)
                    yaw_std = yaw_var ** 0.5
                    pitch_std = pitch_var ** 0.5

                    if yaw_std < 0.35 and pitch_std < 0.35:
                        if not sess["static_face_detected"]:
                            sess["static_face_detected"] = True
                            self._internal_add_event(sess, "STATIC_FACE_WARNING", {
                                "message": "Possible static photo detected: no natural blinks or head micro-movements in 30+ seconds."
                            })

            return True

    def _internal_add_event(self, sess: Dict[str, Any], event_type: str, details: Dict[str, Any]):
        """Helper to append an event without acquiring the lock again."""
        sess["events"].append({
            "id": f"evt_{uuid.uuid4().hex[:8]}",
            "timestamp": round(time.time(), 3),
            "type": event_type,
            "details": details
        })

    def add_event(self, session_id: str, event_type: str, details: Dict[str, Any]) -> bool:
        """Explicitly append a timestamped event."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            self._internal_add_event(sess, event_type, details)
            return True

    def stop_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Stop a session and calculate final summary statistics and independent verdicts."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            
            sess["status"] = "completed"
            sess["end_time"] = time.time()
            start = sess.get("start_time") or sess.get("created_at")
            duration = max(1.0, round(sess["end_time"] - start, 1))
            records = sess.get("records", [])
            if records:
                if duration <= 1.0 and records[-1].get("t_s", 0) > 1.0:
                    duration = round(records[-1]["t_s"], 1)
                summary = evaluate_session_records(
                    records,
                    duration_seconds=duration,
                    bad_baseline=sess.get("bad_baseline", False)
                )
                if not summary.get("events") and sess.get("events"):
                    summary["events"] = sess["events"]
            else:
                summary = {
                    "duration_seconds": duration,
                    "total_frames": sess["total_frames"],
                    "matched_frames": sess["matched_frames"],
                    "unmatched_frames": sess["unmatched_frames"],
                    "no_face_frames": sess["no_face_frames"],
                    "multiple_faces_frames": sess["multiple_faces_frames"],
                    "match_percentage": 0.0,
                    "looking_away_percentage": 0.0,
                    "overall_verdict": "PASSED",
                    "face": {"verdict": "FAIL", "reasons": ["no_data"]},
                    "gaze": {"verdict": "REVIEW", "reasons": ["no_data"]},
                    "metrics": {
                        "valid_frames": 0, "fps": round(1.0 / config.FRAME_PERIOD_S, 1), "different_person_s": 0.0,
                        "multiple_faces_s": 0.0, "no_face_s": 0.0, "no_face_pct": 0.0,
                        "identity_unknown_s": 0.0, "identity_unknown_pct": 0.0,
                        "gaze_valid_frames": 0, "away_s": 0.0, "away_pct": 0.0,
                        "long_away_intervals": 0, "gaze_unknown_pct": 0.0,
                        "bad_baseline": False, "static_face": False,
                    },
                    "intervals": [],
                    "events": sess.get("events", [])
                }

            summary["session_id"] = session_id
            summary["status"] = "completed"

            # Discard reference biometric embedding now that verification session is concluded (L5)
            sess["reference_embedding"] = None

        # Persist final verdicts, metrics, intervals, and failure proofs in PostgreSQL
        try:
            save_session_results(session_id, summary)
        except Exception:
            pass

        return summary

    def cleanup_old_sessions(self, max_age_seconds: int = config.SESSION_MAX_AGE_S):
        """Remove sessions older than max_age_seconds (L5)."""
        now = time.time()
        with self._lock:
            to_delete = [
                sid for sid, s in self._sessions.items()
                if (now - s.get("created_at", now)) > max_age_seconds
            ]
            for sid in to_delete:
                del self._sessions[sid]

# Global singleton database instance
db = SessionDatabase()
