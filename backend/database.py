"""
In-Memory Session Database for Face & Eye-Gaze Verification
Provides thread-safe temporary session storage for reference embeddings,
live verification events, and aggregate session metrics.
"""

import uuid
import time
from typing import Dict, Any, Optional, List
from threading import Lock

class SessionDatabase:
    def __init__(self):
        self._lock = Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, reference_embedding: Any, image_metadata: Dict[str, Any]) -> str:
        """Create a new verification session with the reference face embedding."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": timestamp,
                "status": "initialized",  # initialized | active | completed | terminated
                "reference_embedding": reference_embedding,
                "reference_metadata": image_metadata,
                "start_time": None,
                "end_time": None,
                "total_frames": 0,
                "matched_frames": 0,
                "unmatched_frames": 0,
                "no_face_frames": 0,
                "multiple_faces_frames": 0,
                "looking_away_frames": 0,
                "center_gaze_frames": 0,
                "last_gaze_direction": "CENTER",
                "last_verification_status": "MATCH",
                "events": [],
                "recent_similarities": [],
            }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def start_verification(self, session_id: str) -> bool:
        """Mark a session as active for live verification."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            sess["status"] = "active"
            sess["start_time"] = time.time()
            return True

    def add_frame_result(self, session_id: str, result: Dict[str, Any]) -> bool:
        """Record live frame analysis results, trigger state transition events, and update statistics."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            
            sess["total_frames"] += 1
            status = result.get("verification_status")
            if status == "MATCH":
                sess["matched_frames"] += 1
            elif status == "NO_MATCH":
                sess["unmatched_frames"] += 1
            elif status == "NO_FACE":
                sess["no_face_frames"] += 1
            elif status == "MULTIPLE_FACES":
                sess["multiple_faces_frames"] += 1

            is_looking_away = result.get("is_looking_away", False)
            gaze_dir = result.get("gaze_direction", "CENTER")

            if is_looking_away:
                sess["looking_away_frames"] += 1
            else:
                sess["center_gaze_frames"] += 1

            if "similarity" in result and result["similarity"] is not None:
                sess["recent_similarities"].append(result["similarity"])
                if len(sess["recent_similarities"]) > 60:
                    sess["recent_similarities"].pop(0)

            # Auto-detect state transitions to create human-readable events
            last_status = sess.get("last_verification_status")
            last_gaze = sess.get("last_gaze_direction")
            current_time = round(time.time(), 2)

            if status != last_status:
                if status == "NO_FACE":
                    self._internal_add_event(sess, "FACE_LOST", {
                        "message": "User face disappeared from camera view."
                    })
                elif status == "MULTIPLE_FACES":
                    self._internal_add_event(sess, "MULTIPLE_FACES", {
                        "message": f"Multiple faces detected ({result.get('faces_detected')} people). Potential integrity breach."
                    })
                elif status == "NO_MATCH":
                    self._internal_add_event(sess, "IDENTITY_MISMATCH", {
                        "message": f"Face does not match reference identity (similarity: {result.get('similarity')})."
                    })
                elif status == "MATCH" and last_status in ["NO_FACE", "MULTIPLE_FACES", "NO_MATCH"]:
                    self._internal_add_event(sess, "FACE_RESTORED", {
                        "message": "Verified user face re-established in camera view."
                    })
                sess["last_verification_status"] = status

            # Gaze state transitions
            if gaze_dir != last_gaze:
                if is_looking_away and last_gaze == "CENTER":
                    self._internal_add_event(sess, "LOOKING_AWAY", {
                        "direction": gaze_dir,
                        "yaw": result.get("gaze_yaw"),
                        "pitch": result.get("gaze_pitch"),
                        "message": f"User looked away ({gaze_dir.replace('_', ' ').lower()})."
                    })
                elif not is_looking_away and last_gaze != "CENTER":
                    self._internal_add_event(sess, "GAZE_RETURNED", {
                        "message": "User eye-gaze returned to center focus."
                    })
                sess["last_gaze_direction"] = gaze_dir

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
        """Stop a session and calculate final summary statistics."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            
            sess["status"] = "completed"
            sess["end_time"] = time.time()
            start = sess.get("start_time") or sess.get("created_at")
            duration = max(1.0, round(sess["end_time"] - start, 1))
            total = sess["total_frames"]

            match_pct = round((sess["matched_frames"] / total * 100.0), 1) if total > 0 else 0.0
            gaze_away_pct = round((sess["looking_away_frames"] / total * 100.0), 1) if total > 0 else 0.0

            # Determine overall session verdict
            if total >= 5 and match_pct >= 80.0 and gaze_away_pct <= 35.0 and sess["multiple_faces_frames"] == 0:
                verdict = "PASSED"
            elif match_pct >= 60.0 or gaze_away_pct > 35.0:
                verdict = "SUSPICIOUS"
            else:
                verdict = "FAILED"

            return {
                "session_id": session_id,
                "status": "completed",
                "duration_seconds": duration,
                "total_frames": total,
                "matched_frames": sess["matched_frames"],
                "unmatched_frames": sess["unmatched_frames"],
                "no_face_frames": sess["no_face_frames"],
                "multiple_faces_frames": sess["multiple_faces_frames"],
                "match_percentage": match_pct,
                "looking_away_percentage": gaze_away_pct,
                "overall_verdict": verdict,
                "events": sess["events"]
            }

    def cleanup_old_sessions(self, max_age_seconds: int = 7200):
        """Remove sessions older than max_age_seconds."""
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
