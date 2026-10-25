"""
Pure Evaluator Module for Face & Eye-Gaze Verification (L10 & L7).

Separates extraction from evaluation:
1. Extract (per frame): record t_ms, face count, similarity, pose, iris ratios, eyes open.
2. Evaluate (at stop): pure function over records computing baseline, intervals, metrics, and verdicts.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class IntervalTracker:
    """
    Hysteresis interval tracker for episodes of anomalies (M3, L7, L10).
    - An episode opens after open_after consecutive anomaly frames.
    - Closes after close_after consecutive normal frames.
    - Tracks start_s, end_s, duration_s relative to session start.
    """
    def __init__(
        self,
        interval_type: str = "looking_away",
        open_after: int = 5,
        close_after: int = 3,
        max_gap_s: float = 1.0,
        period_s: float = 0.2
    ):
        self.interval_type = interval_type
        self.open_after = open_after
        self.close_after = close_after
        self.max_gap_s = max_gap_s
        self.period_s = period_s
        self.run_start_s: Optional[float] = None
        self.run_len: int = 0
        self.ok_len: int = 0
        self.last_anomaly_t_s: Optional[float] = None
        self.current: Optional[Dict[str, Any]] = None
        self.intervals: List[Dict[str, Any]] = []

    def update(
        self,
        t_s: float,
        is_anomaly: Any = False,
        is_unknown: bool = False,
        details: str = "",
        thumbnail: Optional[str] = None
    ) -> Optional[str]:
        """Update tracker with frame state at relative time t_s."""
        event = None
        if isinstance(is_anomaly, str):
            state = is_anomaly.lower()
            if state in ("away", "anomaly", "true"):
                is_anomaly = True
                is_unknown = False
            elif state in ("unknown", "gap"):
                is_anomaly = False
                is_unknown = True
            else:
                is_anomaly = False
                is_unknown = False

        if is_unknown:
            if self.last_anomaly_t_s is not None and (t_s - self.last_anomaly_t_s) > self.max_gap_s:
                if self._close(self.last_anomaly_t_s + self.period_s):
                    event = "CLOSED_OK"
                self._reset_run()
            return event

        if is_anomaly:
            self.ok_len = 0
            if self.run_start_s is None:
                self.run_start_s = t_s
            self.run_len += 1
            self.last_anomaly_t_s = t_s
            if self.current is None and self.run_len >= self.open_after:
                self.current = {
                    "type": self.interval_type,
                    "start_s": round(self.run_start_s, 2),
                    "end_s": None,
                    "duration_s": 0.0,
                    "details": details,
                    "thumbnail": thumbnail,
                    "start": round(self.run_start_s, 2),
                    "end": None,
                    "duration": 0.0,
                }
                event = "OPENED"
        else:
            self.ok_len += 1
            if self.ok_len >= self.close_after:
                if self._close(self.last_anomaly_t_s + self.period_s if self.last_anomaly_t_s is not None else t_s):
                    event = "CLOSED_OK"
                self._reset_run()

        return event

    def _close(self, end_time_s: Optional[float] = None) -> bool:
        if self.current is not None:
            end_t = end_time_s if end_time_s is not None else (
                self.last_anomaly_t_s + self.period_s if self.last_anomaly_t_s is not None else 0.0
            )
            self.current["end_s"] = round(end_t, 2)
            self.current["duration_s"] = max(0.0, round(self.current["end_s"] - self.current["start_s"], 2))
            self.current["end"] = self.current["end_s"]
            self.current["duration"] = self.current["duration_s"]
            self.intervals.append(self.current)
            self.current = None
            return True
        return False

    def _reset_run(self):
        self.run_start_s = None
        self.run_len = 0
        self.last_anomaly_t_s = None

    def finalize(self, total_duration_s: float):
        self._close(end_time_s=total_duration_s)
        self._reset_run()


def evaluate_face_verdict(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Independent Face Assessment (H2):
    FAIL conditions:
      - valid_frames == 0 -> no_data
      - different_person_s >= 2.0 -> different_person
      - multiple_faces_s >= 5.0 -> multiple_faces
      - no_face_pct > 30.0 -> no_face
    REVIEW conditions:
      - no_face_pct > 15.0 -> no_face
      - identity_unknown_pct > 25.0 -> identity_unknown
      - static_face -> static_face_detected
    PASS otherwise.
    """
    reasons = []
    first_verdict = None

    if metrics.get("valid_frames", 0) == 0:
        return "FAIL", ["no_data"]

    if metrics.get("different_person_s", 0.0) >= 2.0:
        reasons.append("different_person")
        if not first_verdict:
            first_verdict = "FAIL"

    if metrics.get("multiple_faces_s", 0.0) >= 5.0:
        reasons.append("multiple_faces")
        if not first_verdict:
            first_verdict = "FAIL"

    no_face_pct = metrics.get("no_face_pct", 0.0)
    if no_face_pct > 30.0:
        reasons.append("no_face")
        if not first_verdict:
            first_verdict = "FAIL"
    elif no_face_pct > 15.0:
        reasons.append("no_face")
        if not first_verdict:
            first_verdict = "REVIEW"

    if metrics.get("identity_unknown_pct", 0.0) > 25.0:
        reasons.append("identity_unknown")
        if not first_verdict:
            first_verdict = "REVIEW"

    if metrics.get("static_face", False):
        reasons.append("static_face_detected")
        if not first_verdict:
            first_verdict = "REVIEW"

    if not reasons:
        return "PASS", []

    return first_verdict or "PASS", reasons


def evaluate_gaze_verdict(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Independent Gaze Assessment (H2, M4):
    FAIL conditions:
      - long_away_intervals > 0 -> long_away_intervals
      - away_pct > 35.0 -> looking_away
    REVIEW conditions:
      - bad_baseline -> no_stable_baseline
      - gaze_valid_frames == 0 -> no_data
      - away_pct > 20.0 -> looking_away
      - gaze_unknown_pct > 25.0 -> gaze_unknown
    PASS otherwise.
    """
    reasons = []
    first_verdict = None

    if metrics.get("bad_baseline", False):
        reasons.append("no_stable_baseline")
        return "REVIEW", reasons

    if metrics.get("gaze_valid_frames", 0) == 0:
        reasons.append("no_data")
        return "REVIEW", reasons

    if metrics.get("long_away_intervals", 0) > 0:
        reasons.append("long_away_intervals")
        if not first_verdict or first_verdict == "REVIEW":
            first_verdict = "FAIL"

    away_pct = metrics.get("away_pct", 0.0)
    if away_pct > 35.0:
        reasons.append("looking_away")
        if not first_verdict or first_verdict == "REVIEW":
            first_verdict = "FAIL"
    elif away_pct > 20.0:
        reasons.append("looking_away")
        if not first_verdict:
            first_verdict = "REVIEW"

    if metrics.get("gaze_unknown_pct", 0.0) > 25.0:
        reasons.append("gaze_unknown")
        if not first_verdict:
            first_verdict = "REVIEW"

    if not reasons:
        return "PASS", []

    return first_verdict or "PASS", reasons


def evaluate_session_records(
    records: List[Dict[str, Any]],
    duration_seconds: float = 0.0,
    bad_baseline: bool = False
) -> Dict[str, Any]:
    """
    Pure evaluation function (L10):
    Takes raw recorded frame list and session duration.
    Computes baseline (M4), intervals (M3, L7), metrics, and independent verdicts (H2).
    """
    total = len(records)
    if duration_seconds <= 0.0:
        if total > 0 and "t_s" in records[-1]:
            duration_seconds = max(1.0, round(records[-1]["t_s"], 2))
        else:
            duration_seconds = max(1.0, round(total * 0.2, 2))

    fps = total / duration_seconds if duration_seconds > 0 else 5.0

    # 1. Compute per-subject baseline (first 50 on-screen frames with single face & eyes open)
    baseline_samples = []
    for r in records:
        if len(baseline_samples) >= 50:
            break
        if (
            r.get("faces_detected") == 1
            and r.get("eyes_open", True)
            and r.get("verification_status") in ("MATCH", "NO_MATCH", "UNKNOWN")
            and r.get("yaw") is not None
            and r.get("pitch") is not None
        ):
            baseline_samples.append({
                "yaw": r["yaw"],
                "pitch": r["pitch"],
                "iris_h": r.get("iris_h", 0.5),
                "iris_v": r.get("iris_v", 0.5)
            })

    if len(baseline_samples) >= 30:
        yaw_vals = [s["yaw"] for s in baseline_samples]
        pitch_vals = [s["pitch"] for s in baseline_samples]
        yaw_std = float(np.std(yaw_vals))
        pitch_std = float(np.std(pitch_vals))
        if yaw_std > 8.0 or pitch_std > 8.0:
            bad_baseline = True
    elif total >= 50:
        bad_baseline = True

    # Post-calibration check (M4): if >50% of first 30 frames after calibration are looking away
    if not bad_baseline:
        post_calib_records = records[len(baseline_samples):len(baseline_samples) + 30]
        if len(post_calib_records) >= 15:
            post_away = sum(1 for r in post_calib_records if r.get("is_looking_away") is True)
            if (post_away / len(post_calib_records)) > 0.5:
                bad_baseline = True

    # 2. Track Intervals across all anomaly types (L7)
    gaze_tracker = IntervalTracker("looking_away", open_after=5, close_after=3, period_s=round(1.0 / fps, 3))
    diff_person_tracker = IntervalTracker("different_person", open_after=3, close_after=2, period_s=round(1.0 / fps, 3))
    multi_face_tracker = IntervalTracker("multiple_faces", open_after=3, close_after=2, period_s=round(1.0 / fps, 3))
    no_face_tracker = IntervalTracker("no_face", open_after=3, close_after=2, period_s=round(1.0 / fps, 3))

    matched_frames = 0
    unmatched_frames = 0
    no_face_frames = 0
    multiple_faces_frames = 0
    identity_unknown_frames = 0
    looking_away_frames = 0
    center_gaze_frames = 0
    gaze_unknown_frames = 0

    blink_count = 0
    was_closed = False
    recent_poses = []

    for idx, r in enumerate(records):
        t_s = r.get("t_s", round(idx * (1.0 / fps), 3))
        status = r.get("verification_status", "UNKNOWN")
        faces = r.get("faces_detected", 0)
        thumb = r.get("thumbnail")

        # Identity tracking
        if faces == 0:
            no_face_frames += 1
            no_face_tracker.update(t_s, is_anomaly=True, details="Face missing from camera view", thumbnail=thumb)
            diff_person_tracker.update(t_s, is_anomaly=False)
            multi_face_tracker.update(t_s, is_anomaly=False)
        elif faces > 1:
            multiple_faces_frames += 1
            multi_face_tracker.update(t_s, is_anomaly=True, details=f"Second face detected ({faces} faces visible)", thumbnail=thumb)
            no_face_tracker.update(t_s, is_anomaly=False)
            diff_person_tracker.update(t_s, is_anomaly=False)
        else:
            no_face_tracker.update(t_s, is_anomaly=False)
            multi_face_tracker.update(t_s, is_anomaly=False)

            if status == "MATCH":
                matched_frames += 1
                diff_person_tracker.update(t_s, is_anomaly=False)
            elif status == "NO_MATCH":
                unmatched_frames += 1
                sim = r.get("similarity")
                sim_str = f"similarity: {round(sim, 2)}" if sim is not None else ""
                diff_person_tracker.update(t_s, is_anomaly=True, details=f"Different person in camera view ({sim_str})", thumbnail=thumb)
            elif status == "UNKNOWN":
                identity_unknown_frames += 1
                diff_person_tracker.update(t_s, is_anomaly=False, is_unknown=True)

        # Gaze tracking (only evaluated when exactly 1 face is present, per M1 & H2)
        if faces == 1:
            is_away = r.get("is_looking_away")
            direction = r.get("gaze_direction", "UNKNOWN")
            eyes_open = r.get("eyes_open", True)

            if not eyes_open:
                if not was_closed:
                    blink_count += 1
                    was_closed = True
            else:
                was_closed = False

            if r.get("yaw") is not None and r.get("pitch") is not None:
                recent_poses.append((r["yaw"], r["pitch"]))

            if is_away is True:
                looking_away_frames += 1
                gaze_tracker.update(t_s, is_anomaly=True, details=f"Looked away ({direction.replace('_', ' ')})", thumbnail=thumb)
            elif is_away is False:
                center_gaze_frames += 1
                gaze_tracker.update(t_s, is_anomaly=False)
            else:
                gaze_unknown_frames += 1
                gaze_tracker.update(t_s, is_anomaly=False, is_unknown=True)
        else:
            # When face is absent or multiple faces present, gaze tracker is not in anomaly
            gaze_tracker.update(t_s, is_anomaly=False)

    # Finalize trackers at session end
    gaze_tracker.finalize(duration_seconds)
    diff_person_tracker.finalize(duration_seconds)
    multi_face_tracker.finalize(duration_seconds)
    no_face_tracker.finalize(duration_seconds)

    all_intervals = (
        gaze_tracker.intervals
        + diff_person_tracker.intervals
        + multi_face_tracker.intervals
        + no_face_tracker.intervals
    )
    all_intervals.sort(key=lambda iv: iv.get("start_s", 0.0))

    # Check static face heuristic (M8)
    static_face_detected = False
    if duration_seconds >= 30.0:
        if len(recent_poses) >= 100:
            sample_poses = recent_poses[-150:]
            y_arr = [p[0] for p in sample_poses]
            p_arr = [p[1] for p in sample_poses]
            pose_std = (float(np.std(y_arr)) + float(np.std(p_arr))) / 2.0
            if blink_count == 0 and pose_std < 0.6:
                static_face_detected = True

    # 3. Calculate metrics
    different_person_s = round(unmatched_frames / fps, 1)
    multiple_faces_s = round(multiple_faces_frames / fps, 1)
    no_face_s = round(no_face_frames / fps, 1)
    no_face_pct = round((no_face_frames / total * 100.0), 1) if total > 0 else 0.0

    identity_unknown_s = round(identity_unknown_frames / fps, 1)
    identity_unknown_pct = round((identity_unknown_frames / total * 100.0), 1) if total > 0 else 0.0

    gaze_valid_frames = center_gaze_frames + looking_away_frames
    away_s = round(looking_away_frames / fps, 1)
    away_pct = round((looking_away_frames / gaze_valid_frames * 100.0), 1) if gaze_valid_frames > 0 else 0.0

    long_away_intervals = sum(
        1 for iv in gaze_tracker.intervals
        if iv.get("duration_s", 0.0) >= 10.0
    )

    single_face_frames = center_gaze_frames + looking_away_frames + gaze_unknown_frames
    gaze_unknown_pct = round((gaze_unknown_frames / single_face_frames * 100.0), 1) if single_face_frames > 0 else 0.0

    metrics = {
        "valid_frames": total,
        "fps": round(fps, 1),
        "different_person_s": different_person_s,
        "multiple_faces_s": multiple_faces_s,
        "no_face_s": no_face_s,
        "no_face_pct": no_face_pct,
        "identity_unknown_s": identity_unknown_s,
        "identity_unknown_pct": identity_unknown_pct,
        "gaze_valid_frames": gaze_valid_frames,
        "away_s": away_s,
        "away_pct": away_pct,
        "long_away_intervals": long_away_intervals,
        "gaze_unknown_pct": gaze_unknown_pct,
        "bad_baseline": bad_baseline,
        "static_face": static_face_detected,
    }

    face_verdict, face_reasons = evaluate_face_verdict(metrics)
    gaze_verdict, gaze_reasons = evaluate_gaze_verdict(metrics)

    match_pct = round((matched_frames / total * 100.0), 1) if total > 0 else 0.0

    if face_verdict == "FAIL" or gaze_verdict == "FAIL":
        legacy_verdict = "FAILED"
    elif face_verdict == "REVIEW" or gaze_verdict == "REVIEW":
        legacy_verdict = "SUSPICIOUS"
    else:
        legacy_verdict = "PASSED"

    # Map intervals to legacy events list for backward compatibility
    events = []
    for idx, iv in enumerate(all_intervals):
        evt_type = str(iv.get("type", "EVENT")).upper()
        evt_msg = str(iv.get("details", ""))
        events.append({
            "id": f"evt_{idx}_{int(iv.get('start_s', 0.0) * 10)}",
            "timestamp": iv.get("start_s", 0.0),
            "type": evt_type,
            "message": evt_msg,
            "details": {
                "message": evt_msg,
                "start_s": iv.get("start_s", 0.0),
                "end_s": iv.get("end_s", 0.0),
                "duration_s": iv.get("duration_s", 0.0)
            }
        })

    return {
        "duration_seconds": duration_seconds,
        "total_frames": total,
        "matched_frames": matched_frames,
        "unmatched_frames": unmatched_frames,
        "no_face_frames": no_face_frames,
        "multiple_faces_frames": multiple_faces_frames,
        "match_percentage": match_pct,
        "looking_away_percentage": away_pct,
        "overall_verdict": legacy_verdict,
        "face": {
            "verdict": face_verdict,
            "reasons": face_reasons
        },
        "gaze": {
            "verdict": gaze_verdict,
            "reasons": gaze_reasons
        },
        "metrics": metrics,
        "intervals": all_intervals,
        "events": events
    }
