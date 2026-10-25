"""
Evaluator module: Builds session audit report from intervals and frame records.
"""

import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from intervals import feed_trackers, make_trackers, IntervalTracker
from records import FrameRecord, MATCH, NO_MATCH, UNKNOWN, NO_FACE, MULTIPLE_FACES

SEVERITY = {"PASS": 0, "REVIEW": 1, "FAIL": 2}


def _period(records: List[FrameRecord]) -> float:
    diffs = [b.t_s - a.t_s for a, b in zip(records, records[1:]) if b.t_s > a.t_s]
    return round(statistics.median(diffs), 3) if diffs else config.FRAME_PERIOD_S


def _worst(outcomes: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
    if not outcomes:
        return "PASS", []
    verdict = max((v for v, _ in outcomes), key=SEVERITY.get)
    return verdict, [r for _, r in outcomes]


def face_verdict(m: Dict[str, Any]) -> Dict[str, Any]:
    if m["total_frames"] == 0:
        return {"verdict": "REVIEW", "reasons": ["no_data"]}
    outcomes: List[Tuple[str, str]] = []
    if m["different_person_s"] >= config.FACE_DIFFERENT_PERSON_FAIL_S:
        outcomes.append(("FAIL", "different_person"))
    if m["multiple_faces_s"] >= config.FACE_MULTIPLE_FACES_FAIL_S:
        outcomes.append(("FAIL", "multiple_faces"))
    if m["no_face_pct"] > config.FACE_NO_FACE_FAIL_PCT:
        outcomes.append(("FAIL", "no_face"))
    elif m["no_face_pct"] > config.FACE_NO_FACE_REVIEW_PCT:
        outcomes.append(("REVIEW", "no_face"))
    if m["identity_unknown_pct"] > config.FACE_IDENTITY_UNKNOWN_REVIEW_PCT:
        outcomes.append(("REVIEW", "identity_unknown"))
    if m.get("static_face"):
        outcomes.append(("REVIEW", "static_face"))
    v, reasons = _worst(outcomes)
    return {"verdict": v, "reasons": reasons}


def gaze_verdict(m: Dict[str, Any]) -> Dict[str, Any]:
    if m["total_frames"] == 0:
        return {"verdict": "REVIEW", "reasons": ["no_data"]}
    calib = m.get("calibration", {})
    if calib.get("completed") is False:
        return {"verdict": "REVIEW", "reasons": ["calibration_incomplete"]}
    if calib.get("unstable") or calib.get("post_calibration_away"):
        return {"verdict": "REVIEW", "reasons": ["no_stable_baseline"]}
    if m["gaze_valid_frames"] == 0:
        return {"verdict": "REVIEW", "reasons": ["no_data"]}

    outcomes: List[Tuple[str, str]] = []
    if m["long_away_intervals"] > 0:
        outcomes.append(("FAIL", "long_away"))
    if m["away_pct"] > config.GAZE_AWAY_FAIL_PCT:
        outcomes.append(("FAIL", "looking_away"))
    elif m["away_pct"] > config.GAZE_AWAY_REVIEW_PCT:
        outcomes.append(("REVIEW", "looking_away"))
    if m["gaze_unknown_pct"] > config.GAZE_UNKNOWN_REVIEW_PCT:
        outcomes.append(("REVIEW", "gaze_unknown"))

    v, reasons = _worst(outcomes)
    return {"verdict": v, "reasons": reasons}


def _overall_verdict(face_v: str, gaze_v: str) -> str:
    if face_v == "FAIL" or gaze_v == "FAIL":
        return "FAILED"
    if face_v == "REVIEW" or gaze_v == "REVIEW":
        return "SUSPICIOUS"
    return "PASSED"


def _metrics(records: List[FrameRecord], intervals: List[Dict[str, Any]],
             duration_s: float, period_s: float,
             calibration: Dict[str, bool]) -> Dict[str, Any]:
    total_frames = len(records)
    fps = round(total_frames / duration_s, 1) if duration_s > 0 else 0.0

    matched_frames = sum(1 for r in records if r.identity == MATCH)
    unmatched_frames = sum(1 for r in records if r.identity == NO_MATCH)
    identity_unknown_frames = sum(1 for r in records if r.identity == UNKNOWN)
    no_face_frames = sum(1 for r in records if r.identity == NO_FACE)
    multiple_faces_frames = sum(1 for r in records if r.identity == MULTIPLE_FACES)

    single_face_frames = [r for r in records if r.faces_detected == 1]
    denom_match = matched_frames + unmatched_frames + identity_unknown_frames
    match_pct = round(matched_frames / denom_match * 100.0, 1) if denom_match > 0 else 0.0

    different_person_s = round(sum(iv.get("duration_s", 0.0) or 0.0 for iv in intervals if iv["type"] == "different_person"), 2)
    multiple_faces_s = round(sum(iv.get("duration_s", 0.0) or 0.0 for iv in intervals if iv["type"] == "multiple_faces"), 2)
    no_face_s = round(sum(iv.get("duration_s", 0.0) or 0.0 for iv in intervals if iv["type"] == "no_face"), 2)
    no_face_pct = round(no_face_s / duration_s * 100.0, 1) if duration_s > 0 else 0.0

    identity_unknown_pct = round(identity_unknown_frames / len(single_face_frames) * 100.0, 1) if single_face_frames else 0.0

    gaze_frames = [r for r in single_face_frames if not r.calibrating]
    gaze_valid_frames = sum(1 for r in gaze_frames if r.is_looking_away is not None)
    gaze_unknown_frames = sum(1 for r in gaze_frames if r.is_looking_away is None)
    gaze_valid_s = round(gaze_valid_frames * period_s, 2)

    away_s = round(sum(iv.get("duration_s", 0.0) or 0.0 for iv in intervals if iv["type"] == "looking_away"), 2)
    away_pct = round(min(100.0, away_s / gaze_valid_s * 100.0), 1) if gaze_valid_s > 0 else 0.0
    time_on_screen_pct = round(100.0 - away_pct, 1)

    away_by_direction = {"LEFT": 0.0, "RIGHT": 0.0, "UP": 0.0, "DOWN": 0.0}
    for iv in intervals:
        if iv["type"] == "looking_away":
            dur = iv.get("duration_s", 0.0) or 0.0
            direction = (iv.get("direction") or "").upper().replace("LOOKING_", "")
            if direction in away_by_direction:
                away_by_direction[direction] = round(away_by_direction[direction] + dur, 2)

    long_away_intervals = sum(1 for iv in intervals if iv["type"] == "looking_away" and (iv.get("duration_s", 0.0) or 0.0) >= config.GAZE_LONG_AWAY_S)
    long_away_threshold_s = config.GAZE_LONG_AWAY_S
    gaze_unknown_pct = round(gaze_unknown_frames / len(gaze_frames) * 100.0, 1) if gaze_frames else 0.0

    # Calibration
    gaze_valid_recs = [r for r in gaze_frames if r.is_looking_away is not None]
    first_gaze_valid = gaze_valid_recs[:config.POST_CALIBRATION_CHECK_FRAMES]
    if "post_calibration_away" in calibration:
        post_calib_away = bool(calibration["post_calibration_away"])
    elif len(first_gaze_valid) >= config.POST_CALIBRATION_MIN_FRAMES:
        away_cnt = sum(1 for r in first_gaze_valid if r.is_looking_away is True)
        post_calib_away = (away_cnt / len(first_gaze_valid)) > config.POST_CALIBRATION_AWAY_RATIO
    else:
        post_calib_away = False

    calib_dict = {
        "completed": bool(calibration.get("completed", False)),
        "unstable": bool(calibration.get("unstable", False)),
        "post_calibration_away": post_calib_away,
    }

    # Blink count: number of transitions into eyes_open == False among single-face frames
    blink_count = 0
    was_open = True
    for r in single_face_frames:
        if r.eyes_open is False:
            if was_open:
                blink_count += 1
                was_open = False
        elif r.eyes_open is True:
            was_open = True

    # Static face heuristic
    match_poses = [(r.yaw, r.pitch) for r in records if r.identity == MATCH and r.yaw is not None and r.pitch is not None]
    static_face = False
    if (duration_s >= config.STATIC_FACE_MIN_DURATION_S
            and len(match_poses) >= config.STATIC_FACE_MIN_POSES
            and blink_count == 0):
        window = match_poses[-config.STATIC_FACE_WINDOW:]
        y_arr = [p[0] for p in window]
        p_arr = [p[1] for p in window]
        mean_std = (float(np.std(y_arr)) + float(np.std(p_arr))) / 2.0
        if mean_std < config.STATIC_FACE_MAX_POSE_STD:
            static_face = True

    # Similarity stats
    sims = [r.similarity for r in records if r.similarity is not None]
    thresh = config.SFACE_MATCH_THRESHOLD
    if sims:
        s_arr = np.array(sims, dtype=float)
        sim_stats = {
            "threshold": thresh,
            "count": len(sims),
            "min": round(float(np.min(s_arr)), 3),
            "p05": round(float(np.percentile(s_arr, 5)), 3),
            "median": round(float(np.median(s_arr)), 3),
            "max": round(float(np.max(s_arr)), 3),
            "near_threshold_frames": sum(1 for s in sims if abs(s - thresh) <= config.SIMILARITY_NEAR_MARGIN),
        }
    else:
        sim_stats = {
            "threshold": thresh,
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "max": None,
            "near_threshold_frames": 0,
        }

    return {
        "fps": fps,
        "period_s": period_s,
        "total_frames": total_frames,
        "matched_frames": matched_frames,
        "unmatched_frames": unmatched_frames,
        "identity_unknown_frames": identity_unknown_frames,
        "no_face_frames": no_face_frames,
        "multiple_faces_frames": multiple_faces_frames,
        "match_pct": match_pct,
        "different_person_s": different_person_s,
        "multiple_faces_s": multiple_faces_s,
        "no_face_s": no_face_s,
        "no_face_pct": no_face_pct,
        "identity_unknown_pct": identity_unknown_pct,
        "gaze_valid_frames": gaze_valid_frames,
        "gaze_unknown_frames": gaze_unknown_frames,
        "gaze_valid_s": gaze_valid_s,
        "away_s": away_s,
        "away_pct": away_pct,
        "time_on_screen_pct": time_on_screen_pct,
        "away_by_direction_s": away_by_direction,
        "long_away_intervals": long_away_intervals,
        "long_away_threshold_s": long_away_threshold_s,
        "gaze_unknown_pct": gaze_unknown_pct,
        "calibration": calib_dict,
        "blink_count": blink_count,
        "static_face": static_face,
        "similarity": sim_stats,
    }


def _report(records: List[FrameRecord], intervals: List[Dict[str, Any]],
            metrics: Dict[str, Any], face: Dict[str, Any],
            gaze: Dict[str, Any], duration_s: float) -> Dict[str, Any]:
    return {
        "session_id": "",
        "status": "completed",
        "rules_version": config.RULES_VERSION,
        "duration_s": duration_s,
        "total_frames": len(records),
        "face": face,
        "gaze": gaze,
        "overall_verdict": _overall_verdict(face["verdict"], gaze["verdict"]),
        "metrics": metrics,
        "intervals": intervals,

        # legacy (delete in section 16)
        "duration_seconds": duration_s,
        "matched_frames": metrics["matched_frames"],
        "unmatched_frames": metrics["unmatched_frames"],
        "no_face_frames": metrics["no_face_frames"],
        "multiple_faces_frames": metrics["multiple_faces_frames"],
        "match_percentage": metrics["match_pct"],
        "looking_away_percentage": metrics["away_pct"],
        "events": [],
    }


def _empty_report(duration_s: float, calibration: Dict[str, bool]) -> Dict[str, Any]:
    metrics = _metrics([], [], duration_s, config.FRAME_PERIOD_S, calibration)
    face = face_verdict(metrics)
    gaze = gaze_verdict(metrics)
    return _report([], [], metrics, face, gaze, duration_s)


def evaluate(records: List[FrameRecord], duration_s: float,
             calibration: Dict[str, bool]) -> Dict[str, Any]:
    if not records:
        return _empty_report(duration_s, calibration)
    period = _period(records)
    duration_s = round(max(duration_s, records[-1].t_s + period), 2)

    trackers = make_trackers(period)
    for r in records:
        feed_trackers(trackers, r)
    for t in trackers.values():
        t.finalize()
    intervals = sorted((iv for t in trackers.values() for iv in t.intervals),
                       key=lambda iv: iv["start_s"])

    metrics = _metrics(records, intervals, duration_s, period, calibration)
    face = face_verdict(metrics)
    gaze = gaze_verdict(metrics)
    return _report(records, intervals, metrics, face, gaze, duration_s)


# ---- Backward compatibility wrappers (section 11 rewrites test_rules.py) ----
def evaluate_face_verdict(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    v = face_verdict(metrics)
    reasons = list(v["reasons"])
    if "static_face" in reasons:
        reasons.append("static_face_detected")
    return v["verdict"], reasons


def evaluate_gaze_verdict(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    v = gaze_verdict(metrics)
    reasons = list(v["reasons"])
    if "long_away" in reasons:
        reasons.append("long_away_intervals")
    return v["verdict"], reasons


def evaluate_session_records(
    records: List[Any],
    duration_seconds: float = 0.0,
    bad_baseline: bool = False
) -> Dict[str, Any]:
    frame_records: List[FrameRecord] = []
    for r in records:
        if isinstance(r, FrameRecord):
            frame_records.append(r)
        elif isinstance(r, dict):
            identity = r.get("identity") or r.get("verification_status") or "UNKNOWN"
            frame_records.append(FrameRecord(
                t_s=float(r.get("t_s", 0.0)),
                faces_detected=int(r.get("faces_detected", 1)),
                identity=identity,
                similarity=r.get("similarity"),
                face_box=r.get("face_box"),
                yaw=r.get("yaw"),
                pitch=r.get("pitch"),
                roll=r.get("roll"),
                iris_h=r.get("iris_h"),
                iris_v=r.get("iris_v"),
                openness=r.get("openness"),
                eyes_open=r.get("eyes_open", True),
                gaze_direction=r.get("gaze_direction", "UNKNOWN"),
                is_looking_away=r.get("is_looking_away"),
                calibrating=bool(r.get("calibrating", False)),
                thumbnail=r.get("thumbnail"),
            ))

    if not bad_baseline and len(frame_records) >= 30:
        samples = [r for r in frame_records[:50] if r.faces_detected == 1 and r.yaw is not None and r.pitch is not None]
        if len(samples) >= 30:
            yaw_std = float(np.std([s.yaw for s in samples]))
            pitch_std = float(np.std([s.pitch for s in samples]))
            if yaw_std > config.CALIBRATION_MAX_STD_DEG or pitch_std > config.CALIBRATION_MAX_STD_DEG:
                bad_baseline = True

    calib = {"completed": True, "unstable": bad_baseline, "post_calibration_away": False}
    rep = evaluate(frame_records, duration_seconds, calib)
    rep["metrics"]["bad_baseline"] = bad_baseline
    if "static_face" in rep["face"]["reasons"]:
        rep["face"]["reasons"].append("static_face_detected")
    if "long_away" in rep["gaze"]["reasons"]:
        rep["gaze"]["reasons"].append("long_away_intervals")
    return rep
