from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import config
from records import FrameRecord, MATCH, NO_MATCH, UNKNOWN, NO_FACE, MULTIPLE_FACES

INTERVAL_TYPES = ("looking_away", "different_person", "multiple_faces", "no_face")
DIRECTION_WORDS = {"LOOKING_LEFT": "left", "LOOKING_RIGHT": "right",
                   "LOOKING_UP": "up", "LOOKING_DOWN": "down"}


def interval_message(iv: Dict[str, Any]) -> str:
    t = iv["type"]
    if t == "looking_away":
        word = DIRECTION_WORDS.get(iv.get("direction") or "")
        return f"Looked {word}" if word else "Looked away"
    if t == "no_face":
        return "Face not visible"
    if t == "multiple_faces":
        return "Another person in view"
    if t == "different_person":
        return "Different person in view"
    return t


class IntervalTracker:
    """Opens an interval after open_after anomaly frames, closes it after close_after ok frames.
    'unknown' frames neither count nor reset, but a gap of unknown frames longer than
    max_gap_s after the last anomaly ends the run (and closes an open interval)."""

    def __init__(self, interval_type: str, open_after: int, close_after: int,
                 max_gap_s: float = config.MAX_UNKNOWN_GAP_S, period_s: float = config.FRAME_PERIOD_S) -> None:
        self.interval_type = interval_type
        self.open_after = open_after
        self.close_after = close_after
        self.max_gap_s = max_gap_s
        self.period_s = period_s
        self.intervals: List[Dict[str, Any]] = []
        self.current: Optional[Dict[str, Any]] = None
        self._count = 0
        self._reset_run()

    def _reset_run(self) -> None:
        self.run_start: Optional[float] = None
        self.run_len = 0
        self.ok_len = 0
        self.last_anomaly_t: Optional[float] = None
        self.run_thumbnail: Optional[str] = None
        self.run_labels: Counter = Counter()

    def update(self, t_s: float, state: Optional[str] = None, label: Optional[str] = None,
               thumbnail: Optional[str] = None, **kwargs) -> Optional[Tuple[str, Dict[str, Any]]]:
        """state is 'anomaly', 'ok' or 'unknown'.
        Returns ('opened', interval) or ('closed', interval) or None."""
        if "is_unknown" in kwargs or "is_anomaly" in kwargs:
            if kwargs.get("is_unknown"):
                state = "unknown"
            elif kwargs.get("is_anomaly"):
                state = "anomaly"
            else:
                state = "ok"
        elif state is None:
            state = "ok"

        if label is None and "details" in kwargs:
            label = kwargs["details"]

        if state == "unknown":
            if self.last_anomaly_t is not None and t_s - self.last_anomaly_t > self.max_gap_s:
                closed = self._close()
                self._reset_run()
                return ("closed", closed) if closed else None
            return None

        if state == "anomaly":
            self.ok_len = 0
            if self.run_start is None:
                self.run_start = t_s
                self.run_thumbnail = thumbnail
            elif self.run_thumbnail is None and thumbnail is not None:
                self.run_thumbnail = thumbnail
            self.run_len += 1
            self.last_anomaly_t = t_s
            if label:
                self.run_labels[label] += 1
            if self.current is None and self.run_len >= self.open_after:
                self._count += 1
                self.current = {
                    "id": f"{self.interval_type}-{self._count}",
                    "type": self.interval_type,
                    "start_s": round(self.run_start, 2),
                    "end_s": None,
                    "duration_s": None,
                    "direction": self._direction(),
                    "thumbnail": self.run_thumbnail,
                    "evidence_image": self.run_thumbnail,
                }
                self.current["message"] = interval_message(self.current)
                return ("opened", dict(self.current))
            return None

        # state == "ok"
        self.ok_len += 1
        if self.ok_len >= self.close_after:
            closed = self._close()
            self._reset_run()
            return ("closed", closed) if closed else None
        return None

    def finalize(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Close an interval still open at the end of the session."""
        closed = self._close()
        self._reset_run()
        return closed

    def _direction(self) -> Optional[str]:
        return self.run_labels.most_common(1)[0][0] if self.run_labels else None

    def _close(self) -> Optional[Dict[str, Any]]:
        if self.current is None:
            return None
        end = (self.last_anomaly_t + self.period_s) if self.last_anomaly_t is not None \
            else self.current["start_s"] + self.period_s
        self.current["end_s"] = round(end, 2)
        self.current["duration_s"] = round(max(0.0, self.current["end_s"] - self.current["start_s"]), 2)
        self.current["direction"] = self._direction()
        self.current["message"] = interval_message(self.current)
        thumb = self.current.get("thumbnail") or self.run_thumbnail
        self.current["thumbnail"] = thumb
        self.current["evidence_image"] = thumb
        closed = dict(self.current)
        self.intervals.append(closed)
        self.current = None
        return closed


def make_trackers(period_s: float) -> Dict[str, IntervalTracker]:
    g = dict(open_after=config.GAZE_OPEN_AFTER, close_after=config.GAZE_CLOSE_AFTER,
             max_gap_s=config.MAX_UNKNOWN_GAP_S, period_s=period_s)
    i = dict(open_after=config.IDENTITY_OPEN_AFTER, close_after=config.IDENTITY_CLOSE_AFTER,
             max_gap_s=config.MAX_UNKNOWN_GAP_S, period_s=period_s)
    return {
        "looking_away": IntervalTracker("looking_away", **g),
        "different_person": IntervalTracker("different_person", **i),
        "multiple_faces": IntervalTracker("multiple_faces", **i),
        "no_face": IntervalTracker("no_face", **i),
    }


def feed_trackers(trackers: Dict[str, IntervalTracker],
                  rec: FrameRecord) -> List[Tuple[str, Dict[str, Any]]]:
    """Single source of truth for which frame state counts as what."""
    changes = []

    def push(result):
        if result:
            changes.append(result)

    t, thumb = rec.t_s, rec.thumbnail
    if rec.identity == NO_FACE:
        push(trackers["no_face"].update(t, "anomaly", thumbnail=thumb))
        push(trackers["multiple_faces"].update(t, "ok"))
        push(trackers["different_person"].update(t, "unknown"))
    elif rec.identity == MULTIPLE_FACES:
        push(trackers["multiple_faces"].update(t, "anomaly", thumbnail=thumb))
        push(trackers["no_face"].update(t, "ok"))
        push(trackers["different_person"].update(t, "unknown"))
    else:
        push(trackers["no_face"].update(t, "ok"))
        push(trackers["multiple_faces"].update(t, "ok"))
        if rec.identity == MATCH:
            push(trackers["different_person"].update(t, "ok"))
        elif rec.identity == NO_MATCH:
            push(trackers["different_person"].update(t, "anomaly", thumbnail=thumb))
        else:  # UNKNOWN
            push(trackers["different_person"].update(t, "unknown"))

    # Gaze is not judged during calibration.
    if not rec.calibrating:
        if rec.is_looking_away is True:
            push(trackers["looking_away"].update(t, "anomaly", label=rec.gaze_direction, thumbnail=thumb))
        elif rec.is_looking_away is False:
            push(trackers["looking_away"].update(t, "ok"))
        else:
            push(trackers["looking_away"].update(t, "unknown"))
    return changes


def interval_to_event(iv: Dict[str, Any]) -> Dict[str, Any]:
    """Live-screen event (no thumbnail)."""
    return {k: iv.get(k) for k in ("id", "type", "start_s", "end_s", "duration_s", "direction", "message")}
