import os
import sys
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from intervals import IntervalTracker, make_trackers, feed_trackers
from records import FrameRecord, MATCH, NO_MATCH, NO_FACE, MULTIPLE_FACES, UNKNOWN


def test_tracker_opens_and_closes():
    """5 anomaly frames then 3 ok frames, period 0.2:
    one opened on frame 5, one closed; interval start 0.0, end 1.0 (last anomaly 0.8 + 0.2), duration 1.0"""
    tracker = IntervalTracker("looking_away", open_after=5, close_after=3, period_s=0.2)

    # 5 anomaly frames: t = 0.0, 0.2, 0.4, 0.6, 0.8
    results = []
    for i in range(5):
        res = tracker.update(t_s=round(i * 0.2, 3), state="anomaly", label="LOOKING_LEFT")
        results.append(res)

    # Frame 0 to 3 should not open an interval
    assert all(r is None for r in results[:4])
    # Frame 4 (5th anomaly frame) should open the interval
    assert results[4] is not None
    assert results[4][0] == "opened"
    opened = results[4][1]
    assert opened["start_s"] == 0.0
    assert opened["end_s"] is None

    # 3 ok frames: t = 1.0, 1.2, 1.4
    ok_results = []
    for i in range(5, 8):
        res = tracker.update(t_s=round(i * 0.2, 3), state="ok")
        ok_results.append(res)

    assert ok_results[0] is None
    assert ok_results[1] is None
    # 3rd ok frame closes the interval
    assert ok_results[2] is not None
    assert ok_results[2][0] == "closed"
    closed = ok_results[2][1]

    assert len(tracker.intervals) == 1
    iv = tracker.intervals[0]
    assert iv["start_s"] == 0.0
    assert iv["end_s"] == 1.0   # last anomaly (0.8) + period (0.2)
    assert iv["duration_s"] == 1.0


def test_tracker_short_blip_ignored():
    """4 anomaly frames then 3 ok -> no interval opened or created."""
    tracker = IntervalTracker("looking_away", open_after=5, close_after=3, period_s=0.2)

    for i in range(4):
        res = tracker.update(t_s=round(i * 0.2, 3), state="anomaly")
        assert res is None

    for i in range(4, 7):
        res = tracker.update(t_s=round(i * 0.2, 3), state="ok")
        assert res is None

    assert len(tracker.intervals) == 0
    assert tracker.current is None


def test_tracker_unknown_gap_closes():
    """Open an interval, then unknown frames for 1.2 s -> closed when gap > MAX_UNKNOWN_GAP_S."""
    tracker = IntervalTracker("looking_away", open_after=5, close_after=3, max_gap_s=1.0, period_s=0.2)

    # Open interval with 5 anomalies: t = 0.0, 0.2, 0.4, 0.6, 0.8
    for i in range(5):
        tracker.update(t_s=round(i * 0.2, 3), state="anomaly")

    assert tracker.current is not None

    # Unknown frame at t = 1.2s (gap = 1.2 - 0.8 = 0.4s <= 1.0s) -> should not close yet
    res1 = tracker.update(t_s=1.2, state="unknown")
    assert res1 is None
    assert tracker.current is not None

    # Unknown frame at t = 2.05s (gap = 2.05 - 0.8 = 1.25s > 1.0s MAX_UNKNOWN_GAP_S) -> closes!
    res2 = tracker.update(t_s=2.05, state="unknown")
    assert res2 is not None
    assert res2[0] == "closed"
    closed = res2[1]

    assert len(tracker.intervals) == 1
    assert closed["start_s"] == 0.0
    assert closed["end_s"] == 1.0
    assert closed["duration_s"] == 1.0
    assert tracker.current is None


def test_tracker_direction_majority():
    """Away frames labelled LEFT x6, RIGHT x2 -> closed interval direction == 'LOOKING_LEFT', message 'Looked left'."""
    tracker = IntervalTracker("looking_away", open_after=5, close_after=3, period_s=0.2)

    # 6 frames LEFT: t = 0.0 to 1.0
    for i in range(6):
        tracker.update(t_s=round(i * 0.2, 3), state="anomaly", label="LOOKING_LEFT")

    # 2 frames RIGHT: t = 1.2, 1.4
    tracker.update(t_s=1.2, state="anomaly", label="LOOKING_RIGHT")
    tracker.update(t_s=1.4, state="anomaly", label="LOOKING_RIGHT")

    # 3 frames ok: t = 1.6, 1.8, 2.0
    tracker.update(t_s=1.6, state="ok")
    tracker.update(t_s=1.8, state="ok")
    tracker.update(t_s=2.0, state="ok")

    assert len(tracker.intervals) == 1
    iv = tracker.intervals[0]
    assert iv["direction"] == "LOOKING_LEFT"
    assert iv["message"] == "Looked left"


def test_tracker_finalize_closes_open_interval():
    """An interval that is still open when session finishes is closed by finalize()."""
    tracker = IntervalTracker("different_person", open_after=3, close_after=2, period_s=0.2)
    tracker.update(t_s=0.0, state="anomaly")
    tracker.update(t_s=0.2, state="anomaly")
    tracker.update(t_s=0.4, state="anomaly")
    assert tracker.current is not None

    closed = tracker.finalize()
    assert closed is not None
    assert closed["start_s"] == 0.0
    assert closed["end_s"] == 0.6
    assert closed["duration_s"] == 0.6
    assert len(tracker.intervals) == 1


def test_feed_trackers_behavior():
    trackers = make_trackers(period_s=0.2)

    # 1. NO_FACE frame
    rec_nf = FrameRecord(t_s=0.0, faces_detected=0, identity=NO_FACE)
    feed_trackers(trackers, rec_nf)
    assert trackers["no_face"].run_len == 1
    assert trackers["multiple_faces"].run_len == 0

    # 2. MULTIPLE_FACES frame
    rec_mf = FrameRecord(t_s=0.2, faces_detected=2, identity=MULTIPLE_FACES)
    feed_trackers(trackers, rec_mf)
    assert trackers["multiple_faces"].run_len == 1

    # 3. Calibrating frame does not feed gaze tracker
    rec_cal = FrameRecord(t_s=0.4, faces_detected=1, identity=MATCH, calibrating=True, is_looking_away=True)
    feed_trackers(trackers, rec_cal)
    assert trackers["looking_away"].run_len == 0
