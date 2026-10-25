import os
import sys
from types import SimpleNamespace
import numpy as np
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from vision.face import select_faces
from vision.engine import pick_landmarks
from vision.gaze import GazeBaseline


def test_select_faces_ignores_small_background():
    """Unit test for Section 5: Small background faces (area ratio < SECOND_FACE_MIN_AREA_RATIO) are ignored."""
    # Main face: 200x200 = 40,000 area
    main_face = np.array([100, 100, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95], dtype=np.float32)
    # Small background face: 40x40 = 1,600 area (ratio = 1600 / 40000 = 0.04 < 0.30)
    small_face = np.array([450, 50, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.85], dtype=np.float32)

    # Order with small face first to ensure sorting by area works properly
    faces = np.array([small_face, main_face], dtype=np.float32)

    main, count = select_faces(faces)

    assert count == 1, f"Expected 1 face counted, got {count}"
    assert np.array_equal(main, main_face), "Expected main face to be the largest box"


def test_select_faces_counts_second_large_face():
    """A second face with area ratio >= SECOND_FACE_MIN_AREA_RATIO is counted."""
    # Main face: 200x200 = 40,000 area
    main_face = np.array([100, 100, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95], dtype=np.float32)
    # Second face: 150x150 = 22,500 area (ratio = 22500 / 40000 = 0.5625 >= 0.30)
    second_face = np.array([350, 100, 150, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90], dtype=np.float32)
    # Third small face: 30x30 = 900 area (ignored)
    small_face = np.array([550, 50, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.82], dtype=np.float32)

    faces = np.array([small_face, second_face, main_face], dtype=np.float32)

    main, count = select_faces(faces)

    assert count == 2, f"Expected 2 faces counted, got {count}"
    assert np.array_equal(main, main_face)


def test_select_faces_empty():
    faces = np.empty((0, 15), dtype=np.float32)
    main, count = select_faces(faces)
    assert main is None
    assert count == 0


def _make_mock_face_landmarks(center_x: float, center_y: float, img_w: int, img_h: int):
    # Normalized coords so p.x * img_w = center_x, etc.
    p1 = SimpleNamespace(x=(center_x - 10) / img_w, y=(center_y - 10) / img_h, z=0.0)
    p2 = SimpleNamespace(x=(center_x + 10) / img_w, y=(center_y + 10) / img_h, z=0.0)
    return SimpleNamespace(landmark=[p1, p2])


def test_pick_landmarks_aligns_closest_face():
    img_w, img_h = 640, 480
    box = {"x": 100, "y": 100, "width": 200, "height": 200}
    # Box centre is cx = 200, cy = 200

    # FaceMesh face 1: centre at (500, 400) - far
    lm1 = _make_mock_face_landmarks(500, 400, img_w, img_h)
    # FaceMesh face 2: centre at (202, 198) - very close to box centre
    lm2 = _make_mock_face_landmarks(202, 198, img_w, img_h)

    mesh_results = SimpleNamespace(multi_face_landmarks=[lm1, lm2])

    picked = pick_landmarks(mesh_results, box, img_w, img_h)
    assert picked is lm2, "Should pick the landmark whose centre is closest to the main face box"


def test_pick_landmarks_rejects_far_away():
    img_w, img_h = 640, 480
    box = {"x": 100, "y": 100, "width": 100, "height": 100}
    # Box centre is cx = 150, cy = 150, width = 100

    # Only one face mesh result, but far away (> 100 px away)
    lm = _make_mock_face_landmarks(350, 350, img_w, img_h)
    mesh_results = SimpleNamespace(multi_face_landmarks=[lm])

    picked = pick_landmarks(mesh_results, box, img_w, img_h)
    assert picked is None, "Should reject landmarks more than one box width away"


def test_pick_landmarks_no_faces():
    box = {"x": 100, "y": 100, "width": 100, "height": 100}
    mesh_results = SimpleNamespace(multi_face_landmarks=None)
    assert pick_landmarks(mesh_results, box, 640, 480) is None


def test_baseline_completes_after_time_and_samples():
    """Calibration completes only when both CALIBRATION_SECONDS and CALIBRATION_MIN_SAMPLES are satisfied."""
    baseline = GazeBaseline()
    assert not baseline.ready
    assert baseline.progress(0.0) == 0.0

    # Add 19 samples over 4.75 seconds (insufficient samples and insufficient time)
    for i in range(19):
        t = i * 0.25
        baseline.add_calibration_sample(t, yaw=1.0, pitch=-2.0, iris_h=0.5, iris_v=0.5)
        assert not baseline.ready
        assert baseline.progress(t) < 1.0

    # Add 20th sample at t=4.9s (samples reached 20, but time < 5.0s)
    baseline.add_calibration_sample(4.9, yaw=1.0, pitch=-2.0, iris_h=0.5, iris_v=0.5)
    assert not baseline.ready
    assert baseline.progress(4.9) < 1.0

    # Add 21st sample at t=5.1s (both time >= 5.0s and samples >= 20 met)
    baseline.add_calibration_sample(5.1, yaw=1.0, pitch=-2.0, iris_h=0.5, iris_v=0.5)
    assert baseline.ready
    assert baseline.progress(5.1) == 1.0
    assert baseline.values == {"yaw": 1.0, "pitch": -2.0, "iris_h": 0.5, "iris_v": 0.5}
    assert baseline.unstable is False
    assert baseline.summary() == {"completed": True, "unstable": False}


def test_baseline_trailing_ignores_away():
    """After calibration, on-screen samples update the trailing window while away samples are ignored."""
    baseline = GazeBaseline()

    # Pre-calibration: on-screen samples should be ignored
    baseline.add_on_screen_sample(yaw=5.0, pitch=5.0, iris_h=0.5, iris_v=0.5)
    assert len(baseline.trailing) == 0

    # Complete calibration with yaw=0.0, pitch=0.0
    for i in range(25):
        t = i * 0.25
        baseline.add_calibration_sample(t, yaw=0.0, pitch=0.0, iris_h=0.5, iris_v=0.5)

    assert baseline.ready
    assert baseline.values["yaw"] == 0.0

    # Away samples should never enter the trailing baseline window (caller filters them)
    # Simulate on-screen samples with a shifted center (e.g. user shifted posture slightly: yaw=3.0)
    window_size = config.TRAILING_BASELINE_WINDOW
    for _ in range(window_size - 1):
        baseline.add_on_screen_sample(yaw=3.0, pitch=1.0, iris_h=0.52, iris_v=0.48)
        # Window not full yet, values remain calibrated values
        assert baseline.values["yaw"] == 0.0

    # Adding the final sample fills the trailing window and updates baseline to new median
    baseline.add_on_screen_sample(yaw=3.0, pitch=1.0, iris_h=0.52, iris_v=0.48)
    assert len(baseline.trailing) == window_size
    assert baseline.values["yaw"] == 3.0
    assert baseline.values["pitch"] == 1.0

