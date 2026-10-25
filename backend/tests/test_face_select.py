import os
import sys
import numpy as np
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.face import select_faces


def test_select_faces_ignores_small_background():
    """Rows with areas 10000 and 2000 -> count 1 (2000 < 0.3 x 10000); with 10000 and 4000 -> count 2."""
    # Main face: 100x100 = 10,000 area (row index 2 is width, 3 is height)
    main_face = np.array([50, 50, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95], dtype=np.float32)
    # Small background face: w=50, h=40 -> area 2000 (ratio 2000 / 10000 = 0.20 < 0.30)
    small_face = np.array([200, 50, 50, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.85], dtype=np.float32)

    # First test: 10000 and 2000 -> count 1
    faces_1 = np.array([small_face, main_face], dtype=np.float32)
    main_res_1, count_1 = select_faces(faces_1)
    assert count_1 == 1
    assert np.array_equal(main_res_1, main_face)

    # Second face: w=80, h=50 -> area 4000 (ratio 4000 / 10000 = 0.40 >= 0.30)
    larger_face = np.array([300, 50, 80, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.88], dtype=np.float32)

    # Second test: 10000 and 4000 -> count 2
    faces_2 = np.array([larger_face, main_face], dtype=np.float32)
    main_res_2, count_2 = select_faces(faces_2)
    assert count_2 == 2
    assert np.array_equal(main_res_2, main_face)


def test_select_faces_empty():
    faces = np.empty((0, 15), dtype=np.float32)
    main, count = select_faces(faces)
    assert main is None
    assert count == 0


def test_select_faces_single():
    single_face = np.array([[10, 10, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]], dtype=np.float32)
    main, count = select_faces(single_face)
    assert count == 1
    assert np.array_equal(main, single_face[0])
