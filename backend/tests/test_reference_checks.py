import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.engine import engine


def test_analyze_reference_no_face():
    # Empty / black image with no face
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.analyze_reference(img)

    assert res["success"] is False
    assert res["faces_detected"] == 0
    assert len(res["checks"]) == 7

    checks = res["checks"]
    assert checks[0]["id"] == "face_detected"
    assert checks[0]["passed"] is False
    assert checks[0]["detail"] == "No face found. Face the camera in good light."

    # All subsequent checks should be null (passed is None)
    for c in checks[1:]:
        assert c["passed"] is None
        assert c["detail"] is None


def test_analyze_reference_checks_order():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.analyze_reference(img)
    expected_ids = [
        "face_detected",
        "single_face",
        "face_size",
        "sharpness",
        "lighting",
        "facing_camera",
        "eyes_open",
    ]
    actual_ids = [c["id"] for c in res["checks"]]
    assert actual_ids == expected_ids
