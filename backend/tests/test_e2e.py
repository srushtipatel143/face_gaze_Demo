import base64
import os
import sys
from pathlib import Path
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
PERSON_A_1 = FIXTURES_DIR / "person_a_1.jpg"
PERSON_A_2 = FIXTURES_DIR / "person_a_2.jpg"
PERSON_B = FIXTURES_DIR / "person_b.jpg"

ALL_FIXTURES_EXIST = PERSON_A_1.exists() and PERSON_A_2.exists() and PERSON_B.exists()


def _file_to_data_url(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@pytest.mark.e2e
@pytest.mark.skipif(not ALL_FIXTURES_EXIST, reason="E2E test photos missing in backend/tests/fixtures")
def test_full_verification_e2e():
    """End-to-end verification pipeline test using real fixture photos."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. POST /api/reference with person_a_1 -> success
    ref_b64 = _file_to_data_url(PERSON_A_1)
    resp = client.post("/api/reference", json={"image": ref_b64})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    session_id = data["session_id"]
    assert session_id is not None

    # 2. POST /start
    start_resp = client.post("/start", json={"session_id": session_id})
    assert start_resp.status_code == 200, start_resp.text
    assert start_resp.json()["status"] == "active"

    # 3. POST 10 frames of person_a_2 -> every verification_status == "MATCH"
    a2_b64 = _file_to_data_url(PERSON_A_2)
    for _ in range(10):
        frame_resp = client.post("/frame", json={"session_id": session_id, "image": a2_b64})
        assert frame_resp.status_code == 200, frame_resp.text
        f_data = frame_resp.json()
        assert f_data["verification_status"] == "MATCH"

    # 4. POST 10 frames of person_b -> NO_MATCH
    b_b64 = _file_to_data_url(PERSON_B)
    for _ in range(10):
        frame_resp = client.post("/frame", json={"session_id": session_id, "image": b_b64})
        assert frame_resp.status_code == 200, frame_resp.text
        f_data = frame_resp.json()
        assert f_data["verification_status"] == "NO_MATCH"

    # 5. POST /stop -> report has a different_person interval
    stop_resp = client.post("/stop", json={"session_id": session_id})
    assert stop_resp.status_code == 200, stop_resp.text
    report = stop_resp.json()
    assert report["status"] == "completed"
    diff_person_intervals = [iv for iv in report["intervals"] if iv["type"] == "different_person"]
    assert len(diff_person_intervals) >= 1
