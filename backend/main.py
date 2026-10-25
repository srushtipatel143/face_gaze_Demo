"""
FastAPI Backend Application for Face Verification & Eye-Gaze Tracking.
Endpoints:
- POST /api/reference: Validate reference face snapshot and initialize session.
- POST /api/verification/start: Start live stream verification for a session.
- POST /api/verification/frame: Real-time biometric face & gaze inference on a frame.
- POST /api/verification/stop: Conclude session and return comprehensive statistics.
- GET  /api/session/{session_id}: Query session state.
- GET  /health: Service health check.
"""

import os
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from database import db
from db_models import get_session_from_db, get_session_failures
from models.schemas import (
    ReferencePhotoRequest,
    ReferenceResponse,
    VerificationStartRequest,
    VerificationStartResponse,
    FrameVerificationRequest,
    FrameVerificationResponse,
    VerificationStopRequest,
    SessionSummaryResponse,
    BoundingBox
)
from vision_engine import engine

app = FastAPI(
    title="Face Verification & Eye-Gaze Tracking API",
    description="Real-time biometric facial recognition and head/eye-gaze orientation inference service.",
    version="1.0.0"
)

# Configure CORS with explicit allowed origins and allow_credentials=False (L6)
allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
)
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request, exc):
    print("RESPONSE VALIDATION ERROR:", exc.errors())
    return JSONResponse(
        status_code=500,
        content={"detail": "Response validation error", "errors": exc.errors()}
    )

@app.get("/")
def home():
    return {
        "service": "Face Verification & Eye-Gaze AI API",
        "status": "online",
        "endpoints": [
            "/api/reference",
            "/api/verification/start",
            "/api/verification/frame",
            "/api/verification/stop"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

# ---------------------------------------------------------------------------
# Phase 2: Reference Photo Upload & Session Creation
# ---------------------------------------------------------------------------
@app.post("/api/reference", response_model=ReferenceResponse)
def receive_reference_photo(req: ReferencePhotoRequest):
    """
    Validates uploaded reference snapshot:
    1. Decodes base64 image data
    2. Validates single face presence and quality metrics (blur, lighting, resolution)
    3. Generates invariant 3D landmark biometric reference embedding
    4. Creates a temporary verification session
    """
    # Clean up stale verification sessions (L5)
    db.cleanup_old_sessions()

    try:
        image = engine.decode_base64_image(req.image)
    except Exception as e:
        return ReferenceResponse(
            success=False,
            faces_detected=0,
            quality_score=0.0,
            message=f"Invalid image format: {str(e)}"
        )

    res = engine.process_reference_image(image)

    if not res.get("success", False):
        return ReferenceResponse(
            success=False,
            faces_detected=res.get("faces_detected", 0),
            quality_score=res.get("quality_score"),
            issues=res.get("issues", []),
            message=res.get("message", "Reference face validation failed.")
        )

    # Store reference embedding, image, and metadata in database session
    session_id = db.create_session(
        reference_embedding=res["embedding"],
        image_metadata=res.get("metadata", {}),
        reference_image=req.image
    )

    return ReferenceResponse(
        success=True,
        session_id=session_id,
        faces_detected=1,
        quality_score=res.get("quality_score", 100.0),
        issues=[],
        message="Reference face validated. Session created successfully."
    )

# ---------------------------------------------------------------------------
# Phase 3 & 4: Live Verification Stream & Gaze Tracking
# ---------------------------------------------------------------------------
@app.post("/api/verification/start", response_model=VerificationStartResponse)
def start_live_verification(req: VerificationStartRequest):
    """Marks the specified session as active for live frame ingestion."""
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found. Please capture a reference photo first."
        )

    if sess["status"] == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session has already completed. Please capture a new reference photo to start a new session."
        )

    started = db.start_verification(req.session_id)
    if not started:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to start session verification."
        )

    return VerificationStartResponse(
        session_id=req.session_id,
        status="active",
        message="Live verification session started. Ingesting webcam frames."
    )

@app.post("/api/verification/frame", response_model=FrameVerificationResponse)
def analyze_live_frame(req: FrameVerificationRequest):
    """
    Analyzes an incoming webcam frame:
    - Verifies face match against reference embedding
    - Determines gaze direction & looking-away flag relative to baseline (M4)
    - Calculates 3D head pose (yaw/pitch) and bounding box
    """
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found."
        )

    if sess["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session '{req.session_id}' is {sess['status']}, not active."
        )

    try:
        frame_image = engine.decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode frame image: {str(e)}"
        )

    ref_embedding = sess["reference_embedding"]
    baseline = sess.get("baseline")
    result = engine.process_live_frame(frame_image, ref_embedding, baseline=baseline)

    # Record metrics and detect state change events in session database
    db.add_frame_result(req.session_id, result)

    box = None
    if result.get("face_box"):
        fb = result["face_box"]
        box = BoundingBox(
            x=fb["x"],
            y=fb["y"],
            width=fb["width"],
            height=fb["height"]
        )

    return FrameVerificationResponse(
        session_id=req.session_id,
        faces_detected=result["faces_detected"],
        verification_status=result["verification_status"],
        similarity=result.get("similarity"),
        is_match=result["is_match"],
        gaze_direction=result["gaze_direction"],
        is_looking_away=result["is_looking_away"],
        gaze_yaw=result.get("gaze_yaw"),
        gaze_pitch=result.get("gaze_pitch"),
        iris_h=result.get("iris_h"),
        iris_v=result.get("iris_v"),
        calibrating=result.get("calibrating"),
        face_box=box,
        timestamp=time.time()
    )

# ---------------------------------------------------------------------------
# Phase 5: Stop Verification & Final Audit Summary
# ---------------------------------------------------------------------------
@app.post("/api/verification/stop", response_model=SessionSummaryResponse)
def stop_live_verification(req: VerificationStopRequest):
    """
    Stops live verification and calculates aggregate compliance metrics,
    match percentages, gaze deviations, and timestamped event logs.
    """
    try:
        summary = db.stop_session(req.session_id)
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{req.session_id}' not found."
            )

        return SessionSummaryResponse(**summary)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in stop_live_verification: {tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate session summary: {str(e)}"
        )

@app.get("/api/session/{session_id}")
def get_session_info(session_id: str):
    """Fetch current session progress and frame counts."""
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    # Return safe representation without the numpy embedding
    return {
        "session_id": sess["session_id"],
        "status": sess["status"],
        "created_at": sess["created_at"],
        "start_time": sess["start_time"],
        "total_frames": sess["total_frames"],
        "matched_frames": sess["matched_frames"],
        "unmatched_frames": sess["unmatched_frames"],
        "no_face_frames": sess["no_face_frames"],
        "multiple_faces_frames": sess["multiple_faces_frames"],
        "looking_away_frames": sess["looking_away_frames"],
        "center_gaze_frames": sess["center_gaze_frames"],
        "events_count": len(sess["events"])
    }

@app.get("/api/session/{session_id}/stored")
def get_stored_session_data(session_id: str):
    """Retrieve persisted session record, verdicts, and compliance data from PostgreSQL."""
    data = get_session_from_db(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found in PostgreSQL database."
        )
    return data

@app.get("/api/session/{session_id}/failures")
def get_failure_proof_images(session_id: str):
    """Retrieve all proof-of-failure visual snapshots and violation records for this session."""
    failures = get_session_failures(session_id)
    return {
        "session_id": session_id,
        "total_failures": len(failures),
        "failures": failures
    }

