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

import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from database import db
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

# Enable CORS for local Vite dev server and production preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            message=res.get("message", "Reference face validation failed.")
        )

    # Store reference embedding and metadata in database session
    session_id = db.create_session(
        reference_embedding=res["embedding"],
        image_metadata=res.get("metadata", {})
    )

    return ReferenceResponse(
        success=True,
        session_id=session_id,
        faces_detected=1,
        quality_score=res.get("quality_score", 100.0),
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

    db.start_verification(req.session_id)
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
    - Determines gaze direction & looking-away flag
    - Calculates 3D head pose (yaw/pitch) and bounding box
    """
    sess = db.get_session(req.session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found."
        )

    if sess["status"] not in ["active", "initialized"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is currently {sess['status']}, not active."
        )

    try:
        frame_image = engine.decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode frame image: {str(e)}"
        )

    ref_embedding = sess["reference_embedding"]
    result = engine.process_live_frame(frame_image, ref_embedding)

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
    summary = db.stop_session(req.session_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found."
        )

    return SessionSummaryResponse(**summary)

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
