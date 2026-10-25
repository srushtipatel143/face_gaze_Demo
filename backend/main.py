"""
FastAPI Backend Application for Face Verification & Eye-Gaze Tracking.
Endpoints:
- POST /api/reference: Validate reference face snapshot and initialize session.
- POST /api/verification/start: Start live stream verification for a session.
- POST /api/verification/frame: Real-time biometric face & gaze inference on a frame.
- POST /api/verification/stop: Conclude session and return comprehensive statistics.
- GET  /api/session/{session_id}: Query session state.
- GET  /api/sessions: List past completed sessions.
- GET  /api/session/{session_id}/report: Fetch stored session report.
- GET  /api/session/{session_id}/stored: Query persisted PostgreSQL session.
- GET  /api/session/{session_id}/failures: Query session failure proof images.
- GET  /health: Service health check.
"""

from contextlib import asynccontextmanager, contextmanager
import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
import db_models
from db_models import (
    get_session_from_db,
    get_session_failures,
    save_reference_session,
    save_report,
)
from evaluator import evaluate
from models.schemas import (
    BoundingBox,
    FrameVerificationRequest,
    FrameVerificationResponse,
    LiveEvent,
    LiveStatus,
    ReferencePhotoRequest,
    ReferenceResponse,
    SessionReport,
    VerificationStartRequest,
    VerificationStartResponse,
    VerificationStopRequest,
)
from session_store import (
    SessionCompleted,
    SessionNotActive,
    SessionNotFound,
    store,
)
from vision.engine import decode_base64_image, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("face_gaze.api")


@asynccontextmanager
async def lifespan(app):
    for attempt in range(1, config.DB_CONNECT_ATTEMPTS + 1):
        if db_models.init_db():
            log.info("Database connected")
            break
        log.warning("Database not ready (attempt %d/%d)", attempt, config.DB_CONNECT_ATTEMPTS)
        time.sleep(config.DB_CONNECT_RETRY_S)
    yield


@contextmanager
def session_errors():
    try:
        yield
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found. Capture a new reference photo.")
    except SessionCompleted:
        raise HTTPException(status_code=409, detail="Session already ended. Start a new session.")
    except SessionNotActive as e:
        raise HTTPException(status_code=409, detail=f"Session is not active ({e}).")


app = FastAPI(title="Identity & Attention Check API", version="1.1.0", lifespan=lifespan)

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


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request, exc):
    log.error("Response validation error: %s", exc.errors())
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
            "/api/verification/stop",
            "/api/sessions"
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
    store.cleanup_old()

    try:
        image = decode_base64_image(req.image)
    except Exception as e:
        return ReferenceResponse(
            success=False,
            faces_detected=0,
            quality_score=0.0,
            message=f"Invalid image format: {str(e)}"
        )

    res = engine.analyze_reference(image)

    if not res.get("success", False):
        return ReferenceResponse(
            success=False,
            faces_detected=res.get("faces_detected", 0),
            quality_score=res.get("quality_score"),
            issues=res.get("issues", []),
            message=res.get("message", "Reference face validation failed.")
        )

    # Store reference embedding and metadata in in-memory session store
    session_id = store.create(res["embedding"], res.get("metadata", {}))

    # Persist reference session in PostgreSQL
    try:
        save_reference_session(
            session_id=session_id,
            reference_image=req.image,
            reference_embedding=res["embedding"],
            quality_score=res.get("quality_score"),
            metadata=res.get("metadata", {})
        )
    except Exception:
        pass

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
    with session_errors():
        store.start(req.session_id)
    return VerificationStartResponse(
        session_id=req.session_id,
        status="active",
        message="Live check started."
    )


@app.post("/api/verification/frame", response_model=FrameVerificationResponse)
def analyze_live_frame(req: FrameVerificationRequest):
    with session_errors():
        reference = store.get_reference(req.session_id)
    try:
        image = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode frame: {e}")

    analysis = engine.analyze_frame(image, reference)       # no store lock held here

    with session_errors():
        update = store.add_frame(req.session_id, analysis)

    r = update.record
    return FrameVerificationResponse(
        session_id=req.session_id,
        elapsed_s=update.elapsed_s,
        faces_detected=r.faces_detected,
        verification_status=r.identity,
        similarity=r.similarity,
        is_match=(r.identity == "MATCH"),
        gaze_direction=r.gaze_direction,
        is_looking_away=r.is_looking_away,
        calibrating=r.calibrating,
        calibration_progress=update.calibration_progress,
        gaze_yaw=r.yaw,
        gaze_pitch=r.pitch,
        iris_h=r.iris_h,
        iris_v=r.iris_v,
        face_box=BoundingBox(**r.face_box) if r.face_box else None,
        live_status=LiveStatus(**update.live_status),
        events=[LiveEvent(**e) for e in update.events],
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Phase 5: Stop Verification & Final Audit Summary
# ---------------------------------------------------------------------------
@app.post("/api/verification/stop", response_model=SessionReport)
def stop_live_verification(req: VerificationStopRequest):
    with session_errors():
        session = store.get(req.session_id)
    if session.status == "completed" and session.report:
        return session.report                      # idempotent: second /stop returns same report
    with session_errors():
        data = store.stop(req.session_id)
    report = evaluate(data.records, data.duration_s, data.calibration)
    report["session_id"] = req.session_id
    store.set_report(req.session_id, report)
    save_report(req.session_id, report, data.records)
    return report


@app.get("/api/sessions")
def list_sessions(limit: int = 50):
    return {"sessions": db_models.list_sessions(min(max(limit, 1), 200))}


@app.get("/api/session/{session_id}/report")
def get_session_report(session_id: str):
    report = db_models.get_report(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/api/session/{session_id}")
def get_session_info(session_id: str):
    """Fetch current session progress and frame counts."""
    with session_errors():
        sess = store.get(session_id)
    return {
        "session_id": sess.session_id,
        "status": sess.status,
        "created_at": sess.created_at,
        "frames": len(sess.records),
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
