from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ReferencePhotoRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG/PNG image data or data URL")

class ReferenceResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    faces_detected: int
    quality_score: Optional[float] = None
    issues: List[str] = Field(default_factory=list)
    message: str

class VerificationStartRequest(BaseModel):
    session_id: str

class VerificationStartResponse(BaseModel):
    session_id: str
    status: str
    message: str

class FrameVerificationRequest(BaseModel):
    session_id: str
    image: str = Field(..., description="Base64 encoded live frame image")

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class FrameVerificationResponse(BaseModel):
    session_id: str
    faces_detected: int
    verification_status: str  # "MATCH", "NO_MATCH", "NO_FACE", "MULTIPLE_FACES", "UNKNOWN"
    similarity: Optional[float] = None
    is_match: bool
    gaze_direction: str  # "CENTER", "LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_UP", "LOOKING_DOWN", "CALIBRATING", "UNKNOWN"
    is_looking_away: Optional[bool] = None
    gaze_yaw: Optional[float] = None
    gaze_pitch: Optional[float] = None
    iris_h: Optional[float] = None
    iris_v: Optional[float] = None
    calibrating: Optional[bool] = None
    face_box: Optional[BoundingBox] = None
    timestamp: float

class VerificationStopRequest(BaseModel):
    session_id: str

class GazeEvent(BaseModel):
    id: str
    timestamp: float
    type: str
    details: Dict[str, Any]

class VerdictDetail(BaseModel):
    verdict: str  # "PASS", "FAIL", "REVIEW"
    reasons: List[str] = Field(default_factory=list)

class SessionMetrics(BaseModel):
    valid_frames: int
    fps: float
    different_person_s: float
    multiple_faces_s: float
    no_face_s: float
    no_face_pct: float
    identity_unknown_s: float
    identity_unknown_pct: float
    gaze_valid_frames: int
    away_s: float
    away_pct: float
    long_away_intervals: int
    gaze_unknown_pct: float
    bad_baseline: Optional[bool] = False
    static_face: Optional[bool] = False

class SessionSummaryResponse(BaseModel):
    session_id: str
    status: str
    duration_seconds: float
    total_frames: int
    matched_frames: int
    unmatched_frames: int
    no_face_frames: int
    multiple_faces_frames: int
    match_percentage: float
    looking_away_percentage: float
    face: VerdictDetail
    gaze: VerdictDetail
    metrics: SessionMetrics
    overall_verdict: Optional[str] = None  # Kept for backward compatibility
    events: List[GazeEvent]
    intervals: List[Dict[str, Any]] = []
