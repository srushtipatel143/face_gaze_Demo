from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ReferencePhotoRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG/PNG image data or data URL")

class ReferenceResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    faces_detected: int
    quality_score: Optional[float] = None
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
    verification_status: str  # "MATCH", "NO_MATCH", "NO_FACE", "MULTIPLE_FACES"
    similarity: Optional[float] = None
    is_match: bool
    gaze_direction: str  # "CENTER", "LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_UP", "LOOKING_DOWN", "UNKNOWN"
    is_looking_away: bool
    gaze_yaw: Optional[float] = None
    gaze_pitch: Optional[float] = None
    face_box: Optional[BoundingBox] = None
    timestamp: float

class VerificationStopRequest(BaseModel):
    session_id: str

class GazeEvent(BaseModel):
    id: str
    timestamp: float
    type: str
    details: Dict[str, Any]

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
    overall_verdict: str  # "PASSED", "FAILED", "SUSPICIOUS"
    events: List[GazeEvent]
