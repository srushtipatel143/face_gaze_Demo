from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class ReferencePhotoRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG/PNG image data or data URL")

class ReferenceCheck(BaseModel):
    id: str
    label: str
    passed: Optional[bool] = None
    detail: Optional[str] = None


class ReferenceResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    faces_detected: int
    quality_score: Optional[float] = None
    checks: List[ReferenceCheck]
    issues: List[str] = Field(default_factory=list)   # = details of failed checks, kept for compatibility
    message: str        # "Photo accepted." or "Photo not accepted. See the checks below."

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


class LiveStatus(BaseModel):
    identity: str     # VERIFIED | NOT_MATCHED | NO_FACE | MULTIPLE_FACES | UNCERTAIN | CHECKING
    attention: str    # ON_SCREEN | AWAY | UNKNOWN | CALIBRATING


class LiveEvent(BaseModel):
    id: str                          # e.g. "looking_away-3"; same id is sent again when it closes
    type: str                        # looking_away | different_person | multiple_faces | no_face
    start_s: float
    end_s: Optional[float] = None    # None while open
    duration_s: Optional[float] = None
    direction: Optional[str] = None  # LOOKING_LEFT | LOOKING_RIGHT | LOOKING_UP | LOOKING_DOWN
    message: str


class FrameVerificationResponse(BaseModel):
    session_id: str
    elapsed_s: float = 0.0
    faces_detected: int
    verification_status: str          # MATCH | NO_MATCH | UNKNOWN | NO_FACE | MULTIPLE_FACES
    similarity: Optional[float] = None
    is_match: bool
    gaze_direction: str               # CENTER | LOOKING_* | CALIBRATING | UNKNOWN
    is_looking_away: Optional[bool] = None
    calibrating: bool = False
    calibration_progress: float = 0.0 # 0.0 - 1.0
    gaze_yaw: Optional[float] = None
    gaze_pitch: Optional[float] = None
    iris_h: Optional[float] = None
    iris_v: Optional[float] = None
    face_box: Optional[BoundingBox] = None
    live_status: Optional[LiveStatus] = None
    events: List[LiveEvent] = Field(default_factory=list)
    timestamp: float

class VerificationStopRequest(BaseModel):
    session_id: str

class VerdictDetail(BaseModel):
    verdict: str  # "PASS", "FAIL", "REVIEW"
    reasons: List[str] = Field(default_factory=list)


class SessionReport(BaseModel):
    model_config = ConfigDict(extra="allow")     # legacy keys pass through
    session_id: str
    status: str
    rules_version: str
    duration_s: float
    total_frames: int
    face: VerdictDetail
    gaze: VerdictDetail
    overall_verdict: str
    metrics: Dict[str, Any]
    intervals: List[Dict[str, Any]]

