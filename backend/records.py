from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any

# identity values
MATCH = "MATCH"
NO_MATCH = "NO_MATCH"
UNKNOWN = "UNKNOWN"            # one face, but head turned too far to judge identity
NO_FACE = "NO_FACE"
MULTIPLE_FACES = "MULTIPLE_FACES"

# gaze_direction values
CENTER = "CENTER"
LOOKING_LEFT = "LOOKING_LEFT"
LOOKING_RIGHT = "LOOKING_RIGHT"
LOOKING_UP = "LOOKING_UP"
LOOKING_DOWN = "LOOKING_DOWN"
CALIBRATING = "CALIBRATING"
GAZE_UNKNOWN = "UNKNOWN"


@dataclass
class FrameAnalysis:
    """Raw measurements from one image. Produced by VisionEngine.analyze_frame()."""
    faces_detected: int
    identity: str
    similarity: Optional[float] = None
    face_box: Optional[Dict[str, int]] = None      # {"x","y","width","height"} in sent-image pixels
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    iris_h: Optional[float] = None
    iris_v: Optional[float] = None
    openness: Optional[float] = None
    eyes_open: Optional[bool] = None
    thumbnail: Optional[str] = None                 # small JPEG data URL of the whole frame


@dataclass
class FrameRecord:
    """FrameAnalysis + session time + gaze decision. Stored per session."""
    t_s: float
    faces_detected: int
    identity: str
    similarity: Optional[float] = None
    face_box: Optional[Dict[str, int]] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    iris_h: Optional[float] = None
    iris_v: Optional[float] = None
    openness: Optional[float] = None
    eyes_open: Optional[bool] = None
    gaze_direction: str = GAZE_UNKNOWN
    is_looking_away: Optional[bool] = None          # None = gaze not judged on this frame
    calibrating: bool = False
    thumbnail: Optional[str] = None

    @classmethod
    def from_analysis(cls, a: FrameAnalysis, *, t_s: float, gaze_direction: str,
                      is_looking_away: Optional[bool], calibrating: bool,
                      thumbnail: Optional[str]) -> "FrameRecord":
        data = asdict(a)
        data.update(t_s=t_s, gaze_direction=gaze_direction, is_looking_away=is_looking_away,
                    calibrating=calibrating, thumbnail=thumbnail)
        return cls(**data)

    def to_dict(self, include_thumbnail: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_thumbnail:
            d.pop("thumbnail", None)
        return d
