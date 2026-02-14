"""Pydantic models for YOLO-based basketball event detection."""

from datetime import datetime
from pydantic import BaseModel


class Detection(BaseModel):
    """A single detected object in a frame."""

    track_id: int
    class_name: str  # "player" or "ball"
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    team: int | None = None  # 0 or 1, None for ball


class FrameDetections(BaseModel):
    """All detections from a single video frame."""

    frame_index: int
    timestamp_sec: float
    detections: list[Detection]


class HoopPosition(BaseModel):
    """Detected basketball hoop location."""

    x: int
    y: int
    width: int
    height: int
    side: str  # "left" or "right"


class GameEvent(BaseModel):
    """A detected basketball event."""

    event_type: str  # "score", "rebound", "block", "assist"
    timestamp_sec: float
    frame_index: int
    team: int  # 0 or 1
    player_track_id: int
    details: dict  # e.g. {"points": 3, "shot_type": "3pt"}


class TeamStats(BaseModel):
    """Aggregated statistics for one team."""

    team_id: int
    team_label: str  # "Team A" / "Team B" or detected jersey color
    points: int = 0
    field_goals_made: int = 0
    three_pointers_made: int = 0
    two_pointers_made: int = 0
    rebounds: int = 0
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0
    assists: int = 0
    blocks: int = 0


class Metadata(BaseModel):
    """Parsing metadata."""

    source_url: str
    parsed_at: datetime
    total_frames_processed: int
    detection_model: str
    detection_confidence: float


class GameStats(BaseModel):
    """Complete game statistics output."""

    teams: list[TeamStats]
    events: list[GameEvent]
    metadata: Metadata
