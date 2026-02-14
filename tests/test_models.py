"""Tests for Pydantic models."""

import json
from datetime import datetime, timezone

from sports_video_parser.models import (
    Detection,
    FrameDetections,
    GameEvent,
    GameStats,
    HoopPosition,
    Metadata,
    TeamStats,
)


def test_detection_defaults():
    d = Detection(track_id=1, class_name="player", bbox=(0, 0, 100, 200), confidence=0.8)
    assert d.team is None
    assert d.class_name == "player"


def test_detection_with_team():
    d = Detection(track_id=1, class_name="player", bbox=(0, 0, 100, 200), confidence=0.8, team=0)
    assert d.team == 0


def test_frame_detections():
    fd = FrameDetections(
        frame_index=10,
        timestamp_sec=0.333,
        detections=[
            Detection(track_id=1, class_name="player", bbox=(100, 200, 150, 400), confidence=0.9),
            Detection(track_id=2, class_name="ball", bbox=(300, 100, 320, 120), confidence=0.7),
        ],
    )
    assert fd.frame_index == 10
    assert len(fd.detections) == 2
    assert fd.detections[1].class_name == "ball"


def test_hoop_position():
    h = HoopPosition(x=900, y=100, width=60, height=30, side="right")
    assert h.side == "right"
    assert h.width == 60


def test_game_event():
    e = GameEvent(
        event_type="score",
        timestamp_sec=45.2,
        frame_index=1356,
        team=0,
        player_track_id=5,
        details={"points": 3, "shot_type": "3pt"},
    )
    assert e.event_type == "score"
    assert e.details["points"] == 3


def test_team_stats_defaults():
    ts = TeamStats(team_id=0, team_label="Team A")
    assert ts.points == 0
    assert ts.field_goals_made == 0
    assert ts.rebounds == 0
    assert ts.assists == 0
    assert ts.blocks == 0


def test_game_stats_serialization():
    stats = GameStats(
        teams=[
            TeamStats(team_id=0, team_label="Team A", points=45, field_goals_made=18,
                      three_pointers_made=5, two_pointers_made=13, rebounds=10,
                      offensive_rebounds=3, defensive_rebounds=7, assists=8, blocks=2),
            TeamStats(team_id=1, team_label="Team B", points=42),
        ],
        events=[
            GameEvent(
                event_type="score",
                timestamp_sec=10.5,
                frame_index=315,
                team=0,
                player_track_id=3,
                details={"points": 2, "shot_type": "2pt"},
            ),
        ],
        metadata=Metadata(
            source_url="https://youtube.com/watch?v=test",
            parsed_at=datetime(2024, 12, 26, 10, 30, 0, tzinfo=timezone.utc),
            total_frames_processed=18000,
            detection_model="yolo11n.pt",
            detection_confidence=0.3,
        ),
    )

    data = json.loads(stats.model_dump_json())
    assert len(data["teams"]) == 2
    assert data["teams"][0]["points"] == 45
    assert data["teams"][0]["three_pointers_made"] == 5
    assert data["teams"][1]["points"] == 42
    assert len(data["events"]) == 1
    assert data["events"][0]["event_type"] == "score"
    assert data["metadata"]["total_frames_processed"] == 18000
    assert data["metadata"]["detection_model"] == "yolo11n.pt"
