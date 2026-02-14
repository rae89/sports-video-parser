"""Tests for stats aggregation logic."""

from sports_video_parser.models import GameEvent
from sports_video_parser.stats_aggregator import aggregate_stats


def _event(
    event_type: str = "score",
    team: int = 0,
    player_track_id: int = 1,
    timestamp: float = 0.0,
    frame_index: int = 0,
    details: dict | None = None,
) -> GameEvent:
    return GameEvent(
        event_type=event_type,
        timestamp_sec=timestamp,
        frame_index=frame_index,
        team=team,
        player_track_id=player_track_id,
        details=details or {},
    )


class TestAggregateStats:
    def test_empty_events(self):
        stats = aggregate_stats([], source_url="https://example.com")
        assert len(stats.teams) == 2
        assert stats.teams[0].points == 0
        assert stats.teams[1].points == 0
        assert stats.events == []

    def test_scoring_2pt(self):
        events = [
            _event(event_type="score", team=0, details={"points": 2, "shot_type": "2pt"}),
            _event(event_type="score", team=0, details={"points": 2, "shot_type": "2pt"}),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")
        team_a = stats.teams[0]
        assert team_a.points == 4
        assert team_a.field_goals_made == 2
        assert team_a.two_pointers_made == 2
        assert team_a.three_pointers_made == 0

    def test_scoring_3pt(self):
        events = [
            _event(event_type="score", team=1, details={"points": 3, "shot_type": "3pt"}),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")
        team_b = stats.teams[1]
        assert team_b.points == 3
        assert team_b.field_goals_made == 1
        assert team_b.three_pointers_made == 1
        assert team_b.two_pointers_made == 0

    def test_mixed_scoring(self):
        events = [
            _event(event_type="score", team=0, details={"points": 2}),
            _event(event_type="score", team=1, details={"points": 3}),
            _event(event_type="score", team=0, details={"points": 3}),
            _event(event_type="score", team=1, details={"points": 2}),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")
        assert stats.teams[0].points == 5  # 2 + 3
        assert stats.teams[1].points == 5  # 3 + 2

    def test_rebounds(self):
        events = [
            _event(event_type="rebound", team=0, details={"rebound_type": "offensive"}),
            _event(event_type="rebound", team=0, details={"rebound_type": "defensive"}),
            _event(event_type="rebound", team=1, details={"rebound_type": "defensive"}),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")
        assert stats.teams[0].rebounds == 2
        assert stats.teams[0].offensive_rebounds == 1
        assert stats.teams[0].defensive_rebounds == 1
        assert stats.teams[1].rebounds == 1

    def test_assists(self):
        events = [
            _event(event_type="assist", team=0),
            _event(event_type="assist", team=0),
            _event(event_type="assist", team=1),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")
        assert stats.teams[0].assists == 2
        assert stats.teams[1].assists == 1

    def test_blocks(self):
        events = [
            _event(event_type="block", team=1),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")
        assert stats.teams[0].blocks == 0
        assert stats.teams[1].blocks == 1

    def test_metadata(self):
        stats = aggregate_stats(
            [],
            source_url="https://youtube.com/watch?v=test",
            total_frames=1000,
            model_name="yolo11s.pt",
            confidence=0.5,
        )
        assert stats.metadata.source_url == "https://youtube.com/watch?v=test"
        assert stats.metadata.total_frames_processed == 1000
        assert stats.metadata.detection_model == "yolo11s.pt"
        assert stats.metadata.detection_confidence == 0.5

    def test_full_game(self):
        """A mini game with mixed events."""
        events = [
            _event(event_type="score", team=0, details={"points": 2}, timestamp=10.0),
            _event(event_type="assist", team=0, timestamp=10.0),
            _event(event_type="rebound", team=1, details={"rebound_type": "defensive"}, timestamp=20.0),
            _event(event_type="score", team=1, details={"points": 3}, timestamp=25.0),
            _event(event_type="block", team=0, timestamp=30.0),
            _event(event_type="rebound", team=0, details={"rebound_type": "offensive"}, timestamp=31.0),
            _event(event_type="score", team=0, details={"points": 2}, timestamp=33.0),
        ]
        stats = aggregate_stats(events, source_url="https://example.com")

        team_a = stats.teams[0]
        team_b = stats.teams[1]

        assert team_a.points == 4
        assert team_a.assists == 1
        assert team_a.blocks == 1
        assert team_a.offensive_rebounds == 1

        assert team_b.points == 3
        assert team_b.defensive_rebounds == 1

        assert len(stats.events) == 7

    def test_score_default_points(self):
        """Score event without explicit points should default to 2."""
        events = [_event(event_type="score", team=0, details={})]
        stats = aggregate_stats(events, source_url="https://example.com")
        assert stats.teams[0].points == 2
        assert stats.teams[0].two_pointers_made == 1
