"""Aggregate detected events into per-team statistics."""

from datetime import datetime, timezone

from sports_video_parser.models import GameEvent, GameStats, Metadata, TeamStats


def aggregate_stats(
    events: list[GameEvent],
    source_url: str,
    total_frames: int = 0,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.3,
) -> GameStats:
    """Tally events by type and team into structured game statistics."""
    team_stats = {
        0: TeamStats(team_id=0, team_label="Team A"),
        1: TeamStats(team_id=1, team_label="Team B"),
    }

    for event in events:
        team_id = event.team
        if team_id not in team_stats:
            team_stats[team_id] = TeamStats(
                team_id=team_id, team_label=f"Team {chr(65 + team_id)}"
            )

        ts = team_stats[team_id]

        if event.event_type == "score":
            points = event.details.get("points", 2)
            ts.points += points
            ts.field_goals_made += 1
            if points == 3:
                ts.three_pointers_made += 1
            else:
                ts.two_pointers_made += 1

        elif event.event_type == "rebound":
            ts.rebounds += 1
            rebound_type = event.details.get("rebound_type", "defensive")
            if rebound_type == "offensive":
                ts.offensive_rebounds += 1
            else:
                ts.defensive_rebounds += 1

        elif event.event_type == "assist":
            ts.assists += 1

        elif event.event_type == "block":
            ts.blocks += 1

    return GameStats(
        teams=list(team_stats.values()),
        events=events,
        metadata=Metadata(
            source_url=source_url,
            parsed_at=datetime.now(timezone.utc),
            total_frames_processed=total_frames,
            detection_model=model_name,
            detection_confidence=confidence,
        ),
    )
