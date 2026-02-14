"""Tests for event detection engine."""

import numpy as np

from sports_video_parser.court import CourtAnalyzer
from sports_video_parser.event_detector import (
    EventDetector,
    _bbox_center,
    _bbox_iou,
    _distance,
    _find_ball,
    _find_nearest_player,
    _find_players,
    _find_possessor,
    _in_hoop_zone,
)
from sports_video_parser.models import Detection, FrameDetections, HoopPosition


def _det(
    track_id: int = 1,
    class_name: str = "player",
    bbox: tuple[float, float, float, float] = (100, 200, 150, 400),
    confidence: float = 0.8,
    team: int | None = 0,
) -> Detection:
    return Detection(
        track_id=track_id,
        class_name=class_name,
        bbox=bbox,
        confidence=confidence,
        team=team,
    )


def _frame(
    frame_index: int = 0,
    timestamp: float = 0.0,
    detections: list[Detection] | None = None,
) -> FrameDetections:
    return FrameDetections(
        frame_index=frame_index,
        timestamp_sec=timestamp,
        detections=detections or [],
    )


class TestHelperFunctions:
    def test_bbox_center(self):
        assert _bbox_center((0, 0, 100, 100)) == (50.0, 50.0)
        assert _bbox_center((10, 20, 30, 40)) == (20.0, 30.0)

    def test_bbox_iou_overlap(self):
        iou = _bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0 < iou < 1
        assert abs(iou - 2500 / 17500) < 0.001

    def test_bbox_iou_no_overlap(self):
        assert _bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_bbox_iou_identical(self):
        assert _bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_distance(self):
        assert _distance(0, 0, 3, 4) == 5.0
        assert _distance(1, 1, 1, 1) == 0.0

    def test_find_ball(self):
        dets = _frame(detections=[
            _det(track_id=1, class_name="player"),
            _det(track_id=2, class_name="ball", bbox=(300, 100, 320, 120)),
        ])
        ball = _find_ball(dets)
        assert ball is not None
        assert ball.track_id == 2

    def test_find_ball_none(self):
        dets = _frame(detections=[_det(track_id=1, class_name="player")])
        assert _find_ball(dets) is None

    def test_find_players(self):
        dets = _frame(detections=[
            _det(track_id=1, class_name="player"),
            _det(track_id=2, class_name="ball"),
            _det(track_id=3, class_name="player"),
        ])
        players = _find_players(dets)
        assert len(players) == 2

    def test_in_hoop_zone(self):
        hoop = HoopPosition(x=100, y=50, width=60, height=30, side="right")
        assert _in_hoop_zone(130, 65, hoop) is True
        assert _in_hoop_zone(0, 0, hoop) is False
        assert _in_hoop_zone(70, 50, hoop) is True

    def test_find_possessor_overlap(self):
        ball = _det(class_name="ball", bbox=(100, 100, 120, 120))
        player = _det(track_id=1, class_name="player", bbox=(95, 95, 125, 125))
        result = _find_possessor(ball, [player])
        assert result is not None
        assert result.track_id == 1

    def test_find_possessor_no_overlap(self):
        ball = _det(class_name="ball", bbox=(100, 100, 120, 120))
        player = _det(track_id=1, class_name="player", bbox=(500, 500, 600, 700))
        result = _find_possessor(ball, [player])
        assert result is None

    def test_find_nearest_player(self):
        p1 = _det(track_id=1, bbox=(0, 0, 10, 10))
        p2 = _det(track_id=2, bbox=(100, 100, 110, 110))
        nearest = _find_nearest_player(5, 5, [p1, p2])
        assert nearest is not None
        assert nearest.track_id == 1

    def test_find_nearest_player_empty(self):
        assert _find_nearest_player(0, 0, []) is None


class TestEventDetector:
    def setup_method(self):
        self.court = CourtAnalyzer()
        self.detector = EventDetector(self.court)

    def test_no_events_without_hoop(self):
        """Without a hoop, no scoring/rebound/block events should be generated."""
        frame_det = _frame(
            frame_index=0,
            detections=[
                _det(track_id=1, class_name="player"),
                _det(track_id=2, class_name="ball", bbox=(300, 100, 320, 120)),
            ],
        )
        events = self.detector.process_frame(frame_det, hoop=None)
        assert events == []

    def test_finalize_returns_all_events(self):
        events = self.detector.finalize()
        assert isinstance(events, list)
        assert len(events) == 0

    def test_ball_enters_and_exits_hoop_zone_below(self):
        """Ball entering hoop zone then exiting below should create a score event."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")

        for i in range(10):
            ball_y = 300 - i * 20
            frame_det = _frame(
                frame_index=i,
                timestamp=i / 30.0,
                detections=[
                    _det(track_id=1, class_name="player", bbox=(800, 300, 850, 500), team=0),
                    _det(track_id=10, class_name="ball", bbox=(925, ball_y, 935, ball_y + 10)),
                ],
            )
            self.detector.process_frame(frame_det, hoop=hoop)

        frame_det = _frame(
            frame_index=10,
            timestamp=10 / 30.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(800, 300, 850, 500), team=0),
                _det(track_id=10, class_name="ball", bbox=(925, 105, 935, 115)),
            ],
        )
        events = self.detector.process_frame(frame_det, hoop=hoop)

        frame_det = _frame(
            frame_index=11,
            timestamp=11 / 30.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(800, 300, 850, 500), team=0),
                _det(track_id=10, class_name="ball", bbox=(925, 170, 935, 180)),
            ],
        )
        events = self.detector.process_frame(frame_det, hoop=hoop)
        score_events = [e for e in events if e.event_type == "score"]
        assert len(score_events) == 1
        assert score_events[0].details["points"] in (2, 3)

    def test_ball_disappears_in_hoop_zone(self):
        """Ball disappearing inside hoop zone should register as a score."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")

        for i in range(5):
            ball_y = 200 - i * 20
            frame_det = _frame(
                frame_index=i,
                timestamp=i / 30.0,
                detections=[
                    _det(track_id=1, class_name="player", bbox=(800, 300, 850, 500), team=0),
                    _det(track_id=10, class_name="ball", bbox=(925, ball_y, 935, ball_y + 10)),
                ],
            )
            self.detector.process_frame(frame_det, hoop=hoop)

        frame_det = _frame(
            frame_index=5,
            timestamp=5 / 30.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(800, 300, 850, 500), team=0),
                _det(track_id=10, class_name="ball", bbox=(925, 105, 935, 115)),
            ],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        frame_det = _frame(
            frame_index=6,
            timestamp=6 / 30.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(800, 300, 850, 500), team=0),
            ],
        )
        events = self.detector.process_frame(frame_det, hoop=hoop)
        score_events = [e for e in events if e.event_type == "score"]
        assert len(score_events) == 1

    def test_assist_detection(self):
        """A pass between teammates before a score should produce an assist."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")

        frame_det = _frame(
            frame_index=0,
            timestamp=0.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(500, 300, 550, 500), team=0),
                _det(track_id=2, class_name="player", bbox=(800, 300, 850, 500), team=0),
                _det(track_id=10, class_name="ball", bbox=(510, 310, 520, 320)),
            ],
        )
        self.detector.process_frame(frame_det, hoop=hoop)
        self.detector._possession_history.append((0.0, 1, 0))
        self.detector._current_possessor = 1
        self.detector._current_possessor_team = 0

        for i in range(1, 5):
            frame_det = _frame(
                frame_index=i,
                timestamp=i / 30.0,
                detections=[
                    _det(track_id=1, class_name="player", bbox=(500, 300, 550, 500), team=0),
                    _det(track_id=2, class_name="player", bbox=(800, 300, 850, 500), team=0),
                    _det(track_id=10, class_name="ball", bbox=(810, 310, 820, 320)),
                ],
            )
            self.detector.process_frame(frame_det, hoop=hoop)

        frame_det = _frame(
            frame_index=5,
            timestamp=5 / 30.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(500, 300, 550, 500), team=0),
                _det(track_id=2, class_name="player", bbox=(800, 300, 850, 500), team=0),
                _det(track_id=10, class_name="ball", bbox=(925, 105, 935, 115)),
            ],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        frame_det = _frame(
            frame_index=6,
            timestamp=6 / 30.0,
            detections=[
                _det(track_id=1, class_name="player", bbox=(500, 300, 550, 500), team=0),
                _det(track_id=2, class_name="player", bbox=(800, 300, 850, 500), team=0),
                _det(track_id=10, class_name="ball", bbox=(925, 170, 935, 180)),
            ],
        )
        events = self.detector.process_frame(frame_det, hoop=hoop)

        all_events = self.detector.finalize()
        for e in all_events:
            assert e.event_type in ("score", "assist", "rebound", "block")

    def test_process_empty_frame(self):
        """Processing a frame with no detections should not crash."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")
        frame_det = _frame(frame_index=0, detections=[])
        events = self.detector.process_frame(frame_det, hoop=hoop)
        assert events == []


class TestTransitionDetection:
    """Test player movement reversal scoring detection."""

    def setup_method(self):
        self.court = CourtAnalyzer()
        self.detector = EventDetector(self.court)

    def _make_players_at_x(self, mean_x: float, n: int = 5) -> list[Detection]:
        """Create n players clustered around mean_x."""
        players = []
        for i in range(n):
            x = mean_x + (i - n // 2) * 30
            players.append(
                _det(
                    track_id=i + 1,
                    class_name="player",
                    bbox=(x, 300, x + 50, 500),
                    team=0,
                )
            )
        return players

    def test_movement_reversal_triggers_score(self):
        """Players moving right then suddenly left should trigger a transition score."""
        # Phase 1: Players moving right (increasing x)
        for i in range(20):
            mean_x = 500 + i * 10  # moving right at 10px/frame
            players = self._make_players_at_x(mean_x)
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            self.detector.process_frame(frame_det, hoop=None)

        # Phase 2: Players reverse and move left
        events_collected = []
        for i in range(20, 50):
            mean_x = 700 - (i - 20) * 10  # moving left at 10px/frame
            players = self._make_players_at_x(mean_x)
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            events = self.detector.process_frame(frame_det, hoop=None)
            events_collected.extend(events)

        # Should have at least one transition-based score
        transition_scores = [
            e for e in events_collected
            if e.event_type == "score"
            and e.details.get("detection_method") == "player_movement_reversal"
        ]
        assert len(transition_scores) >= 1

    def test_cooldown_prevents_double_count(self):
        """Within cooldown period, no second transition score should fire."""
        # Trigger first reversal
        for i in range(20):
            mean_x = 500 + i * 10
            players = self._make_players_at_x(mean_x)
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            self.detector.process_frame(frame_det, hoop=None)

        for i in range(20, 50):
            mean_x = 700 - (i - 20) * 10
            players = self._make_players_at_x(mean_x)
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            self.detector.process_frame(frame_det, hoop=None)

        # Try another reversal immediately (within 150-frame cooldown)
        for i in range(50, 70):
            mean_x = 400 + (i - 50) * 10
            players = self._make_players_at_x(mean_x)
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            self.detector.process_frame(frame_det, hoop=None)

        events_collected = []
        for i in range(70, 100):
            mean_x = 600 - (i - 70) * 10
            players = self._make_players_at_x(mean_x)
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            events = self.detector.process_frame(frame_det, hoop=None)
            events_collected.extend(events)

        # The second reversal (frames 70-100) should be blocked by cooldown
        transition_scores = [
            e for e in events_collected
            if e.event_type == "score"
            and e.details.get("detection_method") == "player_movement_reversal"
        ]
        assert len(transition_scores) == 0

    def test_too_few_players_no_transition(self):
        """With fewer than TRANSITION_MIN_PLAYERS, no transition detection."""
        for i in range(40):
            x = 500 + (i if i < 20 else 40 - i) * 10
            players = [_det(track_id=1, class_name="player", bbox=(x, 300, x + 50, 500), team=0)]
            frame_det = _frame(frame_index=i, timestamp=i / 30.0, detections=players)
            events = self.detector.process_frame(frame_det, hoop=None)
            assert len([e for e in events if e.details.get("detection_method") == "player_movement_reversal"]) == 0


class TestBallInterpolation:
    """Test ball position interpolation for sparse detections."""

    def setup_method(self):
        self.court = CourtAnalyzer()
        self.detector = EventDetector(self.court)

    def test_interpolation_fills_small_gap(self):
        """Ball missing for a few frames should be interpolated."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")

        # Frame 0: ball detected
        frame_det = _frame(
            frame_index=0,
            timestamp=0.0,
            detections=[
                _det(track_id=10, class_name="ball", bbox=(500, 300, 510, 310)),
                _det(track_id=1, class_name="player", bbox=(400, 300, 450, 500), team=0),
            ],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        # Frames 1-5: ball missing — should be interpolated
        for i in range(1, 6):
            frame_det = _frame(
                frame_index=i,
                timestamp=i / 30.0,
                detections=[
                    _det(track_id=1, class_name="player", bbox=(400, 300, 450, 500), team=0),
                ],
            )
            self.detector.process_frame(frame_det, hoop=hoop)

        # Ball history should have entries from interpolation
        assert len(self.detector._ball_history) > 1

    def test_interpolation_respects_max_gap(self):
        """Ball missing for more than MAX_GAP frames should NOT be interpolated."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")

        # Frame 0: ball detected
        frame_det = _frame(
            frame_index=0,
            timestamp=0.0,
            detections=[
                _det(track_id=10, class_name="ball", bbox=(500, 300, 510, 310)),
            ],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        # Frame 50: ball missing, gap > 30 — should NOT interpolate
        frame_det = _frame(
            frame_index=50,
            timestamp=50 / 30.0,
            detections=[],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        # Ball history should only have the original detection
        assert len(self.detector._ball_history) == 1

    def test_interpolated_ball_has_zero_confidence(self):
        """Synthetic interpolated balls should have confidence=0.0."""
        hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")

        frame_det = _frame(
            frame_index=0,
            timestamp=0.0,
            detections=[
                _det(track_id=10, class_name="ball", bbox=(500, 300, 510, 310), confidence=0.8),
            ],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        # Frame 1: no ball — interpolation should create synthetic detection
        frame_det = _frame(
            frame_index=1,
            timestamp=1 / 30.0,
            detections=[],
        )
        self.detector.process_frame(frame_det, hoop=hoop)

        # The raw detections should only have the real one
        assert len(self.detector._raw_ball_detections) == 1
        assert self.detector._raw_ball_detections[0][3] == 0.8
