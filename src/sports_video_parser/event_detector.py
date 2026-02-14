"""Event detection engine — scoring, rebounds, blocks, assists from tracked objects."""

from collections import deque

import numpy as np

from sports_video_parser.config import (
    BALL_INTERPOLATION_MAX_GAP,
    EVENT_LOOKBACK_SECONDS,
    POSSESSION_IOU_THRESHOLD,
    SCORING_ZONE_PADDING,
    TRANSITION_COOLDOWN_FRAMES,
    TRANSITION_MIN_PLAYERS,
    TRANSITION_MIN_SWING,
    TRANSITION_MIN_VELOCITY,
    TRANSITION_SMOOTHING_WINDOW,
)
from sports_video_parser.court import CourtAnalyzer
from sports_video_parser.models import Detection, FrameDetections, GameEvent, HoopPosition


class EventDetector:
    """Detects basketball events from per-frame YOLO detections and court info."""

    def __init__(self, court_analyzer: CourtAnalyzer) -> None:
        self.court = court_analyzer
        self._events: list[GameEvent] = []

        # Ball tracking state
        self._ball_history: deque[tuple[int, float, float, float]] = deque(
            maxlen=60
        )  # (frame_idx, x_center, y_center, timestamp)
        self._ball_in_hoop_zone = False
        self._last_shot_frame: int = -999

        # Possession tracking: (timestamp, player_track_id, team)
        self._possession_history: deque[tuple[float, int, int]] = deque(maxlen=300)
        self._current_possessor: int | None = None
        self._current_possessor_team: int | None = None

        # Shot tracking
        self._pending_shot_frame: int | None = None
        self._pending_shooter_id: int | None = None
        self._pending_shooter_team: int | None = None
        self._pending_shot_points: int | None = None

        # Ball interpolation: raw detections for gap filling
        self._raw_ball_detections: list[tuple[int, float, float, float]] = []
        # (frame_idx, x_center, y_center, confidence)

        # Transition detection state
        self._player_x_history: deque[tuple[int, float]] = deque(
            maxlen=TRANSITION_SMOOTHING_WINDOW * 2
        )  # (frame_idx, mean_x)
        self._smoothed_velocity: deque[float] = deque(
            maxlen=TRANSITION_SMOOTHING_WINDOW
        )
        self._last_transition_frame: int = -999

    def process_frame(
        self,
        detections: FrameDetections,
        frame: np.ndarray | None = None,
        hoop: HoopPosition | None = None,
    ) -> list[GameEvent]:
        """Process a single frame's detections and return any new events."""
        frame_events: list[GameEvent] = []
        frame_idx = detections.frame_index
        timestamp = detections.timestamp_sec

        ball = _find_ball(detections)
        players = _find_players(detections)

        # Store raw ball detection for interpolation
        if ball is not None:
            bx, by = _bbox_center(ball.bbox)
            self._raw_ball_detections.append((frame_idx, bx, by, ball.confidence))
            self._ball_history.append((frame_idx, bx, by, timestamp))
            possessor = _find_possessor(ball, players)
            if possessor is not None:
                self._current_possessor = possessor.track_id
                self._current_possessor_team = possessor.team if possessor.team is not None else 0
                self._possession_history.append(
                    (timestamp, possessor.track_id, self._current_possessor_team)
                )
        else:
            # Try ball interpolation — but NOT when ball was in hoop zone
            # (we need the "ball disappeared" path to fire for scoring detection)
            interpolated = None
            if not self._ball_in_hoop_zone:
                interpolated = self._interpolate_ball(frame_idx)
            if interpolated is not None:
                ix, iy = interpolated
                self._ball_history.append((frame_idx, ix, iy, timestamp))
                # Create a synthetic ball detection for downstream logic
                ball = Detection(
                    track_id=-1,
                    class_name="ball",
                    bbox=(ix - 5, iy - 5, ix + 5, iy + 5),
                    confidence=0.0,
                )

        # Transition-based scoring detection (works with 0 ball detections)
        transition_events = self._detect_transition_score(
            players, detections
        )
        frame_events.extend(transition_events)

        if hoop is None:
            self._events.extend(frame_events)
            return frame_events

        # Scoring detection
        scoring_events = self._detect_scoring(
            ball, players, hoop, detections, frame
        )
        frame_events.extend(scoring_events)

        # Rebound detection (after a missed shot)
        rebound_events = self._detect_rebound(
            ball, players, hoop, detections
        )
        frame_events.extend(rebound_events)

        # Block detection
        block_events = self._detect_block(ball, players, hoop, detections)
        frame_events.extend(block_events)

        self._events.extend(frame_events)
        return frame_events

    def finalize(self) -> list[GameEvent]:
        """Return all detected events. Call after processing all frames."""
        return list(self._events)

    def _interpolate_ball(self, frame_idx: int) -> tuple[float, float] | None:
        """Linearly interpolate ball position from nearest before/after detections."""
        if not self._raw_ball_detections:
            return None

        # Find nearest detection before this frame
        before = None
        for det in reversed(self._raw_ball_detections):
            if det[0] < frame_idx:
                before = det
                break

        if before is None:
            return None

        gap = frame_idx - before[0]
        if gap > BALL_INTERPOLATION_MAX_GAP:
            return None

        # Without a future detection, just use the last known position
        # (true interpolation happens when we have both before and after)
        # For now, return last known position for small gaps
        return (before[1], before[2])

    def _detect_transition_score(
        self,
        players: list[Detection],
        detections: FrameDetections,
    ) -> list[GameEvent]:
        """Detect scoring via collective player movement reversal.

        When all players suddenly reverse direction (e.g., transition from
        offense to defense), it often indicates a dead ball / score just happened.
        """
        events: list[GameEvent] = []
        frame_idx = detections.frame_index

        if len(players) < TRANSITION_MIN_PLAYERS:
            return events

        # Compute mean x-position of all players
        mean_x = np.mean([_bbox_center(p.bbox)[0] for p in players])
        self._player_x_history.append((frame_idx, mean_x))

        if len(self._player_x_history) < TRANSITION_SMOOTHING_WINDOW + 1:
            return events

        # Compute smoothed velocity (change in mean x per frame)
        history = list(self._player_x_history)
        velocities = []
        for i in range(1, len(history)):
            dt = history[i][0] - history[i - 1][0]
            if dt > 0:
                velocities.append((history[i][1] - history[i - 1][1]) / dt)

        if len(velocities) < TRANSITION_SMOOTHING_WINDOW:
            return events

        # Split into two halves and check for sign change
        mid = len(velocities) // 2
        first_half = np.mean(velocities[:mid])
        second_half = np.mean(velocities[mid:])

        self._smoothed_velocity.append(second_half)

        # Detect reversal: significant velocity sign change with minimum swing
        reversal = (
            (first_half > TRANSITION_MIN_VELOCITY and second_half < -TRANSITION_MIN_VELOCITY)
            or (first_half < -TRANSITION_MIN_VELOCITY and second_half > TRANSITION_MIN_VELOCITY)
        ) and abs(first_half - second_half) >= TRANSITION_MIN_SWING

        if not reversal:
            return events

        # Cooldown check
        if frame_idx - self._last_transition_frame < TRANSITION_COOLDOWN_FRAMES:
            return events

        self._last_transition_frame = frame_idx
        events.append(
            GameEvent(
                event_type="score",
                timestamp_sec=detections.timestamp_sec,
                frame_index=frame_idx,
                team=0,  # can't determine team from movement alone
                player_track_id=-1,
                details={
                    "points": 2,
                    "shot_type": "2pt",
                    "detection_method": "player_movement_reversal",
                },
            )
        )
        return events

    def _detect_scoring(
        self,
        ball: Detection | None,
        players: list[Detection],
        hoop: HoopPosition,
        detections: FrameDetections,
        frame: np.ndarray | None,
    ) -> list[GameEvent]:
        """Detect made baskets via ball entering hoop zone."""
        events: list[GameEvent] = []

        if ball is None:
            # Ball disappeared — check if it was in hoop zone
            if self._ball_in_hoop_zone and self._pending_shot_frame is not None:
                events.append(self._create_score_event(detections))
                self._clear_shot_state()
            return events

        bx, by = _bbox_center(ball.bbox)
        in_zone = _in_hoop_zone(bx, by, hoop)

        if in_zone and not self._ball_in_hoop_zone:
            # Ball just entered hoop zone
            self._ball_in_hoop_zone = True
            shooter = self._find_shooter(players, hoop, frame)
            if shooter is not None:
                self._pending_shot_frame = detections.frame_index
                self._pending_shooter_id = shooter.track_id
                self._pending_shooter_team = shooter.team if shooter.team is not None else 0
                frame_width = int(ball.bbox[2] + 500)
                if frame is not None:
                    frame_width = frame.shape[1]
                self._pending_shot_points = self._classify_shot(
                    shooter, hoop, frame_width
                )

        elif not in_zone and self._ball_in_hoop_zone:
            self._ball_in_hoop_zone = False
            if self._pending_shot_frame is not None:
                if by > hoop.y + hoop.height:
                    events.append(self._create_score_event(detections))
                    assist = self._detect_assist(detections)
                    if assist is not None:
                        events.append(assist)
                self._clear_shot_state()

        return events

    def _detect_rebound(
        self,
        ball: Detection | None,
        players: list[Detection],
        hoop: HoopPosition,
        detections: FrameDetections,
    ) -> list[GameEvent]:
        """Detect rebounds after missed shots."""
        events: list[GameEvent] = []

        if ball is None or len(self._ball_history) < 5:
            return events

        bx, by = _bbox_center(ball.bbox)
        was_near_hoop = any(
            _in_hoop_zone(hx, hy, hoop)
            for _, hx, hy, _ in list(self._ball_history)[-15:]
        )

        if not was_near_hoop:
            return events

        possessor = _find_possessor(ball, players)
        if possessor is None:
            return events

        min_frames_between = 30
        if detections.frame_index - self._last_shot_frame < min_frames_between:
            return events

        self._last_shot_frame = detections.frame_index

        rebounder_team = possessor.team if possessor.team is not None else 0
        rebound_type = "defensive"
        if self._pending_shooter_team is not None:
            if rebounder_team == self._pending_shooter_team:
                rebound_type = "offensive"

        events.append(
            GameEvent(
                event_type="rebound",
                timestamp_sec=detections.timestamp_sec,
                frame_index=detections.frame_index,
                team=rebounder_team,
                player_track_id=possessor.track_id,
                details={"rebound_type": rebound_type},
            )
        )
        return events

    def _detect_block(
        self,
        ball: Detection | None,
        players: list[Detection],
        hoop: HoopPosition,
        detections: FrameDetections,
    ) -> list[GameEvent]:
        """Detect blocks — ball trajectory reversal near a defender."""
        events: list[GameEvent] = []

        if ball is None or len(self._ball_history) < 5:
            return events

        recent = list(self._ball_history)[-5:]
        if len(recent) < 4:
            return events

        vy_early = recent[-3][2] - recent[-4][2]
        vy_late = recent[-1][2] - recent[-2][2]

        if vy_early < -2 and vy_late > 2:
            bx, by = _bbox_center(ball.bbox)
            attacker_team = self._current_possessor_team
            defenders = [
                p for p in players
                if p.team is not None and p.team != attacker_team
            ]
            nearest = _find_nearest_player(bx, by, defenders)
            if nearest is not None:
                dist = _distance(bx, by, *_bbox_center(nearest.bbox))
                ball_size = ball.bbox[2] - ball.bbox[0]
                if dist < ball_size * 5:
                    events.append(
                        GameEvent(
                            event_type="block",
                            timestamp_sec=detections.timestamp_sec,
                            frame_index=detections.frame_index,
                            team=nearest.team if nearest.team is not None else 0,
                            player_track_id=nearest.track_id,
                            details={},
                        )
                    )

        return events

    def _detect_assist(self, detections: FrameDetections) -> GameEvent | None:
        """Check possession history for a pass leading to the score."""
        if self._pending_shooter_id is None or self._pending_shooter_team is None:
            return None

        timestamp = detections.timestamp_sec
        shooter_id = self._pending_shooter_id
        shooter_team = self._pending_shooter_team

        for ts, pid, team in reversed(self._possession_history):
            if timestamp - ts > EVENT_LOOKBACK_SECONDS:
                break
            if pid != shooter_id and team == shooter_team:
                return GameEvent(
                    event_type="assist",
                    timestamp_sec=ts,
                    frame_index=detections.frame_index,
                    team=shooter_team,
                    player_track_id=pid,
                    details={"to_player": shooter_id},
                )
        return None

    def _find_shooter(
        self,
        players: list[Detection],
        hoop: HoopPosition,
        frame: np.ndarray | None,
    ) -> Detection | None:
        """Find the player who most likely took the shot."""
        if not self._ball_history:
            return None

        history = list(self._ball_history)
        release = history[-min(5, len(history))]
        rx, ry = release[1], release[2]

        return _find_nearest_player(rx, ry, players)

    def _classify_shot(
        self, shooter: Detection, hoop: HoopPosition, frame_width: int
    ) -> int:
        """Classify a shot as 2pt or 3pt based on shooter distance from hoop."""
        sx, sy = _bbox_center(shooter.bbox)
        hx = hoop.x + hoop.width / 2
        hy = hoop.y + hoop.height / 2
        dist = _distance(sx, sy, hx, hy)
        threshold = self.court.estimate_three_point_distance(hoop, frame_width)
        return 3 if dist > threshold else 2

    def _create_score_event(self, detections: FrameDetections) -> GameEvent:
        """Create a scoring GameEvent from the pending shot state."""
        points = self._pending_shot_points or 2
        shot_type = "3pt" if points == 3 else "2pt"
        return GameEvent(
            event_type="score",
            timestamp_sec=detections.timestamp_sec,
            frame_index=detections.frame_index,
            team=self._pending_shooter_team or 0,
            player_track_id=self._pending_shooter_id or -1,
            details={"points": points, "shot_type": shot_type},
        )

    def _clear_shot_state(self) -> None:
        self._pending_shot_frame = None
        self._pending_shooter_id = None
        self._pending_shooter_team = None
        self._pending_shot_points = None
        self._ball_in_hoop_zone = False


def _find_ball(detections: FrameDetections) -> Detection | None:
    for d in detections.detections:
        if d.class_name == "ball":
            return d
    return None


def _find_players(detections: FrameDetections) -> list[Detection]:
    return [d for d in detections.detections if d.class_name == "player"]


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute intersection over union between two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _in_hoop_zone(x: float, y: float, hoop: HoopPosition) -> bool:
    """Check if a point is within the expanded hoop scoring zone."""
    pad = SCORING_ZONE_PADDING
    return (
        hoop.x - pad <= x <= hoop.x + hoop.width + pad
        and hoop.y - pad <= y <= hoop.y + hoop.height + pad
    )


def _find_possessor(
    ball: Detection, players: list[Detection]
) -> Detection | None:
    """Find the player whose bbox most overlaps with the ball."""
    best_player = None
    best_iou = POSSESSION_IOU_THRESHOLD

    for player in players:
        iou = _bbox_iou(ball.bbox, player.bbox)
        if iou > best_iou:
            best_iou = iou
            best_player = player

    return best_player


def _find_nearest_player(
    x: float, y: float, players: list[Detection]
) -> Detection | None:
    """Find the player nearest to a given point."""
    if not players:
        return None

    nearest = None
    min_dist = float("inf")
    for p in players:
        px, py = _bbox_center(p.bbox)
        d = _distance(x, y, px, py)
        if d < min_dist:
            min_dist = d
            nearest = p
    return nearest


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
