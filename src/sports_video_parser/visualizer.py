"""Debug video output module — draws annotations on frames and writes to MP4."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from sports_video_parser.models import Detection, FrameDetections, GameEvent, HoopPosition

# Team colors (BGR)
_TEAM_COLORS = {
    0: (255, 100, 0),   # blue
    1: (0, 0, 255),     # red
}
_BALL_COLOR = (0, 165, 255)  # orange
_HOOP_ZONE_COLOR = (0, 255, 0)  # green
_TEXT_COLOR = (255, 255, 255)  # white
_EVENT_PERSIST_FRAMES = 30  # ~1 second at 30fps


class DebugVisualizer:
    """Draws annotations on video frames and writes to MP4."""

    def __init__(self, output_path: Path | str, fps: float, width: int, height: int) -> None:
        self._output_path = str(output_path)
        self._fps = fps
        self._width = width
        self._height = height
        self._writer = cv2.VideoWriter(
            self._output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for: {output_path}")

        # Ball trajectory trail (last ~3 sec)
        self._ball_trail: deque[tuple[int, int]] = deque(maxlen=int(fps * 3))

        # Active event overlays: (text, expire_frame)
        self._active_events: list[tuple[str, int]] = []

    def annotate_and_write(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        hoop: HoopPosition | None = None,
        events: list[GameEvent] | None = None,
    ) -> None:
        """Draw all annotations on frame and write to video."""
        annotated = frame.copy()
        frame_idx = detections.frame_index
        timestamp = detections.timestamp_sec

        # Draw hoop scoring zone
        if hoop is not None:
            _draw_hoop_zone(annotated, hoop)

        # Draw detections
        for det in detections.detections:
            if det.class_name == "player":
                _draw_player(annotated, det)
            elif det.class_name == "ball":
                _draw_ball(annotated, det)
                cx = int((det.bbox[0] + det.bbox[2]) / 2)
                cy = int((det.bbox[1] + det.bbox[3]) / 2)
                self._ball_trail.append((cx, cy))

        # Draw ball trajectory trail
        _draw_trail(annotated, list(self._ball_trail))

        # Register new events
        if events:
            for ev in events:
                text = f"{ev.event_type.upper()}"
                if ev.details.get("points"):
                    text += f" {ev.details['points']}pt"
                if ev.details.get("detection_method"):
                    text += f" ({ev.details['detection_method']})"
                self._active_events.append((text, frame_idx + _EVENT_PERSIST_FRAMES))

        # Draw persistent event overlays
        self._active_events = [
            (t, exp) for t, exp in self._active_events if exp > frame_idx
        ]
        for i, (text, _) in enumerate(self._active_events):
            y_pos = 80 + i * 40
            cv2.putText(
                annotated, text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )

        # Frame metadata
        meta_text = f"Frame {frame_idx} | {timestamp:.2f}s"
        cv2.putText(
            annotated, meta_text,
            (10, self._height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1,
        )

        self._writer.write(annotated)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _draw_player(frame: np.ndarray, det: Detection) -> None:
    """Draw player bounding box color-coded by team."""
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    team = det.team if det.team is not None else 0
    color = _TEAM_COLORS.get(team, (128, 128, 128))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{det.track_id} {det.confidence:.2f}"
    cv2.putText(
        frame, label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
    )


def _draw_ball(frame: np.ndarray, det: Detection) -> None:
    """Draw ball bounding box in orange."""
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BALL_COLOR, 2)
    label = f"ball {det.confidence:.2f}"
    cv2.putText(
        frame, label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _BALL_COLOR, 1,
    )


def _draw_hoop_zone(frame: np.ndarray, hoop: HoopPosition) -> None:
    """Draw hoop scoring zone rectangle in green."""
    from sports_video_parser.config import SCORING_ZONE_PADDING

    pad = SCORING_ZONE_PADDING
    x1 = hoop.x - pad
    y1 = hoop.y - pad
    x2 = hoop.x + hoop.width + pad
    y2 = hoop.y + hoop.height + pad
    cv2.rectangle(frame, (x1, y1), (x2, y2), _HOOP_ZONE_COLOR, 2)
    cv2.putText(
        frame, "HOOP",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _HOOP_ZONE_COLOR, 1,
    )


def _draw_trail(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    """Draw ball trajectory as a polyline."""
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, _BALL_COLOR, 2)
