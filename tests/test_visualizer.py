"""Tests for debug video visualizer module."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from sports_video_parser.models import Detection, FrameDetections, GameEvent, HoopPosition
from sports_video_parser.visualizer import DebugVisualizer


def _make_frame_detections(
    frame_index: int = 0,
    timestamp: float = 0.0,
    detections: list[Detection] | None = None,
) -> FrameDetections:
    return FrameDetections(
        frame_index=frame_index,
        timestamp_sec=timestamp,
        detections=detections or [],
    )


def _make_detection(
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


class TestDebugVisualizer:
    def test_creates_video_file(self):
        """Visualizer should create a valid MP4 file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            with DebugVisualizer(out_path, fps=30.0, width=640, height=480) as viz:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                dets = _make_frame_detections()
                viz.annotate_and_write(frame, dets)

            assert out_path.exists()
            assert out_path.stat().st_size > 0
        finally:
            out_path.unlink(missing_ok=True)

    def test_annotates_players_and_ball(self):
        """Frame with player and ball detections should be annotated without error."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            with DebugVisualizer(out_path, fps=30.0, width=1920, height=1080) as viz:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                dets = _make_frame_detections(
                    frame_index=0,
                    detections=[
                        _make_detection(track_id=1, class_name="player", bbox=(100, 200, 200, 500), team=0),
                        _make_detection(track_id=2, class_name="player", bbox=(800, 200, 900, 500), team=1),
                        _make_detection(track_id=10, class_name="ball", bbox=(500, 300, 520, 320), team=None),
                    ],
                )
                viz.annotate_and_write(frame, dets)

            assert out_path.stat().st_size > 0
        finally:
            out_path.unlink(missing_ok=True)

    def test_hoop_zone_drawn(self):
        """Hoop zone should be drawn when hoop position is provided."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            hoop = HoopPosition(x=900, y=100, width=60, height=30, side="right")
            with DebugVisualizer(out_path, fps=30.0, width=1920, height=1080) as viz:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                dets = _make_frame_detections()
                viz.annotate_and_write(frame, dets, hoop=hoop)

            assert out_path.stat().st_size > 0
        finally:
            out_path.unlink(missing_ok=True)

    def test_event_overlay_persists(self):
        """Event text overlays should persist for ~30 frames."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            with DebugVisualizer(out_path, fps=30.0, width=640, height=480) as viz:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Frame 0: event occurs
                event = GameEvent(
                    event_type="score",
                    timestamp_sec=0.0,
                    frame_index=0,
                    team=0,
                    player_track_id=1,
                    details={"points": 2, "shot_type": "2pt"},
                )
                dets = _make_frame_detections(frame_index=0)
                viz.annotate_and_write(frame, dets, events=[event])

                # Events are registered
                assert len(viz._active_events) == 1

                # Frame 15: event should still be active
                dets = _make_frame_detections(frame_index=15)
                viz.annotate_and_write(frame, dets)
                assert len(viz._active_events) == 1

                # Frame 35: event should have expired (persist = 30 frames)
                dets = _make_frame_detections(frame_index=35)
                viz.annotate_and_write(frame, dets)
                assert len(viz._active_events) == 0

            assert out_path.stat().st_size > 0
        finally:
            out_path.unlink(missing_ok=True)

    def test_ball_trail_accumulates(self):
        """Ball positions should accumulate in the trail deque."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            with DebugVisualizer(out_path, fps=30.0, width=640, height=480) as viz:
                for i in range(5):
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    dets = _make_frame_detections(
                        frame_index=i,
                        detections=[
                            _make_detection(
                                track_id=10,
                                class_name="ball",
                                bbox=(100 + i * 20, 200, 120 + i * 20, 220),
                                team=None,
                            ),
                        ],
                    )
                    viz.annotate_and_write(frame, dets)

                assert len(viz._ball_trail) == 5

        finally:
            out_path.unlink(missing_ok=True)

    def test_multiple_frames_output(self):
        """Writing multiple frames should produce a larger file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            with DebugVisualizer(out_path, fps=30.0, width=320, height=240) as viz:
                for i in range(30):
                    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                    dets = _make_frame_detections(frame_index=i, timestamp=i / 30.0)
                    viz.annotate_and_write(frame, dets)

            size = out_path.stat().st_size
            assert size > 1000  # should be a non-trivial file
        finally:
            out_path.unlink(missing_ok=True)
