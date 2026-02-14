"""YOLO object detection and tracking for basketball video."""

from collections.abc import Generator
from pathlib import Path

from ultralytics import YOLO

from sports_video_parser.config import (
    BALL_CONFIDENCE,
    DEFAULT_IMGSZ,
    DETECTION_CONFIDENCE,
    PLAYER_CONFIDENCE,
    YOLO_MODEL,
)
from sports_video_parser.models import Detection, FrameDetections

# COCO class IDs
_PERSON_CLASS = 0
_SPORTS_BALL_CLASS = 32

_CLASS_NAMES = {_PERSON_CLASS: "player", _SPORTS_BALL_CLASS: "ball"}


class Detector:
    """Wraps Ultralytics YOLO for player and ball detection with tracking."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        confidence: float = DETECTION_CONFIDENCE,
        imgsz: int = DEFAULT_IMGSZ,
        player_confidence: float = PLAYER_CONFIDENCE,
        ball_confidence: float = BALL_CONFIDENCE,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz
        self.player_confidence = player_confidence
        self.ball_confidence = ball_confidence

    def track_video(
        self, video_path: Path | str
    ) -> Generator[FrameDetections, None, None]:
        """Run YOLO tracking on a video, yielding per-frame detections.

        Filters for person (class 0) and sports_ball (class 32) only.
        Uses BoTSORT tracker with persistent IDs across frames.
        """
        results = self.model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            classes=[_PERSON_CLASS, _SPORTS_BALL_CLASS],
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )

        for frame_idx, result in enumerate(results):
            detections = _parse_result(
                result,
                frame_idx,
                player_confidence=self.player_confidence,
                ball_confidence=self.ball_confidence,
            )
            yield detections


def _parse_result(
    result,
    frame_index: int,
    player_confidence: float = PLAYER_CONFIDENCE,
    ball_confidence: float = BALL_CONFIDENCE,
) -> FrameDetections:
    """Convert a single YOLO result to our FrameDetections model."""
    detections: list[Detection] = []

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # Extract tracking IDs (may be None if tracker lost the object)
        track_ids = boxes.id
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = _CLASS_NAMES.get(cls_id)
            if class_name is None:
                continue

            conf = float(boxes.conf[i].item())

            # Per-class confidence post-filtering
            if class_name == "player" and conf < player_confidence:
                continue
            if class_name == "ball" and conf < ball_confidence:
                continue

            track_id = int(track_ids[i].item()) if track_ids is not None else -1
            bbox = boxes.xyxy[i].tolist()

            detections.append(
                Detection(
                    track_id=track_id,
                    class_name=class_name,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    confidence=conf,
                )
            )

    # Compute timestamp from frame index and video fps
    fps = 30.0  # default
    if hasattr(result, "orig_fps") and result.orig_fps:
        fps = result.orig_fps
    timestamp_sec = frame_index / fps

    return FrameDetections(
        frame_index=frame_index,
        timestamp_sec=round(timestamp_sec, 4),
        detections=detections,
    )
