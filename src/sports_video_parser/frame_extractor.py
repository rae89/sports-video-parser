"""Extract individual frames from video for debug snapshots.

The main pipeline uses YOLO's built-in video streaming instead of pre-extracting
frames. This module is only used for saving debug snapshots or extracting
specific frames by index.
"""

from pathlib import Path

import cv2
import numpy as np


def extract_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    """Extract a single frame from a video by frame index.

    Returns the frame as a BGR numpy array, or None if the frame cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def get_video_info(video_path: Path) -> dict:
    """Get basic video metadata (fps, frame count, dimensions)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


class SequentialFrameReader:
    """Reads frames sequentially from a video file.

    Much faster than per-frame extract_frame() calls because it avoids
    random seeking. Used alongside YOLO tracking when --debug-video is active.
    """

    def __init__(self, video_path: Path | str) -> None:
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self._frame_index = 0

    def read(self) -> tuple[int, np.ndarray | None]:
        """Read the next frame. Returns (frame_index, frame) or (frame_index, None) at EOF."""
        ret, frame = self._cap.read()
        idx = self._frame_index
        self._frame_index += 1
        return (idx, frame if ret else None)

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
