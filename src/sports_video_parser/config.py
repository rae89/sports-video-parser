"""Configuration settings for the sports video parser."""

import os
from pathlib import Path
import tempfile


# YOLO detection
YOLO_MODEL: str = "yolo11n.pt"
DETECTION_CONFIDENCE: float = 0.15  # base YOLO confidence (lowered for ball recall)
PLAYER_CONFIDENCE: float = 0.3  # post-filter threshold for players
BALL_CONFIDENCE: float = 0.15  # post-filter threshold for ball
DEFAULT_IMGSZ: int = 1280  # YOLO input resolution (was 640 default — 2x for small objects)

# Hoop detection (HSV color range for orange rim)
HOOP_HSV_LOW: tuple[int, int, int] = (5, 130, 120)  # raised min S 100→130, min V 100→120
HOOP_HSV_HIGH: tuple[int, int, int] = (22, 255, 255)  # narrowed hue upper 25→22

# Hoop spatial filtering
HOOP_SEARCH_REGION: float = 0.45  # upper fraction of frame to search (was 0.60)
HOOP_CIRCULARITY_MIN: float = 0.3  # reject contours below this circularity

# Hoop temporal accumulator
HOOP_ACCUMULATOR_WINDOW: int = 300  # frames of candidate history
HOOP_CLUSTER_DISTANCE: float = 50.0  # max pixel distance for same-hoop cluster

# Event detection
SCORING_ZONE_PADDING: int = 30  # pixels around hoop bbox for scoring zone
THREE_POINT_DISTANCE_RATIO: float = 0.45  # fraction of frame width
POSSESSION_IOU_THRESHOLD: float = 0.3  # ball-player overlap for possession
EVENT_LOOKBACK_SECONDS: float = 5.0  # time window for assist detection

# Transition-based scoring detection
TRANSITION_SMOOTHING_WINDOW: int = 15  # frames to smooth player velocity
TRANSITION_MIN_PLAYERS: int = 3  # min players needed for reversal signal
TRANSITION_COOLDOWN_FRAMES: int = 150  # ~5 sec cooldown between transition scores

# Ball interpolation
BALL_INTERPOLATION_MAX_GAP: int = 30  # max frames to interpolate across (~1 sec)

# Video download
DEFAULT_VIDEO_FORMAT: str = "best[height<=1080]"
_download_dir = Path(tempfile.gettempdir()) / f"sports_video_parser_{os.getuid()}"
_download_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
DOWNLOAD_DIR: Path = _download_dir

# Output
DEFAULT_OUTPUT_PATH: Path = Path("game_stats.json")
