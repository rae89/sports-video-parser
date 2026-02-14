"""YouTube video download via yt-dlp."""

import re
from pathlib import Path

import yt_dlp

from sports_video_parser.config import DEFAULT_VIDEO_FORMAT, DOWNLOAD_DIR

_YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w-]+",
)


def validate_youtube_url(url: str) -> None:
    """Validate that a URL is a YouTube video URL.

    Raises ValueError if the URL doesn't match expected YouTube patterns.
    """
    if not _YOUTUBE_URL_PATTERN.match(url):
        raise ValueError(
            f"Invalid YouTube URL: {url}\n"
            "Expected format: https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID"
        )


def download_video(url: str, output_dir: Path = DOWNLOAD_DIR, video_format: str = DEFAULT_VIDEO_FORMAT) -> Path:
    """Download a YouTube video and return the path to the downloaded file."""
    validate_youtube_url(url)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": video_format,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        # Verify we got an actual video, not an auth/login page
        duration = info.get("duration", 0)
        if duration is not None and duration < 5:
            raise RuntimeError(
                f"Downloaded video is only {duration}s long — this may be an auth/login "
                "redirect instead of the actual video. Try providing the video file "
                "directly with --video-path."
            )

    video_path = Path(filename)
    if not video_path.exists():
        raise FileNotFoundError(f"Downloaded video not found at {video_path}")

    return video_path
