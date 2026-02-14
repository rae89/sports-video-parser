"""YouTube video download via yt-dlp."""

from pathlib import Path

import yt_dlp

from sports_video_parser.config import DEFAULT_VIDEO_FORMAT, DOWNLOAD_DIR


def download_video(url: str, output_dir: Path = DOWNLOAD_DIR, video_format: str = DEFAULT_VIDEO_FORMAT) -> Path:
    """Download a YouTube video and return the path to the downloaded file."""
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

    video_path = Path(filename)
    if not video_path.exists():
        raise FileNotFoundError(f"Downloaded video not found at {video_path}")

    return video_path
