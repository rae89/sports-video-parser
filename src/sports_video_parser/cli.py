"""CLI entry point for the sports video parser."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from sports_video_parser.config import DEFAULT_IMGSZ, DEFAULT_OUTPUT_PATH, DETECTION_CONFIDENCE, YOLO_MODEL

app = typer.Typer(help="Parse basketball game videos to extract statistics via YOLO detection.")
console = Console()


@app.command()
def parse(
    url: Annotated[str, typer.Argument(help="YouTube URL of the basketball game")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output JSON file path")] = DEFAULT_OUTPUT_PATH,
    model: Annotated[str, typer.Option("--model", help="YOLO model size (e.g. yolo11n, yolo11s, yolo11m)")] = "yolo11n",
    confidence: Annotated[
        float, typer.Option("--confidence", help="Detection confidence threshold")
    ] = DETECTION_CONFIDENCE,
    imgsz: Annotated[
        int, typer.Option("--imgsz", help="YOLO input resolution (higher = better small object detection)")
    ] = DEFAULT_IMGSZ,
    video_path: Annotated[
        Optional[Path], typer.Option("--video-path", help="Path to a local video file (skip download)")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Save per-frame detections to a debug JSON file")] = False,
    debug_video: Annotated[
        Optional[Path], typer.Option("--debug-video", help="Write annotated debug video to this path")
    ] = None,
) -> None:
    """Parse a basketball game video and extract statistics to JSON."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    from sports_video_parser.court import CourtAnalyzer
    from sports_video_parser.detector import Detector
    from sports_video_parser.downloader import download_video
    from sports_video_parser.event_detector import EventDetector
    from sports_video_parser.frame_extractor import SequentialFrameReader, extract_frame, get_video_info
    from sports_video_parser.stats_aggregator import aggregate_stats

    model_file = f"{model}.pt"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Step 1: Download video
        if video_path and video_path.exists():
            console.print(f"[green]Using local video:[/green] {video_path}")
            vid_path = video_path
        else:
            task = progress.add_task("Downloading video...", total=None)
            vid_path = download_video(url)
            progress.update(task, completed=100, total=100)
            console.print(f"[green]Downloaded:[/green] {vid_path}")

        # Get video info for progress tracking
        vid_info = get_video_info(vid_path)
        total_frames = vid_info["frame_count"]

        # Step 2: YOLO tracking
        task = progress.add_task("Running YOLO detection & tracking...", total=total_frames)
        detector = Detector(model_name=model_file, confidence=confidence, imgsz=imgsz)
        court = CourtAnalyzer()
        event_detector = EventDetector(court)

        # Set up debug video if requested
        visualizer = None
        frame_reader = None
        if debug_video is not None:
            from sports_video_parser.visualizer import DebugVisualizer

            frame_reader = SequentialFrameReader(vid_path)
            visualizer = DebugVisualizer(
                debug_video,
                fps=vid_info["fps"],
                width=vid_info["width"],
                height=vid_info["height"],
            )

        debug_log: list[dict] = []
        frame_count = 0

        try:
            for frame_detections in detector.track_video(vid_path):
                frame_count += 1

                # Get frame for hoop detection and/or debug video
                frame = None
                hoop = None

                if frame_reader is not None:
                    # Sequential reader for debug video — read every frame
                    _, frame = frame_reader.read()
                    if frame is not None:
                        hoop = court.detect_hoop(frame, frame_detections.frame_index)
                    if hoop is None:
                        hoop = court._confirmed_hoop
                elif frame_detections.frame_index % 30 == 0:
                    frame = extract_frame(vid_path, frame_detections.frame_index)
                    if frame is not None:
                        hoop = court.detect_hoop(frame, frame_detections.frame_index)
                    if hoop is None:
                        hoop = court._confirmed_hoop
                else:
                    # Use confirmed hoop position
                    hoop = court._confirmed_hoop

                events = event_detector.process_frame(
                    frame_detections, frame=frame, hoop=hoop
                )

                # Write debug video frame
                if visualizer is not None and frame is not None:
                    visualizer.annotate_and_write(
                        frame, frame_detections, hoop=hoop, events=events
                    )

                if debug:
                    debug_log.append({
                        "frame_index": frame_detections.frame_index,
                        "timestamp_sec": frame_detections.timestamp_sec,
                        "detections": [d.model_dump() for d in frame_detections.detections],
                        "hoop": hoop.model_dump() if hoop else None,
                        "events": [e.model_dump() for e in events],
                    })

                progress.update(task, completed=frame_count)
        finally:
            if visualizer is not None:
                visualizer.close()
            if frame_reader is not None:
                frame_reader.close()

        console.print(f"[green]Processed {frame_count} frames[/green]")

        # Step 3: Finalize events
        task = progress.add_task("Aggregating statistics...", total=None)
        all_events = event_detector.finalize()
        game_stats = aggregate_stats(
            all_events,
            source_url=url,
            total_frames=frame_count,
            model_name=model_file,
            confidence=confidence,
        )
        progress.update(task, completed=100, total=100)

    # Write debug output if requested
    if debug:
        debug_path = output.with_suffix(".debug.json")
        debug_path.write_text(json.dumps(debug_log, indent=2, default=str))
        console.print(f"[yellow]Debug output written to {debug_path}[/yellow]")

    if debug_video is not None:
        console.print(f"[yellow]Debug video written to {debug_video}[/yellow]")

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(game_stats.model_dump_json(indent=2))
    console.print(f"\n[bold green]Results written to {output}[/bold green]")

    # Print summary
    for team in game_stats.teams:
        console.print(f"  {team.team_label}: {team.points} pts, "
                      f"{team.field_goals_made} FG, {team.three_pointers_made} 3PT, "
                      f"{team.rebounds} REB, {team.assists} AST, {team.blocks} BLK")
    console.print(f"  Total events detected: {len(all_events)}")


@app.command()
def info(
    url: Annotated[str, typer.Argument(help="YouTube URL to inspect")],
) -> None:
    """Show video info without parsing."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        console.print(f"[bold]Title:[/bold] {info.get('title', 'N/A')}")
        console.print(f"[bold]Duration:[/bold] {info.get('duration', 0)} seconds")
        console.print(f"[bold]Resolution:[/bold] {info.get('width', '?')}x{info.get('height', '?')}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
