# Sports Video Parser

YOLO-based basketball video parser that extracts game statistics from broadcast footage. Detects players, ball, and hoop positions to identify scoring events, rebounds, assists, and blocks.

![Debug video output](demo.gif)

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/rae89/sports-video-parser.git
cd sports-video-parser
uv sync
```

## Usage

### Parse a YouTube video

```bash
uv run sports-video-parser parse "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" -o game.json
```

The tool downloads the video, runs YOLO detection and tracking on every frame, then outputs game statistics as JSON.

### Parse a local video file

If you already have the video downloaded:

```bash
uv run sports-video-parser parse "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --video-path /path/to/video.mp4 \
  -o game.json
```

### Generate a debug video

To see exactly what YOLO detects — player bounding boxes, ball tracking, hoop zone, and event overlays:

```bash
uv run sports-video-parser parse "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --video-path /path/to/video.mp4 \
  -o game.json \
  --debug-video debug.mp4
```

### Save per-frame detection data

```bash
uv run sports-video-parser parse "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --video-path /path/to/video.mp4 \
  -o game.json \
  --debug
```

This writes a `game.debug.json` with every detection, hoop position, and event per frame.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `game_stats.json` | Output JSON file path |
| `--model` | `yolo11n` | YOLO model size (`yolo11n`, `yolo11s`, `yolo11m`) |
| `--confidence` | `0.15` | Base detection confidence threshold |
| `--imgsz` | `1280` | YOLO input resolution (higher = better small object detection) |
| `--video-path` | — | Path to local video file (skips download) |
| `--debug` | off | Save per-frame detections to debug JSON |
| `--debug-video` | — | Write annotated debug video to this path |

### Video info

Check video metadata without parsing:

```bash
uv run sports-video-parser info "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

## Output

The JSON output contains:

- **Team stats** — points, field goals, 3-pointers, rebounds, assists, blocks per team
- **Events** — timestamped list of every detected scoring play, rebound, assist, and block
- **Metadata** — source URL, model used, frames processed

## How it works

1. **YOLO Detection** — Runs YOLOv11 with BoTSORT tracking at 1280px resolution to detect players (class 0) and ball (class 32) with persistent track IDs
2. **Hoop Detection** — HSV color filtering for the orange rim with a temporal accumulator that requires 30+ consistent detections before confirming hoop position
3. **Scoring Detection** — Two methods:
   - Ball entering and exiting below the hoop zone
   - Player movement reversal detection (collective direction change signals a dead ball / score)
4. **Ball Interpolation** — Fills gaps up to 1 second when ball tracking is sparse
5. **Event Classification** — 2pt vs 3pt based on shooter distance, assists from possession history, rebounds from ball trajectory near hoop

## Tests

```bash
uv run pytest tests/ -v
```
