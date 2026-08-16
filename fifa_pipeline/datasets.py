"""Format-compatible loaders for MovieNet and BBC scene-boundary annotations.

These loaders are provided so the pipeline can be pointed at a small local sample
of either dataset later; **this project does not download either dataset**. For
the runnable demo (see the notebook / ``synthetic_data.py``) we instead render a
handful of short synthetic clips with known ground truth.

Expected local layout, if/when a few real clips are added by hand:

    data/movienet/<video_id>/scene_boundaries.json
    data/movienet/<video_id>/<video_id>.mp4
    data/bbc/<video_id>/scenes.csv
    data/bbc/<video_id>/<video_id>.mp4

Both loaders are intentionally forgiving about the exact schema (MovieNet ships
scene boundaries as shot-index ranges, e.g. via ``scene_movie318.json`` files,
while the BBC planar-shot-boundary dataset ships per-video CSVs of timecodes) and
normalize everything to a list of :class:`~fifa_pipeline.config.VideoSegment`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fifa_pipeline.config import VideoSegment

DEFAULT_FPS = 24.0


def load_movienet_scenes(video_dir: Path) -> list[VideoSegment]:
    """Load MovieNet-style scene boundaries for a single local video directory.

    MovieNet scene annotations map a scene index to a contiguous range of shot
    ids / frame ids, e.g. ``{"scene_id": 0, "start_frame": 0, "end_frame": 240}``.
    """

    video_dir = Path(video_dir)
    video_id = video_dir.name
    ann_path = video_dir / "scene_boundaries.json"
    video_path = next(video_dir.glob("*.mp4"), None)

    if not ann_path.exists():
        raise FileNotFoundError(
            f"No MovieNet-style annotation found at {ann_path}. "
            "This loader expects a small locally-added sample; the project does "
            "not auto-download MovieNet."
        )

    with open(ann_path) as f:
        scenes = json.load(f)

    segments = []
    for scene in scenes:
        fps = float(scene.get("fps", DEFAULT_FPS))
        start_frame = int(scene["start_frame"])
        end_frame = int(scene["end_frame"])
        segments.append(
            VideoSegment(
                video_id=video_id,
                segment_id=f"scene_{scene['scene_id']:03d}",
                source="movienet",
                path=video_path or video_dir,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                start_time_s=start_frame / fps,
                end_time_s=end_frame / fps,
            )
        )
    return segments


def load_bbc_scenes(video_dir: Path) -> list[VideoSegment]:
    """Load BBC planar shot/scene-boundary CSVs for a single local video directory.

    Expected columns: ``scene_id, start_time_s, end_time_s`` (timecodes rather
    than frame indices, per the BBC dataset's published format).
    """

    video_dir = Path(video_dir)
    video_id = video_dir.name
    ann_path = video_dir / "scenes.csv"
    video_path = next(video_dir.glob("*.mp4"), None)

    if not ann_path.exists():
        raise FileNotFoundError(
            f"No BBC-style annotation found at {ann_path}. "
            "This loader expects a small locally-added sample; the project does "
            "not auto-download the full BBC dataset."
        )

    segments = []
    with open(ann_path, newline="") as f:
        for row in csv.DictReader(f):
            start_s = float(row["start_time_s"])
            end_s = float(row["end_time_s"])
            fps = float(row.get("fps", DEFAULT_FPS))
            segments.append(
                VideoSegment(
                    video_id=video_id,
                    segment_id=f"scene_{row['scene_id']}",
                    source="bbc",
                    path=video_path or video_dir,
                    start_frame=int(start_s * fps),
                    end_frame=int(end_s * fps),
                    fps=fps,
                    start_time_s=start_s,
                    end_time_s=end_s,
                )
            )
    return segments


def discover_local_samples(root: Path, loader) -> list[VideoSegment]:
    """Apply ``loader`` to every video sub-directory under ``root`` that has one,
    skipping (rather than failing on) videos that are missing annotations."""

    root = Path(root)
    if not root.exists():
        return []

    segments: list[VideoSegment] = []
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            segments.extend(loader(video_dir))
        except FileNotFoundError:
            continue
    return segments
