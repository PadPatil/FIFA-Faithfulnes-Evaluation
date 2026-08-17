"""Generate short synthetic "cinematic" clips with known ground truth.

Downloading and licensing full MovieNet / BBC video files is out of scope for a
runnable demo of this pipeline. Instead, this module renders short (~3-6 second)
synthetic clips with OpenCV that stand in for a scene: a colored background encodes
the *location*, colored shapes encode *characters/objects*, and their motion encodes
an *action* / *relationship*. Because the ground truth is known exactly, the mock
captioner/VQA backends (see ``vlm_captioners.py`` and ``verification.py``) can be
validated against it, which is what lets the rest of the pipeline (fact extraction,
STSDG construction, controlled hallucinations, scoring) be exercised end-to-end
without a GPU, model weights, or a network download.

The real dataset loaders that speak the MovieNet / BBC scene-boundary formats live
in ``datasets.py`` and can be pointed at a small local sample of either dataset
later without changing any downstream code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fifa_pipeline.config import VideoSegment

FRAME_SIZE = (320, 240)  # (width, height)
FPS = 12

LOCATIONS: dict[str, tuple[int, int, int]] = {
    "restaurant": (60, 130, 200),   # warm amber (BGR)
    "hospital": (220, 220, 220),    # sterile white
    "office": (140, 140, 140),      # neutral gray
    "park": (60, 160, 60),          # green
}

CHARACTERS: dict[str, tuple[int, int, int]] = {
    "man": (200, 80, 40),
    "woman": (150, 50, 180),
    "child": (40, 200, 220),
}

ACTIONS = ["enters_and_sits_across", "enters_and_leaves", "enters_and_sits_alone"]


@dataclass
class SyntheticSceneSpec:
    scene_id: str
    location: str
    character_a: str
    character_b: str
    action: str
    duration_s: float = 4.0


def _draw_shape(frame: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    cv2.circle(frame, center, 22, color, thickness=-1)
    cv2.putText(
        frame,
        label[0].upper(),
        (center[0] - 8, center[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_scene(spec: SyntheticSceneSpec, out_dir: Path) -> tuple[Path, dict]:
    """Render ``spec`` to an .mp4 file and return (path, ground_truth_dict)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.scene_id}.mp4"

    width, height = FRAME_SIZE
    num_frames = int(spec.duration_s * FPS)
    bg_color = LOCATIONS[spec.location]
    color_a = CHARACTERS[spec.character_a]
    color_b = CHARACTERS[spec.character_b]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (width, height))

    door_x, door_y = 20, height // 2
    table_x, table_y = width // 2 + 40, height // 2

    for t in range(num_frames):
        progress = t / max(1, num_frames - 1)
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        cv2.rectangle(frame, (0, height - 20), (width, height), (30, 30, 30), thickness=-1)
        cv2.putText(
            frame,
            spec.location,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Character B is stationary at the table for the "sits across" / "sits
        # alone" scenes; absent for "enters_and_leaves".
        if spec.action != "enters_and_leaves":
            _draw_shape(frame, (table_x + 40, table_y), color_b, spec.character_b)

        # Character A always enters through the door; trajectory depends on action.
        if spec.action == "enters_and_leaves":
            # walk in for first half, walk back out for second half
            half = 0.5
            if progress <= half:
                p = progress / half
                x = int(door_x + p * (table_x - door_x))
            else:
                p = (progress - half) / (1 - half)
                x = int(table_x - p * (table_x - door_x))
            pos = (x, door_y)
        elif spec.action == "enters_and_sits_across":
            x = int(door_x + progress * (table_x - door_x))
            pos = (x, table_y)
        else:  # enters_and_sits_alone
            x = int(door_x + progress * (table_x - 60 - door_x))
            pos = (x, table_y)

        _draw_shape(frame, pos, color_a, spec.character_a)
        writer.write(frame)

    writer.release()

    ground_truth = {
        "location": spec.location,
        "characters": [spec.character_a] + ([spec.character_b] if spec.action != "enters_and_leaves" else []),
        "primary_character": spec.character_a,
        "action": spec.action,
        "relationship": (
            "sits_across_from" if spec.action == "enters_and_sits_across" else "alone"
        ),
        "num_frames": num_frames,
        "fps": FPS,
        "duration_s": spec.duration_s,
    }
    return out_path, ground_truth


def build_synthetic_dataset(out_dir: Path, seed: int = 42, num_scenes: int = 6) -> list[VideoSegment]:
    """Render ``num_scenes`` short synthetic clips and wrap them as VideoSegments."""

    rng = random.Random(seed)
    segments: list[VideoSegment] = []
    for i in range(num_scenes):
        location = rng.choice(list(LOCATIONS))
        char_a, char_b = rng.sample(list(CHARACTERS), 2)
        action = rng.choice(ACTIONS)
        spec = SyntheticSceneSpec(
            scene_id=f"scene_{i:02d}",
            location=location,
            character_a=char_a,
            character_b=char_b,
            action=action,
            duration_s=rng.uniform(3.0, 5.0),
        )
        path, ground_truth = render_scene(spec, out_dir)
        segments.append(
            VideoSegment(
                video_id="synthetic_movie_01",
                segment_id=spec.scene_id,
                source="synthetic",
                path=path,
                start_frame=0,
                end_frame=ground_truth["num_frames"],
                fps=float(FPS),
                start_time_s=0.0,
                end_time_s=spec.duration_s,
                ground_truth=ground_truth,
            )
        )
    return segments
