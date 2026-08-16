"""Classical computer-vision "perception" shared by the mock captioner and mock VQA.

Real VLM/VQA backends (QWEN3-VL, Molmo 2) look at pixels with a neural network.
To exercise this pipeline without a GPU or downloaded weights, the mock backends
below also look at *actual pixels* -- just with classical CV (color masking +
contour/blob tracking) instead of a neural net. Crucially, the mock captioner and
mock VQA each independently re-derive the scene from the frames; neither is told
the synthetic ground truth directly. This means a hallucinated description really
can be caught by the mock VQA, which is what makes Section 3.4 (controlled
hallucination evaluation) a meaningful test of the scoring pipeline rather than a
tautology.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from fifa_pipeline.synthetic_data import CHARACTERS, LOCATIONS


def _nearest_palette_color(bgr: np.ndarray, palette: dict[str, tuple[int, int, int]]) -> tuple[str, float]:
    best_name, best_dist = None, float("inf")
    for name, color in palette.items():
        dist = float(np.linalg.norm(bgr.astype(float) - np.array(color, dtype=float)))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name, best_dist


def _detect_blobs(frame: np.ndarray) -> list[tuple[str, tuple[int, int], int]]:
    """Return (character_label, centroid_xy, area) for each recognizable blob."""

    detections = []
    for name, color in CHARACTERS.items():
        lower = np.array([max(0, c - 30) for c in color])
        upper = np.array([min(255, c + 30) for c in color])
        mask = cv2.inRange(frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80:  # filter noise
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            detections.append((name, (cx, cy), int(area)))
    return detections


@dataclass
class SceneObservation:
    location: str
    characters_present: list[str] = field(default_factory=list)
    motion: dict[str, dict] = field(default_factory=dict)  # char -> {"dx": .., "start": .., "end": ..}
    relationship: str = "unknown"
    action: str = "unknown"


def observe_scene(frames: list[np.ndarray]) -> SceneObservation:
    """Independently re-derive location/characters/action from raw frames."""

    # Location: sample background color from the corners of the middle frame,
    # avoiding the caption bar drawn at the bottom of synthetic frames.
    mid = frames[len(frames) // 2]
    h, w = mid.shape[:2]
    corner_px = mid[5:15, 5:15].reshape(-1, 3).mean(axis=0)
    location, _ = _nearest_palette_color(corner_px, LOCATIONS)

    # Characters + motion: track each color's centroid across frames.
    tracks: dict[str, list[tuple[int, int]]] = {}
    for frame in frames:
        for name, centroid, area in _detect_blobs(frame):
            tracks.setdefault(name, []).append(centroid)

    characters_present = sorted(tracks)
    motion = {}
    for name, positions in tracks.items():
        start, end = positions[0], positions[-1]
        motion[name] = {
            "start": start,
            "end": end,
            "dx": end[0] - start[0],
            "dy": end[1] - start[1],
            "displacement": float(np.hypot(end[0] - start[0], end[1] - start[1])),
        }

    # Action / relationship heuristics mirror how render_scene() constructs motion.
    action = "unknown"
    relationship = "unknown"
    if len(characters_present) >= 2:
        # a stationary character + a moving one that ends up nearby => "sits across from"
        moving = [n for n, m in motion.items() if m["displacement"] > 20]
        stationary = [n for n, m in motion.items() if m["displacement"] <= 20]
        if moving and stationary:
            action = "enters_and_sits_across"
            relationship = "sits_across_from"
        elif moving:
            action = "enters_and_leaves"
    elif len(characters_present) == 1:
        name = characters_present[0]
        m = motion[name]
        # if it ends near where it started (relative to frame width), it left again
        if m["end"][0] < w * 0.35 and m["dx"] < 0:
            action = "enters_and_leaves"
        else:
            action = "enters_and_sits_alone"
            relationship = "alone"

    return SceneObservation(
        location=location,
        characters_present=characters_present,
        motion=motion,
        relationship=relationship,
        action=action,
    )
