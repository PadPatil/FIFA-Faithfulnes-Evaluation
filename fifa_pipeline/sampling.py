"""Video sampling & preprocessing.

Implements the two sampling strategies mentioned in the proposal:

* ``uniform_frame_sample`` -- pick N frames spread evenly across a segment.
* ``fixed_duration_clips`` -- split a segment into fixed-length, possibly
  overlapping sub-clips (useful for segments too long to describe in one pass).

Both operate on real video files via OpenCV, and ``preprocess_frames`` turns raw
BGR frames into the RGB, resized ``PIL.Image`` list most VLM front-ends expect.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import cv2

from fifa_pipeline.config import VideoSegment


def uniform_frame_sample(segment: VideoSegment, num_frames: int = 8) -> list[np.ndarray]:
    """Sample ``num_frames`` frames uniformly spaced across ``segment``."""

    cap = cv2.VideoCapture(str(segment.path))
    if not cap.isOpened():
        raise IOError(f"Could not open video at {segment.path}")

    total = segment.num_frames if segment.num_frames > 0 else int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    total = max(total, 1)
    indices = np.linspace(
        segment.start_frame, max(segment.start_frame, segment.end_frame - 1), num=num_frames
    ).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read for segment {segment.uid}")
    return frames


def fixed_duration_clips(
    segment: VideoSegment, clip_duration_s: float = 4.0, stride_s: float | None = None
) -> list[tuple[float, float]]:
    """Split ``segment`` into fixed-duration (start_s, end_s) windows.

    Falls back to stride == clip_duration_s (non-overlapping) if not given.
    """

    stride_s = stride_s or clip_duration_s
    windows = []
    t = segment.start_time_s
    end = segment.end_time_s if segment.end_time_s > segment.start_time_s else segment.num_frames / segment.fps
    while t < end:
        windows.append((t, min(t + clip_duration_s, end)))
        t += stride_s
    return windows or [(segment.start_time_s, end)]


def preprocess_frames(
    frames: list[np.ndarray], size: tuple[int, int] = (224, 224)
) -> list[Image.Image]:
    """Convert BGR OpenCV frames into resized RGB PIL images for VLM input."""

    processed = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(size)
        processed.append(img)
    return processed
