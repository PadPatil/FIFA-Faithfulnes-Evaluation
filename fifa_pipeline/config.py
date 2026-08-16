"""Project-wide configuration and shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MOVIENET_DIR = DATA_DIR / "movienet"
BBC_DIR = DATA_DIR / "bbc"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class PipelineConfig:
    """Global configuration for a pipeline run.

    Centralizing these knobs makes the notebook cells short and lets every stage
    (sampling, generation, evaluation) share one source of truth.
    """

    seed: int = 42
    num_frames_per_segment: int = 8
    frame_size: tuple[int, int] = (224, 224)
    clip_duration_s: float = 4.0
    clip_stride_s: float = 4.0
    captioner_name: str = "mock"  # "mock" | "qwen3-vl" | "molmo2"
    vqa_name: str = "mock"  # "mock" | "qwen3-vl"
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoSegment:
    """A single scene / multi-shot segment sampled from a source video."""

    video_id: str
    segment_id: str
    source: str  # "movienet" | "bbc" | "synthetic"
    path: Path
    start_frame: int
    end_frame: int
    fps: float
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.video_id}::{self.segment_id}"

    @property
    def num_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class Fact:
    """A single descriptive fact extracted from a generated description."""

    fact_id: str
    text: str
    category: str  # "entity" | "action" | "location" | "relationship" | "temporal"
    subject: Optional[str] = None
    predicate: Optional[str] = None
    obj: Optional[str] = None
    order: int = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Fact({self.fact_id}, [{self.category}] '{self.text}')"


@dataclass
class VerificationResult:
    fact: Fact
    question: str
    raw_answer: str
    supported: bool
    confidence: float


@dataclass
class FaithfulnessResult:
    video_uid: str
    description: str
    captioner_name: str
    fact_results: list[VerificationResult]
    description_score: float
    fact_level_score: float

    def to_row(self) -> dict[str, Any]:
        return {
            "video_uid": self.video_uid,
            "captioner": self.captioner_name,
            "description": self.description,
            "num_facts": len(self.fact_results),
            "num_supported": sum(r.supported for r in self.fact_results),
            "fact_level_score": self.fact_level_score,
            "description_score": self.description_score,
        }
