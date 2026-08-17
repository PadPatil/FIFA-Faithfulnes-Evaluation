"""FIFA-for-cinema: an evaluation pipeline for measuring the factual faithfulness
of vision-language-model (VLM) generated descriptions of long-form cinematic video.

This package implements the pipeline described in the project proposal:

    1. Video sampling & preprocessing      -> fifa_pipeline.sampling / synthetic_data / datasets
    2. Description generation (VLMs)       -> fifa_pipeline.vlm_captioners
    3. FIFA faithfulness evaluation        -> fifa_pipeline.fact_extraction / stsdg / verification / fifa_scorer
    4. Controlled hallucination evaluation -> fifa_pipeline.hallucination
    5. Human evaluation validation         -> fifa_pipeline.human_eval

The heavy VLM/VQA backends (QWEN3-VL, Molmo 2) are optional and imported lazily so
that the rest of the pipeline can run end-to-end (e.g. in CI, or on a laptop without
a GPU) using lightweight, deterministic "mock" backends that operate on real pixel
data via classical computer vision instead of a neural network.
"""

from fifa_pipeline.config import PipelineConfig

__all__ = ["PipelineConfig"]

__version__ = "0.1.0"
