# FIFA for Long-Form Cinematic Video

CSED 504 NLP project — Aeden Jameson, Padmanabh Patil.

Evaluates whether [FIFA](https://github.com/du-nlp-lab/FIFA) (Jing et al., 2026) can
identify hallucinations in VLM-generated descriptions of cinematic video. See
`NLP_Proposal.ipynb` for the full write-up and a runnable implementation of every
pipeline stage.

## What's implemented

`fifa_pipeline/` is a small, well-tested package implementing the full pipeline:

| Stage | Module |
|---|---|
| Video sampling & preprocessing | `sampling.py`, `synthetic_data.py`, `datasets.py` |
| Description generation (VLMs) | `vlm_captioners.py` |
| Fact extraction | `fact_extraction.py` |
| Spatio-Temporal Semantic Dependency Graph | `stsdg.py` |
| Fact verification (VideoQA) | `verification.py` |
| Faithfulness scoring | `fifa_scorer.py` |
| Controlled hallucination injection | `hallucination.py` |
| Human evaluation validation | `human_eval.py` |

**No datasets are downloaded.** Instead of MovieNet/BBC, the pipeline renders a
handful of short (3-5s) synthetic clips with known ground truth
(`fifa_pipeline/synthetic_data.py`), which is enough to exercise every stage of
the pipeline end-to-end. The MovieNet/BBC-format loaders in `datasets.py` are
real, format-compatible parsers you can point at a small local sample later.

**No GPU/model weights are required.** The description generator and VideoQA
verifier default to lightweight "mock" backends that use classical computer
vision (color/blob detection) on the actual frames instead of a neural network.
Real `Qwen3-VL` / `Molmo 2` backends are implemented against 🤗 `transformers` in
`vlm_captioners.py` and are used automatically once the optional dependencies in
`requirements-full.txt` are installed and `captioner_name` is set accordingly.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name=fifa-video-faithfulness \
    --display-name "Python 3.11 (fifa-video-faithfulness)"
```

Open `NLP_Proposal.ipynb` and select the "Python 3.11 (fifa-video-faithfulness)"
kernel, or run it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=fifa-video-faithfulness \
    NLP_Proposal.ipynb
```

A quick non-notebook sanity check of the whole pipeline is also available:

```bash
python scripts/smoke_test.py
```

To use the real QWEN3-VL / Molmo 2 backends (requires a CUDA GPU):

```bash
pip install -r requirements-full.txt
```

then set `CONFIG.captioner_name = "qwen3-vl"` (or `"molmo2"`) in the notebook.
# FIFA-Faithfulnes-Evaluation
