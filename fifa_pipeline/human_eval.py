"""Human-evaluated subset validation (Section 3.5).

``load_human_annotations`` reads a simple CSV that a human annotator fills in
(one row per fact: supported / unsupported / ambiguous) -- see
``export_annotation_template`` for how that CSV is produced from a batch of FIFA
results. For the automated, no-human-in-the-loop demo run, ``simulate_human_annotations``
generates a synthetic annotator by starting from the (independently CV-derived)
correctness signal and flipping a small fraction of labels to model realistic
annotator noise/disagreement; this is clearly a stand-in and should be replaced
with ``load_human_annotations`` once real annotations exist.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pointbiserialr

from fifa_pipeline.config import FaithfulnessResult

LABELS = ("supported", "unsupported", "ambiguous")
_LABEL_TO_SCORE = {"supported": 1.0, "ambiguous": 0.5, "unsupported": 0.0}


def export_annotation_template(results: list[FaithfulnessResult], path: Path) -> pd.DataFrame:
    """Write a CSV a human can fill in (``label`` column left blank)."""

    rows = []
    for result in results:
        for fr in result.fact_results:
            rows.append(
                {
                    "video_uid": result.video_uid,
                    "captioner": result.captioner_name,
                    "fact_id": fr.fact.fact_id,
                    "fact_text": fr.fact.text,
                    "category": fr.fact.category,
                    "fifa_supported": fr.supported,
                    "label": "",  # human fills in: supported | unsupported | ambiguous
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_human_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(df["label"].fillna("")) - set(LABELS) - {""}
    if missing:
        raise ValueError(f"Unknown labels in {path}: {missing}. Expected one of {LABELS}.")
    return df


def simulate_human_annotations(
    results: list[FaithfulnessResult], noise_level: float = 0.15, ambiguous_rate: float = 0.1, seed: int = 0
) -> pd.DataFrame:
    """Synthesize a plausible human annotation pass for demo/testing purposes only."""

    rng = random.Random(seed)
    rows = []
    for result in results:
        for fr in result.fact_results:
            true_label = "supported" if fr.supported else "unsupported"
            if rng.random() < ambiguous_rate:
                label = "ambiguous"
            elif rng.random() < noise_level:
                label = "unsupported" if true_label == "supported" else "supported"
            else:
                label = true_label
            rows.append(
                {
                    "video_uid": result.video_uid,
                    "captioner": result.captioner_name,
                    "fact_id": fr.fact.fact_id,
                    "fact_text": fr.fact.text,
                    "category": fr.fact.category,
                    "fifa_supported": fr.supported,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def agreement_metrics(annotations: pd.DataFrame) -> dict:
    """Compare human labels against FIFA's per-fact supported/unsupported calls."""

    df = annotations[annotations["label"].isin(LABELS)].copy()
    df["human_supported"] = df["label"] == "supported"
    df["human_score"] = df["label"].map(_LABEL_TO_SCORE)

    non_ambiguous = df[df["label"] != "ambiguous"]
    agreement_rate = float((non_ambiguous["human_supported"] == non_ambiguous["fifa_supported"]).mean()) if len(non_ambiguous) else float("nan")
    kappa = (
        cohen_kappa_score(non_ambiguous["human_supported"], non_ambiguous["fifa_supported"])
        if len(non_ambiguous) > 1
        else float("nan")
    )
    if df["human_score"].nunique() > 1 and df["fifa_supported"].nunique() > 1:
        corr, p_value = pointbiserialr(df["fifa_supported"].astype(float), df["human_score"])
    else:
        corr, p_value = float("nan"), float("nan")

    return {
        "n_facts": int(len(df)),
        "n_ambiguous": int((df["label"] == "ambiguous").sum()),
        "agreement_rate": agreement_rate,
        "cohen_kappa": float(kappa) if kappa == kappa else float("nan"),
        "point_biserial_r": float(corr) if corr == corr else float("nan"),
        "p_value": float(p_value) if p_value == p_value else float("nan"),
    }
