"""End-to-end smoke test for the FIFA-for-cinema pipeline (no notebook needed)."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fifa_pipeline.config import PipelineConfig, RESULTS_DIR, SYNTHETIC_DIR
from fifa_pipeline.synthetic_data import build_synthetic_dataset
from fifa_pipeline.sampling import uniform_frame_sample, preprocess_frames, fixed_duration_clips
from fifa_pipeline.vlm_captioners import get_captioner
from fifa_pipeline.fact_extraction import extract_facts
from fifa_pipeline.stsdg import build_stsdg
from fifa_pipeline.verification import get_vqa
from fifa_pipeline.fifa_scorer import evaluate_description
from fifa_pipeline.hallucination import generate_controlled_variants
from fifa_pipeline.human_eval import simulate_human_annotations, agreement_metrics


def main():
    cfg = PipelineConfig(seed=42)
    random.seed(cfg.seed)

    print("== 1. Sampling & preprocessing ==")
    segments = build_synthetic_dataset(SYNTHETIC_DIR, seed=cfg.seed, num_scenes=6)
    print(f"Built {len(segments)} synthetic segments")
    seg = segments[0]
    frames = uniform_frame_sample(seg, num_frames=cfg.num_frames_per_segment)
    imgs = preprocess_frames(frames, size=cfg.frame_size)
    clips = fixed_duration_clips(seg, clip_duration_s=cfg.clip_duration_s)
    print(f"Sampled {len(frames)} frames -> {len(imgs)} preprocessed images; {len(clips)} fixed-duration clip window(s)")
    assert len(frames) == cfg.num_frames_per_segment
    assert seg.ground_truth["location"] in {"restaurant", "hospital", "office", "park"}

    print("\n== 2. Description generation ==")
    captioner = get_captioner("mock")
    descriptions = {s.uid: captioner.generate(uniform_frame_sample(s, 8)) for s in segments}
    for uid, desc in descriptions.items():
        print(f"  {uid}: {desc}")

    print("\n== 3. FIFA faithfulness evaluation ==")
    vqa = get_vqa("mock")
    results = []
    for s in segments:
        desc = descriptions[s.uid]
        facts = extract_facts(desc)
        assert len(facts) > 0, f"No facts extracted for: {desc}"
        graph = build_stsdg(facts)
        assert graph.number_of_nodes() == len(facts)
        result = evaluate_description(s, desc, vqa, captioner_name="mock")
        results.append(result)
        print(f"  {s.uid}: fact_level={result.fact_level_score:.2f} desc_score={result.description_score:.2f} n_facts={len(facts)}")

    baseline_scores = [r.description_score for r in results]
    print(f"Baseline description-score mean={sum(baseline_scores)/len(baseline_scores):.3f}")
    assert all(0.0 <= s <= 1.0 for s in baseline_scores)
    # On unperturbed mock-generated descriptions, the mock VQA (which independently
    # re-derives the scene) should mostly agree with the mock captioner.
    assert sum(baseline_scores) / len(baseline_scores) > 0.7

    print("\n== 4. Controlled hallucination evaluation ==")
    seg0, desc0 = segments[0], descriptions[segments[0].uid]
    variants = generate_controlled_variants(desc0, seed=0)
    assert variants, "expected at least one controlled variant to be generated"
    original_result = evaluate_description(seg0, desc0, vqa, captioner_name="mock")
    drops = 0
    for category, variant in variants.items():
        corrupted_result = evaluate_description(seg0, variant.text, vqa, captioner_name="mock")
        delta = original_result.description_score - corrupted_result.description_score
        print(
            f"  [{category}] '{variant.original_span}' -> '{variant.replacement_span}': "
            f"score {original_result.description_score:.2f} -> {corrupted_result.description_score:.2f} (delta={delta:+.2f})"
        )
        if delta > 0:
            drops += 1
    assert drops >= 1, "expected FIFA score to drop for at least one hallucination category"

    print("\n== 5. Human evaluation validation ==")
    human_df = simulate_human_annotations(results, noise_level=0.1, seed=1)
    metrics = agreement_metrics(human_df)
    print(f"  agreement metrics: {metrics}")
    assert metrics["n_facts"] > 0
    assert 0.0 <= metrics["agreement_rate"] <= 1.0

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
