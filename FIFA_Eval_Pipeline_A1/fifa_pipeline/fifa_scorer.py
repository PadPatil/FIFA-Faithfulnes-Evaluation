"""Orchestrates FIFA-style faithfulness scoring: fact extraction -> STSDG ->
question generation -> VQA verification -> dependency-aware score aggregation.
"""

from __future__ import annotations

import networkx as nx

from fifa_pipeline.config import FaithfulnessResult, VideoSegment, VerificationResult
from fifa_pipeline.fact_extraction import extract_facts
from fifa_pipeline.stsdg import build_stsdg, topological_fact_order
from fifa_pipeline.verification import BaseVQA, fact_to_question


def _propagate_dependencies(graph: nx.DiGraph, supported: dict[str, bool]) -> dict[str, bool]:
    """If an ``entity_support`` predecessor is unsupported, its dependents cannot
    be considered faithful either, even if the VQA model answered "yes" for them
    in isolation. This is the STSDG-guided aggregation step."""

    propagated = dict(supported)
    for node in topological_fact_order(graph):
        for pred in graph.predecessors(node):
            if graph.edges[pred, node]["kind"] == "entity_support" and not propagated[pred]:
                propagated[node] = False
    return propagated


def evaluate_description(
    segment: VideoSegment, description: str, vqa: BaseVQA, captioner_name: str = "unknown"
) -> FaithfulnessResult:
    facts = extract_facts(description)
    if not facts:
        return FaithfulnessResult(
            video_uid=segment.uid, description=description, captioner_name=captioner_name,
            fact_results=[], description_score=float("nan"), fact_level_score=float("nan"),
        )

    graph = build_stsdg(facts)
    fact_by_id = {f.fact_id: f for f in facts}

    raw_supported: dict[str, bool] = {}
    results: dict[str, VerificationResult] = {}
    for fact_id in topological_fact_order(graph):
        fact = fact_by_id[fact_id]
        question = fact_to_question(fact)
        raw_answer, supported, confidence = vqa.answer(segment, fact)
        raw_supported[fact_id] = supported
        results[fact_id] = VerificationResult(
            fact=fact, question=question, raw_answer=raw_answer, supported=supported, confidence=confidence,
        )

    propagated = _propagate_dependencies(graph, raw_supported)
    for fact_id, result in results.items():
        result.supported = propagated[fact_id]

    ordered_results = [results[f.fact_id] for f in facts]
    fact_level_score = sum(raw_supported.values()) / len(raw_supported)
    description_score = sum(propagated.values()) / len(propagated)

    return FaithfulnessResult(
        video_uid=segment.uid,
        description=description,
        captioner_name=captioner_name,
        fact_results=ordered_results,
        description_score=description_score,
        fact_level_score=fact_level_score,
    )
