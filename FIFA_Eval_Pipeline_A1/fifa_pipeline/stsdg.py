"""Build the Spatio-Temporal Semantic Dependency Graph (STSDG) over extracted facts.

Two kinds of edges are modeled, mirroring the dependencies FIFA uses to guide
question generation and score aggregation:

* ``entity_support`` -- an *entity* fact (e.g. "There is a woman.") supports
  every *event* fact (action/location/relationship/temporal) that mentions that
  entity as its subject or object. If the entity itself turns out to be
  unsupported by the video, dependent event facts about it cannot be trusted
  either, regardless of what the VQA model says about them in isolation.
* ``temporal_next`` -- consecutive event facts (in the order they were narrated)
  are chained together, capturing the fact that later events are described as
  following earlier ones.

The graph is a DAG by construction (edges only point from earlier-appearing
facts to later ones), which lets us do a single topological pass during scoring.
"""

from __future__ import annotations

import networkx as nx

from fifa_pipeline.config import Fact


def build_stsdg(facts: list[Fact]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for fact in facts:
        graph.add_node(fact.fact_id, fact=fact)

    entity_facts = [f for f in facts if f.category == "entity"]
    event_facts = sorted((f for f in facts if f.category != "entity"), key=lambda f: f.order)

    for ef in entity_facts:
        entity_name = (ef.subject or "").lower()
        if not entity_name:
            continue
        for evf in event_facts:
            mentions = (evf.subject or "").lower() == entity_name or entity_name in (evf.obj or "").lower()
            if mentions and evf.order >= ef.order:
                graph.add_edge(ef.fact_id, evf.fact_id, kind="entity_support")

    for a, b in zip(event_facts, event_facts[1:]):
        graph.add_edge(a.fact_id, b.fact_id, kind="temporal_next")

    return graph


def topological_fact_order(graph: nx.DiGraph) -> list[str]:
    return list(nx.topological_sort(graph))
