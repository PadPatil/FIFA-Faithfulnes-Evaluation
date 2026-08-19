import textwrap

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def build_fifa_graph(
    facts,
    questions,
    dependencies,
):
    G = nx.DiGraph()

    facts_by_id = {
        fact["id"]: fact
        for fact in facts
    }

    all_ids = sorted(
        set(facts_by_id)
        | set(questions)
        | set(dependencies)
    )

    for fact_id in all_ids:
        fact = facts_by_id.get(fact_id, {})

        G.add_node(
            fact_id,
            fact=fact.get("text", ""),
            broad_type=(
                fact.get("category_broad")
                or "unknown"
            ),
            detailed_type=fact.get(
                "category_detailed",
                "",
            ),
            args=fact.get(
                "args",
                [],
            ),
            question=questions.get(
                fact_id,
                "",
            ),
        )

    # FIFA dependency format: child_id -> [parent ids]
    #
    # We reverse that when constructing the graph so the
    # visual arrow reads: prerequisite fact ---> dependent fact
    for child_id, parent_ids in dependencies.items():

        for parent_id in parent_ids:

            # FIFA's 0 means "no dependency".
            if parent_id == 0:
                continue

            # Handle malformed model output where the
            # dependency references an unknown ID.
            if parent_id not in G:
                G.add_node(
                    parent_id,
                    fact="",
                    broad_type="unknown",
                    detailed_type="",
                    args=[],
                    question="",
                )

            G.add_edge(
                parent_id,
                child_id,
            )

    return G


def draw_fifa_graph(G):
    plt.figure(figsize=(24, 16))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=2.2,
        iterations=150,
    )

    categories = [
        "entity",
        "attribute",
        "action",
        "relation",
        "event",
        "other",
        "unknown",
    ]

    node_shapes = {
        "entity": "o",
        "attribute": "s",
        "action": "^",
        "relation": "D",
        "event": "h",
        "other": "v",
        "unknown": "o",
    }

    # Draw nodes grouped by FIFA broad fact type.
    for category in categories:
        nodes = [
            node_id
            for node_id, data in G.nodes(data=True)
            if data.get("broad_type", "unknown") == category
        ]

        if not nodes:
            continue

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_shape=node_shapes[category],
            node_size=7000,
        )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=20,
        width=1.5,
        connectionstyle="arc3,rad=0.05",
    )

    labels = {}

    for node_id, data in G.nodes(data=True):
        fact = textwrap.fill(
            data.get("fact", "") or "",
            width=35,
        )

        question = textwrap.fill(
            data.get("question", "") or "",
            width=40,
        )

        labels[node_id] = (
            f"#{node_id}\n"
            f"{fact}\n\n"
            f"Q: {question}"
        )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=7,
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker=node_shapes[category],
            linestyle="None",
            markersize=10,
            label=category,
        )
        for category in categories
        if any(
            data.get("broad_type", "unknown") == category
            for _, data in G.nodes(data=True)
        )
    ]
    plt.legend(
        handles=legend_handles,
        title="Fact Type",
        loc="upper left",
    )

    plt.title(
        "FIFA Fact / Question Dependency Graph",
        fontsize=16,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()
