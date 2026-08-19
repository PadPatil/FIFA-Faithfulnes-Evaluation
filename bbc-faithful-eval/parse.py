import re


_TUPLE_RE = re.compile(
    r"^(?P<broad>[A-Za-z]+)\s*"
    r"(?:-\s*(?P<detailed>[A-Za-z ]*?))?\s*"
    r"\((?P<args>.*)\)\s*$"
)


def _clean_lines(output_str):
    if "</think>" in output_str:
        output_str = output_str.rsplit("</think>", 1)[1]

    if "output:" in output_str:
        output_str = output_str.split("output:", 1)[1]

    return [
        line.strip()
        for line in output_str.strip().splitlines()
        if line.strip()
        and not line.strip().startswith("```")
    ]


def _split_id(line):
    if "|" not in line:
        return None, None

    head, rest = line.split("|", 1)
    head = head.strip().lstrip("#").strip()

    if not head.isdigit():
        return None, None

    return int(head), rest.strip()


def parse_tuples(output_str):
    facts = []
    dropped = []
    seen = set()

    for line in _clean_lines(output_str):
        tuple_id, payload = _split_id(line)

        if tuple_id is None or tuple_id in seen:
            dropped.append(line)
            continue

        seen.add(tuple_id)

        match = _TUPLE_RE.match(payload)

        facts.append(
            {
                "id": tuple_id,
                "raw": line,
                "text": payload,
                "category_broad": (
                    match.group("broad").strip().lower()
                    if match
                    else None
                ),
                "category_detailed": (
                    (match.group("detailed") or "").strip().lower()
                    if match
                    else ""
                ),
                "args": (
                    [
                        arg.strip()
                        for arg in match.group("args").split(",")
                        if arg.strip()
                    ]
                    if match
                    else []
                ),
            }
        )

    facts.sort(key=lambda fact: fact["id"])

    return facts, dropped


def parse_questions(output_str, valid_ids=None):
    """Parse `id | question` lines into {id: question}.

    Parameters
    ----------
    output_str : str
        Raw LLM output containing one question per line.

    valid_ids : set[int] | None
        Optional set of tuple/fact IDs. If provided, questions whose IDs
        are not present in valid_ids are rejected.

    Returns
    -------
    questions : dict[int, str]
        Parsed question text keyed by fact ID.

    dropped : list[str]
        Lines that could not be parsed, had duplicate IDs, referenced
        unknown IDs, or contained an empty question.
    """
    questions = {}
    dropped = []

    for line in _clean_lines(output_str):
        qid, payload = _split_id(line)

        if qid is None:
            dropped.append(line)
            continue

        if qid in questions:
            dropped.append(line)
            continue

        if valid_ids is not None and qid not in valid_ids:
            dropped.append(line)
            continue

        payload = payload.strip()

        if not payload:
            dropped.append(line)
            continue

        questions[qid] = payload

    return questions, dropped


def parse_dependencies(output_str, valid_ids=None):
    deps = {}
    dropped = []
    invalid_parents = []

    for line in _clean_lines(output_str):
        did, payload = _split_id(line)

        if did is None or did in deps:
            dropped.append(line)
            continue

        parents = []

        for tok in payload.split(","):
            tok = tok.strip()

            if not tok.isdigit():
                continue

            pid = int(tok)

            if pid == 0:
                continue

            if pid == did:
                continue

            if pid in parents:
                continue

            if valid_ids is not None and pid not in valid_ids:
                invalid_parents.append(
                    {
                        "child": did,
                        "parent": pid,
                        "raw": line,
                    }
                )
                continue

            parents.append(pid)

        deps[did] = parents

    return deps, dropped, invalid_parents


def validate_dsg(facts, deps):
    ids = {f["id"] for f in facts}

    report = {
        "missing_entry": sorted(ids - set(deps)),
        "unknown_id": sorted(set(deps) - ids),
        "dangling_parent": sorted(
            {
                p
                for child, parents in deps.items()
                for p in parents
                if p not in ids
            }
        ),
        "roots": sorted(
            child
            for child, parents in deps.items()
            if not parents
        ),
        "forward_ref": sorted(
            child
            for child, parents in deps.items()
            if any(parent > child for parent in parents)
        ),
    }

    colour = {}
    cycles = []

    def walk(node, path):
        if colour.get(node) == 2:
            return

        if colour.get(node) == 1:
            if node in path:
                cycle_start = path.index(node)
                cycles.append(
                    path[cycle_start:] + [node]
                )
            return

        colour[node] = 1

        for parent in deps.get(node, []):
            walk(
                parent,
                path + [node],
            )

        colour[node] = 2

    for node in deps:
        walk(node, [])

    report["cycles"] = cycles

    num_edges = sum(
        len(parents)
        for parents in deps.values()
    )

    print(
        f"nodes={len(ids)} "
        f"dependency_entries={len(deps)} "
        f"edges={num_edges} "
        f"roots={len(report['roots'])}"
    )

    error_fields = [
        "missing_entry",
        "unknown_id",
        "dangling_parent",
        "cycles",
    ]

    info_fields = [
        "forward_ref",
    ]

    for key in error_fields:
        if report[key]:
            print(f"  ERROR {key}: {report[key]}")

    for key in info_fields:
        if report[key]:
            print(f"  INFO {key}: {report[key]}")

    if not any(report[k] for k in error_fields):
        print(
            "  graph is a well-formed DAG "
            "covering all extracted tuples"
        )

    return report
