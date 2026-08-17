"""Inject controlled factual errors into a generated description (Section 3.4).

Targets the controlled vocabulary produced by the captioners in this project
(character names, location names, and a small set of actions/relationships), so
that we can measure whether FIFA's faithfulness score reliably drops for a known,
specific corruption. A production version would swap the regex-based substitution
here for an LLM-based paraphraser capable of perturbing arbitrary free text while
preserving fluency.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from fifa_pipeline.synthetic_data import CHARACTERS, LOCATIONS

EXTRA_ENTITIES = ["dog", "robot"]
ACTION_SWAPS = {"enters": "leaves", "leaves": "enters"}


@dataclass
class HallucinationVariant:
    category: str
    text: str
    original_span: str
    replacement_span: str


def _swap_first(pattern: str, description: str, replacement_fn) -> tuple[str, str, str] | None:
    match = re.search(pattern, description, flags=re.IGNORECASE)
    if not match:
        return None
    original = match.group(0)
    replacement = replacement_fn(match)
    new_text = description[: match.start()] + replacement + description[match.end() :]
    return new_text, original, replacement


def inject_entity_error(description: str, rng: random.Random) -> HallucinationVariant | None:
    pool = list(CHARACTERS) + EXTRA_ENTITIES
    for name in CHARACTERS:
        pattern = rf"\b{name}\b"
        if re.search(pattern, description, flags=re.IGNORECASE):
            choices = [c for c in pool if c != name]
            replacement = rng.choice(choices)
            result = _swap_first(pattern, description, lambda m: replacement)
            if result:
                new_text, original, repl = result
                return HallucinationVariant("entity", new_text, original, repl)
    return None


def inject_location_error(description: str, rng: random.Random) -> HallucinationVariant | None:
    for name in LOCATIONS:
        pattern = rf"\b{name}\b"
        if re.search(pattern, description, flags=re.IGNORECASE):
            choices = [c for c in LOCATIONS if c != name]
            replacement = rng.choice(choices)
            result = _swap_first(pattern, description, lambda m: replacement)
            if result:
                new_text, original, repl = result
                return HallucinationVariant("location", new_text, original, repl)
    return None


def inject_action_error(description: str, rng: random.Random) -> HallucinationVariant | None:
    for word, swap in ACTION_SWAPS.items():
        pattern = rf"\b{word}\b"
        if re.search(pattern, description, flags=re.IGNORECASE):
            result = _swap_first(pattern, description, lambda m: swap)
            if result:
                new_text, original, repl = result
                return HallucinationVariant("action", new_text, original, repl)
    return None


def inject_relationship_error(description: str, rng: random.Random) -> HallucinationVariant | None:
    across_pattern = r"across from (a|the) \w+"
    if re.search(across_pattern, description, flags=re.IGNORECASE):
        result = _swap_first(across_pattern, description, lambda m: "alone")
        if result:
            new_text, original, repl = result
            return HallucinationVariant("relationship", new_text, original, repl)

    alone_pattern = r"\balone\b"
    if re.search(alone_pattern, description, flags=re.IGNORECASE):
        stranger = rng.choice([c for c in CHARACTERS])
        result = _swap_first(alone_pattern, description, lambda m: f"across from a {stranger}")
        if result:
            new_text, original, repl = result
            return HallucinationVariant("relationship", new_text, original, repl)
    return None


INJECTORS = {
    "entity": inject_entity_error,
    "location": inject_location_error,
    "action": inject_action_error,
    "relationship": inject_relationship_error,
}


def generate_controlled_variants(description: str, seed: int = 0) -> dict[str, HallucinationVariant]:
    rng = random.Random(seed)
    variants = {}
    for category, injector in INJECTORS.items():
        variant = injector(description, rng)
        if variant is not None:
            variants[category] = variant
    return variants
