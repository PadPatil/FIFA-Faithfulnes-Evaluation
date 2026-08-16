"""Extract descriptive facts from a generated natural-language description.

FIFA's own fact extraction step uses an LLM. To keep this pipeline runnable
without an LLM API key, we use a lightweight, deterministic, dependency-parse
based extractor (spaCy) that is good enough for the kinds of single/multi-clause
sentences produced by our captioners (real or mock). Each sentence is split into
one "event" per (subject, verb, [object/prep-phrase]) tuple, and each event is
turned into one or more :class:`~fifa_pipeline.config.Fact` objects covering the
categories described in the proposal: entity, action, location, relationship, and
temporal facts.

Swapping in an LLM-based extractor later only requires implementing the same
``extract_facts(text) -> list[Fact]`` signature.
"""

from __future__ import annotations

import functools

import spacy

from fifa_pipeline.config import Fact

LOCATION_WORDS = {
    "restaurant", "hospital", "office", "park", "room", "kitchen", "house",
    "apartment", "school", "station", "home", "street", "car", "garden", "bar",
}
PERSON_WORDS = {
    "man", "woman", "child", "person", "boy", "girl", "someone", "dog", "cat",
    "robot", "kid", "guy", "lady",
}
RELATIONSHIP_PREPS = {"with", "across", "from", "beside", "near", "alone", "opposite"}
LOCATION_PREPS = {"in", "at", "into", "inside", "within", "onto"}
TEMPORAL_MARKERS = {"then", "after", "before", "next", "finally", "later", "afterwards"}


@functools.lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm")


def _noun_phrase(token) -> str:
    words = [t.text for t in token.subtree if t.dep_ not in {"cc"}]
    return " ".join(words)


_DETERMINERS = {"a", "an", "the"}


def _there_is(phrase: str) -> str:
    """Compose a natural 'There is a/an X.' sentence, avoiding double determiners
    when ``phrase`` already starts with one (e.g. spaCy noun-chunk text 'A man')."""

    if not phrase:
        return "There is something."
    phrase = phrase[0].lower() + phrase[1:]  # mid-sentence casing, even if sentence-initial in the source
    first_word = phrase.split(" ", 1)[0]
    if first_word in _DETERMINERS:
        return f"There is {phrase}."
    article = "an" if phrase[:1] in "aeiou" else "a"
    return f"There is {article} {phrase}."


def _resolve_prep_chain(prep_token):
    """Walk a (possibly nested) preposition chain, e.g. 'across' -> 'from' -> pobj.

    Returns (phrase_text, head_noun_token) or (None, None) if no object found.
    """

    node = prep_token
    words = [prep_token.text]
    while True:
        pobj = next((c for c in node.children if c.dep_ == "pobj"), None)
        if pobj is not None:
            words.append(_noun_phrase(pobj))
            return " ".join(words), pobj
        nested_prep = next((c for c in node.children if c.dep_ == "prep"), None)
        if nested_prep is None:
            return None, None
        words.append(nested_prep.text)
        node = nested_prep


def _verb_events(sent):
    """Yield verb tokens representing one 'event' each: the sentence ROOT plus
    any conjuncts of it (e.g. 'enters ... and sits ...')."""

    root = next((t for t in sent if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}), None)
    if root is None:
        return []
    events = [root]
    events.extend(t for t in root.children if t.dep_ == "conj" and t.pos_ == "VERB")
    return events


def _subject_of(verb_token, fallback_subject):
    subj = next((c for c in verb_token.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
    if subj is not None:
        return _noun_phrase(subj), subj
    return fallback_subject


def extract_facts(text: str) -> list[Fact]:
    doc = _nlp()(text)
    facts: list[Fact] = []
    seen_entities: set[str] = set()
    fact_counter = 0

    def add_fact(fact_text: str, category: str, subject=None, predicate=None, obj=None, order: int = 0):
        nonlocal fact_counter
        facts.append(
            Fact(
                fact_id=f"f{fact_counter}",
                text=fact_text,
                category=category,
                subject=subject,
                predicate=predicate,
                obj=obj,
                order=order,
            )
        )
        fact_counter += 1

    for sent in doc.sents:
        events = _verb_events(sent)
        if not events:
            continue

        fallback_subject = None
        event_texts_in_order = []

        for verb in events:
            subject_text, subject_tok = _subject_of(verb, fallback_subject)
            if subject_text is None:
                continue
            fallback_subject = (subject_text, subject_tok)

            if subject_text not in seen_entities:
                add_fact(_there_is(subject_text), "entity", subject=subject_text, order=verb.i)
                seen_entities.add(subject_text)

            particle = next((c.text for c in verb.children if c.dep_ == "prt"), None)
            verb_phrase = f"{verb.lemma_} {particle}" if particle else verb.lemma_

            dobj = next((c for c in verb.children if c.dep_ == "dobj"), None)
            preps = [c for c in verb.children if c.dep_ == "prep"]
            advmods = [c.text for c in verb.children if c.dep_ == "advmod" and c.text.lower() not in TEMPORAL_MARKERS]

            location_phrase, relationship_phrase = None, None

            if dobj is not None:
                dobj_np = _noun_phrase(dobj)
                dobj_head = dobj.lemma_.lower()
                if dobj_head in LOCATION_WORDS:
                    location_phrase = dobj_np
                    if dobj_np not in seen_entities:
                        add_fact(_there_is(dobj_np), "entity", subject=dobj_np, order=dobj.i)
                        seen_entities.add(dobj_np)
                elif dobj_head in PERSON_WORDS:
                    relationship_phrase = f"{verb_phrase} {dobj_np}"
                    if dobj_np not in seen_entities:
                        add_fact(_there_is(dobj_np), "entity", subject=dobj_np, order=dobj.i)
                        seen_entities.add(dobj_np)
                else:
                    # generic action target, e.g. "opens the door"
                    add_fact(
                        f"{subject_text} {verb_phrase} {dobj_np}.", "action",
                        subject=subject_text, predicate=verb_phrase, obj=dobj_np, order=verb.i,
                    )

            for prep in preps:
                phrase, head_tok = _resolve_prep_chain(prep)
                if phrase is None:
                    continue
                if prep.text.lower() in LOCATION_PREPS:
                    location_phrase = _noun_phrase(head_tok)
                elif prep.text.lower() in RELATIONSHIP_PREPS or head_tok.lemma_.lower() in PERSON_WORDS:
                    relationship_phrase = f"{verb_phrase} {phrase}"
                    head_np = _noun_phrase(head_tok)
                    if head_np not in seen_entities:
                        add_fact(_there_is(head_np), "entity", subject=head_np, order=head_tok.i)
                        seen_entities.add(head_np)

            if any(w.lower() == "alone" for w in advmods):
                relationship_phrase = f"{verb_phrase} alone"

            if location_phrase:
                add_fact(
                    f"{subject_text} {verb_phrase} {location_phrase}.", "location",
                    subject=subject_text, predicate=verb_phrase, obj=location_phrase, order=verb.i,
                )
                event_texts_in_order.append((verb.i, f"{subject_text} {verb_phrase} {location_phrase}"))
            elif relationship_phrase:
                add_fact(
                    f"{subject_text} {relationship_phrase}.", "relationship",
                    subject=subject_text, predicate=verb_phrase, obj=relationship_phrase, order=verb.i,
                )
                event_texts_in_order.append((verb.i, f"{subject_text} {relationship_phrase}"))
            elif dobj is None:
                extra = f" {' '.join(advmods)}" if advmods else ""
                add_fact(
                    f"{subject_text} {verb_phrase}{extra}.", "action",
                    subject=subject_text, predicate=verb_phrase, order=verb.i,
                )
                event_texts_in_order.append((verb.i, f"{subject_text} {verb_phrase}{extra}"))

        has_temporal_marker = any(t.text.lower() in TEMPORAL_MARKERS for t in sent)
        if has_temporal_marker and len(event_texts_in_order) >= 2:
            (_, first_text), (_, second_text) = event_texts_in_order[0], event_texts_in_order[-1]
            add_fact(
                f"'{first_text}' happens before '{second_text}'.", "temporal",
                subject=first_text, obj=second_text, order=event_texts_in_order[-1][0],
            )

    facts.sort(key=lambda f: f.order)
    for i, f in enumerate(facts):
        f.fact_id = f"f{i}"
    return facts
