"""Turn facts into verification questions and check them against the source video.

``BaseVQA`` is the common interface (mirrors a real VideoQA model: given a video
segment and a yes/no question, return an answer + confidence). ``MockVQA`` is a
dependency-free stand-in that independently re-derives the scene from raw pixels
(via ``cv_perception.observe_scene``) -- it is never shown the caption or the
synthetic ground truth, only the frames and the question -- so it can genuinely
catch a hallucinated fact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fifa_pipeline.config import Fact, VideoSegment
from fifa_pipeline.cv_perception import SceneObservation, observe_scene
from fifa_pipeline.sampling import uniform_frame_sample
from fifa_pipeline.synthetic_data import CHARACTERS, LOCATIONS

_DETERMINERS = {"a", "an", "the"}


def _strip_det(phrase: str) -> str:
    words = [w for w in phrase.split() if w.lower() not in _DETERMINERS]
    return " ".join(words).lower()


def fact_to_question(fact: Fact) -> str:
    """Generic fact -> yes/no verification question, as FIFA does before VideoQA."""

    declarative = fact.text.rstrip(".")
    return f"Is it true that: {declarative}?"


class BaseVQA(ABC):
    name: str = "base"

    @abstractmethod
    def answer(self, segment: VideoSegment, fact: Fact) -> tuple[str, bool, float]:
        """Return (raw_answer_text, supported, confidence) for ``fact`` against ``segment``."""


class MockVQA(BaseVQA):
    """CV-based stand-in for a neural VideoQA model (e.g. a QWEN3-VL VQA head)."""

    name = "mock"

    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
        self._obs_cache: dict[str, SceneObservation] = {}

    def _observe(self, segment: VideoSegment) -> SceneObservation:
        if segment.uid not in self._obs_cache:
            frames = uniform_frame_sample(segment, num_frames=self.num_frames)
            self._obs_cache[segment.uid] = observe_scene(frames)
        return self._obs_cache[segment.uid]

    def answer(self, segment: VideoSegment, fact: Fact) -> tuple[str, bool, float]:
        obs = self._observe(segment)
        supported, confidence = self._check(fact, obs)
        answer = "yes" if supported else "no"
        return answer, supported, confidence

    @staticmethod
    def _check(fact: Fact, obs: SceneObservation) -> tuple[bool, float]:
        text = fact.text.lower()

        if fact.category == "entity":
            subj = _strip_det(fact.subject or "")
            if subj in LOCATIONS:
                return subj == obs.location, 0.9
            if subj in CHARACTERS:
                return subj in obs.characters_present, 0.9
            # generic person word (e.g. "person") -- supported if anyone is present
            return bool(obs.characters_present), 0.6

        if fact.category == "location":
            loc_phrase = (fact.obj or "").lower()
            location_ok = obs.location in loc_phrase
            subject_ok, predicate_ok = MockVQA._check_subject_and_predicate(fact, obs)
            return (location_ok and subject_ok and predicate_ok), 0.9

        if fact.category == "relationship":
            subj = _strip_det(fact.subject or "")
            subject_ok = (not obs.characters_present) or (subj in obs.characters_present)
            if "alone" in text:
                rel_ok = obs.relationship == "alone"
            elif "across from" in text or " with " in f" {text} ":
                rel_ok = obs.relationship == "sits_across_from"
            else:
                rel_ok = obs.relationship != "unknown"
            return (subject_ok and rel_ok), 0.85

        if fact.category == "action":
            subject_ok, predicate_ok = MockVQA._check_subject_and_predicate(fact, obs)
            return (subject_ok and predicate_ok), 0.8

        if fact.category == "temporal":
            return True, 0.5  # refined by STSDG-based propagation, not raw VQA

        return True, 0.5

    @staticmethod
    def _check_subject_and_predicate(fact: Fact, obs: SceneObservation) -> tuple[bool, bool]:
        """Shared subject/verb check used by both 'location' facts (e.g. "enters
        the restaurant") and 'action' facts (e.g. "sits down"), since a
        location-object fact still asserts a specific action (enter vs. leave)."""

        subj = _strip_det(fact.subject or "")
        subject_ok = (not obs.characters_present) or (subj in obs.characters_present)
        pred = (fact.predicate or "").lower()
        if "enter" in pred:
            predicate_ok = obs.action in {
                "enters_and_sits_across", "enters_and_leaves", "enters_and_sits_alone",
            }
        elif "leave" in pred:
            predicate_ok = obs.action == "enters_and_leaves"
        elif "sit" in pred:
            predicate_ok = obs.action in {"enters_and_sits_across", "enters_and_sits_alone"}
        else:
            predicate_ok = True  # unrecognized predicate: don't over-penalize
        return subject_ok, predicate_ok


_REGISTRY: dict[str, type[BaseVQA]] = {"mock": MockVQA}


def get_vqa(name: str = "mock", **kwargs) -> BaseVQA:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown VQA backend '{name}'. Options: {list(_REGISTRY)}")
    return cls(**kwargs)
