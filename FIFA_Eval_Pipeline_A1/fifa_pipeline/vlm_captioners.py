"""Description generation backends.

``BaseCaptioner`` is the common interface. ``MockCaptioner`` is a dependency-free
stand-in that runs anywhere (no GPU, no downloaded weights, no network) by using
classical CV perception (see ``cv_perception.py``). ``Qwen3VLCaptioner`` and
``Molmo2Captioner`` are real integrations against Hugging Face ``transformers``;
they are imported lazily so that simply importing this module never requires
``torch``/``transformers`` to be installed. Use :func:`get_captioner` to select a
backend by name, with automatic fallback to the mock backend if a real backend's
dependencies aren't installed.
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from fifa_pipeline.cv_perception import observe_scene

ACTION_TEXT = {
    "enters_and_sits_across": "enters the {location} and sits down across from a {b}",
    "enters_and_leaves": "enters the {location} and then leaves again",
    "enters_and_sits_alone": "enters the {location} and sits down alone",
}


class BaseCaptioner(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, frames: list[np.ndarray] | list[Image.Image], prompt: str | None = None) -> str:
        """Return a natural-language description of the given frames."""


class MockCaptioner(BaseCaptioner):
    """Deterministic, CV-based description generator.

    Stands in for a real VLM so the rest of the pipeline (fact extraction, STSDG
    construction, FIFA scoring, hallucination injection) can be exercised without
    a GPU. It only looks at raw pixels via ``observe_scene`` -- it is never given
    the synthetic ground truth directly.
    """

    name = "mock"

    def generate(self, frames: list[np.ndarray] | list[Image.Image], prompt: str | None = None) -> str:
        np_frames = [np.array(f)[:, :, ::-1] if isinstance(f, Image.Image) else f for f in frames]
        obs = observe_scene(np_frames)

        primary = obs.characters_present[0] if obs.characters_present else "person"
        secondary = obs.characters_present[1] if len(obs.characters_present) > 1 else "person"
        template = ACTION_TEXT.get(obs.action, "is in the {location}")
        action_phrase = template.format(location=obs.location, b=secondary)
        return f"A {primary} {action_phrase}."


class Qwen3VLCaptioner(BaseCaptioner):
    """Real QWEN3-VL backend (requires ``transformers`` + model weights + GPU)."""

    name = "qwen3-vl"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "Qwen3VLCaptioner requires the optional 'torch' and 'transformers' "
                "packages. Install them (see requirements-full.txt) or use "
                "captioner_name='mock' to run the pipeline without a GPU."
            ) from exc

        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

    def generate(self, frames: list[np.ndarray] | list[Image.Image], prompt: str | None = None) -> str:
        import torch

        prompt = prompt or (
            "Describe what happens in this video segment in one or two factual "
            "sentences: who is present, what they do, and where the scene takes place."
        )
        images = [Image.fromarray(f[:, :, ::-1]) if isinstance(f, np.ndarray) else f for f in frames]
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in images]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", images=images
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]


class Molmo2Captioner(BaseCaptioner):
    """Real Molmo 2 backend (requires ``transformers`` + model weights + GPU)."""

    name = "molmo2"

    def __init__(self, model_id: str = "allenai/Molmo-2-8B-0126", device: str = "cuda"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Molmo2Captioner requires the optional 'torch' and 'transformers' "
                "packages. Install them (see requirements-full.txt) or use "
                "captioner_name='mock' to run the pipeline without a GPU."
            ) from exc

        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    def generate(self, frames: list[np.ndarray] | list[Image.Image], prompt: str | None = None) -> str:
        import torch

        prompt = prompt or (
            "Describe what happens in this video segment: who is present, what "
            "they do, and where the scene takes place."
        )
        images = [Image.fromarray(f[:, :, ::-1]) if isinstance(f, np.ndarray) else f for f in frames]
        inputs = self.processor.process(images=images, text=prompt)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs, max_new_tokens=128, tokenizer=self.processor.tokenizer
            )
        return self.processor.tokenizer.decode(output[0], skip_special_tokens=True)


_REGISTRY: dict[str, type[BaseCaptioner]] = {
    "mock": MockCaptioner,
    "qwen3-vl": Qwen3VLCaptioner,
    "molmo2": Molmo2Captioner,
}


def get_captioner(name: str = "mock", **kwargs) -> BaseCaptioner:
    """Instantiate a captioner by name, falling back to the mock backend if the
    requested backend's dependencies (torch/transformers/weights) aren't available.
    """

    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown captioner '{name}'. Options: {list(_REGISTRY)}")
    if name == "mock":
        return MockCaptioner()
    try:
        return cls(**kwargs)
    except ImportError as exc:
        warnings.warn(f"Falling back to MockCaptioner: {exc}")
        return MockCaptioner()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
