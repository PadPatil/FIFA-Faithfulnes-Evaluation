from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_EXTRACTOR_SYSTEM_PROMPT = (
    "Follow the user's instructions exactly. "
    "Return only the requested output, with no explanation, "
    "reasoning, markdown, or commentary."
)


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def completion(self, prompt: str, config: dict) -> str:
        pass

    def delete_model(self) -> None:
        pass


class GLM47FlashLLM(BaseLLM):

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.7-Flash",
        torch_dtype="auto",
        device_map="auto",
    ):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.model.eval()

    def delete_model(self) -> None:
        del self.model
        del self.tokenizer

    @torch.inference_mode()
    def completion(self, prompt: str, config: dict) -> str:
        messages = [
            {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=config.get("thinking", False),
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=config["do_sample"],
            temperature=config["temperature"],
            top_p=config["top_p"],
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()


class Qwen3LLM(BaseLLM):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-32B",
        torch_dtype="auto",
        device_map="auto",
    ):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.model.eval()

    def delete_model(self) -> None:
        del self.model
        del self.tokenizer

    @torch.inference_mode()
    def completion(self, prompt: str, config: dict) -> str:
        messages = [
            {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=config.get("thinking", False),
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=config["do_sample"],
            temperature=config["temperature"],
            top_p=config["top_p"],
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()


class OpenaiLLM(BaseLLM):

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        max_retries: int = 8,
    ):
        super().__init__(model_name)

        import openai

        # SDK retries eligible 429s with exponential backoff + honors the
        # Retry-After header (per OpenAI's rate-limits guide). Auth/quota
        # errors propagate as intended so they can be acted on.
        self.client = openai.OpenAI(api_key=api_key, max_retries=max_retries)

    def delete_model(self) -> None:
        self.client = None

    def completion(self, prompt: str, config: dict) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=config.get("temperature", 0),
            max_tokens=config["max_new_tokens"],
            top_p=config.get("top_p", 1),
        )
        return response.choices[0].message.content
