from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from inference.config import load_inference_settings
from inference.contracts import (
    InferenceConfig,
    InferenceRequest,
    InferenceResult,
    ModelSpec,
    Precision,
)


@dataclass(frozen=True)
class LoadedQwenVl:
    model: Any
    processor: Any
    device: str


class QwenVlBackend:
    def __init__(self, model: ModelSpec, config: InferenceConfig):
        self.model_spec = model
        self.config = config
        self.loaded: LoadedQwenVl | None = None

    @classmethod
    def from_config(cls, config_path: Path = Path("config.toml")) -> "QwenVlBackend":
        settings = load_inference_settings(config_path)
        return cls(model=settings.model, config=settings.config)

    def load(self) -> LoadedQwenVl:
        if self.loaded is not None:
            return self.loaded

        accelerator = _build_accelerator()
        processor_cls, model_cls = _transformers_classes()
        source = _model_source(self.model_spec)

        processor = processor_cls.from_pretrained(source)
        model = model_cls.from_pretrained(
            source,
            dtype=_torch_dtype(self.config.precision),
            use_safetensors=True,
        )

        if self.model_spec.adapter_path is not None:
            peft_model_cls = _peft_model_class()
            model = peft_model_cls.from_pretrained(model, self.model_spec.adapter_path)

        model.eval()
        model = accelerator.prepare(model)
        self.loaded = LoadedQwenVl(
            model=model,
            processor=processor,
            device=str(accelerator.device),
        )
        return self.loaded

    def predict(self, request: InferenceRequest) -> InferenceResult:
        loaded = self.load()
        started_at = time.perf_counter()
        messages = _messages(request, self.config)
        prompt_text = loaded.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = _process_vision_inputs(messages)
        inputs = loaded.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(loaded.device)

        generated_ids = loaded.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[-1] :]
        prediction = loaded.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return InferenceResult(
            sample_id=request.sample_id,
            prediction=prediction,
            metadata={
                "device": loaded.device,
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                "model_id": self.model_spec.model_id,
                "precision": self.config.precision,
            },
        )


def torch_dtype_name(precision: Precision) -> str:
    if precision == "auto":
        return "auto"
    return precision


def _torch_dtype(precision: Precision):
    if precision == "auto":
        return "auto"

    import torch

    dtype_by_precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_by_precision[precision]


def _model_source(model_spec: ModelSpec):
    if model_spec.model_path is not None and model_spec.model_path.exists():
        return model_spec.model_path
    return model_spec.model_id


def _messages(
    request: InferenceRequest,
    config: InferenceConfig,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": request.system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": request.image,
                    "max_pixels": config.image_max_pixels,
                },
                {"type": "text", "text": request.prompt},
            ],
        },
    ]


def _build_accelerator():
    from accelerate import Accelerator

    return Accelerator()


def _transformers_classes():
    import transformers

    processor_cls = transformers.AutoProcessor
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = transformers.Qwen2VLForConditionalGeneration
    return processor_cls, model_cls


def _peft_model_class():
    from peft import PeftModel

    return PeftModel


def _process_vision_inputs(messages: list[dict[str, Any]]):
    from qwen_vl_utils import process_vision_info

    return process_vision_info(messages)
