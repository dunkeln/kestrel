from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from training.datasets.contracts import EvalSample

Precision = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_path: Path | None = None
    adapter_path: Path | None = None


@dataclass(frozen=True)
class InferenceRequest:
    sample_id: str
    image: Any
    system_prompt: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_eval_sample(
        cls,
        sample: EvalSample,
        system_prompt: str,
    ) -> "InferenceRequest":
        return cls(
            sample_id=sample.id,
            image=sample.image,
            system_prompt=system_prompt,
            prompt=sample.question,
            metadata={
                "dataset": sample.dataset,
                "answer_type": sample.answer_type,
                "chart_type": sample.chart_type,
                "task_type": sample.task_type,
            },
        )


@dataclass(frozen=True)
class InferenceConfig:
    precision: Precision = "auto"
    max_new_tokens: int = 128


@dataclass(frozen=True)
class InferenceResult:
    sample_id: str
    prediction: str
    metadata: dict[str, Any] = field(default_factory=dict)
