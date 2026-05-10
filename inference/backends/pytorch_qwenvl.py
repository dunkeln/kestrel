from dataclasses import dataclass
from typing import Any

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

    def load(self) -> LoadedQwenVl:
        raise NotImplementedError("Qwen-VL model loading is the next implementation step.")

    def predict(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError("Qwen-VL generation is the next implementation step.")


def torch_dtype_name(precision: Precision) -> str:
    if precision == "auto":
        return "auto"
    return precision
