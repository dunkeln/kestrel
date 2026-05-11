import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.contracts import InferenceConfig, ModelSpec, Precision

MODEL_IDS = {
    "qwen_vl": {
        "tiny": "Qwen/Qwen2-VL-2B-Instruct",
        "small": "Qwen/Qwen2.5-VL-3B-Instruct",
        "aight": "Qwen/Qwen2-VL-7B-Instruct",
    }
}


@dataclass(frozen=True)
class InferenceSettings:
    model: ModelSpec
    config: InferenceConfig


def load_inference_settings(config_path: Path = Path("config.toml")) -> InferenceSettings:
    raw_config = _read_toml(config_path)
    artifacts = raw_config.get("artifacts", {})
    model = raw_config.get("model", {})
    inference = raw_config.get("inference", {})

    family = str(model.get("default_family", "qwen_vl"))
    size = str(model.get("default_size", "tiny"))
    precision = _precision(str(inference.get("precision", "auto")))
    adapter_path = _optional_path(inference.get("adapter_path"))
    pretrained_root = Path(str(artifacts.get("pretrained", "artifacts/pretrained")))

    return InferenceSettings(
        model=ModelSpec(
            model_id=_model_id(family, size),
            model_path=pretrained_root / size,
            adapter_path=adapter_path,
        ),
        config=InferenceConfig(
            precision=precision,
            max_new_tokens=int(inference.get("max_new_tokens", 128)),
            image_max_pixels=int(inference.get("image_max_pixels", 1_003_520)),
        ),
    )


def _read_toml(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _model_id(family: str, size: str) -> str:
    try:
        return MODEL_IDS[family][size]
    except KeyError as error:
        raise ValueError(f"Unknown model family/size: {family}/{size}") from error


def _precision(value: str) -> Precision:
    if value in {"auto", "float32", "float16", "bfloat16"}:
        return value
    raise ValueError(f"Unknown inference precision: {value}")


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)
