import hashlib
import json
from enum import Enum
from typing import Any

from peft import LoraConfig

QWEN2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
QWEN2_5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN2_VL_7B = "Qwen/Qwen2-VL-7B-Instruct"


def _vision_targets(depth: int) -> list[str]:
    return [
        f"model.visual.blocks.{layer}.attn.{module}"
        for layer in range(depth)
        for module in ("qkv", "proj")
    ]


def _language_targets(depth: int) -> list[str]:
    return [
        f"model.language_model.layers.{layer}.self_attn.{module}"
        for layer in range(depth)
        for module in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]


def _mixed_lora(vision_targets: list[str], language_targets: list[str], *, vision_rank: int, language_rank: int) -> LoraConfig:
    return LoraConfig(
        target_modules=vision_targets + language_targets,
        r=language_rank,
        lora_alpha=language_rank * 2,
        lora_dropout=0.05,
        bias="none",
        rank_pattern={target: vision_rank for target in vision_targets},
        alpha_pattern={target: vision_rank for target in vision_targets},
    )


def _config(model_id: str, *, vision_depth: int, language_depth: int, vision_rank: int, language_rank: int) -> dict[str, Any]:
    vision_targets = _vision_targets(vision_depth)
    language_targets = _language_targets(language_depth)
    vision = LoraConfig(
        target_modules=vision_targets,
        r=vision_rank,
        lora_alpha=vision_rank,
        lora_dropout=0.05,
        bias="none",
    )
    language = LoraConfig(
        target_modules=language_targets,
        r=language_rank,
        lora_alpha=language_rank * 2,
        lora_dropout=0.05,
        bias="none",
    )
    return {
        "model_id": model_id,
        "targets": {
            "vision": vision_targets,
            "language": language_targets,
        },
        "vision": vision,
        "language": language,
        "train": _mixed_lora(
            vision_targets,
            language_targets,
            vision_rank=vision_rank,
            language_rank=language_rank,
        ),
    }


QWENVL_LORA_CONFIGS = {
    "tiny": _config(
        QWEN2_VL_2B,
        vision_depth=32,
        language_depth=28,
        vision_rank=16,
        language_rank=32,
    ),
    "small": _config(
        QWEN2_5_VL_3B,
        vision_depth=32,
        language_depth=36,
        vision_rank=16,
        language_rank=32,
    ),
    "aight": _config(
        QWEN2_VL_7B,
        vision_depth=32,
        language_depth=28,
        vision_rank=8,
        language_rank=16,
    ),
}


def lora_config_payload(size: str) -> dict[str, Any]:
    config = QWENVL_LORA_CONFIGS[size]
    return {
        "model_id": config["model_id"],
        "vision": _stable(config["vision"].to_dict()),
        "language": _stable(config["language"].to_dict()),
        "train": _stable(config["train"].to_dict()),
    }


def lora_train_config(size: str) -> LoraConfig:
    return QWENVL_LORA_CONFIGS[size]["train"]


def lora_config_hash(size: str) -> str:
    payload = json.dumps(lora_config_payload(size), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in sorted(value.items())}
    if isinstance(value, set):
        return sorted(_stable(item) for item in value)
    if isinstance(value, list | tuple):
        return [_stable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value
