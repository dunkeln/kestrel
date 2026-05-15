import hashlib
import json
from enum import Enum
from typing import Any

from peft import LoraConfig

QWEN2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
QWEN2_VL_2B_VISION_TARGETS = [
    f"model.visual.blocks.{layer}.attn.{module}"
    for layer in range(32)
    for module in ("qkv", "proj")
]
QWEN2_VL_2B_LANGUAGE_TARGETS = [
    f"model.language_model.layers.{layer}.self_attn.{module}"
    for layer in range(28)
    for module in ("q_proj", "k_proj", "v_proj", "o_proj")
]

QWEN2_VL_2B_VISION_LORA = LoraConfig(
    target_modules=QWEN2_VL_2B_VISION_TARGETS,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

QWEN2_VL_2B_LANGUAGE_LORA = LoraConfig(
    target_modules=QWEN2_VL_2B_LANGUAGE_TARGETS,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
)

QWEN2_VL_2B_MIXED_LORA = LoraConfig(
    target_modules=QWEN2_VL_2B_VISION_TARGETS + QWEN2_VL_2B_LANGUAGE_TARGETS,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    rank_pattern={target: 16 for target in QWEN2_VL_2B_VISION_TARGETS},
    alpha_pattern={target: 16 for target in QWEN2_VL_2B_VISION_TARGETS},
)

QWENVL_LORA_CONFIGS = {
    "tiny": {
        "model_id": QWEN2_VL_2B,
        "targets": {
            "vision": QWEN2_VL_2B_VISION_TARGETS,
            "language": QWEN2_VL_2B_LANGUAGE_TARGETS,
        },
        "vision": QWEN2_VL_2B_VISION_LORA,
        "language": QWEN2_VL_2B_LANGUAGE_LORA,
        "train": QWEN2_VL_2B_MIXED_LORA,
    }
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
