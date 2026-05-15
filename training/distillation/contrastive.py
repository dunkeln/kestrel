from __future__ import annotations

import copy
import hashlib
import math
import re
from typing import Any

from training.distillation.storage import attach_record_key


ANSWER_RE = re.compile(r"(<answer>)(.*?)(</answer>)", flags=re.DOTALL | re.IGNORECASE)


def should_emit_contrastive(
    record: dict[str, Any],
    *,
    rate: float,
    seed: int | None,
) -> bool:
    if not math.isfinite(rate):
        return False
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    key = _record_key(record, seed)
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return value / 0xFFFFFFFFFFFFFFFF < rate


def build_contrastive_record(
    record: dict[str, Any],
    *,
    seed: int | None,
) -> dict[str, Any] | None:
    metadata = record.get("metadata", {})
    answer_type = str(metadata.get("answer_type") or "")
    gold_answer = str(metadata.get("answer") or "")
    perturbation = _perturb_answer(
        gold_answer,
        answer_type=answer_type,
        metadata=metadata,
        seed=seed,
    )
    if perturbation is None:
        return None

    perturbed_answer, perturbation_type = perturbation
    output = _replace_answer(record.get("output", ""), perturbed_answer)
    if output is None:
        return None

    contrastive = copy.deepcopy(record)
    record_key = str(record.get("record_key") or metadata.get("record_key") or "")
    contrastive["output"] = output
    contrastive["metadata"] = {
        **metadata,
        "example_type": "contrastive",
        "gold_output": record.get("output"),
        "gold_answer": gold_answer,
        "perturbed_answer": perturbed_answer,
        "perturbation_type": perturbation_type,
    }
    return attach_record_key(contrastive, record_key) if record_key else contrastive


def _perturb_answer(
    answer: str,
    *,
    answer_type: str,
    metadata: dict[str, Any],
    seed: int | None,
) -> tuple[str, str] | None:
    normalized = answer.strip()
    match answer_type:
        case "yes_no":
            lower = normalized.lower()
            if lower in {"yes", "true"}:
                return ("no", "yes_no_flip")
            if lower in {"no", "false"}:
                return ("yes", "yes_no_flip")
            return None
        case "numeric":
            return _perturb_numeric(normalized)
        case "multiple_choice":
            return _alternate_candidate(normalized, metadata, seed, "choice_swap")
        case "text":
            return _alternate_candidate(normalized, metadata, seed, "text_swap")
        case _:
            return None


def _perturb_numeric(answer: str) -> tuple[str, str] | None:
    try:
        value = float(answer.replace(",", "").rstrip("%"))
    except ValueError:
        return None

    delta = max(abs(value) * 0.05, 1.0 if value.is_integer() else 0.1)
    perturbed = value + delta
    if value.is_integer():
        return (str(int(round(perturbed))), "numeric_shift")
    return (f"{perturbed:.4g}", "numeric_shift")


def _alternate_candidate(
    answer: str,
    metadata: dict[str, Any],
    seed: int | None,
    perturbation_type: str,
) -> tuple[str, str] | None:
    candidates = _candidate_values(metadata)
    if not candidates and re.fullmatch(r"[A-Da-d]", answer):
        candidates = ["A", "B", "C", "D"]
    alternatives = [value for value in candidates if value.lower() != answer.lower()]
    if not alternatives:
        return None
    index = _stable_index(answer, seed, len(alternatives))
    return (alternatives[index], perturbation_type)


def _candidate_values(metadata: dict[str, Any]) -> list[str]:
    source = metadata.get("source_metadata")
    if not isinstance(source, dict):
        return []
    raw = source.get("options") or source.get("choices") or source.get("candidates")
    if isinstance(raw, dict):
        return sorted(str(key) for key in raw.keys())
    if isinstance(raw, list | tuple):
        return [str(value) for value in raw]
    return []


def _replace_answer(output: str, answer: str) -> str | None:
    if not isinstance(output, str):
        return None
    if not ANSWER_RE.search(output):
        return None
    return ANSWER_RE.sub(
        lambda match: f"{match.group(1)}{answer}{match.group(3)}",
        output,
        count=1,
    )


def _record_key(record: dict[str, Any], seed: int | None) -> str:
    metadata = record.get("metadata", {})
    key = record.get("record_key") or metadata.get("record_key")
    if key:
        return f"{seed or 0}:{key}"
    return f"{seed or 0}:{metadata.get('dataset')}:{metadata.get('sample_id')}"


def _stable_index(value: str, seed: int | None, size: int) -> int:
    key = f"{seed or 0}:{value}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % size
