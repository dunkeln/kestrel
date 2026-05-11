from __future__ import annotations

import hashlib
import json
from typing import Any


RECORD_KEY_VERSION = "sft-record-key-v1"


def sample_record_key(
    *,
    dataset: str,
    split: str,
    image_sha256: str | None,
    input_text: str | None,
    answer: str | None,
    answer_type: str | None,
    task_type: str | None,
    sample_id: str | None = None,
) -> str:
    payload = {
        "version": RECORD_KEY_VERSION,
        "dataset": dataset,
        "split": split,
        "image_sha256": image_sha256 or "",
        "input": input_text or "",
        "answer": answer or "",
        "answer_type": answer_type or "",
        "task_type": task_type or "",
        "sample_id_fallback": sample_id if not image_sha256 or not input_text else "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_key_from_record(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    existing = record.get("record_key") or metadata.get("record_key")
    if existing:
        return str(existing)

    return sample_record_key(
        dataset=str(metadata.get("dataset") or record.get("dataset") or ""),
        split=str(metadata.get("split") or record.get("split") or ""),
        image_sha256=_optional_str(metadata.get("image_sha256")),
        input_text=_optional_str(record.get("input") or record.get("question")),
        answer=_optional_str(metadata.get("answer") or record.get("answer")),
        answer_type=_optional_str(metadata.get("answer_type") or record.get("answer_type")),
        task_type=_optional_str(metadata.get("task_type") or record.get("task_type")),
        sample_id=_optional_str(metadata.get("sample_id") or record.get("sample_id")),
    )


def attach_record_key(record: dict[str, Any], record_key: str) -> dict[str, Any]:
    record["record_key"] = record_key
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        metadata["record_key"] = record_key
        metadata["record_key_version"] = RECORD_KEY_VERSION
    return record


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "RECORD_KEY_VERSION",
    "attach_record_key",
    "record_key_from_record",
    "sample_record_key",
]
