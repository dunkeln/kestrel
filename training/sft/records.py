from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.datasets.contracts import EvalSample
from training.datasets.loaders import system_prompt
from training.sft.keys import attach_record_key, sample_record_key
from training.sft.reasoning_pipeline import SampleResult
from training.sft.storage import StoredImage, TrainDataPaths


SCHEMA_VERSION = "sft-train-data-v1"


@dataclass(frozen=True)
class PreparedSample:
    eval_sample: EvalSample
    pipeline_sample: dict[str, Any]
    image: StoredImage


@dataclass
class GenerationStats:
    total: int = 0
    written: int = 0
    skipped: int = 0
    agreement: int = 0
    contrastive: int = 0
    deduped: int = 0

    @property
    def agreement_rate(self) -> float:
        return self.agreement / self.total if self.total else 0.0

    @property
    def write_rate(self) -> float:
        return self.written / self.total if self.total else 0.0

    @property
    def skip_rate(self) -> float:
        return self.skipped / self.total if self.total else 0.0


def prepare_sample(sample: EvalSample, image: StoredImage) -> PreparedSample:
    metadata = {
        **sample.metadata,
        "dataset": sample.dataset,
        "sample_id": sample.id,
        "image_ref": image.ref,
        "image_sha256": image.sha256,
    }
    return PreparedSample(
        eval_sample=sample,
        image=image,
        pipeline_sample={
            "imgname": str(image.path),
            "question": sample.question,
            "answer": sample.answer,
            "answer_type": sample.answer_type,
            "task_type": sample.task_type,
            "chart_type": sample.chart_type,
            "dataset": sample.dataset,
            "sample_id": sample.id,
            "metadata": metadata,
            "system_prompt": system_prompt(sample),
        },
    )


def gold_record(
    result: SampleResult,
    prepared: PreparedSample,
    *,
    split: str,
) -> dict[str, Any]:
    if result.sft_record is None:
        raise ValueError("cannot build gold record for skipped sample")

    sample = prepared.eval_sample
    record_key = sample_record_key(
        dataset=sample.dataset,
        split=split,
        image_sha256=prepared.image.sha256,
        input_text=sample.question,
        answer=sample.answer,
        answer_type=sample.answer_type,
        task_type=sample.task_type,
        sample_id=sample.id,
    )
    record = dict(result.sft_record)
    metadata = {
        **record.get("metadata", {}),
        "schema_version": SCHEMA_VERSION,
        "example_type": "gold",
        "dataset": sample.dataset,
        "split": split,
        "sample_id": sample.id,
        "answer": sample.answer,
        "answer_type": sample.answer_type,
        "chart_type": sample.chart_type,
        "task_type": sample.task_type,
        "image_ref": prepared.image.ref,
        "image_sha256": prepared.image.sha256,
        "source_metadata": sample.metadata,
        "record_key": record_key,
    }
    record.update(
        {
            "image": prepared.image.ref,
            "input": record["input"],
            "output": record["output"],
            "metadata": metadata,
        }
    )
    return attach_record_key(record, record_key)


def skipped_record(
    sample: EvalSample,
    *,
    split: str,
    reason: str,
    image: StoredImage | None = None,
    result: SampleResult | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record_key = (
        sample_record_key(
            dataset=sample.dataset,
            split=split,
            image_sha256=image.sha256,
            input_text=sample.question,
            answer=sample.answer,
            answer_type=sample.answer_type,
            task_type=sample.task_type,
            sample_id=sample.id,
        )
        if image
        else sample_record_key(
            dataset=sample.dataset,
            split=split,
            image_sha256=None,
            input_text=sample.question,
            answer=sample.answer,
            answer_type=sample.answer_type,
            task_type=sample.task_type,
            sample_id=sample.id,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "example_type": "skipped",
        "record_key": record_key,
        "dataset": sample.dataset,
        "split": split,
        "sample_id": sample.id,
        "answer": sample.answer,
        "answer_type": sample.answer_type,
        "task_type": sample.task_type,
        "image_ref": image.ref if image else None,
        "skip_reason": reason,
        "error": error or (result.synthesis_error if result else None),
        "agreed_output": result.agreed_output if result else None,
        "teacher_errors": result.teacher_errors if result else {},
    }


def manifest_record(
    *,
    dataset: str,
    split: str,
    command: dict[str, Any],
    stats: GenerationStats,
    paths: TrainDataPaths,
    output_root: Path,
) -> dict[str, Any]:
    payload = asdict(stats)
    payload.update(
        {
            "agreement_rate": stats.agreement_rate,
            "write_rate": stats.write_rate,
            "skip_rate": stats.skip_rate,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "command": command,
        "stats": payload,
        "paths": {
            "gold": str(paths.gold.relative_to(output_root)),
            "contrastive": str(paths.contrastive.relative_to(output_root)),
            "skipped": str(paths.skipped.relative_to(output_root)),
            "images": str(paths.images.relative_to(output_root)),
        },
    }


__all__ = [
    "GenerationStats",
    "PreparedSample",
    "gold_record",
    "manifest_record",
    "prepare_sample",
    "skipped_record",
]
