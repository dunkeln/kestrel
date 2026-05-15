from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aiofiles
from PIL import Image

from training.datasets.contracts import EvalSample
from training.datasets.images import resolve_sample_image
from training.datasets.loaders import system_prompt


SCHEMA_VERSION = "sft-train-data-v1"
RECORD_KEY_VERSION = "sft-record-key-v1"


@dataclass(frozen=True)
class StoredImage:
    path: Path
    ref: str
    sha256: str


@dataclass(frozen=True)
class PreparedSample:
    eval_sample: EvalSample
    pipeline_sample: dict[str, Any]
    image: StoredImage


@dataclass(frozen=True)
class TrainDataPaths:
    gold: Path
    contrastive: Path
    skipped: Path
    manifest: Path
    images: Path


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


class JsonlWriter:
    def __init__(self, path: Path, *, append: bool = False):
        self.path = path
        self.append = append
        self._handle: Any = None

    async def __aenter__(self) -> "JsonlWriter":
        await asyncio.to_thread(self.path.parent.mkdir, parents=True, exist_ok=True)
        self._handle = await aiofiles.open(self.path, "a" if self.append else "w")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._handle is not None:
            await self._handle.close()

    async def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter is not open")
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        await self._handle.write(payload + "\n")
        await self._handle.flush()


class TrainingDataStore:
    def __init__(self, output_root: Path, dataset: str, split: str, *, resume: bool = False):
        self.output_root = output_root
        self.paths = train_data_paths(output_root, dataset, split)
        self.resume = resume
        self._gold = JsonlWriter(self.paths.gold, append=resume)
        self._contrastive = JsonlWriter(self.paths.contrastive, append=resume)
        self._skipped = JsonlWriter(self.paths.skipped, append=resume)
        self._record_keys: set[str] = set()

    async def __aenter__(self) -> "TrainingDataStore":
        await asyncio.to_thread(self.paths.images.mkdir, parents=True, exist_ok=True)
        if self.resume:
            self._record_keys = await asyncio.to_thread(backfill_record_keys, self.paths)
        await self._gold.__aenter__()
        await self._contrastive.__aenter__()
        await self._skipped.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._gold.__aexit__(*exc)
        await self._contrastive.__aexit__(*exc)
        await self._skipped.__aexit__(*exc)

    async def write_gold(self, record: dict[str, Any]) -> None:
        await self._gold.write(record)
        self.mark_record(record)

    async def write_contrastive(self, record: dict[str, Any]) -> None:
        await self._contrastive.write(record)

    async def write_skipped(self, record: dict[str, Any]) -> None:
        await self._skipped.write(record)

    async def store_image(self, sample: EvalSample) -> StoredImage:
        return await store_sample_image(sample, self.paths.images, self.output_root)

    async def write_manifest(self, manifest: dict[str, Any]) -> None:
        payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
        await asyncio.to_thread(self.paths.manifest.parent.mkdir, parents=True, exist_ok=True)
        async with aiofiles.open(self.paths.manifest, "w") as handle:
            await handle.write(payload + "\n")

    def has_record_key(self, record_key: str) -> bool:
        return record_key in self._record_keys

    def mark_record(self, record: dict[str, Any]) -> None:
        record_key = record.get("record_key")
        if record_key:
            self._record_keys.add(str(record_key))


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


def gold_record(result: Any, prepared: PreparedSample, *, split: str) -> dict[str, Any]:
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
    result: Any | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record_key = sample_record_key(
        dataset=sample.dataset,
        split=split,
        image_sha256=image.sha256 if image else None,
        input_text=sample.question,
        answer=sample.answer,
        answer_type=sample.answer_type,
        task_type=sample.task_type,
        sample_id=sample.id,
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
        answer_type=_optional_str(
            metadata.get("answer_type") or record.get("answer_type")
        ),
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


def train_data_paths(output_root: Path, dataset: str, split: str) -> TrainDataPaths:
    return TrainDataPaths(
        gold=output_root / f"{dataset}_{split}.jsonl",
        contrastive=output_root / f"{dataset}_contrastive_{split}.jsonl",
        skipped=output_root / f"{dataset}_skipped_{split}.jsonl",
        manifest=output_root / f"{dataset}_{split}_manifest.json",
        images=output_root / "images",
    )


def backfill_record_keys(paths: TrainDataPaths) -> set[str]:
    existing_keys: set[str] = set()
    _backfill_jsonl(paths.gold, existing_keys=existing_keys, index_records=True)
    _backfill_jsonl(paths.contrastive, existing_keys=existing_keys, index_records=False)
    _backfill_jsonl(paths.skipped, existing_keys=set(), index_records=False)
    return existing_keys


async def store_sample_image(
    sample: EvalSample,
    image_dir: Path,
    output_root: Path,
) -> StoredImage:
    return await asyncio.to_thread(
        _store_sample_image_sync,
        sample,
        image_dir,
        output_root,
    )


def _backfill_jsonl(
    path: Path,
    *,
    existing_keys: set[str],
    index_records: bool,
) -> None:
    if not path.exists():
        return

    rows: list[str] = []
    changed = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_key = record_key_from_record(record)
            if record.get("record_key") != record_key:
                attach_record_key(record, record_key)
                changed = True
            if index_records:
                existing_keys.add(record_key)
            rows.append(json.dumps(record, sort_keys=True, ensure_ascii=False))

    if changed:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        tmp.replace(path)


def _store_sample_image_sync(
    sample: EvalSample,
    image_dir: Path,
    output_root: Path,
) -> StoredImage:
    resolved = resolve_sample_image(sample)
    image_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(resolved, (str, Path)):
        source = Path(resolved)
        payload = source.read_bytes()
        suffix = source.suffix or ".png"
    elif isinstance(resolved, Image.Image):
        payload = _encode_image(resolved)
        suffix = ".png"
    else:
        raise TypeError(f"Unsupported image type for sample `{sample.id}`.")

    digest = hashlib.sha256(payload).hexdigest()
    target = image_dir / f"{digest}{suffix.lower()}"
    if not target.exists():
        if isinstance(resolved, (str, Path)):
            shutil.copyfile(Path(resolved), target)
        else:
            target.write_bytes(payload)

    output_root_resolved = output_root.resolve()
    target_resolved = target.resolve()
    if not target_resolved.is_relative_to(output_root_resolved):
        raise ValueError(
            f"Image directory `{image_dir}` must be under output root `{output_root}`."
        )

    return StoredImage(
        path=target,
        ref=str(target_resolved.relative_to(output_root_resolved)),
        sha256=digest,
    )


def _encode_image(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
