from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aiofiles
from PIL import Image

from training.datasets.contracts import EvalSample
from training.datasets.images import resolve_sample_image
from training.sft.keys import attach_record_key, record_key_from_record


@dataclass(frozen=True)
class StoredImage:
    path: Path
    ref: str
    sha256: str


@dataclass(frozen=True)
class TrainDataPaths:
    gold: Path
    contrastive: Path
    skipped: Path
    manifest: Path
    images: Path


class JsonlWriter:
    def __init__(self, path: Path, *, append: bool = False):
        self.path = path
        self.append = append
        self._handle: Any = None

    async def __aenter__(self) -> "JsonlWriter":
        await asyncio.to_thread(self.path.parent.mkdir, parents=True, exist_ok=True)
        mode = "a" if self.append else "w"
        self._handle = await aiofiles.open(self.path, mode)
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


__all__ = [
    "StoredImage",
    "TrainDataPaths",
    "TrainingDataStore",
    "store_sample_image",
    "train_data_paths",
]
