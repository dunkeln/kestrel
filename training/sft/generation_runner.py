from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from training.sft.generation_batch import ProcessBatchFn, process_stream_batch
from training.sft.provider_batch_pipeline import process_batch_provider_batched
from training.sft.progress import (
    generation_progress,
    log_event,
    render_summary,
    stats_fields,
)
from training.sft.reasoning_pipeline import process_batch
from training.sft.records import GenerationStats, manifest_record
from training.sft.sample_stream import LoaderFactory, StreamConfig, iter_sample_batches
from training.sft.storage import TrainingDataStore


async def run_generation(
    *,
    dataset: str,
    split: str,
    samples: int | None,
    all_samples: bool,
    batch_size: int,
    shuffle_seed: int,
    shuffle_buffer_size: int,
    contrastive_rate: float,
    output_root: Path,
    console: Console,
    provider_mode: str = "sync",
    provider_poll_interval: float = 60.0,
    resume: bool = False,
    process_batch_fn: ProcessBatchFn | None = None,
    loader_factory: LoaderFactory | None = None,
) -> GenerationStats:
    stream = StreamConfig(
        dataset=dataset,
        split=split,
        samples=samples,
        all_samples=all_samples,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    command = _command(
        dataset,
        split,
        samples,
        all_samples,
        batch_size,
        shuffle_seed,
        shuffle_buffer_size,
        contrastive_rate,
        provider_mode,
        provider_poll_interval,
        resume,
    )
    stats = GenerationStats()
    total = samples if samples is not None else None
    log_event("sft_generation_start", **command, output_root=str(output_root))
    paths = None
    failed = False
    selected_process_batch = process_batch_fn or _process_batch_fn(
        provider_mode,
        provider_poll_interval,
    )

    try:
        async with TrainingDataStore(output_root, dataset, split, resume=resume) as store:
            paths = store.paths
            with generation_progress(console) as progress:
                task_id = progress.add_task("sft", total=total)
                batches = iter_sample_batches(
                    stream,
                    **({"loader_factory": loader_factory} if loader_factory else {}),
                )
                for batch_index, batch in enumerate(batches, start=1):
                    await process_stream_batch(
                        batch=batch,
                        split=split,
                        batch_size=batch_size,
                        contrastive_rate=contrastive_rate,
                        shuffle_seed=shuffle_seed,
                        store=store,
                        stats=stats,
                        process_batch_fn=selected_process_batch,
                    )
                    progress.update(task_id, advance=len(batch))
                    log_event(
                        "sft_batch_complete",
                        batch=batch_index,
                        **stats_fields(stats),
                    )

            await store.write_manifest(
                manifest_record(
                    dataset=dataset,
                    split=split,
                    command=command,
                    stats=stats,
                    paths=store.paths,
                    output_root=output_root,
                )
            )
    except Exception as exc:
        failed = True
        log_event(
            "sft_generation_failed",
            error=f"{type(exc).__name__}: {exc}",
            **stats_fields(stats),
        )
        if paths is not None:
            await _write_failure_manifest(
                dataset=dataset,
                split=split,
                command=command,
                stats=stats,
                paths=paths,
                output_root=output_root,
            )
        render_summary(console, stats, output_root / f"{dataset}_{split}.jsonl")
        raise
    finally:
        log_event("sft_generation_complete", failed=failed, **stats_fields(stats))

    render_summary(console, stats, output_root / f"{dataset}_{split}.jsonl")
    return stats


async def _write_failure_manifest(
    *,
    dataset: str,
    split: str,
    command: dict[str, Any],
    stats: GenerationStats,
    paths: Any,
    output_root: Path,
) -> None:
    store = TrainingDataStore(output_root, dataset, split)
    await store.write_manifest(
        manifest_record(
            dataset=dataset,
            split=split,
            command=command,
            stats=stats,
            paths=paths,
            output_root=output_root,
        )
    )


def _command(*values: Any) -> dict[str, Any]:
    keys = (
        "dataset",
        "split",
        "samples",
        "all_samples",
        "batch_size",
        "shuffle_seed",
        "shuffle_buffer_size",
        "contrastive_rate",
        "provider_mode",
        "provider_poll_interval",
        "resume",
    )
    return dict(zip(keys, values, strict=True))


def _process_batch_fn(provider_mode: str, poll_interval: float) -> ProcessBatchFn:
    if provider_mode == "sync":
        return process_batch
    if provider_mode == "batch":
        async def batched(samples: list[dict], *, config):
            return await process_batch_provider_batched(
                samples,
                config=config,
                poll_interval_seconds=poll_interval,
            )

        return batched
    raise ValueError(f"Unsupported provider mode: {provider_mode}")


__all__ = ["run_generation", "render_summary"]
