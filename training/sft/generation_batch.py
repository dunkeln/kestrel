from __future__ import annotations

from typing import Awaitable, Callable

from training.sft.contrastive import build_contrastive_record, should_emit_contrastive
from training.sft.keys import sample_record_key
from training.sft.reasoning_pipeline import PipelineConfig, SampleResult, process_batch
from training.sft.records import (
    GenerationStats,
    PreparedSample,
    gold_record,
    prepare_sample,
    skipped_record,
)
from training.sft.storage import TrainingDataStore


ProcessBatchFn = Callable[..., Awaitable[list[SampleResult]]]


async def process_stream_batch(
    *,
    batch: list,
    split: str,
    batch_size: int,
    contrastive_rate: float,
    shuffle_seed: int,
    store: TrainingDataStore,
    stats: GenerationStats,
    process_batch_fn: ProcessBatchFn = process_batch,
) -> None:
    prepared = []
    for sample in batch:
        stats.total += 1
        try:
            image = await store.store_image(sample)
            record_key = sample_record_key(
                dataset=sample.dataset,
                split=split,
                image_sha256=image.sha256,
                input_text=sample.question,
                answer=sample.answer,
                answer_type=sample.answer_type,
                task_type=sample.task_type,
                sample_id=sample.id,
            )
            if store.has_record_key(record_key):
                stats.deduped += 1
                continue
            prepared.append(prepare_sample(sample, image))
        except Exception as exc:
            stats.skipped += 1
            await store.write_skipped(
                skipped_record(
                    sample,
                    split=split,
                    reason="image_prepare_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if not prepared:
        return

    results = await process_batch_fn(
        [item.pipeline_sample for item in prepared],
        config=PipelineConfig(input_batch_size=batch_size),
    )
    for item, result in zip(prepared, results, strict=True):
        await write_result(
            item=item,
            result=result,
            split=split,
            contrastive_rate=contrastive_rate,
            shuffle_seed=shuffle_seed,
            store=store,
            stats=stats,
        )


async def write_result(
    *,
    item: PreparedSample,
    result: SampleResult,
    split: str,
    contrastive_rate: float,
    shuffle_seed: int,
    store: TrainingDataStore,
    stats: GenerationStats,
) -> None:
    if result.agreed_output is not None:
        stats.agreement += 1
    if not result.written:
        stats.skipped += 1
        await store.write_skipped(
            skipped_record(
                item.eval_sample,
                split=split,
                image=item.image,
                result=result,
                reason=result.skip_reason or "unknown",
            )
        )
        return

    record = gold_record(result, item, split=split)
    await store.write_gold(record)
    stats.written += 1
    if should_emit_contrastive(record, rate=contrastive_rate, seed=shuffle_seed):
        contrastive = build_contrastive_record(record, seed=shuffle_seed)
        if contrastive is not None:
            await store.write_contrastive(contrastive)
            stats.contrastive += 1


__all__ = ["ProcessBatchFn", "process_stream_batch", "write_result"]
