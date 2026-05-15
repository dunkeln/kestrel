from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from training.distillation.contrastive import (
    build_contrastive_record,
    should_emit_contrastive,
)
from training.distillation.prompts import build_stage_prompt
from training.distillation.providers import (
    CLAUDE_ADJUDICATION_MODEL,
    CLAUDE_TEACHER_MODEL,
    OPENAI_MODEL,
    ProviderBatchRequest,
    ProviderBatchResult,
    SynthesisResult,
    TeacherCallResult,
    TeacherName,
    adjudicate_plotqa_numeric,
    call_all_teachers,
    call_teacher,
    encode_image,
    run_anthropic_message_batch,
    run_openai_chat_batch,
    run_plotqa_numeric_adjudication_batches,
    run_synthesis_batches,
    synthesize_with_retries,
)
from training.distillation.sample_stream import (
    LoaderFactory,
    StreamConfig,
    iter_sample_batches,
)
from training.distillation.storage import (
    GenerationStats,
    PreparedSample,
    TrainingDataStore,
    gold_record,
    manifest_record,
    prepare_sample,
    sample_record_key,
    skipped_record,
)
from training.distillation.validation import (
    contest,
    dissenting_providers,
    retry_prompt,
    stage1_requires_compute,
    validate_reasoning,
    validation_error,
    vote_map,
)


logger = logging.getLogger(__name__)
ProcessBatchFn = Callable[..., Awaitable[list["SampleResult"]]]
OpenAIBatchFn = Callable[..., Awaitable[dict[str, ProviderBatchResult]]]
AnthropicBatchFn = Callable[..., Awaitable[dict[str, ProviderBatchResult]]]


@dataclass(frozen=True)
class PipelineConfig:
    stage: int = 1
    input_batch_size: int = 8
    max_concurrent_samples: int = 4
    max_api_attempts: int = 3
    max_teacher_repair_attempts: int = 2
    max_synthesis_validation_attempts: int = 3


@dataclass(frozen=True)
class SampleResult:
    sample: dict[str, Any]
    sft_record: dict[str, Any] | None
    agreed_output: str | None
    teacher_outputs: dict[str, str | None]
    teacher_errors: dict[str, str | None]
    synthesis_error: str | None = None
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def written(self) -> bool:
        return self.sft_record is not None


@dataclass(frozen=True)
class BatchItem:
    index: int
    sample: dict
    image_b64: str
    prompt: str
    requires_compute: bool


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
    log_event("distillation_start", **command, output_root=str(output_root))
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
                task_id = progress.add_task("distill", total=total)
                batches = iter_sample_batches(
                    stream,
                    **({"loader_factory": loader_factory} if loader_factory else {}),
                )
                for batch_index, batch in enumerate(batches, start=1):
                    await _process_stream_batch(
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
                        "distillation_batch_complete",
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
            "distillation_failed",
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
        log_event("distillation_complete", failed=failed, **stats_fields(stats))

    render_summary(console, stats, output_root / f"{dataset}_{split}.jsonl")
    return stats


async def process_batch(
    samples: list[dict],
    *,
    config: PipelineConfig | None = None,
) -> list[SampleResult]:
    cfg = config or PipelineConfig()
    semaphore = asyncio.Semaphore(cfg.max_concurrent_samples)

    async def guarded(sample: dict) -> SampleResult:
        async with semaphore:
            return await process_sample(sample, config=cfg)

    return list(await asyncio.gather(*(guarded(sample) for sample in samples)))


async def process_sample(
    sample: dict,
    *,
    config: PipelineConfig | None = None,
) -> SampleResult:
    cfg = config or PipelineConfig()
    try:
        image_b64 = await encode_image(sample["imgname"])
        prompt = build_stage_prompt(sample, stage=cfg.stage)
        requires_compute = cfg.stage == 1 and stage1_requires_compute(sample)
        teacher_results = await _teacher_round(
            image_b64=image_b64,
            prompt=prompt,
            stage=cfg.stage,
            requires_compute=requires_compute,
            config=cfg,
        )
        reasonings = _teacher_texts(teacher_results)
        agreed_output = contest(
            reasonings,
            stage=cfg.stage,
            requires_compute=requires_compute,
        )
        if agreed_output is None:
            adjudication = await _plotqa_numeric_adjudication(
                image_b64=image_b64,
                sample=sample,
                reasonings=reasonings,
                teacher_results=teacher_results,
                requires_compute=requires_compute,
                config=cfg,
            )
            if adjudication is None:
                return _skipped(sample, teacher_results, "no_teacher_majority")
            return adjudication

        repaired = await _repair_single_dissenter(
            image_b64=image_b64,
            prompt=prompt,
            teacher_results=teacher_results,
            agreed_output=agreed_output,
            stage=cfg.stage,
            requires_compute=requires_compute,
            config=cfg,
        )
        teacher_results.update(repaired)
        reasonings = _teacher_texts(teacher_results)
        agreed_output = contest(
            reasonings,
            stage=cfg.stage,
            requires_compute=requires_compute,
        )
        if agreed_output is None:
            return _skipped(sample, teacher_results, "no_teacher_majority_after_repair")

        if not validate_reasoning(
            reasonings.get("openai"),
            stage=cfg.stage,
            requires_compute=requires_compute,
        ):
            return _skipped(sample, teacher_results, "missing_openai_trace")

        synthesis = await synthesize_with_retries(
            image_b64=image_b64,
            claude_reasoning=reasonings.get("claude") or "",
            gpt_reasoning=reasonings.get("openai") or "",
            agreed_output=agreed_output,
            stage=cfg.stage,
            requires_compute=requires_compute,
            max_validation_attempts=cfg.max_synthesis_validation_attempts,
            sample=sample,
        )
        if synthesis.text is None:
            return _skipped(
                sample,
                teacher_results,
                "synthesis_failed",
                agreed_output=agreed_output,
                synthesis_error=synthesis.error,
            )

        return _successful_result(
            sample=sample,
            synthesized=synthesis.text,
            teacher_results=teacher_results,
            agreed_output=agreed_output,
            stage=cfg.stage,
            requires_compute=requires_compute,
            synthesis=synthesis,
        )
    except Exception as exc:
        logger.exception("sample failed before completion")
        return SampleResult(
            sample=sample,
            sft_record=None,
            agreed_output=None,
            teacher_outputs={},
            teacher_errors={},
            skip_reason="sample_exception",
            synthesis_error=f"{type(exc).__name__}: {exc}",
        )


async def process_batch_provider_batched(
    samples: list[dict],
    *,
    config: PipelineConfig | None = None,
    poll_interval_seconds: float = 60.0,
    anthropic_batch_fn: AnthropicBatchFn = run_anthropic_message_batch,
    openai_batch_fn: OpenAIBatchFn = run_openai_chat_batch,
) -> list[SampleResult]:
    cfg = config or PipelineConfig()
    items = await _prepare_provider_batch_items(samples, cfg)
    teacher_results = await _teacher_batches(
        items,
        poll_interval_seconds=poll_interval_seconds,
        anthropic_batch_fn=anthropic_batch_fn,
        openai_batch_fn=openai_batch_fn,
    )
    outputs: list[SampleResult | None] = [None] * len(samples)
    synthesis_inputs = []
    adjudication_inputs = []

    for item in items:
        teachers = teacher_results[item.index]
        reasonings = _teacher_texts(teachers)
        agreed_output = contest(
            reasonings,
            stage=cfg.stage,
            requires_compute=item.requires_compute,
        )
        if agreed_output is None:
            if _can_adjudicate_plotqa_numeric(item, teachers, cfg):
                adjudication_inputs.append((item, teachers))
                continue
            outputs[item.index] = _skipped(item.sample, teachers, "no_teacher_majority")
            continue
        synthesis_inputs.append((item, teachers, agreed_output))

    adjudicated = await run_plotqa_numeric_adjudication_batches(
        adjudication_inputs,
        max_attempts=cfg.max_synthesis_validation_attempts,
        poll_interval_seconds=poll_interval_seconds,
        anthropic_batch_fn=anthropic_batch_fn,
    )
    for item, teachers in adjudication_inputs:
        result = adjudicated[item.index]
        agreed_output = str(item.sample.get("answer") or "")
        if result.text is None:
            outputs[item.index] = _skipped(
                item.sample,
                teachers,
                "plotqa_numeric_adjudication_failed",
                agreed_output=agreed_output,
                synthesis_error=result.error,
            )
            continue
        metadata = _trace_metadata(
            teacher_results=teachers,
            agreed_output=agreed_output,
            stage=1,
            requires_compute=True,
            synthesis_attempts=result.attempts,
            synthesis_lint_errors=result.lint_errors,
        )
        metadata.update(
            {
                "provider_batch": True,
                "plotqa_numeric_adjudicated": True,
                "adjudicator_model": CLAUDE_ADJUDICATION_MODEL,
            }
        )
        outputs[item.index] = SampleResult(
            sample=item.sample,
            sft_record=_sft_record(item.sample, result.text, stage=1, metadata=metadata),
            agreed_output=agreed_output,
            teacher_outputs=_teacher_texts(teachers),
            teacher_errors=_teacher_errors(teachers),
            metadata=metadata,
        )

    synthesized = await run_synthesis_batches(
        synthesis_inputs,
        max_attempts=cfg.max_synthesis_validation_attempts,
        stage=cfg.stage,
        poll_interval_seconds=poll_interval_seconds,
        anthropic_batch_fn=anthropic_batch_fn,
    )
    for item, teachers, agreed_output in synthesis_inputs:
        result = synthesized[item.index]
        if result.text is None:
            outputs[item.index] = _skipped(
                item.sample,
                teachers,
                "synthesis_failed",
                agreed_output=agreed_output,
                synthesis_error=result.error,
            )
            continue
        metadata = _trace_metadata(
            teacher_results=teachers,
            agreed_output=agreed_output,
            stage=cfg.stage,
            requires_compute=item.requires_compute,
            synthesis_attempts=result.attempts,
            synthesis_lint_errors=result.lint_errors,
        )
        metadata["provider_batch"] = True
        outputs[item.index] = SampleResult(
            sample=item.sample,
            sft_record=_sft_record(
                item.sample,
                result.text,
                stage=cfg.stage,
                metadata=metadata,
            ),
            agreed_output=agreed_output,
            teacher_outputs=_teacher_texts(teachers),
            teacher_errors=_teacher_errors(teachers),
            metadata=metadata,
        )

    return [result for result in outputs if result is not None]


async def _process_stream_batch(
    *,
    batch: list,
    split: str,
    batch_size: int,
    contrastive_rate: float,
    shuffle_seed: int,
    store: TrainingDataStore,
    stats: GenerationStats,
    process_batch_fn: ProcessBatchFn,
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
        await _write_result(
            item=item,
            result=result,
            split=split,
            contrastive_rate=contrastive_rate,
            shuffle_seed=shuffle_seed,
            store=store,
            stats=stats,
        )


async def _write_result(
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


async def _teacher_round(
    *,
    image_b64: str,
    prompt: str,
    stage: int,
    requires_compute: bool,
    config: PipelineConfig,
) -> dict[TeacherName, TeacherCallResult]:
    results = await call_all_teachers(
        image_b64,
        prompt,
        max_api_attempts=config.max_api_attempts,
    )
    for _ in range(config.max_teacher_repair_attempts):
        invalid = {
            provider: result
            for provider, result in results.items()
            if not validate_reasoning(
                result.text,
                stage=stage,
                requires_compute=requires_compute,
            )
        }
        if not invalid:
            break
        repaired = await asyncio.gather(
            *(
                _repair_teacher(
                    provider=provider,
                    image_b64=image_b64,
                    base_prompt=prompt,
                    previous=result,
                    error=result.error
                    or validation_error(
                        result.text,
                        stage=stage,
                        requires_compute=requires_compute,
                    )
                    or "invalid trace",
                    stage=stage,
                    requires_compute=requires_compute,
                    config=config,
                )
                for provider, result in invalid.items()
            )
        )
        results.update({result.provider: result for result in repaired})
    return results


async def _repair_single_dissenter(
    *,
    image_b64: str,
    prompt: str,
    teacher_results: dict[TeacherName, TeacherCallResult],
    agreed_output: str,
    stage: int,
    requires_compute: bool,
    config: PipelineConfig,
) -> dict[TeacherName, TeacherCallResult]:
    dissenters = dissenting_providers(
        _teacher_texts(teacher_results),
        agreed_output=agreed_output,
        stage=stage,
        requires_compute=requires_compute,
    )
    if len(dissenters) != 1:
        return {}

    provider = dissenters[0]
    previous = teacher_results[provider]
    value_error = f"{provider} output did not match agreed value `{agreed_output}`"
    result = await _repair_teacher(
        provider=provider,
        image_b64=image_b64,
        base_prompt=prompt,
        previous=previous,
        error=value_error,
        stage=stage,
        requires_compute=requires_compute,
        config=config,
        agreed_output=agreed_output,
    )
    return {result.provider: result}


async def _repair_teacher(
    *,
    provider: str,
    image_b64: str,
    base_prompt: str,
    previous: TeacherCallResult,
    error: str,
    stage: int,
    requires_compute: bool,
    config: PipelineConfig,
    agreed_output: str | None = None,
) -> TeacherCallResult:
    prompt = retry_prompt(
        base_prompt,
        error=error,
        previous_response=previous.text,
        agreed_output=agreed_output,
    )
    result = await call_teacher(
        provider,  # type: ignore[arg-type]
        image_b64,
        prompt,
        max_api_attempts=config.max_api_attempts,
    )
    if not validate_reasoning(
        result.text,
        stage=stage,
        requires_compute=requires_compute,
    ):
        logger.info(
            "%s repair failed validation: %s",
            provider,
            validation_error(
                result.text,
                stage=stage,
                requires_compute=requires_compute,
            )
            or result.error,
        )
    return result


async def _plotqa_numeric_adjudication(
    *,
    image_b64: str,
    sample: dict,
    reasonings: dict[str, str | None],
    teacher_results: dict[TeacherName, TeacherCallResult],
    requires_compute: bool,
    config: PipelineConfig,
) -> SampleResult | None:
    if not _is_plotqa_numeric_stage1(sample, config):
        return None
    if not all(
        validate_reasoning(
            reasonings.get(provider),
            stage=1,
            requires_compute=requires_compute,
        )
        for provider in ("claude", "openai")
    ):
        return None

    synthesis = await adjudicate_plotqa_numeric(
        image_b64=image_b64,
        claude_reasoning=reasonings.get("claude") or "",
        gpt_reasoning=reasonings.get("openai") or "",
        sample=sample,
        max_validation_attempts=config.max_synthesis_validation_attempts,
    )
    if synthesis.text is None:
        return _skipped(
            sample,
            teacher_results,
            "plotqa_numeric_adjudication_failed",
            agreed_output=str(sample.get("answer") or ""),
            synthesis_error=synthesis.error,
        )
    return _successful_result(
        sample=sample,
        synthesized=synthesis.text,
        teacher_results=teacher_results,
        agreed_output=str(sample.get("answer") or ""),
        stage=1,
        requires_compute=True,
        synthesis=synthesis,
        extra_metadata={
            "plotqa_numeric_adjudicated": True,
            "adjudicator_model": CLAUDE_ADJUDICATION_MODEL,
        },
    )


async def _prepare_provider_batch_items(
    samples: list[dict],
    config: PipelineConfig,
) -> list[BatchItem]:
    async def prepare(index: int, sample: dict) -> BatchItem:
        return BatchItem(
            index=index,
            sample=sample,
            image_b64=await encode_image(sample["imgname"]),
            prompt=build_stage_prompt(sample, stage=config.stage),
            requires_compute=config.stage == 1 and stage1_requires_compute(sample),
        )

    return list(await asyncio.gather(*(prepare(index, sample) for index, sample in enumerate(samples))))


async def _teacher_batches(
    items: list[BatchItem],
    *,
    poll_interval_seconds: float,
    anthropic_batch_fn: AnthropicBatchFn,
    openai_batch_fn: OpenAIBatchFn,
) -> dict[int, dict]:
    claude_requests = [
        ProviderBatchRequest(str(item.index), item.image_b64, item.prompt, CLAUDE_TEACHER_MODEL)
        for item in items
    ]
    openai_requests = [
        ProviderBatchRequest(str(item.index), item.image_b64, item.prompt, OPENAI_MODEL)
        for item in items
    ]
    claude, openai = await asyncio.gather(
        anthropic_batch_fn(
            claude_requests,
            poll_interval_seconds=poll_interval_seconds,
            label="teacher_claude",
        ),
        openai_batch_fn(
            openai_requests,
            poll_interval_seconds=poll_interval_seconds,
            label="teacher_openai",
        ),
    )
    return {
        item.index: {
            "claude": _teacher_result("claude", item.prompt, claude[str(item.index)]),
            "openai": _teacher_result("openai", item.prompt, openai[str(item.index)]),
        }
        for item in items
    }


def _teacher_result(provider: str, prompt: str, result: ProviderBatchResult) -> TeacherCallResult:
    return TeacherCallResult(provider=provider, text=result.text, error=result.error, prompt=prompt, attempts=1)


def _can_adjudicate_plotqa_numeric(item: BatchItem, teachers: dict, config: PipelineConfig) -> bool:
    reasonings = _teacher_texts(teachers)
    return (
        _is_plotqa_numeric_stage1(item.sample, config)
        and all(
            validate_reasoning(
                reasonings.get(provider),
                stage=1,
                requires_compute=item.requires_compute,
            )
            for provider in ("claude", "openai")
        )
    )


def _is_plotqa_numeric_stage1(sample: dict, config: PipelineConfig) -> bool:
    return (
        config.stage == 1
        and sample.get("dataset") == "plotqa_qa"
        and sample.get("answer_type") == "numeric"
    )


def _successful_result(
    *,
    sample: dict,
    synthesized: str,
    teacher_results: dict[TeacherName, TeacherCallResult],
    agreed_output: str,
    stage: int,
    requires_compute: bool,
    synthesis: SynthesisResult,
    extra_metadata: dict[str, Any] | None = None,
) -> SampleResult:
    metadata = _trace_metadata(
        teacher_results=teacher_results,
        agreed_output=agreed_output,
        stage=stage,
        requires_compute=requires_compute,
        synthesis_attempts=synthesis.attempts,
        synthesis_lint_errors=synthesis.lint_errors,
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    return SampleResult(
        sample=sample,
        sft_record=_sft_record(sample, synthesized, stage=stage, metadata=metadata),
        agreed_output=agreed_output,
        teacher_outputs=_teacher_texts(teacher_results),
        teacher_errors=_teacher_errors(teacher_results),
        metadata=metadata,
    )


def _sft_record(
    sample: dict,
    synthesized: str,
    *,
    stage: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if stage == 1:
        return {
            "image": sample["imgname"],
            "input": sample["question"],
            "output": synthesized,
            "metadata": metadata,
        }
    return {
        "image": sample["imgname"],
        "input": f"Evaluate this claim: {sample['claim']}",
        "output": synthesized,
        "metadata": metadata,
    }


def _skipped(
    sample: dict,
    teacher_results: dict[TeacherName, TeacherCallResult],
    reason: str,
    *,
    agreed_output: str | None = None,
    synthesis_error: str | None = None,
) -> SampleResult:
    return SampleResult(
        sample=sample,
        sft_record=None,
        agreed_output=agreed_output,
        teacher_outputs=_teacher_texts(teacher_results),
        teacher_errors=_teacher_errors(teacher_results),
        synthesis_error=synthesis_error,
        skip_reason=reason,
    )


def _teacher_texts(
    teacher_results: dict[TeacherName, TeacherCallResult],
) -> dict[str, str | None]:
    return {provider: result.text for provider, result in teacher_results.items()}


def _teacher_errors(
    teacher_results: dict[TeacherName, TeacherCallResult],
) -> dict[str, str | None]:
    return {provider: result.error for provider, result in teacher_results.items()}


def _trace_metadata(
    *,
    teacher_results: dict[TeacherName, TeacherCallResult],
    agreed_output: str,
    stage: int,
    requires_compute: bool,
    synthesis_attempts: int,
    synthesis_lint_errors: tuple[str, ...],
) -> dict[str, Any]:
    reasonings = _teacher_texts(teacher_results)
    return {
        "agreed_output": agreed_output,
        "teacher_votes": vote_map(
            reasonings,
            stage=stage,
            requires_compute=requires_compute,
        ),
        "teacher_valid": {
            provider: validate_reasoning(
                result.text,
                stage=stage,
                requires_compute=requires_compute,
            )
            for provider, result in teacher_results.items()
        },
        "reasoning_schema": "stage1_compute" if requires_compute else f"stage{stage}",
        "teacher_errors": _teacher_errors(teacher_results),
        "teacher_attempts": {
            provider: result.attempts for provider, result in teacher_results.items()
        },
        "synthesis_attempts": synthesis_attempts,
        "synthesis_lint_errors": list(synthesis_lint_errors),
    }


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


def generation_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]distill[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def render_summary(console: Console, stats: GenerationStats, gold_path: Path) -> None:
    table = Table(title="distillation summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total", str(stats.total))
    table.add_row("written", str(stats.written))
    table.add_row("skipped", str(stats.skipped))
    table.add_row("deduped", str(stats.deduped))
    table.add_row("agreement", f"{stats.agreement_rate:.1%}")
    table.add_row("contrastive", str(stats.contrastive))
    table.add_row("records", str(gold_path))
    console.print(table)


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def stats_fields(stats: GenerationStats) -> dict[str, Any]:
    return {
        "total": stats.total,
        "written": stats.written,
        "skipped": stats.skipped,
        "deduped": stats.deduped,
        "agreement_rate": stats.agreement_rate,
        "contrastive": stats.contrastive,
    }
