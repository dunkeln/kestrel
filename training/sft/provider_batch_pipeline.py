from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from training.sft.provider_batch_contracts import (
    ProviderBatchRequest,
    ProviderBatchResult,
)
from training.sft.provider_batches import (
    run_anthropic_message_batch,
    run_openai_chat_batch,
)
from training.sft.provider_batch_synthesis import (
    AnthropicBatchFn,
    run_plotqa_numeric_adjudication_batches,
    run_synthesis_batches,
)
from training.sft.reasoning_pipeline import (
    PipelineConfig,
    SampleResult,
    _is_plotqa_numeric_stage1,
    _skipped,
    _sft_record,
    _teacher_errors,
    _teacher_texts,
    _trace_metadata,
)
from training.sft.teacher_ensemble import (
    CLAUDE_ADJUDICATION_MODEL,
    CLAUDE_TEACHER_MODEL,
    OPENAI_MODEL,
    TeacherCallResult,
    build_stage_prompt,
    contest,
    encode_image,
    stage1_requires_compute,
    validate_reasoning,
)


OpenAIBatchFn = Callable[..., Awaitable[dict[str, ProviderBatchResult]]]


@dataclass(frozen=True)
class _BatchItem:
    index: int
    sample: dict
    image_b64: str
    prompt: str
    requires_compute: bool


async def process_batch_provider_batched(
    samples: list[dict],
    *,
    config: PipelineConfig | None = None,
    poll_interval_seconds: float = 60.0,
    anthropic_batch_fn: AnthropicBatchFn = run_anthropic_message_batch,
    openai_batch_fn: OpenAIBatchFn = run_openai_chat_batch,
) -> list[SampleResult]:
    cfg = config or PipelineConfig()
    items = await _prepare_items(samples, cfg)
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
        config=cfg,
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
        config=cfg,
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
        outputs[item.index] = SampleResult(
            sample=item.sample,
            sft_record=_sft_record(
                item.sample,
                result.text,
                stage=cfg.stage,
                metadata={**metadata, "provider_batch": True},
            ),
            agreed_output=agreed_output,
            teacher_outputs=_teacher_texts(teachers),
            teacher_errors=_teacher_errors(teachers),
            metadata={**metadata, "provider_batch": True},
        )

    return [result for result in outputs if result is not None]


async def _prepare_items(samples: list[dict], config: PipelineConfig) -> list[_BatchItem]:
    async def prepare(index: int, sample: dict) -> _BatchItem:
        return _BatchItem(
            index=index,
            sample=sample,
            image_b64=await encode_image(sample["imgname"]),
            prompt=build_stage_prompt(sample, stage=config.stage),
            requires_compute=config.stage == 1 and stage1_requires_compute(sample),
        )

    return list(await asyncio.gather(*(prepare(index, sample) for index, sample in enumerate(samples))))


async def _teacher_batches(
    items: list[_BatchItem],
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
        anthropic_batch_fn(claude_requests, poll_interval_seconds=poll_interval_seconds, label="teacher_claude"),
        openai_batch_fn(openai_requests, poll_interval_seconds=poll_interval_seconds, label="teacher_openai"),
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


def _can_adjudicate_plotqa_numeric(item: _BatchItem, teachers: dict, config: PipelineConfig) -> bool:
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


__all__ = ["process_batch_provider_batched"]
