from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from training.sft.provider_batch_contracts import (
    ProviderBatchRequest,
    ProviderBatchResult,
)
from training.sft.reasoning_pipeline import PipelineConfig
from training.sft.reasoning_synthesis import (
    PLOTQA_NUMERIC_ADJUDICATION_PROMPT,
    build_synthesis_prompt,
    lint_synthesis,
)
from training.sft.teacher_ensemble import (
    CLAUDE_ADJUDICATION_MODEL,
    CONSTITUTIONAL_CONSTRAINT,
    CLAUDE_SYNTHESIS_MODEL,
    retry_prompt,
    validation_error,
)


AnthropicBatchFn = Callable[..., Awaitable[dict[str, ProviderBatchResult]]]


@dataclass(frozen=True)
class SynthesisBatchOutput:
    text: str | None
    error: str | None
    attempts: int
    lint_errors: tuple[str, ...] = ()


async def run_synthesis_batches(
    inputs: list[tuple],
    *,
    config: PipelineConfig,
    poll_interval_seconds: float,
    anthropic_batch_fn: AnthropicBatchFn,
) -> dict[int, SynthesisBatchOutput]:
    pending = {
        item.index: synthesis_request(item, teachers, agreed, config)
        for item, teachers, agreed in inputs
    }
    outputs: dict[int, SynthesisBatchOutput] = {}
    for attempt in range(1, config.max_synthesis_validation_attempts + 1):
        if not pending:
            break
        results = await anthropic_batch_fn(
            list(pending.values()),
            poll_interval_seconds=poll_interval_seconds,
            label=f"synthesis_attempt_{attempt}",
        )
        next_pending = {}
        for item, _teachers, agreed in inputs:
            if item.index not in pending:
                continue
            result = results[str(item.index)]
            output = validate_synthesis(item, result, config, attempt)
            if output.text is not None or attempt == config.max_synthesis_validation_attempts:
                outputs[item.index] = output
                continue
            retry = retry_prompt(
                pending[item.index].prompt,
                error=output.error or "invalid synthesis",
                previous_response=result.text,
                agreed_output=agreed,
            )
            next_pending[item.index] = ProviderBatchRequest(
                str(item.index),
                item.image_b64,
                retry,
                CLAUDE_ADJUDICATION_MODEL,
            )
        pending = next_pending
    return outputs


async def run_plotqa_numeric_adjudication_batches(
    inputs: list[tuple],
    *,
    config: PipelineConfig,
    poll_interval_seconds: float,
    anthropic_batch_fn: AnthropicBatchFn,
) -> dict[int, SynthesisBatchOutput]:
    pending = {
        item.index: plotqa_numeric_adjudication_request(item, teachers)
        for item, teachers in inputs
    }
    outputs: dict[int, SynthesisBatchOutput] = {}
    for attempt in range(1, config.max_synthesis_validation_attempts + 1):
        if not pending:
            break
        results = await anthropic_batch_fn(
            list(pending.values()),
            poll_interval_seconds=poll_interval_seconds,
            label=f"plotqa_numeric_adjudication_attempt_{attempt}",
        )
        next_pending = {}
        for item, _teachers in inputs:
            if item.index not in pending:
                continue
            result = results[str(item.index)]
            output = validate_plotqa_numeric_adjudication(item, result, attempt)
            if output.text is not None or attempt == config.max_synthesis_validation_attempts:
                outputs[item.index] = output
                continue
            retry = retry_prompt(
                pending[item.index].prompt,
                error=output.error or "invalid adjudication",
                previous_response=result.text,
                agreed_output=str(item.sample.get("answer") or ""),
            )
            next_pending[item.index] = ProviderBatchRequest(
                str(item.index),
                item.image_b64,
                retry,
                CLAUDE_SYNTHESIS_MODEL,
            )
        pending = next_pending
    return outputs


def synthesis_request(
    item,
    teachers: dict,
    agreed_output: str,
    config: PipelineConfig,
) -> ProviderBatchRequest:
    prompt = build_synthesis_prompt(
        claude_reasoning=teachers["claude"].text or "",
        gpt_reasoning=teachers["openai"].text or "",
        agreed_output=agreed_output,
        stage=config.stage,
        requires_compute=item.requires_compute,
    )
    return ProviderBatchRequest(str(item.index), item.image_b64, prompt, CLAUDE_ADJUDICATION_MODEL)


def validate_synthesis(
    item,
    result: ProviderBatchResult,
    config: PipelineConfig,
    attempt: int,
) -> SynthesisBatchOutput:
    structure_error = validation_error(
        result.text,
        stage=config.stage,
        requires_compute=item.requires_compute,
    )
    lint_errors = lint_synthesis(result.text, stage=config.stage, sample=item.sample)
    error = result.error or structure_error or (" ".join(lint_errors) if lint_errors else None)
    if result.text is not None and error is None:
        return SynthesisBatchOutput(result.text, None, attempt, ())
    return SynthesisBatchOutput(None, error or "invalid synthesis response", attempt, lint_errors)


def plotqa_numeric_adjudication_request(item, teachers: dict) -> ProviderBatchRequest:
    prompt = PLOTQA_NUMERIC_ADJUDICATION_PROMPT.format(
        claude_reasoning=teachers["claude"].text or "",
        gpt_reasoning=teachers["openai"].text or "",
        gold_answer=str(item.sample.get("answer") or ""),
        question=str(item.sample.get("question") or ""),
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
    )
    return ProviderBatchRequest(str(item.index), item.image_b64, prompt, CLAUDE_SYNTHESIS_MODEL)


def validate_plotqa_numeric_adjudication(
    item,
    result: ProviderBatchResult,
    attempt: int,
) -> SynthesisBatchOutput:
    structure_error = validation_error(result.text, stage=1, requires_compute=True)
    lint_errors = lint_synthesis(result.text, stage=1, sample=item.sample)
    error = result.error or structure_error or (" ".join(lint_errors) if lint_errors else None)
    if result.text is not None and error is None:
        return SynthesisBatchOutput(result.text, None, attempt, ())
    return SynthesisBatchOutput(None, error or "invalid adjudication response", attempt, lint_errors)


__all__ = [
    "AnthropicBatchFn",
    "SynthesisBatchOutput",
    "run_plotqa_numeric_adjudication_batches",
    "run_synthesis_batches",
]
