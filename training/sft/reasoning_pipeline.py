from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiofiles

from training.sft.reasoning_synthesis import (
    SynthesisResult,
    adjudicate_plotqa_numeric,
    synthesize_with_retries,
)
from training.sft.teacher_ensemble import (
    CLAUDE_ADJUDICATION_MODEL,
    TeacherCallResult,
    TeacherName,
    build_stage_prompt,
    call_all_teachers,
    call_teacher,
    contest,
    dissenting_providers,
    encode_image,
    retry_prompt,
    stage1_requires_compute,
    validate_reasoning,
    validation_error,
    vote_map,
)


logger = logging.getLogger(__name__)


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


async def build_sft_dataset(
    samples: list[dict],
    output_file: str,
    *,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Process samples in input batches and stream valid SFT rows to JSONL."""
    cfg = config or PipelineConfig()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "written": 0,
        "skipped": 0,
        "agreement": 0,
        "total": len(samples),
        "skip_reasons": {},
    }

    async with aiofiles.open(output_path, "w") as handle:
        for batch in _chunks(samples, cfg.input_batch_size):
            results = await process_batch(batch, config=cfg)
            for result in results:
                if result.agreed_output is not None:
                    stats["agreement"] += 1
                if result.sft_record is None:
                    stats["skipped"] += 1
                    reason = result.skip_reason or "unknown"
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                    continue
                await handle.write(json.dumps(result.sft_record, sort_keys=True) + "\n")
                stats["written"] += 1

    stats["agreement_rate"] = stats["agreement"] / stats["total"] if stats["total"] else 0.0
    stats["write_rate"] = stats["written"] / stats["total"] if stats["total"] else 0.0
    logger.info(
        "constitutional ensemble complete written=%s skipped=%s agreement_rate=%s",
        stats["written"],
        stats["skipped"],
        stats["agreement_rate"],
    )
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
            max_validation_attempts=cfg.max_synthesis_validation_attempts,
            sample=sample,
            requires_compute=requires_compute,
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


def _chunks(items: list[dict], size: int):
    if size <= 0:
        raise ValueError("input_batch_size must be greater than 0")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def sample_result_to_dict(result: SampleResult) -> dict[str, Any]:
    return asdict(result)


__all__ = [
    "PipelineConfig",
    "SampleResult",
    "build_sft_dataset",
    "process_batch",
    "process_sample",
    "sample_result_to_dict",
]
