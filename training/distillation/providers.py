from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from training.distillation.prompts import (
    CONSTITUTIONAL_CONSTRAINT,
    PLOTQA_NUMERIC_ADJUDICATION_PROMPT,
    build_synthesis_prompt,
)
from training.distillation.validation import (
    lint_error_message,
    lint_synthesis,
    retry_prompt,
    validation_error,
)


logger = logging.getLogger(__name__)

TeacherName = Literal["claude", "openai"]
CLAUDE_TEACHER_MODEL = "claude-haiku-4-5"
CLAUDE_SYNTHESIS_MODEL = "claude-sonnet-4-5"
CLAUDE_ADJUDICATION_MODEL = CLAUDE_TEACHER_MODEL
OPENAI_MODEL = "gpt-4o-mini"
SYNTHESIS_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class TeacherCallResult:
    provider: TeacherName
    text: str | None
    error: str | None
    prompt: str
    attempts: int

    @property
    def ok(self) -> bool:
        return self.text is not None and self.error is None


@dataclass(frozen=True)
class SynthesisResult:
    text: str | None
    error: str | None
    attempts: int
    prompt: str
    lint_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.text is not None and self.error is None


@dataclass(frozen=True)
class ProviderBatchRequest:
    custom_id: str
    image_b64: str
    prompt: str
    model: str


@dataclass(frozen=True)
class ProviderBatchResult:
    custom_id: str
    text: str | None
    error: str | None


async def encode_image(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


async def call_all_teachers(
    image_b64: str,
    prompt: str,
    *,
    max_api_attempts: int = 3,
) -> dict[TeacherName, TeacherCallResult]:
    results = await asyncio.gather(
        call_teacher("claude", image_b64, prompt, max_api_attempts=max_api_attempts),
        call_teacher("openai", image_b64, prompt, max_api_attempts=max_api_attempts),
    )
    return {result.provider: result for result in results}


async def call_teacher(
    provider: TeacherName,
    image_b64: str,
    prompt: str,
    *,
    max_api_attempts: int = 3,
) -> TeacherCallResult:
    async def call() -> str:
        match provider:
            case "claude":
                return await _call_claude(
                    image_b64,
                    prompt,
                    model=CLAUDE_TEACHER_MODEL,
                )
            case "openai":
                return await _call_openai(image_b64, prompt)

    text, error, attempts = await _with_backoff(
        label=provider,
        call=call,
        max_attempts=max_api_attempts,
    )
    return TeacherCallResult(provider, text, error, prompt, attempts)


async def synthesize_with_retries(
    *,
    image_b64: str,
    claude_reasoning: str,
    gpt_reasoning: str,
    agreed_output: str,
    stage: int,
    requires_compute: bool,
    max_validation_attempts: int = 3,
    sample: dict | None = None,
) -> SynthesisResult:
    base_prompt = build_synthesis_prompt(
        claude_reasoning=claude_reasoning,
        gpt_reasoning=gpt_reasoning,
        agreed_output=agreed_output,
        stage=stage,
        requires_compute=requires_compute,
    )
    return await _call_claude_until_valid(
        image_b64=image_b64,
        base_prompt=base_prompt,
        agreed_output=agreed_output,
        stage=stage,
        requires_compute=requires_compute,
        sample=sample,
        max_validation_attempts=max_validation_attempts,
        call_label="synthesis",
        call_model=CLAUDE_SYNTHESIS_MODEL,
    )


async def adjudicate_plotqa_numeric(
    *,
    image_b64: str,
    claude_reasoning: str,
    gpt_reasoning: str,
    sample: dict,
    max_validation_attempts: int = 3,
) -> SynthesisResult:
    gold_answer = str(sample.get("answer", ""))
    base_prompt = PLOTQA_NUMERIC_ADJUDICATION_PROMPT.format(
        claude_reasoning=claude_reasoning,
        gpt_reasoning=gpt_reasoning,
        gold_answer=gold_answer,
        question=str(sample.get("question", "")),
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
    )
    return await _call_claude_until_valid(
        image_b64=image_b64,
        base_prompt=base_prompt,
        agreed_output=gold_answer,
        stage=1,
        requires_compute=True,
        sample=sample,
        max_validation_attempts=max_validation_attempts,
        call_label="adjudication",
        call_model=CLAUDE_ADJUDICATION_MODEL,
    )


async def run_anthropic_message_batch(
    requests: list[ProviderBatchRequest],
    *,
    poll_interval_seconds: float,
    label: str,
) -> dict[str, ProviderBatchResult]:
    if not requests:
        return {}
    client = AsyncAnthropic()
    batch = await client.messages.batches.create(
        requests=[_anthropic_request(request) for request in requests]
    )
    _log_event("provider_batch_submitted", provider="anthropic", label=label, id=batch.id)

    while batch.processing_status != "ended":
        await asyncio.sleep(poll_interval_seconds)
        batch = await client.messages.batches.retrieve(batch.id)
        _log_event(
            "provider_batch_poll",
            provider="anthropic",
            label=label,
            id=batch.id,
            status=batch.processing_status,
        )

    result_stream = await client.messages.batches.results(batch.id)
    results: dict[str, ProviderBatchResult] = {}
    async for line in result_stream:
        results[line.custom_id] = _anthropic_batch_result(line)
    return _with_missing_results(requests, results)


async def run_openai_chat_batch(
    requests: list[ProviderBatchRequest],
    *,
    poll_interval_seconds: float,
    label: str,
) -> dict[str, ProviderBatchResult]:
    if not requests:
        return {}
    client = AsyncOpenAI()
    uploaded = await client.files.create(
        file=("distillation_openai_batch.jsonl", _openai_jsonl(requests)),
        purpose="batch",
    )
    batch = await client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"label": label},
    )
    _log_event("provider_batch_submitted", provider="openai", label=label, id=batch.id)

    while batch.status not in {"completed", "failed", "expired", "cancelled"}:
        await asyncio.sleep(poll_interval_seconds)
        batch = await client.batches.retrieve(batch.id)
        _log_event(
            "provider_batch_poll",
            provider="openai",
            label=label,
            id=batch.id,
            status=batch.status,
        )

    results: dict[str, ProviderBatchResult] = {}
    if batch.output_file_id:
        output = await client.files.content(batch.output_file_id)
        results.update(_openai_results((await output.aread()).decode("utf-8")))
    if batch.error_file_id:
        errors = await client.files.content(batch.error_file_id)
        results.update(_openai_results((await errors.aread()).decode("utf-8")))
    return _with_missing_results(requests, results)


async def run_synthesis_batches(
    inputs: list[tuple],
    *,
    max_attempts: int,
    stage: int,
    poll_interval_seconds: float,
    anthropic_batch_fn: Callable[..., Awaitable[dict[str, ProviderBatchResult]]],
) -> dict[int, SynthesisResult]:
    pending = {
        item.index: _synthesis_request(item, teachers, agreed, stage)
        for item, teachers, agreed in inputs
    }
    outputs: dict[int, SynthesisResult] = {}
    for attempt in range(1, max_attempts + 1):
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
            output = _validate_batch_synthesis(item, result, stage, attempt)
            if output.text is not None or attempt == max_attempts:
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
                CLAUDE_SYNTHESIS_MODEL,
            )
        pending = next_pending
    return outputs


async def run_plotqa_numeric_adjudication_batches(
    inputs: list[tuple],
    *,
    max_attempts: int,
    poll_interval_seconds: float,
    anthropic_batch_fn: Callable[..., Awaitable[dict[str, ProviderBatchResult]]],
) -> dict[int, SynthesisResult]:
    pending = {
        item.index: _plotqa_numeric_adjudication_request(item, teachers)
        for item, teachers in inputs
    }
    outputs: dict[int, SynthesisResult] = {}
    for attempt in range(1, max_attempts + 1):
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
            output = _validate_batch_adjudication(item, result, attempt)
            if output.text is not None or attempt == max_attempts:
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
                CLAUDE_ADJUDICATION_MODEL,
            )
        pending = next_pending
    return outputs


async def _call_claude_until_valid(
    *,
    image_b64: str,
    base_prompt: str,
    agreed_output: str,
    stage: int,
    requires_compute: bool,
    sample: dict | None,
    max_validation_attempts: int,
    call_label: str,
    call_model: str,
) -> SynthesisResult:
    prompt = base_prompt
    last_error = f"{call_label} did not run"
    last_lint_errors: tuple[str, ...] = ()
    for attempt in range(1, max_validation_attempts + 1):
        try:
            text = await asyncio.wait_for(
                _call_claude(image_b64, prompt, model=call_model),
                timeout=SYNTHESIS_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            text = None
            last_lint_errors = ()
            last_error = f"{call_label} timed out after {SYNTHESIS_TIMEOUT_SECONDS:.0f}s"
            logger.warning("%s timed out on attempt %s", call_label, attempt)
            prompt = retry_prompt(
                base_prompt,
                error=last_error,
                previous_response=None,
                agreed_output=agreed_output,
            )
            continue

        structure_error = validation_error(
            text,
            stage=stage,
            requires_compute=requires_compute,
        )
        last_lint_errors = lint_synthesis(text, stage=stage, sample=sample)
        last_error = structure_error or lint_error_message(last_lint_errors) or ""
        if text is not None and not structure_error and not last_lint_errors:
            return SynthesisResult(text, None, attempt, prompt, ())

        logger.info("%s validation failed on attempt %s: %s", call_label, attempt, last_error)
        prompt = retry_prompt(
            base_prompt,
            error=last_error,
            previous_response=text,
            agreed_output=agreed_output,
        )

    return SynthesisResult(
        text=None,
        error=last_error or f"invalid {call_label} response",
        attempts=max_validation_attempts,
        prompt=prompt,
        lint_errors=last_lint_errors,
    )


async def _call_claude(image_b64: str, prompt: str, *, model: str) -> str:
    client = AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _image_media_type(image_b64),
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return _anthropic_text(response)


async def _call_openai(image_b64: str, prompt: str) -> str:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(image_b64)},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


async def _with_backoff(
    *,
    label: str,
    call: Callable[[], Awaitable[str]],
    max_attempts: int,
) -> tuple[str | None, str | None, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            return await call(), None, attempt
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "%s call failed on attempt %s/%s: %s",
                label,
                attempt,
                max_attempts,
                error,
            )
            if attempt == max_attempts:
                return None, error, attempt
            await asyncio.sleep(2 ** (attempt - 1) + random.uniform(0, 0.5))
    return None, "unknown retry failure", max_attempts


def _synthesis_request(item, teachers: dict, agreed_output: str, stage: int) -> ProviderBatchRequest:
    prompt = build_synthesis_prompt(
        claude_reasoning=teachers["claude"].text or "",
        gpt_reasoning=teachers["openai"].text or "",
        agreed_output=agreed_output,
        stage=stage,
        requires_compute=item.requires_compute,
    )
    return ProviderBatchRequest(str(item.index), item.image_b64, prompt, CLAUDE_SYNTHESIS_MODEL)


def _validate_batch_synthesis(
    item,
    result: ProviderBatchResult,
    stage: int,
    attempt: int,
) -> SynthesisResult:
    structure_error = validation_error(
        result.text,
        stage=stage,
        requires_compute=item.requires_compute,
    )
    lint_errors = lint_synthesis(result.text, stage=stage, sample=item.sample)
    error = result.error or structure_error or lint_error_message(lint_errors)
    if result.text is not None and error is None:
        return SynthesisResult(result.text, None, attempt, "", ())
    return SynthesisResult(None, error or "invalid synthesis response", attempt, "", lint_errors)


def _plotqa_numeric_adjudication_request(item, teachers: dict) -> ProviderBatchRequest:
    prompt = PLOTQA_NUMERIC_ADJUDICATION_PROMPT.format(
        claude_reasoning=teachers["claude"].text or "",
        gpt_reasoning=teachers["openai"].text or "",
        gold_answer=str(item.sample.get("answer") or ""),
        question=str(item.sample.get("question") or ""),
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
    )
    return ProviderBatchRequest(str(item.index), item.image_b64, prompt, CLAUDE_ADJUDICATION_MODEL)


def _validate_batch_adjudication(
    item,
    result: ProviderBatchResult,
    attempt: int,
) -> SynthesisResult:
    structure_error = validation_error(result.text, stage=1, requires_compute=True)
    lint_errors = lint_synthesis(result.text, stage=1, sample=item.sample)
    error = result.error or structure_error or lint_error_message(lint_errors)
    if result.text is not None and error is None:
        return SynthesisResult(result.text, None, attempt, "", ())
    return SynthesisResult(None, error or "invalid adjudication response", attempt, "", lint_errors)


def _anthropic_request(request: ProviderBatchRequest) -> dict[str, Any]:
    return {
        "custom_id": request.custom_id,
        "params": {
            "model": request.model,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _image_media_type(request.image_b64),
                                "data": request.image_b64,
                            },
                        },
                        {"type": "text", "text": request.prompt},
                    ],
                }
            ],
        },
    }


def _anthropic_batch_result(line: Any) -> ProviderBatchResult:
    result = line.result
    if result.type == "succeeded":
        return ProviderBatchResult(line.custom_id, _anthropic_text(result.message), None)
    error = getattr(result, "error", None)
    return ProviderBatchResult(
        line.custom_id,
        None,
        getattr(error, "message", None) or result.type,
    )


def _openai_jsonl(requests: list[ProviderBatchRequest]) -> bytes:
    lines = []
    for request in requests:
        lines.append(
            json.dumps(
                {
                    "custom_id": request.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": request.model,
                        "temperature": 0,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": request.prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": _data_url(request.image_b64)},
                                    },
                                ],
                            }
                        ],
                    },
                },
                sort_keys=True,
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _openai_results(payload: str) -> dict[str, ProviderBatchResult]:
    results = {}
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row["custom_id"]
        error = row.get("error")
        response = row.get("response")
        if error:
            results[custom_id] = ProviderBatchResult(custom_id, None, str(error))
            continue
        body = (response or {}).get("body", {})
        text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
        results[custom_id] = ProviderBatchResult(custom_id, text or "", None)
    return results


def _with_missing_results(
    requests: list[ProviderBatchRequest],
    results: dict[str, ProviderBatchResult],
) -> dict[str, ProviderBatchResult]:
    for request in requests:
        results.setdefault(
            request.custom_id,
            ProviderBatchResult(request.custom_id, None, "missing provider batch result"),
        )
    return results


def _data_url(image_b64: str) -> str:
    return f"data:{_image_media_type(image_b64)};base64,{image_b64}"


def _image_media_type(image_b64: str) -> str:
    try:
        header = base64.b64decode(image_b64[:64], validate=False)[:12]
    except Exception:
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _anthropic_text(response) -> str:
    return "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )


def _log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))
