from __future__ import annotations

import asyncio
import json
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from training.sft.provider_batch_contracts import (
    ProviderBatchRequest,
    ProviderBatchResult,
)
from training.sft.progress import log_event
from training.sft.teacher_ensemble import (
    _anthropic_text,
    _data_url,
    _image_media_type,
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
    log_event("provider_batch_submitted", provider="anthropic", label=label, id=batch.id)

    while batch.processing_status != "ended":
        await asyncio.sleep(poll_interval_seconds)
        batch = await client.messages.batches.retrieve(batch.id)
        log_event(
            "provider_batch_poll",
            provider="anthropic",
            label=label,
            id=batch.id,
            status=batch.processing_status,
        )

    result_stream = await client.messages.batches.results(batch.id)
    results: dict[str, ProviderBatchResult] = {}
    async for line in result_stream:
        results[line.custom_id] = _anthropic_result(line)
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
    payload = _openai_jsonl(requests)
    uploaded = await client.files.create(
        file=("sft_openai_batch.jsonl", payload),
        purpose="batch",
    )
    batch = await client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"label": label},
    )
    log_event("provider_batch_submitted", provider="openai", label=label, id=batch.id)

    while batch.status not in {"completed", "failed", "expired", "cancelled"}:
        await asyncio.sleep(poll_interval_seconds)
        batch = await client.batches.retrieve(batch.id)
        log_event(
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


def _anthropic_result(line: Any) -> ProviderBatchResult:
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


__all__ = [
    "run_anthropic_message_batch",
    "run_openai_chat_batch",
]
