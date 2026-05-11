from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from training.datasets.contracts import EvalSample
from training.datasets.loaders import system_prompt


logger = logging.getLogger(__name__)

TeacherName = Literal["claude", "openai"]

CLAUDE_TEACHER_MODEL = "claude-haiku-4-5"
CLAUDE_SYNTHESIS_MODEL = "claude-sonnet-4-5"
CLAUDE_ADJUDICATION_MODEL = CLAUDE_TEACHER_MODEL
CLAUDE_MODEL = CLAUDE_TEACHER_MODEL
OPENAI_MODEL = "gpt-4o-mini"

ROLE_ASSIGNMENTS = {
    "claude": "constitutional teacher — structure, consistency, conservatism",
    "openai": "numerical analyst — precise value extraction, axis reading",
}

CONSTITUTIONAL_CONSTRAINT = """
STRICT RULES:
- Only reference what is physically visible in the chart
- Never use external knowledge about the data topic
- If a value is ambiguous, state the uncertainty explicitly
- Complete every XML tag — partial responses are invalid
- Never reverse-engineer reasoning from the answer
"""

STAGE1_REASONING_PROMPT = """You are analyzing a chart to answer a question.

{constitutional_constraint}

Question: {question}
Known correct answer: {answer}

Generate grounded reasoning following this exact structure:

<perceive>Chart type, axis labels, units, legend items visible in the image</perceive>
<extract>Task-relevant chart structure: only the visible labels, series, x/category values, y/numeric values, or visual relation needed to answer</extract>
<answer>{answer}</answer>

Complete all tags. Be concise but specific. Do not serialize the full chart unless the question explicitly asks for structure."""

STAGE1_COMPUTE_REASONING_PROMPT = """You are analyzing a chart to answer a question.

{constitutional_constraint}

Question: {question}
Known correct answer: {answer}

Generate grounded reasoning following this exact structure:

<perceive>Chart type, axis labels, units, legend items visible in the image</perceive>
<extract>Task-relevant chart structure: only the visible labels, series, x/category values, and y/numeric values needed for the calculation</extract>
<compute>Arithmetic or comparison performed only from the extracted chart values</compute>
<answer>{answer}</answer>

Complete all tags. Be concise but specific. Do not serialize the full chart unless the question explicitly asks for structure."""

STAGE2_REASONING_PROMPT = """You are a chart faithfulness judge.

{constitutional_constraint}

Claim: {claim}
Known correct verdict: {verdict}

Evaluate the claim using this exact structure:

<perceive>Chart type, axis labels, units, legend items visible in the image</perceive>
<extract>Task-relevant chart structure for the claim: only visible values, labels, trends, or visual relations needed to judge it</extract>
<compare>How extracted values support or contradict each component of the claim</compare>
<verdict>{verdict}</verdict>
<confidence>0.0-1.0</confidence>
<rubric>
  numerical: PASS/FAIL
  trend: PASS/FAIL
  label: PASS/FAIL
  scope: PASS/FAIL
</rubric>

Complete all tags. Never skip a claim component in compare."""


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


async def encode_image(image_path: str) -> str:
    """Read image file and return base64 encoded string."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def build_stage_prompt(sample: dict | EvalSample, *, stage: int) -> str:
    task_prompt = _task_prompt(sample)
    if stage == 1:
        template = (
            STAGE1_COMPUTE_REASONING_PROMPT
            if stage1_requires_compute(sample)
            else STAGE1_REASONING_PROMPT
        )
        prompt = template.format(
            constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
            question=_sample_value(sample, "question"),
            answer=_sample_value(sample, "answer"),
        )
        return _with_task_prompt(prompt, task_prompt)

    prompt = STAGE2_REASONING_PROMPT.format(
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
        claim=_sample_value(sample, "claim"),
        verdict=_sample_value(sample, "verdict"),
    )
    return _with_task_prompt(prompt, task_prompt)


async def get_claude_reasoning(image_b64: str, prompt: str) -> str | None:
    """Call Claude with image and prompt. Return text response."""
    return (await call_teacher("claude", image_b64, prompt)).text


async def get_claude_synthesis(
    image_b64: str,
    prompt: str,
    *,
    max_api_attempts: int = 3,
) -> str | None:
    """Call Claude Sonnet for privileged synthesis."""
    async def call() -> str:
        return await _call_claude(
            image_b64,
            prompt,
            model=CLAUDE_SYNTHESIS_MODEL,
        )

    text, error, _attempts = await _with_backoff(
        label="claude_synthesis",
        call=call,
        max_attempts=max_api_attempts,
    )
    if error:
        logger.warning("claude synthesis failed: %s", error)
    return text


async def get_claude_adjudication(
    image_b64: str,
    prompt: str,
    *,
    max_api_attempts: int = 3,
) -> str | None:
    """Call Claude Haiku for narrow adjudication tasks."""
    async def call() -> str:
        return await _call_claude(
            image_b64,
            prompt,
            model=CLAUDE_ADJUDICATION_MODEL,
        )

    text, error, _attempts = await _with_backoff(
        label="claude_adjudication",
        call=call,
        max_attempts=max_api_attempts,
    )
    if error:
        logger.warning("claude adjudication failed: %s", error)
    return text


async def get_openai_reasoning(image_b64: str, prompt: str) -> str | None:
    """Call GPT with image and prompt. Return text response."""
    return (await call_teacher("openai", image_b64, prompt)).text


async def get_all_reasonings(image_b64: str, prompt: str) -> dict[str, str | None]:
    """Call both teacher models in parallel via asyncio.gather.
    Return {"claude": str, "openai": str}"""
    results = await call_all_teachers(image_b64, prompt)
    return {provider: result.text for provider, result in results.items()}


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
    return TeacherCallResult(
        provider=provider,
        text=text,
        error=error,
        prompt=prompt,
        attempts=attempts,
    )


def validate_reasoning(
    reasoning: str | None,
    stage: int = 1,
    *,
    requires_compute: bool = False,
) -> bool:
    """Check all required XML tags are present.
    Stage 1 requires: perceive, extract, answer
    Stage 1 compute samples require: perceive, extract, compute, answer
    Stage 2 requires: perceive, extract, compare, verdict, confidence, rubric"""
    return validation_error(
        reasoning,
        stage=stage,
        requires_compute=requires_compute,
    ) is None


def validation_error(
    reasoning: str | None,
    stage: int = 1,
    *,
    requires_compute: bool = False,
) -> str | None:
    if not reasoning:
        return "empty response"

    missing = [
        tag
        for tag in required_tags(stage, requires_compute=requires_compute)
        if _tag_value(reasoning, tag) is None
    ]
    if missing:
        return f"missing or empty XML tags: {', '.join(missing)}"
    return None


def required_tags(stage: int, *, requires_compute: bool = False) -> tuple[str, ...]:
    if stage == 1:
        if requires_compute:
            return ("perceive", "extract", "compute", "answer")
        return ("perceive", "extract", "answer")
    return ("perceive", "extract", "compare", "verdict", "confidence", "rubric")


def extract_verdict(reasoning: str) -> str | None:
    """Extract PASS or FAIL from <verdict> tag via regex."""
    verdict = _tag_value(reasoning, "verdict")
    if verdict is None:
        return None
    match = re.search(r"\b(PASS|FAIL)\b", verdict, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).upper()


def extract_answer(reasoning: str) -> str | None:
    """Extract content from <answer> tag via regex."""
    return _tag_value(reasoning, "answer")


def extract_vote(reasoning: str, *, stage: int) -> str | None:
    return extract_answer(reasoning) if stage == 1 else extract_verdict(reasoning)


def contest(
    reasonings: dict,
    stage: int = 1,
    *,
    requires_compute: bool = False,
) -> str | None:
    """Majority vote on verdict (stage 2) or answer (stage 1).
    Validate each reasoning before including in vote.
    Return agreed value when at least two valid teachers agree, else None."""
    votes = vote_map(
        reasonings,
        stage=stage,
        requires_compute=requires_compute,
    )
    if len(votes) < 2:
        return None

    counts = Counter(_normalize_vote(value) for value in votes.values())
    agreed, count = counts.most_common(1)[0]
    if count < 2:
        return None
    return next(value for value in votes.values() if _normalize_vote(value) == agreed)


def vote_map(
    reasonings: dict,
    *,
    stage: int,
    requires_compute: bool = False,
) -> dict[str, str]:
    votes = {}
    for provider, reasoning in reasonings.items():
        if not validate_reasoning(
            reasoning,
            stage=stage,
            requires_compute=requires_compute,
        ):
            continue
        value = extract_vote(reasoning, stage=stage)
        if value is not None:
            votes[str(provider)] = value.strip()
    return votes


def dissenting_providers(
    reasonings: dict,
    *,
    agreed_output: str,
    stage: int,
    requires_compute: bool = False,
) -> list[str]:
    agreed = _normalize_vote(agreed_output)
    dissenters = []
    for provider, reasoning in reasonings.items():
        if not validate_reasoning(
            reasoning,
            stage=stage,
            requires_compute=requires_compute,
        ):
            dissenters.append(str(provider))
            continue
        value = extract_vote(reasoning, stage=stage)
        if value is None or _normalize_vote(value) != agreed:
            dissenters.append(str(provider))
    return dissenters


def retry_prompt(
    prompt: str,
    *,
    error: str,
    previous_response: str | None = None,
    agreed_output: str | None = None,
) -> str:
    parts = [
        prompt,
        "",
        "RETRY: Your previous response could not be used for distillation.",
        f"Error: {error}",
    ]
    if agreed_output is not None:
        parts.append(f"Required agreed output: {agreed_output}")
    if previous_response:
        parts.extend(["Previous response:", previous_response])
    parts.append("Regenerate the full response using the exact required XML structure.")
    return "\n".join(parts)


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
            await asyncio.sleep(_retry_delay(attempt))
    return None, "unknown retry failure", max_attempts


def _retry_delay(attempt: int) -> float:
    return 2 ** (attempt - 1) + random.uniform(0, 0.5)


def _tag_value(reasoning: str, tag: str) -> str | None:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        reasoning,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _normalize_vote(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _sample_value(sample: dict | EvalSample, key: str) -> str:
    if isinstance(sample, EvalSample):
        if key == "question":
            return sample.question
        if key == "answer":
            return sample.answer
        raise KeyError(f"EvalSample does not provide `{key}` for stage 2.")
    return str(sample[key])


def stage1_requires_compute(sample: dict | EvalSample) -> bool:
    answer_type = _sample_optional_value(sample, "answer_type")
    answer = _sample_optional_value(sample, "answer")
    if answer_type in {"yes_no", "text", "multiple_choice", "structure"}:
        return False
    if answer_type is None and answer is not None and not _looks_numeric(answer):
        return False

    metadata = _sample_metadata(sample)
    if metadata.get("plotqa_answer_mode") == "computed_oov":
        return True

    task_type = _sample_optional_value(sample, "task_type")
    if task_type == "arithmetic":
        return True

    question = str(_sample_optional_value(sample, "question") or "").lower()
    arithmetic_terms = (
        "difference",
        "ratio",
        "average",
        "total",
        "sum",
        "combined",
        "how much more",
        "how much less",
    )
    return bool(_looks_numeric(answer) and any(term in question for term in arithmetic_terms))


def _sample_optional_value(sample: dict | EvalSample, key: str) -> str | None:
    if isinstance(sample, EvalSample):
        value = getattr(sample, key, None)
    else:
        value = sample.get(key)
    if value is None:
        return None
    return str(value)


def _sample_metadata(sample: dict | EvalSample) -> dict:
    if isinstance(sample, EvalSample):
        return sample.metadata
    metadata = sample.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _looks_numeric(value: str | None) -> bool:
    if value is None:
        return False
    try:
        float(str(value).strip().replace(",", "").rstrip("%").rstrip("."))
    except ValueError:
        return False
    return True


def _task_prompt(sample: dict | EvalSample) -> str | None:
    if isinstance(sample, EvalSample):
        return system_prompt(sample)
    value = sample.get("task_prompt") or sample.get("system_prompt")
    if value is None:
        return None
    return str(value)


def _with_task_prompt(prompt: str, task_prompt: str | None) -> str:
    if not task_prompt:
        return prompt
    return "\n".join(
        [
            prompt,
            "",
            "Dataset task guidance:",
            task_prompt,
            "Use this guidance only for answer focus; the XML structure above is still required.",
        ]
    )


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


__all__ = [
    "ROLE_ASSIGNMENTS",
    "CLAUDE_MODEL",
    "CLAUDE_ADJUDICATION_MODEL",
    "CLAUDE_TEACHER_MODEL",
    "CLAUDE_SYNTHESIS_MODEL",
    "OPENAI_MODEL",
    "CONSTITUTIONAL_CONSTRAINT",
    "STAGE1_REASONING_PROMPT",
    "STAGE1_COMPUTE_REASONING_PROMPT",
    "STAGE2_REASONING_PROMPT",
    "TeacherCallResult",
    "build_stage_prompt",
    "call_all_teachers",
    "call_teacher",
    "contest",
    "dissenting_providers",
    "encode_image",
    "extract_answer",
    "extract_verdict",
    "get_all_reasonings",
    "get_claude_adjudication",
    "get_claude_synthesis",
    "required_tags",
    "retry_prompt",
    "stage1_requires_compute",
    "validate_reasoning",
    "validation_error",
    "vote_map",
]
