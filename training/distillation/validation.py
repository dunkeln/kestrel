from __future__ import annotations

import re
from collections import Counter

from training.datasets.contracts import EvalSample


def validate_reasoning(
    reasoning: str | None,
    stage: int = 1,
    *,
    requires_compute: bool = False,
) -> bool:
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
        if tag_value(reasoning, tag) is None
    ]
    if missing:
        return f"missing or empty XML tags: {', '.join(missing)}"
    return None


def required_tags(stage: int, *, requires_compute: bool = False) -> tuple[str, ...]:
    if stage == 1:
        return ("perceive", "extract", "compute", "answer") if requires_compute else (
            "perceive",
            "extract",
            "answer",
        )
    return ("perceive", "extract", "compare", "verdict", "confidence", "rubric")


def extract_verdict(reasoning: str) -> str | None:
    verdict = tag_value(reasoning, "verdict")
    if verdict is None:
        return None
    match = re.search(r"\b(PASS|FAIL)\b", verdict, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_answer(reasoning: str) -> str | None:
    return tag_value(reasoning, "answer")


def extract_vote(reasoning: str, *, stage: int) -> str | None:
    return extract_answer(reasoning) if stage == 1 else extract_verdict(reasoning)


def contest(
    reasonings: dict,
    stage: int = 1,
    *,
    requires_compute: bool = False,
) -> str | None:
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


def lint_synthesis(
    text: str | None,
    *,
    stage: int,
    sample: dict | None = None,
) -> tuple[str, ...]:
    if text is None:
        return ()
    extract = tag_value(text, "extract")
    if extract is None:
        return ()

    errors: list[str] = []
    if stage == 1 and sample is not None:
        question = str(sample.get("question", ""))
        answer = str(sample.get("answer", ""))
        extra_numbers = _extra_numbers(extract, allowed_text=f"{question} {answer}")
        if len(extra_numbers) > _allowed_extra_number_count(question):
            errors.append(
                "extract includes unnecessary numeric/date evidence: "
                + ", ".join(extra_numbers)
                + ". Keep only facts needed to answer the question."
            )
        if _simple_question(question) and _word_count(extract) > 42:
            errors.append(
                "extract is too broad for a simple question. Keep only the minimal evidence needed."
            )
    return tuple(errors)


def lint_error_message(errors: tuple[str, ...]) -> str | None:
    return " ".join(errors) if errors else None


def tag_value(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _normalize_vote(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


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


def _extra_numbers(text: str, *, allowed_text: str) -> list[str]:
    allowed = set(_numbers(allowed_text))
    return [number for number in _numbers(text) if number not in allowed]


def _numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def _allowed_extra_number_count(question: str) -> int:
    lowered = question.lower()
    if any(term in lowered for term in ("how many", "count", "number of")):
        return 1
    if any(
        term in lowered
        for term in (
            "increase",
            "decrease",
            "change",
            "difference",
            "higher",
            "lower",
            "minimum",
            "maximum",
        )
    ):
        return 2
    return 1


def _simple_question(question: str) -> bool:
    lowered = question.lower()
    return lowered.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "how many"))


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
