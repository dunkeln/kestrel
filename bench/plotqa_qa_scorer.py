import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping

from training.datasets.contracts import AnswerType


@dataclass(frozen=True)
class PlotQAQAScore:
    correct: bool
    expected: str
    actual: str
    score: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def score_plotqa_qa(
    *,
    answer_type: AnswerType,
    gold: str,
    prediction: str,
    task_type: str | None = None,
) -> PlotQAQAScore:
    expected = _normalize_by_type(answer_type, gold)
    actual = _normalize_by_type(answer_type, prediction)
    correct = expected == actual
    gold_number = _parse_decimal(gold) if answer_type == "numeric" else None
    predicted_number = _parse_decimal(prediction) if answer_type == "numeric" else None
    metadata: dict[str, Any] = {
        "plotqa_qa_answer_mode": _answer_mode(answer_type, task_type),
        "plotqa_qa_task_type": task_type,
        "plotqa_qa_exact_match": correct,
    }

    if answer_type == "numeric":
        metadata.update(
            {
                "plotqa_qa_numeric_gold_parse_ok": gold_number is not None,
                "plotqa_qa_numeric_prediction_parse_ok": predicted_number is not None,
            }
        )
        if gold_number is not None and predicted_number is not None:
            absolute_error = abs(predicted_number - gold_number)
            denominator = abs(gold_number)
            relative_error = (
                Decimal("0")
                if absolute_error == 0
                else absolute_error / denominator
                if denominator != 0
                else None
            )
            metadata["plotqa_qa_absolute_error"] = float(absolute_error)
            metadata["plotqa_qa_relative_error"] = (
                float(relative_error) if relative_error is not None else None
            )

    return PlotQAQAScore(
        correct=correct,
        expected=expected,
        actual=actual,
        score=1.0 if correct else 0.0,
        metadata=metadata,
    )


def _normalize_by_type(answer_type: AnswerType, value: str) -> str:
    if answer_type == "numeric":
        return _normalize_numeric(value)
    if answer_type == "yes_no":
        return _normalize_yes_no(value)
    return _normalize_text(value)


def _normalize_numeric(value: str) -> str:
    parsed = _parse_decimal(value)
    if parsed is None:
        return _normalize_text(value)
    return _decimal_to_text(parsed)


def _normalize_yes_no(value: str) -> str:
    text = _normalize_text(value)
    if text == "true":
        return "yes"
    if text == "false":
        return "no"
    return text


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().strip(" \t\r\n.!?").split())


def _parse_decimal(value: str) -> Decimal | None:
    text = str(value).strip().replace(",", "")
    text = text.rstrip(".").rstrip("%").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text, re.IGNORECASE):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _decimal_to_text(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized, "f")
    return format(normalized, "f").rstrip("0").rstrip(".")


def _answer_mode(answer_type: AnswerType, task_type: str | None) -> str:
    if answer_type in {"yes_no", "text", "multiple_choice"}:
        return "fixed_vocabulary"
    if task_type == "arithmetic":
        return "computed_oov"
    return "extractive_or_count"
