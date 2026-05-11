from dataclasses import dataclass, field
from typing import Any

from training.datasets.contracts import AnswerType

from bench.plotqa_qa_scorer import score_plotqa_qa
from bench.plotqa_structure_scorer import score_plotqa_structure


@dataclass(frozen=True)
class Score:
    correct: bool
    expected: str
    actual: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def score_prediction(
    *,
    answer_type: AnswerType,
    gold: str,
    prediction: str,
    dataset: str | None = None,
    task_type: str | None = None,
) -> Score:
    if dataset == "plotqa_structure":
        plotqa_score = score_plotqa_structure(gold, prediction)
        return Score(
            correct=plotqa_score.score == 1.0,
            expected="plotqa_structure",
            actual="plotqa_structure",
            score=plotqa_score.score,
            metadata=dict(plotqa_score.metadata),
        )

    if dataset == "plotqa_qa":
        plotqa_score = score_plotqa_qa(
            answer_type=answer_type,
            gold=gold,
            prediction=prediction,
            task_type=task_type,
        )
        return Score(
            correct=plotqa_score.correct,
            expected=plotqa_score.expected,
            actual=plotqa_score.actual,
            score=plotqa_score.score,
            metadata=dict(plotqa_score.metadata),
        )

    expected = _normalize_by_type(answer_type, gold)
    actual = _normalize_by_type(answer_type, prediction)
    correct = expected == actual
    return Score(
        correct=correct,
        expected=expected,
        actual=actual,
        score=1.0 if correct else 0.0,
    )


def _normalize_by_type(answer_type: AnswerType, value: str) -> str:
    if answer_type == "numeric":
        return _normalize_numeric(value)
    if answer_type == "yes_no":
        return _normalize_yes_no(value)
    return _normalize_text(value)


def _normalize_numeric(value: str) -> str:
    text = str(value).strip().replace(",", "").rstrip("%")
    try:
        return str(float(text))
    except ValueError:
        return _normalize_text(value)


def _normalize_yes_no(value: str) -> str:
    text = _normalize_text(value)
    if text == "true":
        return "yes"
    if text == "false":
        return "no"
    return text


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().strip(" \t\r\n.!?").split())
