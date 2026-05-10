from dataclasses import dataclass, field
from typing import Any, Literal

AnswerType = Literal["yes_no", "numeric", "text", "multiple_choice", "structure"]
SupervisionType = Literal["qa", "classification", "structure", "benchmark"]


@dataclass(frozen=True)
class EvalSample:
    id: str
    dataset: str
    image: Any
    question: str
    answer: str
    answer_type: AnswerType
    supervision: SupervisionType
    chart_type: str | None = None
    task_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def infer_answer_type(answer: Any, has_options: bool = False) -> AnswerType:
    if has_options:
        return "multiple_choice"

    answer_text = str(answer).strip().lower().rstrip("!.?")
    if answer_text in {"yes", "no"}:
        return "yes_no"

    try:
        float(answer_text.replace(",", "").rstrip("%"))
    except ValueError:
        return "text"

    return "numeric"


def infer_task_type(question: str) -> str | None:
    question_text = question.lower()
    if question_text.startswith(("is ", "are ", "does ", "do ", "was ", "were ")):
        return "yes_no"
    if any(term in question_text for term in ("difference", "sum", "average", "ratio")):
        return "arithmetic"
    if any(
        term in question_text for term in ("highest", "lowest", "maximum", "minimum")
    ):
        return "global_extrema"
    if any(term in question_text for term in ("how many", "what is the value")):
        return "value_extraction"
    if "chart" in question_text or "graph" in question_text:
        return "chart_reasoning"
    return None
