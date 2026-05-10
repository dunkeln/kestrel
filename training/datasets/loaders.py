from io import BytesIO
from hashlib import sha1

from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from PIL import Image

from training.datasets.contracts import EvalSample, infer_answer_type, infer_task_type

load_dotenv(find_dotenv())


DATASET_CARDS = {
    "chartqa": {
        "path": "HuggingFaceM4/ChartQA",
        "splits": {
            "train": "train",
            "validation": "val",
            "test": "test",
        },
        "supervision": "benchmark",
    },
    "plotqa": {
        "path": "achang/plot_qa",
        "splits": {
            "train": "train",
            "validation": "validation",
            "test": "test",
        },
        "supervision": "structure",
    },
    "figureqa": {
        "path": "vikhyatk/figureqa",
        "splits": {
            "train": "train",
            "test": "train",
        },
        "supervision": "qa",
    },
    "chartbench": {
        "path": "SincereX/ChartBench",
        "name": "chart_bench",
        "splits": {
            "train": "train_data",
            "test": "test_data",
        },
        "supervision": "benchmark",
    },
    "mmc_benchmark": {
        "path": "xywang1/MMC",
        "name": "MMC-Benchmark",
        "splits": {
            "train": "test",
            "test": "test",
        },
        "supervision": "benchmark",
    },
}


DEFAULT_SYSTEM_PROMPT = (
    "Read the chart carefully. Answer only the question. Keep the answer concise."
)

TASK_SYSTEM_PROMPTS = {
    "value_extraction": "Read the chart value requested. Return only the value.",
    "arithmetic": "Compute from chart values. Return only the final answer.",
    "global_extrema": "Find the requested maximum or minimum. Return only the answer.",
    "yes_no": "Judge the chart statement. Answer only yes or no.",
    "analysis": "Judge the chart statement against the image. Answer only true or false.",
    "structure_extraction": "Extract the chart structure in the requested serialized format.",
    "chart_reasoning": "Use visual chart evidence. Return a concise answer.",
}

ANSWER_TYPE_SYSTEM_PROMPTS = {
    "yes_no": "Answer only yes or no.",
    "numeric": "Return only the numeric answer.",
    "multiple_choice": "Return only the selected option.",
    "structure": "Return only the requested structured representation.",
    "text": DEFAULT_SYSTEM_PROMPT,
}


def _system_prompt(task_type=None, answer_type=None):
    if task_type in TASK_SYSTEM_PROMPTS:
        return TASK_SYSTEM_PROMPTS[task_type]
    if answer_type in ANSWER_TYPE_SYSTEM_PROMPTS:
        return ANSWER_TYPE_SYSTEM_PROMPTS[answer_type]
    return DEFAULT_SYSTEM_PROMPT


def system_prompt(sample: EvalSample):
    return _system_prompt(
        task_type=sample.task_type,
        answer_type=sample.answer_type,
    )


class DatasetLoader:
    def __init__(self, dataset_name, split=None, streaming=True):
        self.dataset_name = dataset_name
        self.dataset_card = DATASET_CARDS[dataset_name]
        self.split = split or "train"
        self.streaming = streaming

    def raw(self):
        return load_dataset(**self._load_kwargs(), streaming=self.streaming)

    def samples(self):
        for source_index, row in enumerate(self.raw()):
            yield from _adapt_row(
                self.dataset_name,
                row,
                source_index=source_index,
            )

    def batch(self, offset=0, limit=None):
        dataset = self.raw()
        if offset:
            dataset = dataset.skip(offset)
        if limit is not None:
            dataset = dataset.take(limit)

        for source_index, row in enumerate(dataset, start=offset):
            yield from _adapt_row(
                self.dataset_name,
                row,
                source_index=source_index,
            )

    def _load_kwargs(self):
        load_kwargs = {
            key: value
            for key, value in self.dataset_card.items()
            if key in {"path", "name"}
        }
        load_kwargs["split"] = self._resolve_split()
        return load_kwargs

    def _resolve_split(self):
        splits = self.dataset_card["splits"]
        return splits.get(self.split, self.split)


def _stable_id(value):
    return sha1(str(value).encode("utf-8")).hexdigest()[:12]


def _decode_image(image):
    if isinstance(image, dict) and image.get("bytes") is not None:
        return Image.open(BytesIO(image["bytes"]))
    return image


def _adapt_row(dataset_name, row, source_index=None):
    match dataset_name:
        case "chartqa":
            return [_adapt_chartqa(row)]
        case "figureqa":
            return list(_adapt_figureqa(row, source_index))
        case "plotqa":
            return [_adapt_plotqa(row, source_index)]
        case "chartbench":
            return list(_adapt_chartbench(row))
        case "mmc_benchmark":
            return [_adapt_mmc_benchmark(row)]
        case _:
            raise KeyError(f"Unknown dataset: {dataset_name}")


def _adapt_chartqa(row):
    question = row["query"]
    answer = row["label"][0]
    return EvalSample(
        id=f"chartqa:{_stable_id([question, answer])}",
        dataset="chartqa",
        image=row["image"],
        question=question,
        answer=str(answer),
        answer_type=infer_answer_type(answer),
        supervision="benchmark",
        task_type=infer_task_type(question),
        tag="real",
        metadata={"source_type": row.get("human_or_machine")},
    )


def _adapt_figureqa(row, source_index):
    source_id = source_index if source_index is not None else _stable_id(row["qa"])
    for index, qa in enumerate(row["qa"]):
        question = qa["question"]
        answer = qa["answer"]
        yield EvalSample(
            id=f"figureqa:{source_id}:{index}",
            dataset="figureqa",
            image=_decode_image(row["image"]),
            question=question,
            answer=str(answer),
            answer_type=infer_answer_type(answer),
            supervision="qa",
            task_type=infer_task_type(question),
            tag="real",
        )


def _adapt_plotqa(row, source_index):
    source_id = source_index if source_index is not None else _stable_id(row["text"])
    return EvalSample(
        id=f"plotqa:{source_id}",
        dataset="plotqa",
        image=row["image"],
        question="Extract the chart structure.",
        answer=row["text"],
        answer_type="structure",
        supervision="structure",
        task_type="structure_extraction",
        tag="real",
    )


def _adapt_chartbench(row):
    source_id = row.get("id") if row.get("id") is not None else _stable_id(row)
    image_ref = row["image"]
    image_archive = "data/train.zip" if "/train/" in image_ref else "data/test.zip"
    chart_type = row.get("type", {}).get("chart")
    task_type = row.get("type", {}).get("task")
    for index, turn in enumerate(row["conversation"]):
        answer = turn["label"]
        yield EvalSample(
            id=f"chartbench:{source_id}:{index}",
            dataset="chartbench",
            image=image_ref,
            question=turn["query"],
            answer=str(answer),
            answer_type=infer_answer_type(answer),
            supervision="benchmark",
            chart_type=chart_type,
            task_type=task_type,
            tag="real",
            metadata={
                "type": row.get("type"),
                "image_ref": image_ref,
                "image_archive": image_archive,
            },
        )


def _adapt_mmc_benchmark(row):
    question = row.get("instruction") or row.get("question") or row.get("query") or ""
    answer = row.get("label") or row.get("answer") or ""
    options = row.get("options") or row.get("choices")
    source_id = row.get("image_id") or row.get("id") or _stable_id(row)
    return EvalSample(
        id=f"mmc_benchmark:{source_id}:{_stable_id(question)}",
        dataset="mmc_benchmark",
        image=row.get("image") or row.get("image_id"),
        question=question,
        answer=str(answer),
        answer_type=infer_answer_type(answer, has_options=bool(options)),
        supervision="benchmark",
        task_type=row.get("task") or infer_task_type(question),
        tag="real",
        metadata={
            "options": options,
            "task": row.get("task"),
            "image_ref": row.get("image_id"),
            "image_archive": "MMC-Benchmark/mmc_benchmark_images.tar.gz",
        },
    )
