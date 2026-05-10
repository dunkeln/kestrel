from hashlib import sha1

from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

from training.datasets.contracts import EvalSample, infer_answer_type, infer_task_type

load_dotenv(find_dotenv())


DATASET_CARDS = {
    "chartqa": {
        "path": "docintel/ChartQA",
        "split": "test",
        "supervision": "benchmark",
    },
    "plotqa": {
        "path": "achang/plot_qa",
        "split": "train",
        "supervision": "structure",
    },
    "figureqa": {
        "path": "vikhyatk/figureqa",
        "split": "train",
        "supervision": "qa",
    },
    "chartbench": {
        "path": "SincereX/ChartBench",
        "name": "chart_bench",
        "split": "train_data",
        "supervision": "benchmark",
    },
    "mmc_benchmark": {
        "path": "xywang1/MMC",
        "name": "MMC-Benchmark",
        "split": "test",
        "supervision": "benchmark",
    },
}


def load_dataset_card(dataset_name, split=None, streaming=False):
    dataset_card = DATASET_CARDS[dataset_name]
    load_kwargs = {
        key: value
        for key, value in dataset_card.items()
        if key in {"path", "name", "split"}
    }
    if split is not None:
        load_kwargs["split"] = split
    return load_dataset(**load_kwargs, streaming=streaming)


def adapt_row(dataset_name, row, source_index=None):
    if dataset_name == "chartqa":
        return [_adapt_chartqa(row)]
    if dataset_name == "figureqa":
        return list(_adapt_figureqa(row, source_index))
    if dataset_name == "plotqa":
        return [_adapt_plotqa(row, source_index)]
    if dataset_name == "chartbench":
        return list(_adapt_chartbench(row))
    if dataset_name == "mmc_benchmark":
        return [_adapt_mmc_benchmark(row)]
    raise KeyError(f"Unknown dataset: {dataset_name}")


def iter_eval_samples(dataset_name, split=None, limit=None, offset=0, streaming=True):
    dataset = load_dataset_card(dataset_name, split=split, streaming=streaming)
    if offset:
        dataset = dataset.skip(offset)
    if limit is not None:
        dataset = dataset.take(limit)

    for source_index, row in enumerate(dataset, start=offset):
        yield from adapt_row(dataset_name, row, source_index=source_index)


def _adapt_chartqa(row):
    question = row["question"]
    answer = row["answer"]
    return EvalSample(
        id=f"chartqa:{row['id']}",
        dataset="chartqa",
        image=row["image"],
        question=question,
        answer=str(answer),
        answer_type=infer_answer_type(answer),
        supervision="benchmark",
        task_type=infer_task_type(question),
        metadata={"source_type": row.get("type")},
    )


def _adapt_figureqa(row, source_index):
    source_id = source_index if source_index is not None else _stable_id(row["qa"])
    for index, qa in enumerate(row["qa"]):
        question = qa["question"]
        answer = qa["answer"]
        yield EvalSample(
            id=f"figureqa:{source_id}:{index}",
            dataset="figureqa",
            image=row["image"],
            question=question,
            answer=str(answer),
            answer_type=infer_answer_type(answer),
            supervision="qa",
            task_type=infer_task_type(question),
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
    )


def _adapt_chartbench(row):
    source_id = row.get("id") or _stable_id(row)
    chart_type = row.get("type", {}).get("chart")
    task_type = row.get("type", {}).get("task")
    for index, turn in enumerate(row["conversation"]):
        answer = turn["label"]
        yield EvalSample(
            id=f"chartbench:{source_id}:{index}",
            dataset="chartbench",
            image=row["image"],
            question=turn["query"],
            answer=str(answer),
            answer_type=infer_answer_type(answer),
            supervision="benchmark",
            chart_type=chart_type,
            task_type=task_type,
            metadata={"type": row.get("type")},
        )


def _adapt_mmc_benchmark(row):
    question = row.get("question") or row.get("query") or ""
    answer = row.get("answer") or row.get("label") or ""
    options = row.get("options") or row.get("choices")
    source_id = row.get("image_id") or row.get("id") or _stable_id(row)
    return EvalSample(
        id=f"mmc_benchmark:{source_id}",
        dataset="mmc_benchmark",
        image=row.get("image") or row.get("image_id"),
        question=question,
        answer=str(answer),
        answer_type=infer_answer_type(answer, has_options=bool(options)),
        supervision="benchmark",
        task_type=infer_task_type(question),
        metadata={"options": options},
    )


def _stable_id(value):
    return sha1(str(value).encode("utf-8")).hexdigest()[:12]
