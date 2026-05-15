import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from training.distillation.storage import SCHEMA_VERSION, train_data_paths

BENCHMARK_DATASETS = ("chartqa", "figureqa", "plotqa_qa", "chartbench", "mmc_benchmark")


def load_distilled_rows(dataset: str, split: str, data_root: Path) -> list[dict]:
    path = train_data_paths(data_root, dataset, split).gold
    return _load_jsonl_rows(path)


def load_contrastive_rows(dataset: str, split: str, data_root: Path) -> list[dict]:
    path = train_data_paths(data_root, dataset, split).contrastive
    return [_as_contrastive_critique(row) for row in _load_jsonl_rows(path)]


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = [row.get("record_key", "<missing>") for row in rows if row["metadata"].get("schema_version") != SCHEMA_VERSION]
    if bad:
        raise ValueError(f"Unsupported SFT schema in {path}: {bad[:3]}")
    return rows


def load_mixed_distilled_rows(
    data_root: Path,
    max_samples_per_dataset: int | None = None,
    val_split: float = 0.05,
    seed: int = 13,
    include_contrastive: bool = False,
) -> tuple[list[dict], list[dict]]:
    train_rows, val_rows = [], []
    for index, dataset in enumerate(BENCHMARK_DATASETS):
        dataset_rows = load_distilled_rows(dataset, "train", data_root)
        random.Random(seed + index).shuffle(dataset_rows)
        if max_samples_per_dataset is not None:
            dataset_rows = dataset_rows[:max_samples_per_dataset]
        cutoff = int(len(dataset_rows) * (1 - val_split))
        dataset_train = dataset_rows[:cutoff]
        dataset_val = dataset_rows[cutoff:]
        if include_contrastive:
            contrastive = {row["record_key"]: row for row in load_contrastive_rows(dataset, "train", data_root)}
            dataset_train += [contrastive[key] for key in _record_keys(dataset_train) if key in contrastive]
            dataset_val += [contrastive[key] for key in _record_keys(dataset_val) if key in contrastive]
        train_rows.extend(dataset_train)
        val_rows.extend(dataset_val)
    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 10_000).shuffle(val_rows)
    return train_rows, val_rows


def _record_keys(rows: list[dict]) -> list[str]:
    return [row["record_key"] for row in rows]


def _as_contrastive_critique(row: dict) -> dict:
    metadata = row["metadata"]
    row = dict(row)
    row["input"] = (
        f"{row['input']}\n"
        f"Proposed answer: {metadata.get('perturbed_answer', '')}\n"
        "Is the proposed answer faithful to the chart?"
    )
    row["output"] = (
        "<verdict>FAIL</verdict>\n"
        f"<correct_answer>{metadata.get('gold_answer', metadata.get('answer', ''))}</correct_answer>\n"
        "<correction>\n"
        f"{metadata.get('gold_output', '')}\n"
        "</correction>"
    )
    return row


class DistilledChartSFTLoader(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self) -> dict:
        if self.cursor >= len(self.rows):
            raise StopIteration
        row = self.rows[self.cursor]
        self.cursor += 1
        return row


def collate_distilled_chart_sft(rows: list[dict], processor, data_root: Path) -> dict:
    texts, prompt_texts, images = [], [], []
    for row in rows:
        images.append(Image.open(data_root / row["image"]).convert("RGB"))
        prompt = (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{row['input']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_texts.append(prompt)
        texts.append(f"{prompt}{row['output']}<|im_end|>")
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    prompt_batch = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["labels"] == processor.tokenizer.pad_token_id] = -100
    padding_side = getattr(processor.tokenizer, "padding_side", "right")
    for index, prompt_mask in enumerate(prompt_batch["attention_mask"]):
        prompt_len = int(prompt_mask.sum().item())
        full_len = int(batch["attention_mask"][index].sum().item())
        start = batch["labels"].shape[1] - full_len if padding_side == "left" else 0
        batch["labels"][index, start : start + prompt_len] = -100
    return batch
