from pathlib import Path
from collections import Counter
import os

import click
import mlflow
from dotenv import find_dotenv, load_dotenv
from mlflow.tracking import MlflowClient
from peft import get_peft_model
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments
from models.qwen_vl import load_model
from training.sft.adapters.qwenvl import lora_config_hash, lora_config_payload, lora_train_config
from training.sft.loader import DistilledChartSFTLoader, collate_distilled_chart_sft, load_mixed_distilled_rows

load_dotenv(find_dotenv())

TRAIN_DATA_ROOT = Path("artifacts/train_data")
MLFLOW_ARTIFACT_ROOT = Path("artifacts/mlflow")
MLFLOW_EXPERIMENT = "kestrel-sft"


def _artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


@click.command()
@click.option("--size", default="tiny", type=click.Choice(["tiny", "small", "aight"]))
@click.option("--max-samples-per-dataset", type=int)
@click.option("--val-split", default=0.05, show_default=True)
@click.option("--seed", default=13, show_default=True)
@click.option("--include-contrastive", is_flag=True)
@click.option("--epochs", default=1.0, show_default=True)
@click.option("--batch-size", default=1, show_default=True)
@click.option("--eval-batch-size", default=1, show_default=True)
@click.option("--grad-accum", default=8, show_default=True)
@click.option("--lr", default=2e-5, show_default=True)
@click.option("--warmup-steps", default=0, show_default=True)
@click.option("--logging-steps", default=10, show_default=True)
@click.option("--eval-steps", default=100, show_default=True)
@click.option("--save-steps", default=100, show_default=True)
@click.option("--save-total-limit", default=2, show_default=True)
@click.option("--max-steps", type=int)
@click.option("--early-stopping-patience", type=int)
@click.option("--max-image-pixels", default=1024 * 28 * 28, show_default=True)
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True, show_default=True)
@click.option("--run-name")
@click.option("--output-dir", default=Path("artifacts/adapters/sft-smoke"), type=click.Path(path_type=Path))
def main(
    size,
    max_samples_per_dataset,
    val_split,
    seed,
    include_contrastive,
    epochs,
    batch_size,
    eval_batch_size,
    grad_accum,
    lr,
    warmup_steps,
    logging_steps,
    eval_steps,
    save_steps,
    save_total_limit,
    max_steps,
    early_stopping_patience,
    max_image_pixels,
    gradient_checkpointing,
    run_name,
    output_dir,
) -> None:
    model, processor = load_model(size=size)
    _configure_image_budget(processor, max_image_pixels)
    if gradient_checkpointing:
        model.config.use_cache = False
    model = get_peft_model(model, lora_train_config(size))
    model.print_trainable_parameters()
    train_rows, val_rows = load_mixed_distilled_rows(TRAIN_DATA_ROOT, max_samples_per_dataset, val_split, seed, include_contrastive)
    resolved_run_name = run_name or f"qwen-vl-{size}-sft"
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps or -1,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
        run_name=resolved_run_name,
    )
    _ensure_mlflow_experiment()
    with mlflow.start_run(run_name=resolved_run_name):
        mlflow.log_params({
            "size": size,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "max_samples_per_dataset": max_samples_per_dataset,
            "val_split": val_split,
            "seed": seed,
            "include_contrastive": include_contrastive,
            "early_stopping_patience": early_stopping_patience,
            "max_image_pixels": max_image_pixels,
            "gradient_checkpointing": gradient_checkpointing,
            "lora_config_hash": lora_config_hash(size),
        })
        mlflow.log_metrics(_row_metrics(train_rows, "train") | _row_metrics(val_rows, "val"), step=0)
        mlflow.log_dict(_row_mix(train_rows, val_rows), "data/mix.json")
        mlflow.log_dict(lora_config_payload(size), f"lora/{_artifact_name(resolved_run_name)}.json")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=DistilledChartSFTLoader(train_rows),
            eval_dataset=DistilledChartSFTLoader(val_rows),
            data_collator=lambda batch: collate_distilled_chart_sft(batch, processor, TRAIN_DATA_ROOT),
            callbacks=_callbacks(early_stopping_patience),
        )
        try:
            trainer.train()
        except KeyboardInterrupt:
            checkpoint = output_dir / "interrupted"
            trainer.save_model(checkpoint)
            processor.save_pretrained(checkpoint)
            mlflow.log_artifacts(checkpoint, artifact_path="interrupted_checkpoint")
            raise


def _row_mix(train_rows: list[dict], val_rows: list[dict]) -> dict:
    return {
        "train": dict(Counter(row["metadata"]["dataset"] for row in train_rows)),
        "val": dict(Counter(row["metadata"]["dataset"] for row in val_rows)),
    }


def _callbacks(early_stopping_patience: int | None) -> list:
    callbacks = [MLflowMetricsCallback()]
    if early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    return callbacks


class MLflowMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        for key, value in logs.items():
            if isinstance(value, int | float):
                mlflow.log_metric(_metric_name(key), float(value), step=step)


def _configure_image_budget(processor, max_image_pixels: int | None) -> None:
    if max_image_pixels is None:
        return
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if size is None:
        return
    size["longest_edge"] = max_image_pixels


def _row_metrics(rows: list[dict], prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {f"{prefix}/rows": float(len(rows))}
    for field in ("dataset", "answer_type", "reasoning_schema", "example_type"):
        counts = Counter(row["metadata"].get(field, "unknown") for row in rows)
        total = sum(counts.values()) or 1
        for key, count in counts.items():
            metrics[f"{prefix}/{field}/{_metric_name(str(key))}/count"] = float(count)
            metrics[f"{prefix}/{field}/{_metric_name(str(key))}/ratio"] = count / total
    return metrics


def _metric_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def _ensure_mlflow_experiment() -> None:
    if os.environ.get("MLFLOW_EXPERIMENT_ID") or os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        return
    tracking_uri = mlflow.get_tracking_uri()
    artifact_root = MLFLOW_ARTIFACT_ROOT.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    artifact_location = artifact_root.as_uri()
    if experiment is None:
        client.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifact_location)
    elif tracking_uri.startswith("sqlite") and experiment.artifact_location.startswith("mlflow-artifacts:"):
        fallback_name = f"{MLFLOW_EXPERIMENT}-local"
        if client.get_experiment_by_name(fallback_name) is None:
            client.create_experiment(fallback_name, artifact_location=artifact_location)
        mlflow.set_experiment(fallback_name)
        return
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


if __name__ == "__main__":
    main()
