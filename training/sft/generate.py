from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from training.sft.generation_runner import run_generation


logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", required=True, help="DatasetLoader dataset key.")
@click.option("--split", default="train", show_default=True)
@click.option("--samples", type=click.IntRange(min=0), default=None)
@click.option("--all", "all_samples", is_flag=True, help="Process the full split.")
@click.option("--batch-size", type=click.IntRange(min=1), default=8, show_default=True)
@click.option(
    "--provider-mode",
    type=click.Choice(["sync", "batch"]),
    default="sync",
    show_default=True,
    help="Use sync API calls or provider-native async batch jobs.",
)
@click.option(
    "--provider-poll-interval",
    type=click.FloatRange(min=1.0),
    default=60.0,
    show_default=True,
    help="Seconds between provider batch status polls.",
)
@click.option("--shuffle-seed", type=int, default=0, show_default=True)
@click.option("--shuffle-buffer-size", type=click.IntRange(min=0), default=1024, show_default=True)
@click.option("--contrastive-rate", type=click.FloatRange(0.0, 1.0), default=0.25, show_default=True)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/train_data"),
    show_default=True,
)
@click.option(
    "--resume",
    is_flag=True,
    help="Append to existing artifacts after backfilling and indexing record keys.",
)
def main(
    dataset: str,
    split: str,
    samples: int | None,
    all_samples: bool,
    batch_size: int,
    provider_mode: str,
    provider_poll_interval: float,
    shuffle_seed: int,
    shuffle_buffer_size: int,
    contrastive_rate: float,
    output_root: Path,
    resume: bool,
) -> None:
    """Generate per-benchmark constitutional SFT JSONL artifacts."""
    if (samples is not None) == all_samples:
        raise click.UsageError("Pass exactly one of --samples or --all.")

    _configure_logging()
    asyncio.run(
        run_generation(
            dataset=dataset,
            split=split,
            samples=samples,
            all_samples=all_samples,
            batch_size=batch_size,
            provider_mode=provider_mode,
            provider_poll_interval=provider_poll_interval,
            shuffle_seed=shuffle_seed,
            shuffle_buffer_size=shuffle_buffer_size,
            contrastive_rate=contrastive_rate,
            output_root=output_root,
            resume=resume,
            console=Console(),
        )
    )


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
        }
        message = record.getMessage()
        if message.startswith("{"):
            try:
                payload.update(json.loads(message))
            except json.JSONDecodeError:
                payload["message"] = message
        else:
            payload["message"] = message
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    main()
