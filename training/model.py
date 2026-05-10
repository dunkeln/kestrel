from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import click
from rich.console import Console
from dotenv import load_dotenv, find_dotenv
from accelerate import Accelerator

load_dotenv(find_dotenv())

console = Console()
DEFAULT_ARTIFACT_PATH = Path("artifacts/pretrained")
MODEL_MAP = {
    "tiny": "Qwen/Qwen2-VL-2B-Instruct",
    "small": "Qwen/Qwen2-VL-3B-Instruct",
    "aight": "Qwen/Qwen2-VL-7B-Instruct",
}


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--size",
    type=click.Choice(["tiny", "small", "aight"], case_sensitive=False),
    default="tiny",
    show_default=True,
)
@click.option(
    "--artifact-path",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_ARTIFACT_PATH,
    show_default=True,
)
@click.option(
    "--precision",
    type=click.Choice(["auto", "cpu", "mps", "gpu"], case_sensitive=False),
    default="auto",
    show_default=True,
)
def load(size, artifact_path, precision):
    model = None
    processor = None
    size = size.lower()
    model_id = MODEL_MAP[size]
    model_path = artifact_path / size
    device = Accelerator().device
    precision_map = {
        "cpu": torch.float32,
        "mps": torch.float16,
        "gpu": torch.bfloat16,
    }
    precision = precision.lower()
    if precision == "auto":
        precision = "gpu" if device.type == "cuda" else device.type
    dtype = precision_map.get(precision, "auto")

    # INFO:
    # model loads from artifact_path first
    # if path doesn't exist, fetch model form huggingface hub
    try:
        if not (model_path / "config.json").exists():
            console.print(f"Downloading {model_id} to {model_path}")
            model_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
            )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            attn_implementation=(
                "flash_attention_2" if device.type == "cuda" else "sdpa"
            ),
        )
        processor = AutoProcessor.from_pretrained(model_path)

        console.print(f"[bold green]Model loaded from {model_path}[/bold green]")

    except Exception as e:
        console.print(f"[red]{e}[/red]")

    return model, processor


cli.add_command(load)


if __name__ == "__main__":
    cli()
