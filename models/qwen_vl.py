from pathlib import Path

import click
import torch
from accelerate import Accelerator
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import snapshot_download
from rich.console import Console
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

load_dotenv(find_dotenv())

console = Console()
DEFAULT_ARTIFACT_PATH = Path("artifacts/pretrained")


def load_model(size="tiny", artifact_path=DEFAULT_ARTIFACT_PATH, precision="auto"):
    model_map = {
        "tiny": "Qwen/Qwen2-VL-2B-Instruct",
        "small": "Qwen/Qwen2-VL-3B-Instruct",
        "aight": "Qwen/Qwen2-VL-7B-Instruct",
    }
    size = size.lower()
    precision = precision.lower()
    model_id = model_map[size]
    model_path = Path(artifact_path) / size
    device = Accelerator().device
    dtype = _resolve_dtype(precision, device.type)

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
        attn_implementation=_resolve_attention(device.type),
    )
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def _resolve_dtype(precision, device_type):
    precision_map = {
        "cpu": torch.float32,
        "mps": torch.float16,
        "gpu": torch.bfloat16,
    }
    if precision == "auto":
        precision = "gpu" if device_type == "cuda" else device_type
    return precision_map.get(precision, "auto")


def _resolve_attention(device_type):
    if device_type != "cuda":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


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
    model, processor = load_model(
        size=size,
        artifact_path=artifact_path,
        precision=precision,
    )
    console.print(f"[bold green]Model loaded from {Path(artifact_path) / size}[/bold green]")
    return model, processor


cli.add_command(load)


if __name__ == "__main__":
    cli()
