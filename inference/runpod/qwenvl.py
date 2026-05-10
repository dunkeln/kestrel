from pathlib import Path

from inference.backends.pytorch_qwenvl import QwenVlBackend
from inference.config import load_inference_settings


def build_backend(config_path: Path = Path("config.toml")) -> QwenVlBackend:
    settings = load_inference_settings(config_path)
    return QwenVlBackend(model=settings.model, config=settings.config)
