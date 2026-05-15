import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.qwen_vl import (
    DEFAULT_ARTIFACT_PATH,
    cli,
    load,
    load_model,
)


if __name__ == "__main__":
    cli()
