FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/huggingface \
    MPLCONFIGDIR=/workspace/cache/matplotlib \
    UV_CACHE_DIR=/workspace/cache/uv \
    VIRTUAL_ENV=/opt/kestrel-venv \
    PATH=/opt/kestrel-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    KESTREL_ROOT=/workspace/kestrel

WORKDIR /workspace/kestrel

COPY docker/runpod/constraints.txt /tmp/kestrel-runpod-constraints.txt
COPY docker/runpod/requirements.txt /tmp/kestrel-runpod-requirements.txt
COPY docker/runpod/start_mlflow.sh /usr/local/bin/kestrel-start-mlflow

RUN /usr/local/bin/python - <<'PY'
from pathlib import Path

import torch
import torchvision

assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torchvision.__version__.startswith("0.23.0"), torchvision.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert "/usr/local/lib/python3.12/dist-packages" in str(Path(torch.__file__)), torch.__file__
print(
    {
        "base_torch": torch.__version__,
        "base_torchvision": torchvision.__version__,
        "torch_file": torch.__file__,
    }
)
PY

RUN /usr/local/bin/python -m venv --system-site-packages /opt/kestrel-venv && \
    chmod +x /usr/local/bin/kestrel-start-mlflow && \
    /opt/kestrel-venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/kestrel-venv/bin/python -m pip install \
      --only-binary=:all: \
      --upgrade-strategy only-if-needed \
      --constraint /tmp/kestrel-runpod-constraints.txt \
      --requirement /tmp/kestrel-runpod-requirements.txt && \
    /opt/kestrel-venv/bin/python -m pip check && \
    /opt/kestrel-venv/bin/python - <<'PY'
from pathlib import Path

import aiofiles
import datasets
import mlflow
import peft
import pyarrow.dataset
import qwen_vl_utils
import torch
import torchvision
import transformers

assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torchvision.__version__.startswith("0.23.0"), torchvision.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert "/usr/local/lib/python3.12/dist-packages" in str(Path(torch.__file__)), torch.__file__
assert "kestrel-venv" in str(Path(mlflow.__file__)), mlflow.__file__
print(
    {
        "ready": True,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_file": torch.__file__,
        "cuda_runtime": torch.version.cuda,
        "mlflow": mlflow.__version__,
        "mlflow_file": mlflow.__file__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
    }
)
PY

RUN set -eux; \
    mkdir -p /tmp/mlflow-artifacts; \
    mlflow server \
      --backend-store-uri sqlite:////tmp/mlflow-smoke.db \
      --default-artifact-root file:///tmp/mlflow-artifacts \
      --host 127.0.0.1 \
      --port 5000 \
      --allowed-hosts "*" \
      --cors-allowed-origins "*" \
      --x-frame-options NONE \
      --disable-security-middleware \
      > /tmp/mlflow-smoke.log 2>&1 & \
    pid="$!"; \
    for i in $(seq 1 60); do \
      if python -c 'import urllib.request; data = urllib.request.urlopen("http://127.0.0.1:5000/", timeout=2).read(4096); assert b"Unable to display MLflow UI" not in data, data[:200]; assert b"<!doctype html>" in data.lower(), data[:200]'; \
      then \
        kill "$pid"; \
        wait "$pid" 2>/dev/null || true; \
        rm -rf /tmp/mlflow-smoke.db /tmp/mlflow-artifacts /tmp/mlflow-smoke.log; \
        exit 0; \
      fi; \
      sleep 2; \
    done; \
    cat /tmp/mlflow-smoke.log; \
    kill "$pid" 2>/dev/null || true; \
    exit 1

CMD ["sleep", "infinity"]
