#!/usr/bin/env bash
set -euo pipefail

root="${KESTREL_ROOT:-/workspace/kestrel}"
port="${MLFLOW_PORT:-5000}"
backend_store_uri="${MLFLOW_BACKEND_STORE_URI:-sqlite:////workspace/kestrel/artifacts/mlflow/mlflow.db}"
artifact_root="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-file:///workspace/kestrel/artifacts/mlflow/artifacts}"

mkdir -p "${root}/artifacts/mlflow" "${root}/artifacts/train_runs"

exec mlflow server \
  --backend-store-uri "${backend_store_uri}" \
  --default-artifact-root "${artifact_root}" \
  --host 0.0.0.0 \
  --port "${port}" \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  --x-frame-options NONE \
  --disable-security-middleware
