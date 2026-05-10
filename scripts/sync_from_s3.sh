#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

if [[ -z "${S3_URI:-}" ]]; then
  echo "S3_URI is not set. Add it to .env or export it before running." >&2
  exit 2
fi

S3_ROOT="${S3_URI%/}"
LOCAL_ROOT="${1:-artifacts}"

aws s3 sync "${S3_ROOT}/artifacts/" "${LOCAL_ROOT}/"
