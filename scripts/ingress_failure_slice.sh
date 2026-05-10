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

SLICE_NAME="${1:-}"
LOCAL_EVAL_ROOT="${2:-artifacts/evals}"

if [[ -z "${SLICE_NAME}" ]]; then
  echo "Usage: scripts/ingress_failure_slice.sh <failure_slice_name> [local_eval_root]" >&2
  exit 2
fi

S3_ROOT="${S3_URI%/}"
S3_PATH="${S3_ROOT#s3://}"
S3_BUCKET="${S3_PATH%%/*}"
S3_BUCKET_URI="s3://${S3_BUCKET}"
LOCAL_SLICE_DIR="${LOCAL_EVAL_ROOT}/failures/${SLICE_NAME}"
EXPECTED_S3_PREFIX="${S3_ROOT}/artifacts/evals/failures/${SLICE_NAME}/"
SEARCH_ROOT="${S3_ROOT}/artifacts/evals/failures/"

has_failure_slice() {
  [[ -f "${1}/manifest.json" && -f "${1}/failures.jsonl" ]]
}

if has_failure_slice "${LOCAL_SLICE_DIR}"; then
  echo "${LOCAL_SLICE_DIR}"
  exit 0
fi

mkdir -p "${LOCAL_SLICE_DIR}"

if aws s3 sync "${EXPECTED_S3_PREFIX}" "${LOCAL_SLICE_DIR}/" --only-show-errors; then
  if has_failure_slice "${LOCAL_SLICE_DIR}"; then
    echo "${LOCAL_SLICE_DIR}"
    exit 0
  fi
fi

FOUND_MANIFEST="$(
  aws s3 ls "${SEARCH_ROOT}" --recursive \
    | awk '{print $4}' \
    | grep -E "/${SLICE_NAME}/manifest[.]json$|(^|/)${SLICE_NAME}/manifest[.]json$" \
    | head -n 1 || true
)"

if [[ -n "${FOUND_MANIFEST}" ]]; then
  FOUND_PREFIX="${FOUND_MANIFEST%/manifest.json}/"
  FOUND_S3_PREFIX="${S3_BUCKET_URI}/${FOUND_PREFIX}"

  aws s3 sync "${FOUND_S3_PREFIX}" "${LOCAL_SLICE_DIR}/" --only-show-errors
  if has_failure_slice "${LOCAL_SLICE_DIR}"; then
    echo "${LOCAL_SLICE_DIR}"
    exit 0
  fi
fi

echo "Failure slice not found locally or in S3: ${SLICE_NAME}" >&2
echo "Checked: ${LOCAL_SLICE_DIR}" >&2
echo "Checked: ${EXPECTED_S3_PREFIX}" >&2
echo "Searched: ${SEARCH_ROOT}" >&2
exit 1
