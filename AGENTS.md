# Kestrel Agent Instructions

## Project Shape

Kestrel is a uv-managed Python project for chart-reasoning VLM training, evaluation, and failure-slice replay.

Keep source code under:

- `training/` for model, dataset, adapter, and eval runtime code.
- `training/datasets/` for dataset contracts, loaders, adapters, and replay loaders.
- `training/evals/` for evaluation artifacts, failure egress, replay, metrics, and eval orchestration.
- `training/adapters/` for PEFT/LoRA construction, loading, and saving.
- `tests/` for root-level tests.

Keep generated runtime outputs under:

- `artifacts/pretrained/` for downloaded base model snapshots.
- `artifacts/adapters/` for PEFT adapter outputs.
- `artifacts/evals/images/` for eval image cache.
- `artifacts/evals/failures/{user_name}_{timestamp}/` for failure JSONL slices.

Do not put generated weights, datasets, eval outputs, or downloaded image archives under source folders.

## Python And uv

- Use `uv` for dependency and execution workflows.
- Do not manually edit dependency lists in `pyproject.toml`; use `uv add`, `uv remove`, or `uv lock`.
- Run Python checks through `uv run`.
- Do not use print debugging in production paths.
- Long-running processes must emit structured logs.
- Prefer small dataclasses and typed functions over broad abstractions.

## Dataset Contract

The canonical runtime sample is `training.datasets.contracts.EvalSample`.

Dataset loaders must normalize raw rows into this contract:

- `id`
- `dataset`
- `image`
- `question`
- `answer`
- `answer_type`
- `supervision`
- `chart_type`
- `task_type`
- `metadata`

Rules:

- Keep `EvalSample.answer` as the gold answer.
- Store predictions and error metadata outside `EvalSample`, usually in eval/failure records.
- Do not store PIL images in JSONL. Store relative `image_ref` values.
- Use `artifacts/evals` as the root for eval-local relative paths.
- Preserve stable IDs. Do not use Python object ids for persistent sample IDs.
- Windowing may operate on raw dataset rows; document when adapters expand one raw row into multiple `EvalSample`s.

## Dataset Loader API

Use `DatasetLoader` from `training/datasets/loaders.py`.

Preferred usage:

```python
from training.datasets.loaders import DatasetLoader

loader = DatasetLoader("chartqa", streaming=True)
samples = loader.batch(offset=0, limit=32)
```

Do not reintroduce older public loader helpers unless there is a clear need.

Dataset-specific raw quirks should stay inside private adapter functions in `loaders.py` or focused adjacent modules.

## Failure Egress And Replay

Failure ingress/egress uses `training/evals/failures.py`.

Canonical flow:

```text
EvalSample -> FailureRecord -> failures.jsonl -> FailureSliceLoader -> EvalSample
```

Failure batch output shape:

```text
artifacts/evals/
  images/
  failures/
    {user_assigned_name}_{timestamp}/
      manifest.json
      failures.jsonl
```

Rules:

- Failure folders must be timestamped.
- Keep user-assigned failure batch names as provided.
- JSONL rows should be portable and inspectable.
- Use relative image refs such as `images/chartqa/chartqa_42.png`.
- Do not use pickle for durable eval or failure artifacts.
- `manifest.json` should include at least schema version, original name, batch name, creation timestamp, and count.

## Models And PEFT

- Base model snapshots go under `artifacts/pretrained/`.
- PEFT adapters go under `artifacts/adapters/`.
- PEFT implementation code belongs in `training/adapters/`, not eval scripts.
- Eval scripts may call PEFT helpers but should not define adapter logic.
- Local MPS/CPU paths should not require CUDA-only dependencies.
- Only request `flash_attention_2` when running on CUDA and `flash-attn` is actually available.

## Tests

- Root-level tests belong in `tests/`.
- Prefer synthetic/local tests over network-dependent tests.
- Use `tmp_path` for filesystem artifact tests.
- Verify targeted changes with `uv run pytest <test path>`.
- Do not modify unrelated tests.

## Permissions

### Allowed Without Prompting

- Read files and inspect project structure.
- Edit source files under `training/` when directly needed for the requested task.
- Edit docs such as `README.md`, `GUIDE.md`, and `AGENTS.md` when directly requested or when documenting a user-facing API change.
- Add or update focused tests under `tests/` when changing contracts, dataset adapters, failure egress/replay, or other shared behavior.
- Run deterministic local checks such as `uv run python -m py_compile ...` and targeted `uv run pytest <test path>`.
- Use `uv add`, `uv remove`, and `uv lock` for dependency changes when the user asks for a package or dependency behavior change.

### Strictly Require Human Approval

- Create and/or modify unrelated tests or broad test infrastructure.
- Delete, rename, or move user-created files outside the requested scope.
- Run destructive commands such as `rm`, `git reset`, or `git checkout --`.
- Push commits, create branches, open PRs, or modify remote resources.
- Install CUDA/GPU-specific dependencies that require compilation or external system tooling.
- Trigger large model or dataset downloads unless the user explicitly asks for that action.
- Sync, upload, or delete external storage such as S3, RunPod volumes, Hugging Face repos, or cloud buckets.

## README And GUIDE

- Keep `README.md` updated when user-facing contracts, artifact layouts, or usage APIs change.
- Maintain a `GUIDE.md` alongside `README.md` for broader project operation notes unless the project becomes a marimo-only project.

## Git And Files

- The worktree may be dirty. Do not revert unrelated changes.
- Respect `.gitignore`; generated artifacts, local env files, caches, and downloaded models should stay untracked.
- Avoid destructive git commands.
- Keep edits minimal and explicit.

> [!IMPORTANT]
> Log learnt failures or successful steps from runpod into `RUNPOD_GUIDE.md` and reference it on consecutive runs in runpod.
