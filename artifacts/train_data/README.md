# First-Pass SFT Train Data

This directory contains the first-pass constitutional chart-reasoning SFT artifacts generated from streamed benchmark samples.

Use the `*_train.jsonl` files as the primary SFT rows for this pass. Keep `*_contrastive_train.jsonl` separate unless the training objective explicitly consumes contrastive examples. Use `*_skipped_train.jsonl` for diagnostics only.

## Dataset Sizes

| dataset | gold SFT rows | contrastive rows | skipped rows |
|---|---:|---:|---:|
| chartbench | 440 | 91 | 72 |
| chartqa | 607 | 118 | 33 |
| figureqa | 613 | 0 | 27 |
| mmc_benchmark | 489 | 121 | 14 |
| plotqa_qa | 976 | 17 | 854 |

## Totals

| split | rows |
|---|---:|
| gold SFT | 3,125 |
| contrastive | 347 |
| skipped diagnostics | 1,000 |

## Notes

- `record_key` is the canonical stable key for gold rows.
- PlotQA numeric rows may include `plotqa_numeric_adjudicated: true` when recovered through the scoped adjudication path.
- Images are content-addressed under `images/` and referenced by JSONL `image` / `image_ref` fields.
- This is the first-pass SFT set; sizes reflect the current accepted rows, not the full source dataset sizes.
