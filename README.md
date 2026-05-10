# Kestrel

Judge-VLM capable of judging other VLMs' outputs on chart reasoning.


## Benchmarks & Datasets

+ ChartQA — 20K+ charts, question-answer pairs, strong ground truth
+ PlotQA — 28M synthetic QA pairs on plots
+ FigureQA — yes/no questions on scientific figures
+ ChartBench — specifically for chart understanding evaluation
+ MMC-Benchmark — multi-task chart comprehension

### Dataset Contract

Dataset rows are normalized into `EvalSample` objects from `training/datasets/contracts.py`.

```python
EvalSample(
    id="chartqa:0",
    dataset="chartqa",
    image=<PIL image or image reference>,
    question="How many food item is shown in the bar graph?",
    answer="14",
    answer_type="numeric",
    supervision="benchmark",
    chart_type=None,
    task_type="value_extraction",
    tag="real",
    metadata={"source_type": "human_test"},
)
```

Common fields:

+ `image` — PIL image when the dataset row provides image bytes; string reference when the dataset only provides archived image paths.
+ `answer_type` — one of `yes_no`, `numeric`, `text`, `multiple_choice`, `structure`.
+ `supervision` — one of `qa`, `classification`, `structure`, `benchmark`.
+ `tag` — `real` for benchmark/source datasets, `synthetic` for generated samples.
+ `chart_type` — chart family when provided by the dataset.
+ `task_type` — dataset-provided or inferred task label such as `value_extraction`, `arithmetic`, `yes_no`, `analysis`, or `structure_extraction`.

### Normalized Examples

ChartQA becomes standard benchmark QA:

```python
{
    "question": "What is the difference in value between Lamb and Corn?",
    "answer": "0.57",
    "answer_type": "numeric",
    "supervision": "benchmark",
    "task_type": "arithmetic",
    "image_type": "PngImageFile",
}
```

FigureQA becomes multiple yes/no QA samples per image:

```python
{
    "question": "Is Pale Green the minimum?",
    "answer": "No.",
    "answer_type": "yes_no",
    "supervision": "qa",
    "task_type": "yes_no",
    "image_type": "PngImageFile",
}
```

ChartBench keeps benchmark metadata:

```python
{
    "question": "This graph is a area chart, instead of pie chart.",
    "answer": "Yes",
    "answer_type": "yes_no",
    "supervision": "benchmark",
    "chart_type": "area",
    "task_type": "CR",
    "metadata": {
        "image_ref": "./data/train/area/area/chart_0/image.png",
        "image_archive": "data/train.zip",
    },
}
```

MMC-Benchmark uses instruction/label rows:

```python
{
    "question": "' There was a significant increase in something specific in 2014.' Please answer whether the description is true or false according to the image.",
    "answer": "true",
    "answer_type": "yes_no",
    "supervision": "benchmark",
    "task_type": "analysis",
    "metadata": {
        "image_ref": "image_benchmark-4000.png",
        "image_archive": "MMC-Benchmark/mmc_benchmark_images.tar.gz",
    },
}
```

PlotQA is structure supervision, not normal QA:

```python
{
    "question": "Extract the chart structure.",
    "answer": "<s><s_y>87.4244<sep/>87.75354...</s_y><s_x>0<sep/>1...</s_x><s_name>Age 65(female)</s_name>...</s>",
    "answer_type": "structure",
    "supervision": "structure",
    "task_type": "structure_extraction",
}
```

Use the loader API for batched, no-store reads:

```python
from training.datasets.loaders import DatasetLoader

loader = DatasetLoader("chartqa", streaming=True)
batch = list(loader.batch(offset=0, limit=32))
```

## Model

model card: `qwen2.5 2B/3B/7B`
