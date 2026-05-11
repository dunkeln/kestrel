from pathlib import Path
from io import BytesIO
import tarfile
import zipfile

import click
import pytest
from PIL import Image

from inference.contracts import InferenceRequest, InferenceResult
from training.datasets.contracts import EvalSample
from training.datasets import images as dataset_images

from bench.bench import (
    MissingSampleImageError,
    _resolve_sample_limit,
    request_from_sample,
    resolve_sample_image,
    run_bench,
    score_prediction,
)


class FakeModel:
    def __init__(self, predictions: dict[str, str]):
        self.predictions = predictions
        self.requests: list[InferenceRequest] = []

    def predict(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        return InferenceResult(
            sample_id=request.sample_id,
            prediction=self.predictions[request.sample_id],
            metadata={"latency_ms": 10},
        )


class FakeLoader:
    calls = []
    batch_calls = []

    def __init__(self, dataset_name, split=None, streaming=True):
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.calls.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "streaming": streaming,
            }
        )

    def batch(self, offset=0, limit=None):
        self.batch_calls.append({"offset": offset, "limit": limit})
        samples = [
            EvalSample(
                id="s1",
                dataset="chartqa",
                image="chart-1.png",
                question="How many bars?",
                answer="1,000",
                answer_type="numeric",
                supervision="benchmark",
                task_type="value_extraction",
            ),
            EvalSample(
                id="s2",
                dataset="chartqa",
                image="chart-2.png",
                question="Is this increasing?",
                answer="true",
                answer_type="yes_no",
                supervision="benchmark",
                task_type="yes_no",
            ),
        ]
        selected = samples[offset:]
        if limit is not None:
            selected = selected[:limit]
        yield from selected


class ExpandingFakeLoader(FakeLoader):
    def batch(self, offset=0, limit=None):
        self.batch_calls.append({"offset": offset, "limit": limit})
        yield EvalSample(
            id="s1",
            dataset="figureqa",
            image="chart-1.png",
            question="Is A higher?",
            answer="true",
            answer_type="yes_no",
            supervision="qa",
            task_type="yes_no",
        )
        yield EvalSample(
            id="s2",
            dataset="figureqa",
            image="chart-1.png",
            question="Is B higher?",
            answer="false",
            answer_type="yes_no",
            supervision="qa",
            task_type="yes_no",
        )
        yield EvalSample(
            id="s3",
            dataset="figureqa",
            image="chart-2.png",
            question="Is C higher?",
            answer="true",
            answer_type="yes_no",
            supervision="qa",
            task_type="yes_no",
        )


class FakeReporter:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.records = []

    def start(self):
        self.started = True

    def advance(self, record):
        self.records.append(record)

    def stop(self):
        self.stopped = True


def test_request_from_sample_uses_sample_specific_system_prompt():
    sample = EvalSample(
        id="s1",
        dataset="chartqa",
        image="chart.png",
        question="How many bars?",
        answer="3",
        answer_type="numeric",
        supervision="benchmark",
        task_type="value_extraction",
    )

    request = request_from_sample(sample)

    assert request.sample_id == "s1"
    assert request.image == "chart.png"
    assert request.prompt == "How many bars?"
    assert request.system_prompt == "Read the chart value requested. Return only the value."
    assert request.metadata["answer_type"] == "numeric"
    assert request.metadata["task_type"] == "value_extraction"


def test_resolve_sample_image_finds_dataset_artifact_path(tmp_path: Path):
    image_path = tmp_path / "chartbench" / "data" / "test" / "area" / "chart.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("not an actual image")
    sample = EvalSample(
        id="chartbench:1",
        dataset="chartbench",
        image="./data/test/area/chart.png",
        question="Is this area?",
        answer="yes",
        answer_type="yes_no",
        supervision="benchmark",
        metadata={"image_ref": "./data/test/area/chart.png"},
    )

    assert resolve_sample_image(sample, dataset_root=tmp_path) == str(image_path)


def test_resolve_sample_image_reads_chartbench_zip_without_extracting(tmp_path: Path):
    archive_path = tmp_path / "chartbench" / "data" / "test.zip"
    archive_path.parent.mkdir(parents=True)
    member = "data/test/area/area/chart_0/image.png"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, _png_bytes())
    sample = EvalSample(
        id="chartbench:1",
        dataset="chartbench",
        image="./data/test/area/area/chart_0/image.png",
        question="Is this area?",
        answer="yes",
        answer_type="yes_no",
        supervision="benchmark",
        metadata={
            "image_ref": "./data/test/area/area/chart_0/image.png",
            "image_archive": "data/test.zip",
        },
    )

    image = resolve_sample_image(sample, dataset_root=tmp_path)

    assert image.size == (2, 2)
    assert not (tmp_path / "chartbench" / member).exists()


def test_resolve_sample_image_reads_mmc_tar_without_extracting(tmp_path: Path):
    archive_path = (
        tmp_path
        / "mmc_benchmark"
        / "MMC-Benchmark"
        / "mmc_benchmark_images.tar.gz"
    )
    archive_path.parent.mkdir(parents=True)
    payload = _png_bytes()
    info = tarfile.TarInfo("image_benchmark-4000.png")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, BytesIO(payload))
    sample = EvalSample(
        id="mmc_benchmark:1",
        dataset="mmc_benchmark",
        image="image_benchmark-4000.png",
        question="Is this true?",
        answer="true",
        answer_type="yes_no",
        supervision="benchmark",
        metadata={
            "image_ref": "image_benchmark-4000.png",
            "image_archive": "MMC-Benchmark/mmc_benchmark_images.tar.gz",
        },
    )

    image = resolve_sample_image(sample, dataset_root=tmp_path)

    assert image.size == (2, 2)
    assert not (tmp_path / "mmc_benchmark" / "image_benchmark-4000.png").exists()


def test_resolve_sample_image_reports_missing_archive_path(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(dataset_images, "_download_archive", lambda *args: None)
    sample = EvalSample(
        id="mmc_benchmark:1",
        dataset="mmc_benchmark",
        image="image_benchmark-4000.png",
        question="Is this true?",
        answer="true",
        answer_type="yes_no",
        supervision="benchmark",
        metadata={
            "image_ref": "image_benchmark-4000.png",
            "image_archive": "MMC-Benchmark/mmc_benchmark_images.tar.gz",
        },
    )

    with pytest.raises(MissingSampleImageError, match="Expected archive"):
        resolve_sample_image(sample, dataset_root=tmp_path)


def test_run_bench_streams_loader_batches_with_fake_model(monkeypatch, tmp_path: Path):
    FakeLoader.calls = []
    FakeLoader.batch_calls = []
    monkeypatch.setattr("bench.bench.DatasetLoader", FakeLoader)
    model = FakeModel({"s1": "1000", "s2": "yes"})
    reporter = FakeReporter()
    output_path = tmp_path / "records.jsonl"

    summary = run_bench(
        model,
        dataset="chartqa",
        split="test",
        samples=2,
        batch_size=1,
        output_path=output_path,
        reporter=reporter,
    )

    assert summary.total == 2
    assert summary.correct == 2
    assert summary.accuracy == 1.0
    assert summary.average_latency_ms == 10
    assert FakeLoader.calls == [
        {
            "dataset_name": "chartqa",
            "split": "test",
            "streaming": True,
        }
    ]
    assert FakeLoader.batch_calls == [{"offset": 0, "limit": None}]
    assert [request.sample_id for request in model.requests] == ["s1", "s2"]
    assert output_path.read_text().count("\n") == 2
    assert reporter.started is True
    assert reporter.stopped is True
    assert [record.sample_id for record in reporter.records] == ["s1", "s2"]


def test_run_bench_all_samples_uses_unbounded_loader_limit(monkeypatch):
    FakeLoader.calls = []
    FakeLoader.batch_calls = []
    monkeypatch.setattr("bench.bench.DatasetLoader", FakeLoader)
    model = FakeModel({"s1": "1000", "s2": "yes"})

    summary = run_bench(
        model,
        dataset="chartqa",
        split="test",
        samples=None,
        batch_size=2,
    )

    assert summary.total == 2
    assert FakeLoader.batch_calls == [{"offset": 0, "limit": None}]


def test_run_bench_samples_limit_applies_to_normalized_samples(monkeypatch):
    ExpandingFakeLoader.calls = []
    ExpandingFakeLoader.batch_calls = []
    monkeypatch.setattr("bench.bench.DatasetLoader", ExpandingFakeLoader)
    model = FakeModel({"s1": "yes", "s2": "no", "s3": "yes"})

    summary = run_bench(
        model,
        dataset="figureqa",
        split="test",
        samples=2,
        batch_size=8,
    )

    assert summary.total == 2
    assert ExpandingFakeLoader.batch_calls == [{"offset": 0, "limit": None}]
    assert [request.sample_id for request in model.requests] == ["s1", "s2"]


def test_resolve_sample_limit_requires_exactly_one_mode():
    assert _resolve_sample_limit(samples=3, all_samples=False) == 3
    assert _resolve_sample_limit(samples=None, all_samples=True) is None

    with pytest.raises(click.UsageError, match="exactly one"):
        _resolve_sample_limit(samples=None, all_samples=False)

    with pytest.raises(click.UsageError, match="exactly one"):
        _resolve_sample_limit(samples=3, all_samples=True)

    with pytest.raises(click.UsageError, match="greater than 0"):
        _resolve_sample_limit(samples=0, all_samples=False)


def test_score_prediction_uses_typed_normalized_exact_match():
    assert score_prediction(
        answer_type="numeric",
        gold="1,000",
        prediction="1000",
    ).correct
    assert score_prediction(
        answer_type="yes_no",
        gold="false",
        prediction="No",
    ).correct
    assert score_prediction(
        answer_type="yes_no",
        gold="Yes.",
        prediction="yes",
    ).correct
    assert score_prediction(
        answer_type="yes_no",
        gold="No.",
        prediction="no",
    ).correct
    assert score_prediction(
        answer_type="text",
        gold="  Bar Chart  ",
        prediction="bar chart",
    ).correct
    assert score_prediction(
        answer_type="multiple_choice",
        gold="Option A",
        prediction="option a",
    ).correct
    assert score_prediction(
        answer_type="structure",
        gold="<s> value </s>",
        prediction="<S>  VALUE </S>",
    ).correct
    assert not score_prediction(
        answer_type="numeric",
        gold="12",
        prediction="13",
    ).correct


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()
