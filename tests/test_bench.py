from pathlib import Path

from inference.contracts import InferenceRequest, InferenceResult
from training.datasets.contracts import EvalSample

from bench.bench import request_from_sample, run_bench, score_prediction


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


def test_run_bench_streams_loader_batches_with_fake_model(monkeypatch, tmp_path: Path):
    FakeLoader.calls = []
    FakeLoader.batch_calls = []
    monkeypatch.setattr("bench.bench.DatasetLoader", FakeLoader)
    model = FakeModel({"s1": "1000", "s2": "yes"})
    output_path = tmp_path / "records.jsonl"

    summary = run_bench(
        model,
        dataset="chartqa",
        split="test",
        samples=2,
        batch_size=1,
        output_path=output_path,
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
    assert FakeLoader.batch_calls == [{"offset": 0, "limit": 2}]
    assert [request.sample_id for request in model.requests] == ["s1", "s2"]
    assert output_path.read_text().count("\n") == 2


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
