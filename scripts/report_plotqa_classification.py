import json
import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.plotqa_report import (
    classify_plotqa_record,
    render_plotqa_classification_report,
)
from bench.scorers import score_prediction


@click.command()
@click.argument(
    "jsonl_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
)
def main(jsonl_path: Path) -> None:
    """Print a PlotQA classification report from a bench JSONL file."""
    records = _read_records(jsonl_path)
    classified = []
    for record in records:
        score = score_prediction(
            answer_type="structure",
            gold=record["gold"],
            prediction=record["prediction"],
        )
        classified.append(
            classify_plotqa_record(
                score=score.score,
                metadata=score.metadata,
                prediction=str(record["prediction"]),
            )
        )

    render_plotqa_classification_report(
        console=Console(),
        records=classified,
        title=str(jsonl_path),
    )


def _read_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise click.ClickException(f"No bench records found in {path}.")

    datasets = {record["dataset"] for record in records}
    if datasets != {"plotqa"}:
        raise click.ClickException(f"Expected only PlotQA records in {path}; found {sorted(datasets)}.")
    return records


if __name__ == "__main__":
    main()
