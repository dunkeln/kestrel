from __future__ import annotations

import random
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from training.datasets.contracts import EvalSample
from training.datasets.loaders import DatasetLoader


LoaderFactory = Callable[..., DatasetLoader]
LOG_EVERY_BATCHES = 10
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamConfig:
    dataset: str
    split: str
    batch_size: int
    samples: int | None = None
    all_samples: bool = False
    shuffle_seed: int | None = 0
    shuffle_buffer_size: int = 1024


def iter_sample_batches(
    config: StreamConfig,
    *,
    loader_factory: LoaderFactory = DatasetLoader,
) -> Iterator[list[EvalSample]]:
    """Stream EvalSample batches from DatasetLoader with optional buffer shuffle."""
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config.samples is not None and config.samples < 0:
        raise ValueError("samples must be non-negative")
    if config.samples is not None and config.all_samples:
        raise ValueError("cannot set both samples and all_samples")
    if config.samples is None and not config.all_samples:
        raise ValueError("set samples or all_samples")

    _log_event(
        "sample_stream_start",
        dataset=config.dataset,
        split=config.split,
        batch_size=config.batch_size,
        samples=config.samples,
        all_samples=config.all_samples,
        shuffle_seed=config.shuffle_seed,
        shuffle_buffer_size=config.shuffle_buffer_size,
    )
    loader = loader_factory(config.dataset, split=config.split, streaming=True)
    stream: Iterable[EvalSample] = loader.samples()
    stream = _shuffle_buffer(
        stream,
        seed=config.shuffle_seed,
        buffer_size=config.shuffle_buffer_size,
    )
    if config.samples is not None:
        stream = islice(stream, config.samples)

    total_samples = 0
    total_batches = 0
    try:
        batch: list[EvalSample] = []
        for sample in stream:
            total_samples += 1
            batch.append(sample)
            if len(batch) == config.batch_size:
                total_batches += 1
                _log_batch_progress(total_samples, total_batches)
                yield batch
                batch = []
        if batch:
            total_batches += 1
            _log_batch_progress(total_samples, total_batches)
            yield batch
    finally:
        _log_event(
            "sample_stream_complete",
            total_samples_processed=total_samples,
            total_batches_yielded=total_batches,
        )


def _log_batch_progress(total_samples: int, total_batches: int) -> None:
    if total_batches == 1 or total_batches % LOG_EVERY_BATCHES == 0:
        _log_event(
            "sample_stream_progress",
            total_samples_processed=total_samples,
            total_batches_yielded=total_batches,
        )


def _shuffle_buffer(
    samples: Iterable[EvalSample],
    *,
    seed: int | None,
    buffer_size: int,
) -> Iterator[EvalSample]:
    if seed is None or buffer_size <= 1:
        yield from samples
        return

    rng = random.Random(seed)
    buffer: list[EvalSample] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample

    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _log_event(event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))
