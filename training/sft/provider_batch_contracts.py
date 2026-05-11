from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderBatchRequest:
    custom_id: str
    image_b64: str
    prompt: str
    model: str


@dataclass(frozen=True)
class ProviderBatchResult:
    custom_id: str
    text: str | None
    error: str | None


__all__ = ["ProviderBatchRequest", "ProviderBatchResult"]
