"""Model registry, cost estimates, and the spend cap."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from margin.cache import Cache

DEFAULT_CONFIG = Path("configs/models.yaml")
CHARS_PER_TOKEN = 3.5
"""Deliberately low (real English is ~4), so estimates err high."""


class BudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str | None
    price_in: float | None
    price_out: float | None
    concurrency: int = 8
    price_cache_read: float | None = None
    """$/M for cache hits; None = billed at price_in."""
    price_cache_write: float | None = None
    """$/M for cache writes (Anthropic); None = billed at price_in."""
    reasoning_allowance: int = 0
    """Expected hidden reasoning tokens per call, added to output in cost estimates."""
    params: dict = field(default_factory=dict)
    """Provider request parameters fixed for this model (reasoning effort, thinking mode, ...)."""

    def cost(
        self, input_tokens: int | None, output_tokens: int | None, cache_read: int | None = 0, cache_write: int | None = 0
    ) -> float:
        """`input_tokens` excludes cached tokens, which are passed separately."""
        if self.price_in is None or self.price_out is None:
            raise BudgetError(f"model {self.name!r} has no price in the registry; verify and fill it in first")
        read = self.price_in if self.price_cache_read is None else self.price_cache_read
        write = self.price_in if self.price_cache_write is None else self.price_cache_write
        return (
            (input_tokens or 0) * self.price_in
            + (cache_read or 0) * read
            + (cache_write or 0) * write
            + (output_tokens or 0) * self.price_out
        ) / 1e6


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelSpec]
    cap_usd: float
    per_run_cap_usd: float

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Registry:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        models = {name: ModelSpec(name=name, **spec) for name, spec in raw["models"].items()}
        budget = raw.get("budget", {})
        return cls(models, float(budget.get("cap_usd", 100.0)), float(budget.get("per_run_cap_usd", 40.0)))

    def __getitem__(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(f"unknown model {name!r}; known: {sorted(self.models)}")
        spec = self.models[name]
        if spec.model is None:
            raise BudgetError(f"model {name!r} has no model id in the registry yet")
        return spec


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def check_cap(registry: Registry, cache: Cache, estimated_usd: float) -> None:
    """Raise before a run whose estimate would break the per-run or global cap."""
    if estimated_usd > registry.per_run_cap_usd:
        raise BudgetError(f"run estimate ${estimated_usd:.2f} exceeds per-run cap ${registry.per_run_cap_usd:.2f}")
    spent = cache.spent()
    if spent + estimated_usd > registry.cap_usd:
        raise BudgetError(
            f"spent ${spent:.2f} + estimate ${estimated_usd:.2f} would exceed global cap ${registry.cap_usd:.2f}"
        )
