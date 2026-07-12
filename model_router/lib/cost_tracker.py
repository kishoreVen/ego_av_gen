
"""Thread-local cost accumulator for tracking LLM costs within a pipeline step.

This module provides a context manager pattern for accumulating costs from
LLM calls made during a pipeline step. The ModelRouter automatically registers
costs with the active accumulator (if any).

Usage:
    from model_router.lib.cost_tracker import track_step_costs

    with track_step_costs("generate_concept") as accumulator:
        # ... make LLM calls via ModelRouter ...
    # accumulator now has total input_tokens, output_tokens, cost_usd
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


@dataclass
class StepCostAccumulator:
    """Accumulates costs for a single pipeline step."""

    step_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0

    def add(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Add cost from an LLM call to this step's total."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost
        self.call_count += 1


_thread_local = threading.local()


def get_current_accumulator() -> StepCostAccumulator | None:
    """Return the active StepCostAccumulator, or None if not inside a tracking context."""
    return getattr(_thread_local, "accumulator", None)


@contextmanager
def track_step_costs(step_name: str) -> Generator[StepCostAccumulator, None, None]:
    """Context manager to track all LLM costs within a named step.

    Any LLM calls made via ModelRouter while inside this context will
    automatically have their costs registered with the accumulator.

    Args:
        step_name: Name of the pipeline step (e.g., "generate_concept")

    Yields:
        StepCostAccumulator that collects costs from all LLM calls.
    """
    accumulator = StepCostAccumulator(step_name=step_name)
    _thread_local.accumulator = accumulator
    try:
        yield accumulator
    finally:
        _thread_local.accumulator = None
