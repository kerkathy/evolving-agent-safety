"""Causal analysis module.

Provides utilities to compute prompt-level causal sufficiency and necessity
scores by:

1. Collecting previously generated prompts / results from MLflow runs.
2. Creating intervention variants of each prompt (ablations or perturbations).
3. (Re-)evaluating model outputs on variants (with optional caching).
4. Aggregating scores into per-prompt causal effect estimates.

Lightweight public API:

    from .pipeline import run_causal_analysis

Configuration is optional. If the causal section is absent or disabled, the
rest of the project remains unaffected.
"""

from .pipeline import run_causal_analysis  # noqa: F401
from .optimization import optimize_instructions, OptimizationResult  # noqa: F401

__all__ = [
    "run_causal_analysis",
    "optimize_instructions",
    "OptimizationResult",
]
