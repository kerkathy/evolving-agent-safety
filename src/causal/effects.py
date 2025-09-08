"""Sufficiency and necessity effect computations.

Given evaluation results for original and intervention variants, compute
per-prompt causal metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable
import math

from .evaluator import EvalResult


@dataclass(slots=True)
class PromptEffects:
    prompt_id: str
    base_score: float | None
    sufficiency: float | None
    necessity: float | None
    n_variants: int


def _safe_mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_effects(results: Iterable[EvalResult]) -> list[PromptEffects]:
    by_prompt: dict[str, list[EvalResult]] = defaultdict(list)
    # Group results by prompt_id
    for r in results:
        by_prompt[r.prompt_id].append(r)
    out: list[PromptEffects] = []
    # For each prompt, compute sufficiency and necessity
    for pid, group in by_prompt.items():
        base = next((g for g in group if g.variant_kind == "original"), None)
        base_score = base.score if base else None
        ablated_scores = [g.score for g in group if g.variant_kind != "original"]
        mean_ablated = _safe_mean([s for s in ablated_scores if s is not None])
        if base_score is None or mean_ablated is None:
            suff = None
            nec = None
        else:
            suff = base_score - mean_ablated
            # Necessity: average positive drop relative to base
            drops = [base_score - s for s in ablated_scores if s is not None]
            drops = [d for d in drops if d > 0]
            nec = _safe_mean(drops)
        out.append(PromptEffects(
            prompt_id=pid,
            base_score=base_score,
            sufficiency=suff,
            necessity=nec,
            n_variants=len(group) - 1,
        ))
    return out
