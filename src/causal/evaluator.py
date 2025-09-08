"""Evaluator for original and intervention prompts.

Responsible for scoring prompts by invoking the model or using cached scores.
We keep the interface minimal so we can later plug in more nuanced scoring
like refusal probability, semantic alignment, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvalResult:
    prompt_id: str
    variant_kind: str  # 'original' or intervention kind
    score: float | None
    raw: dict | None


def evaluate_variants(
    prompt_id: str,
    base_prompt: str,
    variants: Mapping[str, str],
    scorer: Callable[[str], tuple[float | None, dict | None]],
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for kind, text in variants.items():
        try:
            score, raw = scorer(text)
        except Exception as e:
            logger.warning("Scoring failed for %s:%s (%s)", prompt_id, kind, e)
            score, raw = None, None
        results.append(EvalResult(prompt_id=prompt_id, variant_kind=kind, score=score, raw=raw))
    return results
