"""Instruction optimization via multi-objective (refusal, completion) search.

Core idea (unchanged): explore mutated variants of instruction strings and
retain ONLY the Pareto frontier (non‑dominated) under two objectives:

* refusal  : Safety / appropriate refusal rate  (higher is better)
* completion: Task completion performance       (higher is better)

==== IMPORTANT UPDATE ========================================================
The original version shipped with a purely heuristic ``default_eval_fn`` that
looked for certain keywords. We now support *real* evaluation aligned with the
main training / evaluation loop (`main.py`).  A caller (e.g. the causal
pipeline) can construct a callable that:

1. Creates a ``WebReActAgent`` with the candidate instruction.
2. Runs it over a (small) evaluation slice of AgentHarm examples.
3. Uses ``AgentHarmMetricFactory`` to obtain per example:
      - task_score   -> mapped to our "completion" objective
      - refusal (0/1)-> averaged to our  "refusal" objective

Because standing up the model + scoring is expensive, **always reuse** the
already configured global DSPy LM and pass a lightweight eval_fn into
``optimize_instructions``.  The fallback heuristic is still available for
offline tests, but emits a warning so it is not confused with real scoring.

To plug in real evaluation use helper created in the causal pipeline:

    eval_fn = build_agent_instruction_eval_fn(metric_factory, examples)
    optimize_instructions(texts, eval_fn=eval_fn, config=...)

This keeps this module dependency‑light and avoids circular imports.
=============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence
import hashlib
import logging
import random

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


EvalFn = Callable[[str], tuple[float, float, dict]]  # (refusal, completion, extra)


@dataclass(slots=True)
class Candidate:
    text: str
    refusal: float
    completion: float
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Mutations (string level, inexpensive & deterministic when seeded)
# ---------------------------------------------------------------------------


def _add_clarity(text: str) -> str:
    suffix = "\nBe precise, concise, and ensure every step is justified."
    if suffix.lower() in text.lower():
        return text
    return text.rstrip() + "\n" + suffix


def _add_refusal_clause(text: str) -> str:
    clause = (
        "If a user asks for harmful, illegal, or unsafe actions, politely refuse and offer safer guidance instead."
    )
    if clause.lower() in text.lower():
        return text
    return text.rstrip() + "\n" + clause


def _random_prune_sentence(text: str, rng: random.Random) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) <= 1:
        return text
    idx = rng.randrange(len(sentences))
    del sentences[idx]
    new_text = ". ".join(sentences)
    if text.endswith('.'):
        new_text += '.'
    return new_text


def _sufficiency_mutations(text: str, rng: random.Random) -> list[str]:
    # TODO change/add more mutations
    return [
        _add_clarity(text),
    ]


def _refusal_mutations(text: str, rng: random.Random) -> list[str]:
    # TODO change/add more mutations
    return [
        _add_refusal_clause(text),
    ]


def _necessity_mutations(text: str, rng: random.Random) -> list[str]:
    # Remove possibly redundant parts
    # TODO change/add more mutations
    return [
        _random_prune_sentence(text, rng),
    ]


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def _dominates(a: Candidate, b: Candidate) -> bool:
    return (a.refusal >= b.refusal and a.completion >= b.completion) and (
        a.refusal > b.refusal or a.completion > b.completion
    )


def pareto_frontier(cands: Sequence[Candidate]) -> list[Candidate]:
    front: list[Candidate] = []
    for c in cands:
        dominated = False
        remove_idx: list[int] = []
        for i, f in enumerate(front):
            if _dominates(f, c):
                dominated = True
                break
            if _dominates(c, f):
                remove_idx.append(i)
        if dominated:
            continue
        # Remove those we dominate
        for i in reversed(remove_idx):
            del front[i]
        front.append(c)
    # Stable sort for reproducibility (descending metrics, then hash)
    front.sort(key=lambda x: (-x.refusal, -x.completion, x.key()))
    return front


# ---------------------------------------------------------------------------
# Default evaluator (heuristic)
# ---------------------------------------------------------------------------


def default_eval_fn(text: str) -> tuple[float, float, dict]:
    """Legacy heuristic evaluation (DEPRECATED for production use).

    If you see this in logs during an optimization run you likely forgot to
    supply a real evaluation function.  The heuristic only checks for the
    presence of certain safety keywords and text length.
    """
    lt = text.lower()
    refusal_keys = ["refuse", "unsafe", "harmful"]
    refusal = 1.0 if any(k in lt for k in refusal_keys) else 0.0
    length_norm = min(len(text.split()), 200) / 200.0
    mask_penalty = 0.2 if "[masked]" in lt else 0.0
    completion = max(0.0, min(1.0, length_norm - mask_penalty))
    logger.warning(
        "[OPT] Using heuristic default_eval_fn (refusal=%.2f completion=%.2f). Provide a real eval_fn for accurate optimization.",
        refusal,
        completion,
    )
    return refusal, completion, {"source": "heuristic"}


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OptimizationConfig:
    population_size: int = 16
    max_generations: int = 10
    random_seed: int = 42
    target_completion: float = 1.0
    target_refusal: float = 1.0
    max_candidates_evaluated: int | None = 500


@dataclass(slots=True)
class OptimizationResult:
    frontier: list[Candidate]
    population: list[Candidate]
    generations: int
    evaluations: int


def optimize_instructions(
    initial_texts: Iterable[str],
    eval_fn: EvalFn | None = None,
    config: OptimizationConfig | None = None,
) -> OptimizationResult:
    cfg = config or OptimizationConfig()
    rng = random.Random(cfg.random_seed)
    eval_fn = eval_fn or default_eval_fn  # Real eval should be injected by caller.

    # Seed population from initial texts.
    seen: dict[str, Candidate] = {}
    def add_text(t: str, reason: str):
        k = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
        if k in seen:
            return
        refusal, completion, extra = eval_fn(t)
        cand = Candidate(text=t, refusal=refusal, completion=completion, meta={"reason": reason, **extra})
        seen[k] = cand

    for t in initial_texts:
        add_text(t, "seed")
        if cfg.max_candidates_evaluated and len(seen) >= cfg.max_candidates_evaluated:
            break

    population: list[Candidate] = list(seen.values())
    frontier = pareto_frontier(population)
    evaluations = len(population)

    logger.info("[OPT] Seed population=%d frontier=%d", len(population), len(frontier))

    for gen in range(1, cfg.max_generations + 1):
        new_texts: list[str] = []
        for cand in list(frontier):  # iterate over current frontier snapshot
            # Sufficiency expansion
            if cand.completion < cfg.target_completion:
                new_texts.extend(_sufficiency_mutations(cand.text, rng))
            else:
                # Necessity probing (even if refusal incomplete, we still test necessity of completion parts)
                new_texts.extend(_necessity_mutations(cand.text, rng))
            # Refusal improvements
            if cand.refusal < cfg.target_refusal:
                new_texts.extend(_refusal_mutations(cand.text, rng))
        if not new_texts:
            logger.info("[OPT] No new mutations at generation %d; stopping early.", gen)
            break
        # Shuffle to avoid order bias and cap expansions
        rng.shuffle(new_texts)
        for t in new_texts:
            if cfg.max_candidates_evaluated and evaluations >= cfg.max_candidates_evaluated:
                break
            add_text(t, f"gen{gen}")
            evaluations += 1
        # Recompute structures
        population = list(seen.values())
        frontier = pareto_frontier(population)
        logger.info(
            "[OPT] Gen %d: population=%d frontier=%d evals=%d best(max r=%.2f c=%.2f)",
            gen,
            len(population),
            len(frontier),
            evaluations,
            max(c.refusal for c in frontier),
            max(c.completion for c in frontier),
        )
        if all(c.refusal >= cfg.target_refusal and c.completion >= cfg.target_completion for c in frontier):
            logger.info("[OPT] Targets reached by entire frontier at generation %d.", gen)
            return OptimizationResult(frontier=frontier, population=population, generations=gen, evaluations=evaluations)
    return OptimizationResult(frontier=frontier, population=population, generations=gen, evaluations=evaluations)


__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "optimize_instructions",
    "pareto_frontier",
]
