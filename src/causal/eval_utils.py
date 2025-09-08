"""Utilities for constructing instruction evaluation functions used in
optimization (aligns with main.py evaluation path).

This isolates the heavier logic from `pipeline.py` keeping orchestration
readable. We avoid external numeric deps (numpy) by using pure Python stats.
"""
from __future__ import annotations

from typing import Iterable, List, Optional
import logging

from src.agent import WebReActAgent
from src.metrics import AgentHarmMetricFactory

logger = logging.getLogger(__name__)

try:  # Optional import; not strictly required at runtime until functions used
    from src.data import load_agentharm_data, build_dspy_examples
except Exception:  # pragma: no cover - defensive
    load_agentharm_data = None  # type: ignore
    build_dspy_examples = None  # type: ignore


def load_eval_examples(config, limit: int = 20):
    """Load a lightweight slice of AgentHarm examples for instruction eval.

    Falls back to empty list (triggering heuristic scoring) if anything fails.
    """
    if not (load_agentharm_data and build_dspy_examples):  # import failure
        logger.warning("[EVAL] Data loading modules unavailable; using heuristic eval only.")
        return []
    try:
        data_cfg = config.data
        raw = load_agentharm_data(
            behavior_ids=list(data_cfg.behavior_ids or []),
            sample_ids=(list(data_cfg.sample_ids or [])[:limit] or None),
            task_name=data_cfg.task_name,
            split=data_cfg.split,
            detailed_behaviors=data_cfg.detailed_behaviors,
            hint_included=data_cfg.hint_included,
            n_irrelevant_tools=data_cfg.n_irrelevant_tools,
        )
        examples = build_dspy_examples(raw)
        if len(examples) > limit:
            examples = examples[:limit]
        logger.info("[EVAL] Loaded %d eval examples (limit=%d)", len(examples), limit)
        return examples
    except Exception as e:  # pragma: no cover - runtime resilience
        logger.warning("[EVAL] Failed to load eval examples: %s", e, exc_info=True)
        return []


def build_agent_instruction_eval_fn(
    examples: Iterable,
    metric_factory: AgentHarmMetricFactory,
    agent: Optional[WebReActAgent] = None,
):
    """Return callable(instr) -> (refusal_mean, completion_mean, extra_meta).

    completion = mean(task_score)
    refusal    = mean(refusal_flag)   (0/1 per example)

    Improvements over prior version:
      * Reuses a single ``WebReActAgent`` instance (passed in or lazily created)
        and updates its instruction via ``update_instruction`` instead of
        constructing a brand new agent per candidate (performance win).
      * Backwards compatible: if no examples provided falls back to heuristic
        default evaluator from ``optimization`` module.
    """
    from .optimization import default_eval_fn  # local import to avoid cycle

    examples_list: List = list(examples)
    # Reuse provided agent or create one baseline instance once.
    agent_instance: WebReActAgent = agent or WebReActAgent()

    def _mean(values):
        return (sum(values) / len(values)) if values else 0.0

    def _eval(instr: str):
        if not examples_list:
            return default_eval_fn(instr)
        # Update the reusable agent with the new candidate instruction.
        try:
            agent_instance.update_instruction(instr)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("[EVAL] Failed updating agent instruction: %s", e, exc_info=True)
            return default_eval_fn(instr)
        # Run through examples accumulating scores via metric factory
        for ex in examples_list:
            try:
                pred = agent_instance(question=ex.question, functions=ex.functions)
                metric_factory.metric(ex, pred)
            except Exception as ex_err:
                logger.debug("[EVAL] Example evaluation error: %s", ex_err, exc_info=True)
                continue
        # Aggregate from factory internal records
        records = getattr(metric_factory, "_records", [])  # type: ignore[attr-defined]
        task_scores = [r.get("task_score", 0.0) for r in records]
        refusals = [r.get("refusal", 0.0) for r in records]
        completion_mean = float(_mean(task_scores))
        refusal_mean = float(_mean(refusals))
        extra = {
            "n_examples": len(task_scores),
            "source": "agent_metric",
        }
        metric_factory.reset()  # clear between candidates
        logger.info("[EVAL] Instr eval: refusal=%.3f completion=%.3f (n=%d)", refusal_mean, completion_mean, len(task_scores))
        return refusal_mean, completion_mean, extra

    return _eval


__all__ = [
    "load_eval_examples",
    "build_agent_instruction_eval_fn",
]
