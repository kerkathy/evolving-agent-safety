"""Utilities for constructing instruction evaluation functions used in
optimization (aligns with main.py evaluation path).

This isolates the heavier logic from `pipeline.py` keeping orchestration
readable. We avoid external numeric deps (numpy) by using pure Python stats.
"""
from __future__ import annotations

from typing import Iterable, List
import logging

from src.agent import WebReActAgent
from src.metrics import AgentHarmMetricFactory
import dspy  # needed for dspy.Evaluate based evaluation

logger = logging.getLogger(__name__)

try:  # Optional import; not strictly required at runtime until functions used
    from src.data import load_agentharm_data, build_dspy_examples
except Exception:  # pragma: no cover - defensive
    load_agentharm_data = None  # type: ignore
    build_dspy_examples = None  # type: ignore


def load_eval_examples(config):
    """Load a lightweight slice of AgentHarm examples for instruction eval.

    Falls back to empty list (triggering heuristic scoring) if anything fails.
    """
    if not (load_agentharm_data and build_dspy_examples):  # import failure
        logger.warning("[EVAL] Data loading modules unavailable; using heuristic eval only.")
        return []
    limit = config.causal.optimization.train_data_size
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
    agent: WebReActAgent,
):
    """Return callable(instr) -> (refusal_mean, completion_mean, extra_meta).

    Simplified: no internal agent instantiation & no heuristic fallback.
    Caller MUST supply:
      * non-empty iterable of evaluation examples
      * a configured metric_factory (state will be reset each call)
      * a pre-built agent instance whose instruction will be mutated in-place

    Raises:
        ValueError: if examples is empty.
        RuntimeError: if evaluation fails unexpectedly.
    """
    examples_list: List = list(examples)
    if not examples_list:
        raise ValueError("build_agent_instruction_eval_fn received 0 examples; provide at least one for evaluation.")

    evaluator = dspy.Evaluate(
        devset=examples_list,
        metric=metric_factory.metric,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=999,
        provide_traceback=False,
    )
    for idx, ex in enumerate(examples_list):  # dataset traceability
        q = getattr(ex, "question", "")
        logger.info("[EVAL][DATA] Example %d len=%d preview=%s", idx, len(q), q[:140].replace('\n', ' '))

    def _mean(values):
        return (sum(values) / len(values)) if values else 0.0

    def _eval(instr: str):
        metric_factory.reset()  # ensure clean state per candidate
        agent.update_instruction(instr)
        instr_hash = hash(instr)
        logger.info(
            "[EVAL] Evaluating candidate instruction hash=%s len=%d preview=%s", instr_hash, len(instr), instr[:160].replace('\n', ' ')
        )
        logger.info("[EVAL] Devset size=%d", len(examples_list))
        try:
            evaluator(agent)
        except Exception as run_err:  # pragma: no cover
            logger.exception("[EVAL] Evaluation failure")
            raise RuntimeError(f"Evaluation failed: {run_err}") from run_err

        records = list(getattr(metric_factory, "_records", []))  # type: ignore[attr-defined]
        task_scores = [r.get("task_score", 0.0) for r in records]
        refusals = [r.get("refusal", 0.0) for r in records]
        completion_mean = float(_mean(task_scores))
        refusal_mean = float(_mean(refusals))

        for i, r in enumerate(records):
            logger.info(
                "[EVAL][REC] #%d task_score=%.3f refusal=%.3f grading_fn=%s", i, r.get("task_score", 0.0), r.get("refusal", 0.0), r.get("grading_function")
            )
        extra = {"n_examples": len(task_scores), "source": "agent_metric_evaluate"}
        logger.info(
            "[EVAL] Candidate summary: refusal=%.3f completion=%.3f n=%d", refusal_mean, completion_mean, len(task_scores)
        )
        metric_factory.reset()  # leave factory clean for caller's next use
        return refusal_mean, completion_mean, extra

    return _eval


__all__ = [
    "load_eval_examples",
    "build_agent_instruction_eval_fn",
]
