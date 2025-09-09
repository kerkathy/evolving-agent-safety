"""Utilities for constructing instruction evaluation functions used in
optimization (aligns with main.py evaluation path).

This isolates the heavier logic from `pipeline.py` keeping orchestration
readable. We avoid external numeric deps (numpy) by using pure Python stats.
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Tuple
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


def load_eval_examples(config) -> Dict[str, List]:
    """Load separate benign + harmful example slices for dual-metric eval.

    Returns a dict with keys:
      - "benign": examples used to compute completion/task_score
      - "harmful": examples used to compute refusal metric

    Any failure results in empty lists for the affected key.
    """
    out: Dict[str, List] = {"benign": [], "harmful": []}
    if not (load_agentharm_data and build_dspy_examples):  # import failure
        logger.warning("[EVAL] Data loading modules unavailable; using heuristic eval only.")
        return out
    limit = config.causal.optimization.train_data_size
    data_cfg = config.data

    common_kwargs = dict(
        behavior_ids=list(getattr(data_cfg, "behavior_ids", []) or []),
        sample_ids=None,  # handle limit after building examples
        split=getattr(data_cfg, "split", "val"),
        detailed_behaviors=getattr(data_cfg, "detailed_behaviors", True),
        hint_included=getattr(data_cfg, "hint_included", False),
        n_irrelevant_tools=getattr(data_cfg, "n_irrelevant_tools", 0),
    )

    for task_name in ("benign", "harmful"):
        raw = load_agentharm_data(task_name=task_name, **common_kwargs)  # type: ignore[arg-type]
        examples = build_dspy_examples(raw)
        if len(examples) > limit:
            examples = examples[:limit]
        out[task_name] = examples
        logger.info(
            "[EVAL] Loaded %d %s eval examples (limit=%d)", len(examples), task_name, limit
        )
    return out


def build_dual_agent_instruction_eval_fn(
    examples_by_task: Dict[str, List],
    benign_metric_factory: AgentHarmMetricFactory,
    harmful_metric_factory: AgentHarmMetricFactory,
    agent: WebReActAgent,
):
    """Return callable(instr) -> (refusal_mean, completion_mean, extra_meta) using dual datasets.

    Logic:
      * completion_mean: average task_score over BENIGN examples (no refusal aggregation here)
      * refusal_mean: average refusal over HARMFUL examples (task_score ignored for objective)

    We run two independent evaluation passes per candidate instruction to keep
    factory state isolated and reduce cross-contamination of records.
    """
    benign_examples = list(examples_by_task["benign"])
    harmful_examples = list(examples_by_task["harmful"])
    if not benign_examples:
        raise ValueError("Dual eval requires at least one benign example for completion metric.")
    if not harmful_examples:
        raise ValueError("Dual eval requires at least one harmful example for refusal metric.")

    benign_eval = dspy.Evaluate(
        devset=benign_examples,
        metric=benign_metric_factory.metric,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=999,
        provide_traceback=False,
    )
    harmful_eval = dspy.Evaluate(
        devset=harmful_examples,
        metric=harmful_metric_factory.metric,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=999,
        provide_traceback=False,
    )

    def _mean(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    def _eval(instr: str) -> Tuple[float, float, Dict[str, int]]:
        agent.update_instruction(instr)
        logger.info(
            "[EVAL][DUAL] Evaluating instruction (len=%d) benign_n=%d harmful_n=%d", len(instr), len(benign_examples), len(harmful_examples)
        )

        # --- Completion (benign) pass ---
        benign_metric_factory.reset()
        try:
            benign_eval(agent)
        except Exception as e:  # pragma: no cover
            logger.exception("[EVAL][DUAL] Benign evaluation failed")
            raise RuntimeError(f"Benign evaluation failed: {e}") from e
        benign_records = list(getattr(benign_metric_factory, "_records", []))  # type: ignore[attr-defined]
        completion_scores = [r.get("task_score", 0.0) for r in benign_records]
        completion_mean = float(_mean(completion_scores))
        benign_metric_factory.reset()

        # --- Refusal (harmful) pass ---
        harmful_metric_factory.reset()
        try:
            harmful_eval(agent)
        except Exception as e:  # pragma: no cover
            logger.exception("[EVAL][DUAL] Harmful evaluation failed")
            raise RuntimeError(f"Harmful evaluation failed: {e}") from e
        harmful_records = list(getattr(harmful_metric_factory, "_records", []))  # type: ignore[attr-defined]
        refusal_flags = [r.get("refusal", 0.0) for r in harmful_records]
        refusal_mean = float(_mean(refusal_flags))
        harmful_metric_factory.reset()

        logger.info(
            "[EVAL][DUAL] Summary: refusal(harmful)=%.3f completion(benign)=%.3f", refusal_mean, completion_mean
        )
        extra = {
            "n_benign": len(completion_scores),
            "n_harmful": len(refusal_flags),
            "source": "dual_agent_metric_evaluate",
        }
        return refusal_mean, completion_mean, extra

    return _eval


__all__ = [
    "load_eval_examples",
    "build_dual_agent_instruction_eval_fn",
]
