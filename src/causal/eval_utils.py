"""Utilities for constructing instruction evaluation functions used in
optimization (aligns with main.py evaluation path).

This isolates the heavier logic from `pipeline.py` keeping orchestration
readable. We avoid external numeric deps (numpy) by using pure Python stats.
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Tuple, Callable, Optional
import random as _global_random  # for typing/reference
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


def load_eval_examples(config, *, full: bool = False) -> Dict[str, List]:
    """Load separate benign + harmful example slices for dual-metric eval.

    Args:
        config: global config
        full: if True, do NOT apply the training size limit (use all examples).

    Returns:
        dict with keys 'benign' and 'harmful'. Empty lists if loaders unavailable.
    """
    out: Dict[str, List] = {"benign": [], "harmful": []}
    if not (load_agentharm_data and build_dspy_examples):  # import failure
        logger.warning("[EVAL] Data loading modules unavailable; using heuristic eval only.")
        return out
    limit = None if full else config.causal.optimization.train_data_size
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
        if limit is not None and len(examples) > limit:
            examples = examples[:limit]
        out[task_name] = examples
        logger.info(
            "[EVAL] Loaded %d %s eval examples (limit=%s)", len(examples), task_name, limit if limit is not None else "ALL"
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


class MiniBatchDualAgentInstructionEvaluator:
    """Minibatch evaluator that resamples a subset of examples per generation.

    Motivation: During instruction optimization we want stochastic fitness to
    reduce overfitting to a fixed tiny slice and obtain a better exploration
    signal. For *each generation* we draw a fresh minibatch (same for all
    candidates in that generation for fairness). Seeds (gen=0) also get an
    initial batch.

    We avoid retaining per-candidate state to keep memory low. The full
    example pools are held once; per resample we construct new dspy.Evaluate
    objects referencing only the selected subset.
    """

    def __init__(
        self,
        full_examples_by_task: Dict[str, List],
        benign_metric_factory: AgentHarmMetricFactory,
        harmful_metric_factory: AgentHarmMetricFactory,
        agent: WebReActAgent,
        minibatch_size: int,
    rng: Optional[_global_random.Random],
    ) -> None:
        import random as _random  # local import to avoid global dependency if unused

        self.pool_benign = list(full_examples_by_task["benign"])  # copy
        self.pool_harmful = list(full_examples_by_task["harmful"])
        if not self.pool_benign or not self.pool_harmful:
            raise ValueError("Minibatch evaluator requires non-empty benign and harmful pools.")
        self.agent = agent
        self.minibatch_size = max(1, int(minibatch_size))
        self.rng = rng or _random.Random(0)
        self._benign_factory = benign_metric_factory
        self._harmful_factory = harmful_metric_factory
        self._benign_eval: Optional[dspy.Evaluate] = None
        self._harmful_eval: Optional[dspy.Evaluate] = None
        self.current_gen: int = -1
        self.batch_id: str = ""
        self._benign_batch: List = []
        self._harmful_batch: List = []
        # Initial sample
        self.resample(gen=0)

    # Public hook recognized by optimization loop
    def resample(self, gen: int) -> None:  # noqa: D401 - simple
        """Resample a new minibatch (benign + harmful) for a given generation."""
        import hashlib, json as _json

        self.current_gen = gen
        # Sample without replacement; if pool smaller than size use full pool
        bsize = min(self.minibatch_size, len(self.pool_benign))
        hsize = min(self.minibatch_size, len(self.pool_harmful))
        self._benign_batch = self.rng.sample(self.pool_benign, bsize) if len(self.pool_benign) > bsize else self.pool_benign
        self._harmful_batch = self.rng.sample(self.pool_harmful, hsize) if len(self.pool_harmful) > hsize else self.pool_harmful
        # Rebuild dspy.Evaluate objects with new devsets
        self._benign_eval = dspy.Evaluate(
            devset=self._benign_batch,
            metric=self._benign_factory.metric,
            num_threads=1,
            display_progress=False,
            display_table=0,
            max_errors=999,
            provide_traceback=False,
        )
        self._harmful_eval = dspy.Evaluate(
            devset=self._harmful_batch,
            metric=self._harmful_factory.metric,
            num_threads=1,
            display_progress=False,
            display_table=0,
            max_errors=999,
            provide_traceback=False,
        )
        # Derive a deterministic batch id (hash of indices repr)
        raw_sig = _json.dumps({
            "gen": gen,
            "benign_ids": [getattr(x, "id", i) for i, x in enumerate(self._benign_batch)],
            "harmful_ids": [getattr(x, "id", i) for i, x in enumerate(self._harmful_batch)],
        }, sort_keys=True)
        self.batch_id = hashlib.sha256(raw_sig.encode()).hexdigest()[:12]
        logger.info(
            "[EVAL][MB] Resampled minibatch gen=%d benign=%d harmful=%d batch_id=%s", gen, len(self._benign_batch), len(self._harmful_batch), self.batch_id
        )

    def __call__(self, instr: str) -> Tuple[float, float, Dict[str, int]]:  # noqa: D401
        if self._benign_eval is None or self._harmful_eval is None:
            raise RuntimeError("Evaluator not initialized; call resample first.")
        # Update instruction
        self.agent.update_instruction(instr)
        # Completion pass (benign)
        self._benign_factory.reset()
        try:
            self._benign_eval(self.agent)
        except Exception as e:  # pragma: no cover
            logger.exception("[EVAL][MB] Benign minibatch evaluation failed")
            raise RuntimeError(f"Benign minibatch evaluation failed: {e}") from e
        records_b = list(getattr(self._benign_factory, "_records", []))  # type: ignore[attr-defined]
        completion_scores = [r.get("task_score", 0.0) for r in records_b]
        self._benign_factory.reset()

        # Refusal pass (harmful)
        self._harmful_factory.reset()
        try:
            self._harmful_eval(self.agent)
        except Exception as e:  # pragma: no cover
            logger.exception("[EVAL][MB] Harmful minibatch evaluation failed")
            raise RuntimeError(f"Harmful minibatch evaluation failed: {e}") from e
        records_h = list(getattr(self._harmful_factory, "_records", []))  # type: ignore[attr-defined]
        refusal_flags = [r.get("refusal", 0.0) for r in records_h]
        self._harmful_factory.reset()

        def _mean(vals: List[float]) -> float:
            return (sum(vals) / len(vals)) if vals else 0.0

        completion_mean = float(_mean(completion_scores))
        refusal_mean = float(_mean(refusal_flags))
        extra = {
            "n_benign": len(completion_scores),
            "n_harmful": len(refusal_flags),
            "source": "dual_agent_metric_evaluate_minibatch",
            "batch_gen": self.current_gen,
            "batch_id": self.batch_id,
        }
        logger.info(
            "[EVAL][MB] Gen=%d batch=%s refusal=%.3f completion=%.3f", self.current_gen, self.batch_id, refusal_mean, completion_mean
        )
        return refusal_mean, completion_mean, extra


def build_minibatch_dual_agent_instruction_eval_fn(
    full_examples_by_task: Dict[str, List],
    benign_metric_factory: AgentHarmMetricFactory,
    harmful_metric_factory: AgentHarmMetricFactory,
    agent: WebReActAgent,
    minibatch_size: int,
    rng=None,
):
    """Factory returning a MiniBatchDualAgentInstructionEvaluator instance."""
    return MiniBatchDualAgentInstructionEvaluator(
        full_examples_by_task,
        benign_metric_factory=benign_metric_factory,
        harmful_metric_factory=harmful_metric_factory,
        agent=agent,
        minibatch_size=minibatch_size,
        rng=rng,
    )


def build_full_eval_fn(
    examples_by_task_full: Dict[str, List],
    benign_metric_factory: AgentHarmMetricFactory,
    harmful_metric_factory: AgentHarmMetricFactory,
    agent: WebReActAgent,
) -> Callable[[str], Tuple[float, float, Dict[str, int]]]:
    """Construct a full evaluation function on the entire eval set.

    Mirrors build_dual_agent_instruction_eval_fn but intended for larger eval
    slices (no training-size truncation). Factories are passed as NEW instances
    by the caller to avoid cross contamination with train-limited eval.
    """
    return build_dual_agent_instruction_eval_fn(
        examples_by_task_full,
        benign_metric_factory=benign_metric_factory,
        harmful_metric_factory=harmful_metric_factory,
        agent=agent,
    )


__all__ = [
    "load_eval_examples",
    "build_dual_agent_instruction_eval_fn",
    "build_full_eval_fn",
    "build_minibatch_dual_agent_instruction_eval_fn",
]
