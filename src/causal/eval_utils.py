"""Utilities for constructing instruction evaluation functions used in
optimization (aligns with main.py evaluation path).

This isolates the heavier logic from `pipeline.py` keeping orchestration
readable. We avoid external numeric deps (numpy) by using pure Python stats.
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Tuple, Callable, Optional
import random as _global_random  # for typing/reference
import random
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


def load_eval_examples(config, *, full: bool = False) -> Dict[str, Dict[str, List]]:
    """Load separate benign + harmful example slices for dual-metric eval.

    Args:
        config: global config
        full: if True, do NOT apply the training size limit (use all examples).

    Returns:
        dict with keys 'benign' and 'harmful', each containing 'train', 'eval', 'test' lists.
    """
    out: Dict[str, Dict[str, List]] = {"benign": {"train": [], "eval": [], "test": []}, "harmful": {"train": [], "eval": [], "test": []}}
    if not (load_agentharm_data and build_dspy_examples):  # import failure
        logger.warning("[EVAL] Data loading modules unavailable; using heuristic eval only.")
        return out
    limit = None if full else config.causal.optimization.max_data_size
    data_cfg = config.data

    common_kwargs = dict(
        behavior_ids=list(getattr(data_cfg, "behavior_ids", []) or []),
        sample_ids=getattr(data_cfg, "sample_ids", None),
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
        if not examples:
            out[task_name] = {"train": [], "eval": [], "test": []}
            continue
        # Split into train, eval, test
        testset_ids = getattr(data_cfg, "testset_ids", None)
        if testset_ids:
            if getattr(data_cfg, "train_fraction", None) is not None:
                raise ValueError("Cannot set train_fraction when testset_ids is set")
            if getattr(data_cfg, "sample_ids", None) is not None:
                raise ValueError("Cannot set sample_ids when testset_ids is set")
            test_ids = testset_ids
            test_examples = [e for e in examples if getattr(e, 'sample_id', None) in test_ids]
            remaining_examples = [e for e in examples if getattr(e, 'sample_id', None) not in test_ids]
            random_seed = getattr(data_cfg, "shuffle_seed", 0)
            random.seed(random_seed)
            random.shuffle(remaining_examples)
            total_remaining = len(remaining_examples)
            # Use default train_fraction = 0.8 for remaining
            train_fraction = 0.8
            train_eval_count = int(total_remaining * train_fraction)
            train_count = train_eval_count * 8 // 10  # 80% of train_eval for train
            eval_count = train_eval_count - train_count
            if train_count < 1 or eval_count < 1:
                raise ValueError(f"Not enough remaining {task_name} examples ({total_remaining}) to split into train/eval")
            train = remaining_examples[:train_count]
            eval_ = remaining_examples[train_count:train_count + eval_count]
            test_ = test_examples
            logger.info("Loaded test ids...")
        else:
            train_fraction = getattr(data_cfg, "train_fraction", 0.8)
            random_seed = getattr(data_cfg, "shuffle_seed", 0)
            total = len(examples)
            train_eval_count = int(total * train_fraction)
            train_count = train_eval_count * 8 // 10  # 80% of train_eval for train
            eval_count = train_eval_count - train_count
            test_count = total - train_eval_count
            if train_count < 1 or eval_count < 1 or test_count < 1:
                raise ValueError(f"Not enough {task_name} examples ({total}) to split into train/eval/test with fractions {train_fraction}/(1-{train_fraction})")
            random.seed(random_seed)
            random.shuffle(examples)
            train = examples[:train_count]
            eval_ = examples[train_count:train_count + eval_count]
            test_ = examples[train_count + eval_count:]
        out[task_name] = {"train": train, "eval": eval_, "test": test_}
        logger.info(
            "[EVAL] Loaded %d %s eval examples (train=%d, eval=%d, test=%d)", len(examples), task_name, len(train), len(eval_), len(test_)
        )
    return out


def build_dual_agent_instruction_eval_fn(
    examples_by_task: Dict[str, Dict[str, List]],
    benign_metric_factory: AgentHarmMetricFactory,
    harmful_metric_factory: AgentHarmMetricFactory,
    agent: WebReActAgent,
    split: str = 'train',
):
    """Return callable(instr) -> (refusal_mean, completion_mean, extra_meta) using dual datasets.

    Logic:
      * completion_mean: average task_score over BENIGN examples (no refusal aggregation here)
      * refusal_mean: average refusal over HARMFUL examples (task_score ignored for objective)

    We run two independent evaluation passes per candidate instruction to keep
    factory state isolated and reduce cross-contamination of records.
    """
    benign_examples = list(examples_by_task["benign"][split])
    harmful_examples = list(examples_by_task["harmful"][split])
    if not benign_examples:
        raise ValueError("Dual eval requires at least one benign example for completion task_score_metric.")
    if not harmful_examples:
        raise ValueError("Dual eval requires at least one harmful example for refusal task_score_metric.")

    benign_eval = dspy.Evaluate(
        devset=benign_examples,
        metric=benign_metric_factory.task_score_metric,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=999,
        provide_traceback=False,
    )
    harmful_eval = dspy.Evaluate(
        devset=harmful_examples,
        metric=harmful_metric_factory.refusal_metric, # TODO not sure...
        # metric=harmful_metric_factory.task_score_metric,
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
        examples_by_task: Dict[str, Dict[str, List]],
        benign_metric_factory: AgentHarmMetricFactory,
        harmful_metric_factory: AgentHarmMetricFactory,
        agent: WebReActAgent,
        minibatch_size: int,
        rng: Optional[_global_random.Random],
        split: str = 'train',
    ) -> None:
        import random as _random  # local import to avoid global dependency if unused

        self.pool_benign = list(examples_by_task["benign"][split])  # copy
        self.pool_harmful = list(examples_by_task["harmful"][split])
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
        if bsize != hsize:
            raise ValueError("Minibatch evaluator requires equal benign and harmful minibatch sizes.")
        # use the same idx for both benign and harmful to keep eval fair
        indices = self.rng.sample(range(len(self.pool_benign)), bsize)
        self._benign_batch = [self.pool_benign[i] for i in indices]
        self._harmful_batch = [self.pool_harmful[i] for i in indices]
        # Rebuild dspy.Evaluate objects with new devsets
        self._benign_eval = dspy.Evaluate(
            devset=self._benign_batch,
            metric=self._benign_factory.task_score_metric,
            num_threads=1,
            display_progress=False,
            display_table=0,
            max_errors=999,
            provide_traceback=False,
        )
        self._harmful_eval = dspy.Evaluate(
            devset=self._harmful_batch,
            metric=self._harmful_factory.refusal_metric, # TODO not sure...
            # metric=self._harmful_factory.task_score_metric,
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


def build_minibatch_eval_fn(
    examples_by_task: Dict[str, Dict[str, List]],
    benign_metric_factory: AgentHarmMetricFactory,
    harmful_metric_factory: AgentHarmMetricFactory,
    agent: WebReActAgent,
    minibatch_size: int,
    rng=None,
    split: str = 'train',
):
    """Factory returning a MiniBatchDualAgentInstructionEvaluator instance."""
    return MiniBatchDualAgentInstructionEvaluator(
        examples_by_task,
        benign_metric_factory=benign_metric_factory,
        harmful_metric_factory=harmful_metric_factory,
        agent=agent,
        minibatch_size=minibatch_size,
        rng=rng,
        split=split,
    )


def build_full_eval_fn(
    examples_by_task: Dict[str, Dict[str, List]],
    benign_metric_factory: AgentHarmMetricFactory,
    harmful_metric_factory: AgentHarmMetricFactory,
    agent: WebReActAgent,
    split: str = 'eval'
) -> Callable[[str], Tuple[float, float, Dict[str, int]]]:
    """Construct a full evaluation function on the specified split.

    Mirrors build_dual_agent_instruction_eval_fn but intended for larger eval
    slices (no training-size truncation). Factories are passed as NEW instances
    by the caller to avoid cross contamination with train-limited eval.
    """
    return build_dual_agent_instruction_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_metric_factory,
        harmful_metric_factory=harmful_metric_factory,
        agent=agent,
        split=split
    )


__all__ = [
    "load_eval_examples",
    "build_dual_agent_instruction_eval_fn",
    "build_full_eval_fn",
    "build_minibatch_eval_fn",
]
