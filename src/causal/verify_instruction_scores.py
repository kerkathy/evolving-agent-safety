"""Verification script for instruction evaluation reproducibility.

Purpose
-------
Given an MLflow experiment and either a parent run name (``mlflow.runName`` tag)
or identifying model parameter (``model_lm_name``), this script:

1. Collects unique instruction prompts from child evaluation runs (using the
   same logic as ``collector.collect_prompts``).
2. Reconstructs the original evaluation environment (dataset slice, agent,
   metric factory, DSPy LM) from logged parent run parameters.
3. Re-runs evaluation for each collected instruction (one at a time) on the
   dev/eval split and computes the mean ``task_score``.
4. Compares the recomputed score to the stored metric in the *child* run's
   ``raw.source_metrics`` (default key: ``eval``) to ensure reproducibility.

The script reports any mismatches beyond a configurable tolerance and exits
with a non‑zero status if mismatches are found (unless ``--allow-drift`` is
specified).

Usage
-----
python -m src.causal.verify_instruction_scores \
    --experiment agentharm_experiment_harmful_val_detail_True_hint_False \
    --run-name SOME_PARENT_RUN_NAME \
    --metric-key eval \
    --tracking-uri http://127.0.0.1:5000 \
    --abs-tol 1e-6

Or using model name selection instead of run name:
python -m src.causal.verify_instruction_scores \
    --experiment agentharm_experiment_harmful_val_detail_True_hint_False \
    --model-lm-name openai/gpt-5-nano

Notes
-----
* Exact reproducibility requires deterministic model behavior. If the original
  run used a non-zero temperature and no fixed seed, minor drift is expected.
  Use ``--abs-tol`` / ``--rel-tol`` or ``--allow-drift`` to soften strictness.
* This script purposefully evaluates each instruction independently (resetting
  metric factory state) to mirror how candidate instructions are scored during
  optimization / eval phases.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Dict
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient

import dspy

from .collector import collect_prompts, PromptRecord
from .runtime_setup import configure_dspy, build_agent_and_metric
from src.data import load_agentharm_data, build_dspy_examples, split_examples

logger = logging.getLogger("verify_instruction_scores")


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalContext:
    parent_run_id: str
    parent_params: dict
    dev_examples: list


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _extract_parent(client: MlflowClient, experiment_id: str, run_name: str | None, model_lm_name: str | None):
    if run_name:
        runs = client.search_runs(
            [experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            # filter_string=f"tags.mlflow.runName = '{run_name}' AND params.optim_run_optimization = 'True'",
            max_results=1,
        )
    else:
        if not model_lm_name:
            raise ValueError("Either --run-name or --model-lm-name must be provided")
        runs = client.search_runs(
            [experiment_id],
            filter_string=f"params.model_lm_name = '{model_lm_name}' AND params.optim_run_optimization = 'True'",
            max_results=1,
        )
    if not runs:
        raise ValueError("No matching parent run found")
    return runs[0]


def _reconstruct_dev_examples(parent_params: dict) -> list:
    """Rebuild the dev (evaluation) examples used originally.

    Mirrors ``prepare_data`` in ``main.py``: load full dataset, then split with
    train_fraction + shuffle_seed, returning the devset.
    """
    # Required data params (use defaults if absent)
    task_name = parent_params.get("task_name") or parent_params.get("task_name") or "harmful"
    split = parent_params.get("split", "val")
    detailed_behaviors = _coerce_bool(parent_params.get("detailed_behaviors", True))
    hint_included = _coerce_bool(parent_params.get("hint_included", False))
    n_irrelevant_tools = int(parent_params.get("n_irrelevant_tools", 0))
    train_fraction = float(parent_params.get("train_fraction", 0.2))
    shuffle_seed = int(parent_params.get("shuffle_seed", 0))
    behavior_ids_raw = parent_params.get("behavior_ids")
    if behavior_ids_raw in (None, "", "None"):  # treat as no filter
        behavior_ids = []
    else:
        # Could be comma-separated or list-like string; be forgiving.
        if isinstance(behavior_ids_raw, str):
            behavior_ids = [b.strip() for b in behavior_ids_raw.split(",") if b.strip()]
        else:
            behavior_ids = list(behavior_ids_raw)

    sample_ids_raw = parent_params.get("data_sample_ids")
    if sample_ids_raw in (None, "", "None"):
        sample_ids = []
    else:
        if isinstance(sample_ids_raw, str):
            sample_ids = [s.strip() for s in sample_ids_raw.split(",") if s.strip()]
        else:
            sample_ids = list(sample_ids_raw)

    raw = load_agentharm_data(
        behavior_ids=behavior_ids or None,
        sample_ids=sample_ids or None,
        task_name=task_name,
        split=split,
        detailed_behaviors=detailed_behaviors,
        hint_included=hint_included,
        n_irrelevant_tools=n_irrelevant_tools,
    )
    examples = build_dspy_examples(raw)
    trainset, devset = split_examples(examples, train_fraction=train_fraction, seed=shuffle_seed)
    logger.info(
        "Reconstructed dataset: total=%d train=%d dev=%d (task=%s split=%s)",
        len(examples), len(trainset), len(devset), task_name, split,
    )
    return devset


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _compute_task_score(instruction: str, dev_examples, metric_factory, agent) -> float:
    """Compute mean task_score for a single instruction over dev examples.

    This inlined variant mirrors ``build_agent_instruction_eval_fn`` but avoids
    extra closure layers so we can run one instruction at a time.
    """
    metric_factory.reset()
    agent.update_instruction(instruction)
    evaluator = dspy.Evaluate(
        devset=dev_examples,
        metric=metric_factory.metric,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=999,
        provide_traceback=False,
    )
    evaluator(agent)
    records = list(getattr(metric_factory, "_records", []))  # type: ignore[attr-defined]
    scores = [r.get("task_score", 0.0) for r in records]
    metric_factory.reset()
    return sum(scores) / len(scores) if scores else 0.0


def verify_scores(
    experiment_name: str,
    run_name: str | None,
    model_lm_name: str | None,
    param_key: str,
    child_prefix: str,
    metric_key: str,
    abs_tol: float,
    rel_tol: float,
    allow_drift: bool,
    run_is_optim: bool,
) -> dict:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    parent = _extract_parent(client, experiment.experiment_id, run_name, model_lm_name)
    parent_id = parent.info.run_id
    parent_params = dict(parent.data.params)
    logger.info("Parent run selected: %s (name=%s)", parent_id, parent.data.tags.get("mlflow.runName"))

    # Configure DSPy with original model settings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set; cannot run model evaluation")
    # Minimal model config emulation (only fields used by create_enhanced_dspy_lm)
    def _coerce_int(v):  # local helper for robust casting
        if v in (None, "", "None", "none"):  # treat empty / literal None as absent
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    class _ModelCfg:  # lightweight shim
        lm_name = parent_params.get("model_lm_name", "openai/gpt-5-nano")
        lm_temperature = float(parent_params.get("model_lm_temperature", 1.0))
        max_tokens = int(parent_params.get("model_max_tokens", 2048))
        refusal_judge_model = parent_params.get("model_refusal_judge_model", lm_name)
        semantic_judge_model = parent_params.get("model_semantic_judge_model", lm_name)
        api_base = parent_params.get("model_api_base")
        headers = None
        # MLflow stores params as strings; ensure numeric seed like main.py's config loader.
        seed = _coerce_int(parent_params.get("model_seed"))

    class _DataCfg:
        task_name = parent_params.get("task_name", "harmful")

    class _ConfigShim:
        data = _DataCfg()
        models = _ModelCfg()

    configure_dspy(_ConfigShim().models, api_key)
    agent, metric_factory = build_agent_and_metric(_ConfigShim())

    dev_examples = _reconstruct_dev_examples(parent_params)
    if not dev_examples:
        raise RuntimeError("No dev examples reconstructed; cannot verify scores")

    # Collect instruction prompts
    prompts: List[PromptRecord] = collect_prompts(
        experiment_name,
        run_name=run_name,
        model_lm_name=model_lm_name,
        param_key=param_key,
        child_prefix=child_prefix,
        run_is_optim=run_is_optim,
    )
    logger.info("Collected %d unique instruction prompts for verification", len(prompts))

    results: List[Dict] = []
    mismatches = 0
    for idx, rec in enumerate(prompts):
        stored_metrics = rec.raw.get("source_metrics", {})
        if metric_key not in stored_metrics:
            logger.warning("[%d/%d] Run %s missing metric '%s' -> skipping", idx+1, len(prompts), rec.run_id, metric_key)
            continue
        expected = float(stored_metrics[metric_key])
        computed = _compute_task_score(rec.prompt_text, dev_examples, metric_factory, agent)
        abs_err = abs(computed - expected)
        rel_err = abs_err / (abs(expected) + 1e-12)
        match = (abs_err <= abs_tol) or (rel_err <= rel_tol)
        if not match:
            mismatches += 1
        logger.info(
            "[%d/%d] prompt_id=%s expected=%.6f computed=%.6f abs_err=%.3g rel_err=%.3g %s",
            idx + 1,
            len(prompts),
            rec.prompt_id,
            expected,
            computed,
            abs_err,
            rel_err,
            "OK" if match else "MISMATCH",
        )
        results.append(
            {
                "prompt_id": rec.prompt_id,
                "run_id": rec.run_id,
                "expected": expected,
                "computed": computed,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "match": match,
            }
        )

    summary = {
        "n_records": len(results),
        "metric_key": metric_key,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "mismatches": mismatches,
        "deterministic": float(_ConfigShim().models.lm_temperature) == 0.0,
    }
    if mismatches and not allow_drift:
        logger.error("Verification failed: %d mismatches (abs_tol=%g rel_tol=%g)", mismatches, abs_tol, rel_tol)
        exit_code = 1
    else:
        exit_code = 0
    return {"summary": summary, "results": results, "exit_code": exit_code}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify instruction evaluation reproducibility against stored MLflow metrics.")
    ap.add_argument("--experiment", required=True, help="MLflow experiment name (same as used in original run).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-name", help="Parent run name (tag mlflow.runName) identifying optimization run.")
    ap.add_argument("--run-is-optim", action="store_true", help="When set, restrict parent run search to optimization runs (params.optim_run_optimization = 'True').")
    group.add_argument("--model-lm-name", help="Alternative parent selector via logged parameter model_lm_name.")
    ap.add_argument("--param-key", default="WebReActAgent.react.predict.signature.instructions", help="Parameter key holding instruction text in child runs.")
    ap.add_argument("--child-prefix", default="eval_full_", help="Child run name prefix to filter evaluation runs.")
    ap.add_argument("--metric-key", default="eval", help="Metric key inside child run metrics to compare against recomputed task_score.")
    ap.add_argument("--tracking-uri", help="Optional MLflow tracking URI (overrides environment).")
    ap.add_argument("--abs-tol", type=float, default=1e-6, help="Absolute tolerance for score equality.")
    ap.add_argument("--rel-tol", type=float, default=1e-4, help="Relative tolerance for score equality.")
    ap.add_argument("--allow-drift", action="store_true", help="Do not exit non-zero even if mismatches occur.")
    ap.add_argument("--json-out", help="Optional path to write JSON report.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)

    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    # print all args except api_key
    for k, v in vars(args).items():
        if k != "api_key":
            logger.info("Arg %s: %s", k, v)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        logger.info("Set MLflow tracking URI to %s", args.tracking_uri)

    outcome = verify_scores(
        experiment_name=args.experiment,
        run_name=args.run_name,
        model_lm_name=args.model_lm_name,
        param_key=args.param_key,
        child_prefix=args.child_prefix,
        metric_key=args.metric_key,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        allow_drift=args.allow_drift,
        run_is_optim=args.run_is_optim,
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(outcome, f, indent=2)
        logger.info("Wrote JSON report to %s", args.json_out)
    logger.info("Summary: %s", outcome["summary"])
    return int(outcome["exit_code"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
