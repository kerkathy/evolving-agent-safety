"""Orchestrates full causal analysis pipeline.

High-level function: run_causal_analysis(config, experiment_name)

Steps:
 1. Collect prompts from MLflow (collector.collect_prompts)
 2. Generate interventions for each prompt
 3. Evaluate original + variants (evaluator.evaluate_variants)
 4. Compute effects (effects.compute_effects)
 5. Persist results to disk and (optionally) MLflow artifacts

Scoring Strategy:
  - For now, naive scoring = reuse existing label if present else 0.0.
  - Placeholder for plugging in a model-based scorer later.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import json
import logging
from datetime import datetime

import mlflow

from .collector import collect_prompts, PromptRecord
from .optimization import optimize_instructions
from .io_utils import write_optimization_outputs
from .eval_utils import (
    load_eval_examples,
    build_full_eval_fn,
    build_minibatch_dual_agent_instruction_eval_fn,
)
from src.metrics import AgentHarmMetricFactory
from .runtime_setup import configure_dspy, build_agent_and_dual_metric

logger = logging.getLogger(__name__)

def run_causal_analysis(config, experiment_name: str) -> None:
    mlflow.set_tracking_uri(config.experiment.uri)  # Adjust as needed

    cconf = config.causal
    if not cconf or not cconf.enabled:
        logger.info("Causal analysis disabled; skipping.")
        return

    logger.info("[CAUSAL] Collecting instruction prompts (max=%s)...", cconf.max_collected_prompts)
    if not (cconf.run_name or config.models.lm_name):
        raise ValueError("Either run_name or model_lm_name must be set in causal config.")
    records: list[PromptRecord] = collect_prompts(
        experiment_name,
        run_name=cconf.run_name,
        model_lm_name=config.models.lm_name,
        param_key=cconf.param_key,
        child_prefix=cconf.child_prefix,
        limit=cconf.max_collected_prompts,
    )
    if not records:
        logger.warning("[CAUSAL] No prompts collected; aborting causal analysis.")
        return
    logger.info("[CAUSAL] Collected %d unique prompts.", len(records))
    for rec in records:
        logger.info("[CAUSAL] Sample prompt: %s", rec.prompt_text[:200].replace('\n', ' ') + ("..." if len(rec.prompt_text) > 200 else ""))

    # Create a unique timestamped output directory for this run (both modes)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_out_dir = Path(f"{cconf.output_dir}_{timestamp}")
    if run_out_dir.exists():
        raise FileExistsError(f"Output directory {run_out_dir.resolve()} already exists. Aborting to avoid overwrite.")
    run_out_dir.mkdir(parents=True, exist_ok=False)
    logger.info("[CAUSAL] Results will be written to %s", run_out_dir.resolve())

    # Run optimization over collected instruction prompts
    logger.info("[CAUSAL][OPT] Starting multi-objective instruction optimization over %d candidates", len(records))
    initial_texts = [r.prompt_text for r in records]

    # Configure runtime (DSPy + agent + metric)
    api_key = os.getenv("OPENAI_API_KEY") or ""
    main_lm = configure_dspy(config.models, api_key)
    agent, benign_metric_factory, harmful_metric_factory = build_agent_and_dual_metric(config)

    # Load examples (separate benign/harmful)
    # We now load the FULL pool and rely on a per-generation minibatch sampler
    # to reduce overfitting and memory footprint.
    examples_by_task = load_eval_examples(config, full=True)
    if not examples_by_task.get("benign") or not examples_by_task.get("harmful"):
        raise ValueError("[CAUSAL][OPT] Missing benign or harmful examples; cannot proceed with dual-metric optimization.")
    train_eval_fn = build_minibatch_dual_agent_instruction_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_metric_factory,
        harmful_metric_factory=harmful_metric_factory,
        agent=agent,
        minibatch_size=cconf.optimization.minibatch_size,
        rng=None,
        split='train',
    )
    # Full eval slice for post-generation frontier assessment
    # Recreate fresh metric factories for full evaluation to avoid mixing records with train-slice eval.
    benign_full_factory = AgentHarmMetricFactory(
        task_name=benign_metric_factory.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    harmful_full_factory = AgentHarmMetricFactory(
        task_name=harmful_metric_factory.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    dev_eval_fn = build_full_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_full_factory,
        harmful_metric_factory=harmful_full_factory,
        agent=agent,
        split='eval',
    )

    opt_result = optimize_instructions(
        initial_texts,
        main_lm=main_lm,
        train_eval_fn=train_eval_fn,
        dev_eval_fn=dev_eval_fn,
        config=cconf.optimization,
        checkpoint_dir=str(run_out_dir),
    )

    # Evaluate final frontier on test set
    logger.info("[CAUSAL][TEST] Evaluating final frontier on test set (size=%d)", len(opt_result.frontier))
    test_eval_fn = build_full_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_full_factory,
        harmful_metric_factory=harmful_full_factory,
        agent=agent,
        split='test',
    )
    test_results = []
    for cand in opt_result.frontier:
        try:
            r, c, extra = test_eval_fn(cand.text)
            test_results.append({
                "hash": cand.hash(),
                "text": cand.text,
                "refusal_test": r,
                "completion_test": c,
                **extra,
            })
        except Exception as e:
            logger.warning("[CAUSAL][TEST] Failed to evaluate candidate %s: %s", cand.hash(), e)
            test_results.append({
                "hash": cand.hash(),
                "text": cand.text,
                "refusal_test": 0.0,
                "completion_test": 0.0,
                "error": str(e),
            })
    logger.info("[CAUSAL][TEST] Test evaluation completed for %d candidates", len(test_results))

    summary = {
        "n_seeds": len(records),
        "generations": opt_result.generations,
        "evaluations": opt_result.num_evaluations,
        "frontier_size": len(opt_result.frontier),
        "config": asdict(config),
        "segment_effects": opt_result.segment_effects,  # Added segment effects
        "test_results": test_results,
        "test_summary": {
            "best_refusal_test": max((r["refusal_test"] for r in test_results), default=0.0),
            "best_completion_test": max((r["completion_test"] for r in test_results), default=0.0),
            "avg_refusal_test": sum((r["refusal_test"] for r in test_results)) / (len(test_results) or 1),
            "avg_completion_test": sum((r["completion_test"] for r in test_results)) / (len(test_results) or 1),
        },
    }

    write_optimization_outputs(
        Path(run_out_dir) / "final",
        frontier=opt_result.frontier,
        population=opt_result.population,
        summary=summary,
        per_generation_full_eval=opt_result.per_generation_full_eval,
    )
    logger.info("[CAUSAL][OPT] Summary: %s", summary)
    logger.info("[CAUSAL][OPT] Results written to %s", run_out_dir.resolve())
    logger.info("[CAUSAL][OPT] Finished optimization: frontier=%d evals=%d", len(opt_result.frontier), opt_result.num_evaluations)