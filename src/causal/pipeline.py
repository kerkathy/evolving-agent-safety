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
from .eval_utils import load_eval_examples, build_agent_instruction_eval_fn
from .runtime_setup import configure_dspy, build_agent_and_metric

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
    agent, metric_factory = build_agent_and_metric(config)

    # Load examples
    examples = load_eval_examples(config)
    if not examples:
        raise ValueError("[CAUSAL][OPT] No evaluation examples loaded; cannot proceed with optimization.")
    eval_fn = build_agent_instruction_eval_fn(examples, metric_factory, agent=agent)
    opt_result = optimize_instructions(initial_texts, main_lm=main_lm, eval_fn=eval_fn, config=cconf.optimization)
    out_dir = run_out_dir
    frontier_json = [
        {
            "text": c.text,
            "refusal": c.refusal,
            "completion": c.completion,
            "meta": c.meta,
        }
        for c in opt_result.frontier
    ]
    population_json = [
        {
            "text": c.text,
            "refusal": c.refusal,
            "completion": c.completion,
            "meta": c.meta,
        }
        for c in opt_result.population
    ]
    with open(out_dir / "optimization_frontier.json", "w") as f:
        json.dump(frontier_json, f, indent=2)
    with open(out_dir / "optimization_population.json", "w") as f:
        json.dump(population_json, f, indent=2)
    summary = {
        "n_seeds": len(records),
        "generations": opt_result.generations,
        "evaluations": opt_result.num_evaluations,
        "frontier_size": len(opt_result.frontier),
        "config": asdict(config),
        "segment_effects": opt_result.segment_effects,  # Added segment effects
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[CAUSAL][OPT] Summary: %s", summary)
    logger.info("[CAUSAL][OPT] Results written to %s", out_dir.resolve())
    logger.info("[CAUSAL][OPT] Finished optimization: frontier=%d evals=%d", len(opt_result.frontier), opt_result.num_evaluations)