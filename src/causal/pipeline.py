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
from typing import Sequence
from datetime import datetime

import mlflow

from .collector import collect_prompts, PromptRecord
from .optimization import optimize_instructions, OptimizationConfig
from .eval_utils import load_eval_examples, build_agent_instruction_eval_fn
from .runtime_setup import configure_dspy, build_agent_and_metric

logger = logging.getLogger(__name__)

def run_causal_analysis(config, experiment_name: str) -> None:
    mlflow.set_tracking_uri(config.experiment.uri)  # Adjust as needed

    cconf = config.causal
    if not cconf or not cconf.enabled:
        logger.info("Causal analysis disabled; skipping.")
        return

    logger.info("[CAUSAL] Collecting instruction prompts (max=%s)...", cconf.max_prompts)
    if not (cconf.run_name or cconf.model_lm_name):
        logger.warning("[CAUSAL] Neither run_name nor model_lm_name provided; prompt collection will likely fail.")
    records: list[PromptRecord] = collect_prompts(
        experiment_name,
        run_name=cconf.run_name,
        model_lm_name=cconf.model_lm_name,
        param_key=cconf.param_key,
        child_prefix=cconf.child_prefix,
        limit=cconf.max_prompts,
        run_is_optim=cconf.run_is_optim,
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
    run_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[CAUSAL] Results will be written to %s", run_out_dir.resolve())

    # Run optimization over collected instruction prompts (treat each prompt as seed candidate)
    # Adapt nested causal optimization config (CausalOptimizationConfig) to optimization.OptimizationConfig
    # The internal optimization loop expects a config with segmentation_model/openai_api_base etc.
    # We'll map overlapping field names; extras ignored.
    if cconf.optimization is not None:
        opt_src = cconf.optimization
        # Build mapping dict (attribute-style) keeping only keys present in OptimizationConfig
        valid_keys = {f.name for f in OptimizationConfig.__dataclass_fields__.values()}  # type: ignore
        src_dict = {k: getattr(opt_src, k) for k in dir(opt_src) if not k.startswith('_') and hasattr(opt_src, k)}
        filtered = {k: v for k, v in src_dict.items() if k in valid_keys}
        opt_cfg = OptimizationConfig(**filtered)
    else:
        opt_cfg = OptimizationConfig()
    logger.info("[CAUSAL][OPT] Starting multi-objective instruction optimization over %d candidates", len(records))
    initial_texts = [r.prompt_text for r in records]

    # Configure runtime (DSPy + agent + metric)
    api_key = os.getenv("OPENAI_API_KEY") or ""
    configure_dspy(config.models, api_key)
    agent, metric_factory = build_agent_and_metric(config)

    # Load examples (must be non-empty now)
    examples = load_eval_examples(config)
    if not examples:
        raise ValueError("[CAUSAL][OPT] No evaluation examples loaded; cannot proceed with optimization.")
    eval_fn = build_agent_instruction_eval_fn(examples, metric_factory, agent=agent)
    opt_result = optimize_instructions(initial_texts, eval_fn=eval_fn, config=opt_cfg)
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
        "mode": "optimization",
        "n_seeds": len(records),
        "generations": opt_result.generations,
        "evaluations": opt_result.num_evaluations,
        "frontier_size": len(opt_result.frontier),
        "config": asdict(cconf),
        "optimization_config": asdict(opt_cfg),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[CAUSAL][OPT] Summary: %s", summary)
    logger.info("[CAUSAL][OPT] Results written to %s", out_dir.resolve())
    logger.info("[CAUSAL][OPT] Finished optimization: frontier=%d evals=%d", len(opt_result.frontier), opt_result.num_evaluations)