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

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
from typing import Sequence

import mlflow

from .collector import collect_prompts, PromptRecord
from .interventions import generate_interventions
from .evaluator import evaluate_variants, EvalResult
from .effects import compute_effects
from .optimization import optimize_instructions, OptimizationConfig
from src.metrics import AgentHarmMetricFactory
import dspy
import os
from src.utils.enhanced_dspy import create_enhanced_dspy_lm
from src.adapter import FunctionCallAdapter
from .eval_utils import load_eval_examples, build_agent_instruction_eval_fn
from src.agent import WebReActAgent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CausalConfig:
    enabled: bool = False
    run_name: str | None = None          # Explicit mlflow run name
    model_lm_name: str | None = None    # alternative selector if parent_run_name absent
    param_key: str = "WebReActAgent.react.predict.signature.instructions"
    child_prefix: str = "eval_full_"
    max_prompts: int = 200  # passed as limit
    intervention_types: Sequence[str] = ("drop_instruction", "shuffle_order", "mask_step")
    seed: int = 42
    output_dir: str = "results/causal"
    # Removed artifact_pattern/cache_results since prompt collection now instruction-based
    optimize: bool = False                 # If true run multi-objective optimization instead of static effects
    optimization: dict | None = None       # Nested config for OptimizationConfig


def _infer_causal_config(config) -> CausalConfig | None:
    # Config may or may not have a 'causal' attribute; we stay defensive.
    raw = getattr(config, "causal", None)
    if raw is None:
        raise ValueError("Config missing 'causal' section")
    if isinstance(raw, CausalConfig):
        return raw
    try:
        return CausalConfig(**{k: v for k, v in vars(raw).items() if not k.startswith("_")})
    except Exception:
        # Raw might be a simple namespace/dict
        if isinstance(raw, dict):
            return CausalConfig(**raw)
    return None


def run_causal_analysis(config, experiment_name: str) -> None:
    mlflow.set_tracking_uri(config.experiment.uri)  # Adjust as needed

    cconf = _infer_causal_config(config)
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
    )
    if not records:
        logger.warning("[CAUSAL] No prompts collected; aborting causal analysis.")
        return
    logger.info("[CAUSAL] Collected %d unique prompts.", len(records))
    for rec in records:
        logger.info("[CAUSAL] Sample prompt: %s", rec.prompt_text[:200].replace('\n', ' ') + ("..." if len(rec.prompt_text) > 200 else ""))

    if cconf.optimize:
        # Run optimization over collected instruction prompts (treat each prompt as seed candidate)
        opt_cfg_raw = cconf.optimization or {}
        # Map raw dict into OptimizationConfig (filter unknown keys)
        valid_keys = {f.name for f in OptimizationConfig.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in opt_cfg_raw.items() if k in valid_keys}
        opt_cfg = OptimizationConfig(**filtered)
        logger.info("[CAUSAL][OPT] Starting multi-objective instruction optimization over %d candidates", len(records))
        initial_texts = [r.prompt_text for r in records]
        # --- Ensure DSPy LM is configured (was missing: caused zero eval records) ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not dspy.settings or not getattr(dspy.settings, "lm", None):  # type: ignore
            if not api_key:
                raise ValueError("[CAUSAL][OPT] OPENAI_API_KEY not set; evaluations will be skipped.")
            else:
                try:
                    dspy.configure(
                        lm=create_enhanced_dspy_lm(config.models, api_key),
                        adapter=FunctionCallAdapter(),
                    )
                    logger.info("[CAUSAL][OPT] Configured DSPy LM for instruction evaluation.")
                except Exception as e:
                    logger.warning("[CAUSAL][OPT] Failed to configure DSPy LM: %s", e, exc_info=True)

        # Build real evaluation function aligned with main.py (now modularized)
        examples = load_eval_examples(config, limit=20)
        metric_factory = AgentHarmMetricFactory(
            task_name=config.data.task_name,
            refusal_judge_model=config.models.refusal_judge_model,
            semantic_judge_model=config.models.semantic_judge_model,
        )
        # Initialize a single reusable agent instance; its instruction will be
        # updated in-place for each candidate inside the eval function to avoid
        # repeated instantiation overhead.
        reusable_agent = WebReActAgent()
        eval_fn = build_agent_instruction_eval_fn(examples, metric_factory, agent=reusable_agent)
        opt_result = optimize_instructions(initial_texts, eval_fn=eval_fn, config=opt_cfg)
        if not examples:
            logger.warning("[CAUSAL][OPT] No eval examples loaded; completion/refusal metrics will remain zero.")
        out_dir = Path(cconf.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
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
            "evaluations": opt_result.evaluations,
            "frontier_size": len(opt_result.frontier),
            "config": asdict(cconf),
            "optimization_config": asdict(opt_cfg),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("[CAUSAL][OPT] Summary: %s", summary)
        logger.info("[CAUSAL][OPT] Results written to %s", out_dir.resolve())
        # try:
        #     mlflow.log_artifacts(str(out_dir), artifact_path="causal_analysis")
        # except Exception:
        #     logger.debug("[CAUSAL][OPT] Failed logging artifacts to MLflow", exc_info=True)
        logger.info("[CAUSAL][OPT] Finished optimization: frontier=%d evals=%d", len(opt_result.frontier), opt_result.evaluations)
    else:
        results: list[EvalResult] = []
        for rec in records:
            interventions = generate_interventions(rec.prompt_text, cconf.intervention_types, seed=cconf.seed)
            variant_map = {"original": rec.prompt_text} | {i.kind: i.variant_text for i in interventions}

            def scorer(txt: str):  # placeholder scoring
                if txt == rec.prompt_text and rec.label is not None:
                    return rec.label, {"source": "label"}
                return 0.0, {"source": "default"}

            evals = evaluate_variants(rec.prompt_id, rec.prompt_text, variant_map, scorer)
            results.extend(evals)

        effects = compute_effects(results)
        out_dir = Path(cconf.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        effects_json = [asdict(e) for e in effects]
        with open(out_dir / "prompt_effects.json", "w") as f:
            json.dump(effects_json, f, indent=2)
        summary = {
            "mode": "effects",
            "n_prompts": len(records),
            "n_effects": len(effects),
            "config": asdict(cconf),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        try:
            mlflow.log_artifacts(str(out_dir), artifact_path="causal_analysis")
        except Exception:
            logger.debug("[CAUSAL] Failed logging artifacts to MLflow", exc_info=True)
        logger.info("[CAUSAL] Finished causal analysis: %d prompts", len(records))
