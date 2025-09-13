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
import random

import mlflow

from .collector import collect_prompts, PromptRecord, _hash_text
from .optimization import optimize_instructions, Candidate
from .io_utils import write_optimization_outputs
from .eval_utils import (
    load_eval_examples,
    build_full_eval_fn,
    build_minibatch_eval_fn,
)
from src.metrics import AgentHarmMetricFactory
from .runtime_setup import configure_dspy
from src.agent import WebReActAgent

logger = logging.getLogger(__name__)

def _load_candidates_from_json(file_path: Path) -> list[Candidate]:
    """Load candidates from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    candidates = []
    for item in data:
        candidates.append(Candidate(
            text=item["text"],
            refusal=item["refusal"],
            completion=item["completion"],
            meta=item.get("meta", {})
        ))
    return candidates

def _find_latest_gen_folder(resume_folder: Path) -> tuple[Path, int]:
    """Find the latest gen_00x folder and return the folder path and generation number."""
    gen_folders = [f for f in resume_folder.iterdir() if f.is_dir() and f.name.startswith("gen_")]
    if not gen_folders:
        raise ValueError(f"No gen_ folders found in {resume_folder}")
    # Sort by generation number
    gen_folders.sort(key=lambda x: int(x.name.split("_")[1]))
    latest = gen_folders[-1]
    gen_num = int(latest.name.split("_")[1])
    return latest, gen_num

def _load_resume_state(resume_folder: Path) -> tuple[list[Candidate], list[Candidate], int, dict | None]:
    """Load population, frontier, generation, and segment effects from resume folder."""
    latest_gen_folder, gen_num = _find_latest_gen_folder(resume_folder)
    checkpoints_dir = latest_gen_folder / "checkpoints"
    
    population_file = checkpoints_dir / "optimization_population.json"
    frontier_file = checkpoints_dir / "optimization_frontier.json"
    summary_file = checkpoints_dir / "summary.json"
    
    if not population_file.exists() or not frontier_file.exists():
        raise FileExistsError(f"Missing checkpoint files in {checkpoints_dir}")
    
    population = _load_candidates_from_json(population_file)
    frontier = _load_candidates_from_json(frontier_file)
    
    segment_effects = None
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        seg_eff = summary.get("segment_effects")
        # Convert avg-based effects stored in checkpoints to sum-based format expected by optimizer
        if isinstance(seg_eff, dict):
            converted: dict[str, dict[str, float]] = {}
            for key, rec in seg_eff.items():
                if not isinstance(rec, dict):
                    continue
                try:
                    count = float(rec.get("count", 0.0) or 0.0)
                except Exception:
                    count = 0.0
                # Prefer existing sums if already present; else derive from averages
                eff_r = rec.get("effect_on_refusal_sum")
                eff_c = rec.get("effect_on_completion_sum")
                if eff_r is None:
                    avg_r = rec.get("avg_effect_on_refusal")
                    try:
                        eff_r = float(avg_r) * count if avg_r is not None else 0.0
                    except Exception:
                        eff_r = 0.0
                if eff_c is None:
                    avg_c = rec.get("avg_effect_on_completion")
                    try:
                        eff_c = float(avg_c) * count if avg_c is not None else 0.0
                    except Exception:
                        eff_c = 0.0
                try:
                    eff_r = float(eff_r or 0.0)
                except Exception:
                    eff_r = 0.0
                try:
                    eff_c = float(eff_c or 0.0)
                except Exception:
                    eff_c = 0.0
                converted[key] = {
                    "count": count,
                    "effect_on_refusal_sum": eff_r,
                    "effect_on_completion_sum": eff_c,
                }
            segment_effects = converted
        else:
            segment_effects = None
            raise ValueError("segment_effects in summary.json is not a dict; cannot resume.")
    logger.info(
        "[CAUSAL] Resuming effect: total refusal (%.5f), total completion (%.5f)",
        float(segment_effects["system_role"]["effect_on_refusal_sum"]),
        float(segment_effects["system_role"]["effect_on_completion_sum"])
    )
    
    return population, frontier, gen_num, segment_effects

def run_causal_analysis(config, experiment_name: str) -> None:
    mlflow.set_tracking_uri(config.experiment.uri)  # Adjust as needed

    cconf = config.causal
    if not cconf or not cconf.enabled:
        logger.info("Causal analysis disabled; skipping.")
        return

    # Check if resuming from previous run
    resume_mode = cconf.resume_folder is not None
    if resume_mode:
        resume_folder = Path(cconf.resume_folder)
        if not resume_folder.exists():
            raise FileNotFoundError(f"Resume folder {resume_folder} does not exist")
        logger.info("[CAUSAL] Resuming from %s", resume_folder.resolve())
        run_out_dir = resume_folder
        initial_population, initial_frontier, start_generation, initial_segment_effects = _load_resume_state(resume_folder)
        records = []  # Not needed for resume
        baseline_results = []
        # Load baseline results if exists
        baseline_file = resume_folder / "baseline_results.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            baseline_results = baseline_data.get("baseline_results", [])
            logger.info("[CAUSAL] Loaded %d baseline results", len(baseline_results))
    else:
        logger.info("[CAUSAL] Collecting instruction prompts (max=%s)...", cconf.max_collected_prompts)
        if cconf.collect_from:
            if cconf.collect_from.startswith("mlflow-artifacts:/"):
                cconf.collect_from = cconf.collect_from.replace("mlflow-artifacts:/", "mlartifacts/")
            with open(cconf.collect_from, 'r') as f:
                data = json.load(f)
            instructions = [data["react.predict"]["signature"]["instructions"]]
            records = []
            for instr in instructions[:cconf.max_collected_prompts]:
                prompt_id = _hash_text(instr)
                records.append(PromptRecord(
                    run_id="file",
                    prompt_id=prompt_id,
                    prompt_text=instr,
                    result_text=None,
                    label=None,
                    raw={"source": "json_file", "file": cconf.collect_from}
                ))
        else:
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
    if resume_mode:
        logger.info("[CAUSAL][OPT] Resuming multi-objective instruction optimization from generation %d", start_generation)
        initial_texts = []  # Not needed for resume
    else:
        logger.info("[CAUSAL][OPT] Starting multi-objective instruction optimization over %d candidates", len(records))
        initial_texts = [r.prompt_text for r in records]

    # Configure runtime (DSPy + agent + metric)
    api_key = os.getenv("OPENAI_API_KEY") or ""
    main_lm = configure_dspy(config.models, api_key)
    # agent, benign_metric_factory, harmful_metric_factory = build_agent_and_dual_metric(config)
    
    agent = WebReActAgent()
    benign_metric_factory = AgentHarmMetricFactory(
        task_name="benign",
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    harmful_metric_factory = AgentHarmMetricFactory(
        task_name="harmful",
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    # Recreate fresh metric factories for full evaluation to avoid mixing records with train-slice eval.
    benign_full_metric_factory = AgentHarmMetricFactory(
        task_name="benign",
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    harmful_full_metric_factory = AgentHarmMetricFactory(
        task_name="harmful",
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    logger.info("[SETUP] Created agent and dual metric factories (benign_model=%s, harmful_model=%s)", config.models.refusal_judge_model, config.models.semantic_judge_model)

    # Load examples (separate benign/harmful)
    # We now load the FULL pool and rely on a per-generation minibatch sampler
    # to reduce overfitting and memory footprint.
    examples_by_task = load_eval_examples(config, full=True)
    if not examples_by_task.get("benign") or not examples_by_task.get("harmful"):
        raise ValueError("[CAUSAL][OPT] Missing benign or harmful examples; cannot proceed with dual-metric optimization.")
    train_eval_fn = build_minibatch_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_metric_factory,
        harmful_metric_factory=harmful_metric_factory,
        agent=agent,
        minibatch_size=cconf.optimization.minibatch_size,
        rng=random.Random(cconf.optimization.random_seed),
        split='train',
    )
    dev_eval_fn = build_full_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_full_metric_factory,
        harmful_metric_factory=harmful_full_metric_factory,
        agent=agent,
        split='eval',
    )
    test_eval_fn = build_full_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_full_metric_factory,
        harmful_metric_factory=harmful_full_metric_factory,
        agent=agent,
        split='test',
    )

    # Evaluate baseline prompts on test set (skip if resuming)
    if not resume_mode:
        logger.info("[CAUSAL][BASELINE] Evaluating baseline prompts on test set (size=%d)", len(records))
        baseline_results = []
        for rec in records:
            try:
                r, c, extra = test_eval_fn(rec.prompt_text)
                baseline_results.append({
                    "text": rec.prompt_text,
                    "refusal_test": r,
                    "completion_test": c,
                    **extra,
                })
            except Exception as e:
                logger.warning("[CAUSAL][BASELINE] Failed to evaluate prompt %s: %s", rec.prompt_id, e)
                baseline_results.append({
                    "text": rec.prompt_text,
                    "refusal_test": 0.0,
                    "completion_test": 0.0,
                    "error": str(e),
                })
        logger.info("[CAUSAL][BASELINE] Baseline evaluation completed for %d prompts", len(baseline_results))

        # Save baseline results immediately
        baseline_summary = {
            "best_refusal_test": max((r["refusal_test"] for r in baseline_results), default=0.0),
            "best_completion_test": max((r["completion_test"] for r in baseline_results), default=0.0),
            "avg_refusal_test": sum((r["refusal_test"] for r in baseline_results)) / (len(baseline_results) or 1),
            "avg_completion_test": sum((r["completion_test"] for r in baseline_results)) / (len(baseline_results) or 1),
            "baseline_results": baseline_results,
            
        }
        with open(run_out_dir / "baseline_results.json", 'w') as f:
            json.dump({"baseline_results": baseline_results, "baseline_summary": baseline_summary}, f, indent=2)
        logger.info("[CAUSAL][BASELINE] Baseline results saved to %s", run_out_dir / "baseline_results.json")

    opt_result = optimize_instructions(
        initial_texts,
        main_lm=main_lm,
        train_eval_fn=train_eval_fn,
        dev_eval_fn=dev_eval_fn,
        config=cconf.optimization,
        checkpoint_dir=str(run_out_dir),
        initial_population=initial_population if resume_mode else None,
        initial_frontier=initial_frontier if resume_mode else None,
        start_generation=start_generation if resume_mode else 0,
        initial_segment_effects=initial_segment_effects if resume_mode else None,
    )

    # Evaluate final frontier on test set
    logger.info("[CAUSAL][TEST] Evaluating final frontier on test set (size=%d)", len(opt_result.frontier))
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
            "avg_refusal_test": sum((r["refusal_test"] for r in test_results)) / (len(test_results) or 1),
            "avg_completion_test": sum((r["completion_test"] for r in test_results)) / (len(test_results) or 1),
            "best_refusal_test": max((r["refusal_test"] for r in test_results), default=0.0),
            "best_completion_test": max((r["completion_test"] for r in test_results), default=0.0),
        },
    }

    write_optimization_outputs(
        Path(run_out_dir) / "final",
        frontier=opt_result.frontier,
        population=opt_result.population,
        summary=summary,
    )
    logger.info("[CAUSAL][OPT] Summary: %s", summary)
    logger.info("[CAUSAL][OPT] Results written to %s", run_out_dir.resolve())
    logger.info("[CAUSAL][OPT] Finished optimization: frontier=%d evals=%d", len(opt_result.frontier), opt_result.num_evaluations)