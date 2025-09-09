#!/usr/bin/env python
"""Entry point orchestrating data prep, evaluation, and optimization.

Refactored: data loading, metrics, and configuration are now handled in
dedicated modules under `src/` to keep this script concise.
"""
from __future__ import annotations


import os
import argparse
from typing import Literal, cast
from dotenv import load_dotenv
import logging

import dspy
import mlflow

from src.config import load_config, Config
from src.data import load_agentharm_data, build_dspy_examples, split_examples
from src.metrics import AgentHarmMetricFactory
from src.agent import WebReActAgent
from src.adapter import FunctionCallAdapter
from src.utils.logging_setup import setup_logging, mlflow_setup
from src.utils.enhanced_dspy import create_enhanced_dspy_lm
from src.evaluation import evaluate_safety_panels
from src.optimization import optimize_agent


# %%
# Logging setup
logging_level = logging.INFO
logging.basicConfig(level=logging_level)
setup_logging()
logger = logging.getLogger("main")

# Set specific loggers to appropriate levels for debugging
logging.getLogger("agentharm_metric").setLevel(logging_level)
logging.getLogger("grading_utils").setLevel(logging_level)
logging.getLogger("combined_scorer").setLevel(logging_level)
logging.getLogger("trajectory_to_messages").setLevel(logging_level)


def prepare_data(config):
    cfg = config.data
    raw_data = load_agentharm_data(
        behavior_ids=list(cfg.behavior_ids or []),
        sample_ids=list(cfg.sample_ids or []),
        task_name=cfg.task_name,
        split=cfg.split,
        detailed_behaviors=cfg.detailed_behaviors,
        hint_included=cfg.hint_included,
        n_irrelevant_tools=cfg.n_irrelevant_tools,
    )
    examples = build_dspy_examples(raw_data)
    trainset, devset = split_examples(
        examples, train_fraction=cfg.train_fraction, seed=cfg.shuffle_seed
    )
    logger.info(
        "Loaded %d examples (task=%s) | train=%d dev=%d",
        len(examples),
        cfg.task_name,
        len(trainset),
        len(devset),
    )
    return trainset, devset


def main():
    parser = argparse.ArgumentParser(description="Run evolving-agent-safety main script.")
    parser.add_argument(
        "--config",
        type=str,
        default="./src/config/config.yaml",
        help="Path to the config YAML file (default: ./src/config/config.yaml)",
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Configuration
    config: Config = load_config(config_path=args.config)

    # MLflow setup
    experiment_name = f"{config.experiment.name}_{config.data.task_name}_{config.data.split}_detail_{config.data.detailed_behaviors}_hint_{config.data.hint_included}"
    mlflow_setup(config.experiment.uri, experiment_name)

    # Language model setup with enhanced timeout and retry handling
    # Custom adapter for better output parsing
    dspy.configure(
        lm=create_enhanced_dspy_lm(config.models, api_key), 
        adapter=FunctionCallAdapter()
    )

    # Main logic
    trainset, devset = prepare_data(config)

    logger.info("Compiling baseline agent...")
    agent = WebReActAgent()
    logger.info("Setting up metric factory...")
    metric_factory = AgentHarmMetricFactory(
        task_name=config.data.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    # Select metric depending on optimization target (task success vs refusal)
    optimize_refusal = getattr(config.optimization, "optimize_refusal", False)
    algo = config.optimization.algorithm
    if optimize_refusal:
        if algo == "gepa":
            metric_fn = metric_factory.refusal_metric_with_feedback
        elif algo in ["mipro", "copro"]:
            metric_fn = metric_factory.refusal_metric
        else:
            raise ValueError(f"Unknown optimization algorithm: {algo}")
        logger.info("Optimization target: REFUSAL (maximize refusal rate)")
    else:
        if algo == "gepa":
            metric_fn = metric_factory.metric_with_feedback
        elif algo in ["mipro", "copro"]:
            metric_fn = metric_factory.metric
        else:
            raise ValueError(f"Unknown optimization algorithm: {algo}")
        logger.info("Optimization target: TASK SCORE")

    logger.info("Setting up evaluation...")
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric_fn,
        num_threads=1,  # Use single thread to avoid async issues
        display_progress=True,
        display_table=0,
        max_errors=999,
        provide_traceback=True,
    )
    params = config.as_flat_dict()

    # Log config params once
    safe_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
    mlflow.log_params(safe_params)

    # Log source snapshot
    if os.path.exists("./src"):
        mlflow.log_artifacts("./src", artifact_path="source_code")
    if os.path.exists(__file__):
        mlflow.log_artifact(__file__, artifact_path="source_code")

    # ---- Baseline Eval ----
    logger.info("Evaluating baseline agent...")
    evaluate(agent)
    metric_factory.log_detailed_results("eval_baseline_detailed_results", reset=False)
    metric_factory.summarize_and_log("eval_baseline", reset=True)

    # Baseline safety panel at step=0
    evaluate_safety_panels(agent, config, step=0)

    # Stop if optimization is disabled
    if config.optimization.run_optimization is False:
        logger.info("Skipping optimization as per config.")
        return
    
    # ---- Optimization ----
    logger.info("Optimizing agent...")
    optimized_agent = optimize_agent(agent, trainset, config, metric_fn, api_key)

    # ---- Optimized Eval ----
    metric_factory.reset()
    logger.info("Evaluating optimized agent...")
    evaluate(optimized_agent, metric=metric_fn)
    metric_factory.log_detailed_results("eval_final_detailed_results", reset=False)
    metric_factory.summarize_and_log("eval_final", reset=True)

    # Post-optimization safety panel at step=1
    evaluate_safety_panels(optimized_agent, config, step=1)

    logger.info("Run complete")

if __name__ == "__main__":
    main()