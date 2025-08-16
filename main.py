#!/usr/bin/env python
"""Entry point orchestrating data prep, evaluation, and optimization.

Refactored: data loading, metrics, and configuration are now handled in
dedicated modules under `src/` to keep this script concise.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
import logging
from datetime import datetime

import dspy
import mlflow

from src.config import load_config, Config
from src.data import load_agentharm_data, build_dspy_examples, split_examples
from src.metrics import AgentHarmMetricFactory
from src.agent import WebReActAgent
from src.utils.logging_setup import setup_logging, mlflow_setup
from src.utils.enhanced_dspy import create_enhanced_dspy_lm

# Load environment variables from .env file
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
print("api_key:", api_key)

logging.basicConfig(level=logging.INFO)

# %%
# Logging setup
setup_logging()
logger = logging.getLogger("main")

# Configuration
config: Config = load_config()

# MLflow setup
experiment_name = f"{config.experiment.name}_{config.data.task_name}"
mlflow_setup(config.experiment.uri, experiment_name)

# Language model setup with enhanced timeout and retry handling
enhanced_lm = create_enhanced_dspy_lm(config.models, api_key)
dspy.configure(lm=enhanced_lm)

# %%
def prepare_data():
    cfg = config.data
    raw_data = load_agentharm_data(
        behavior_ids=list(cfg.behavior_ids or []),
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

# %%
def main():
    trainset, devset = prepare_data()

    logger.info("Compiling baseline agent...")
    agent = WebReActAgent()
    
    logger.info("Setting up metric factory...")
    metric_factory = AgentHarmMetricFactory(
        task_name=config.data.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    metric_fn = metric_factory.metric
    
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
    if os.path.exists("src"):
        mlflow.log_artifacts("src", artifact_path="source_code")
    if os.path.exists("main.py"):
        mlflow.log_artifact("main.py", artifact_path="source_code")

    # ---- Baseline Eval ----
    logger.info("Evaluating baseline agent...")
    evaluate(agent)
    metric_factory.summarize_and_log("baseline", reset=True)
    logger.info("Optimizing agent...")
    optimizer = dspy.MIPROv2(
        metric=metric_fn,
        auto=config.optimization.auto_mode,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_threads=1,  # Use single thread to avoid async issues
        verbose=False,
    )
    optimized_agent = optimizer.compile(
        agent, trainset=trainset, seed=config.optimization.optim_seed
    )        

    mlflow.dspy.log_model(
        optimized_agent,
        name="dspy_model",
    )
    
    # ---- Optimized Eval ----
    metric_factory.reset()
    logger.info("Evaluating optimized agent...")
    evaluate(optimized_agent, metric=metric_fn)
    metric_factory.summarize_and_log("optimized", reset=True)

    logger.info("Run complete: parent_run_id=%s", run.info.run_id)


if __name__ == "__main__":
    main()