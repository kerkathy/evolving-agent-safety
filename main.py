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
from src.data import load_agentharm_data, build_dspy_examples, split_indices
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
    main_task = cast(Literal["harmful", "benign", "chat"], cfg.task_name)
    another_task = "benign" if main_task == "harmful" else "harmful"

    main_raw_data = load_agentharm_data(
        behavior_ids=list(cfg.behavior_ids or []),
        sample_ids=list(cfg.sample_ids or []),
        task_name=main_task,
        split=cfg.split,
        detailed_behaviors=cfg.detailed_behaviors,
        hint_included=cfg.hint_included,
        n_irrelevant_tools=cfg.n_irrelevant_tools,
    )
    examples = build_dspy_examples(main_raw_data)

    another_raw_data = load_agentharm_data(
        behavior_ids=list(cfg.behavior_ids or []),
        sample_ids=list(cfg.sample_ids or []),
        task_name=another_task,
        split=cfg.split,
        detailed_behaviors=cfg.detailed_behaviors,
        hint_included=cfg.hint_included,
        n_irrelevant_tools=cfg.n_irrelevant_tools,
    )
    another_examples = build_dspy_examples(another_raw_data)

    train_idx, test_idx = split_indices(
        list(range(len(examples))), train_fraction=cfg.train_fraction, seed=cfg.shuffle_seed
    )
    trainset = [examples[i] for i in train_idx]
    testset = {"harmful": [], "benign": []}
    testset[main_task] = [examples[i] for i in test_idx]
    testset[another_task] = [another_examples[i] for i in test_idx]

    logger.info(
        "Loaded %d examples (task=%s) | train=%d test=%d",
        len(examples),
        cfg.task_name,
        len(train_idx),
        len(test_idx),
    )
    return trainset, testset


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
    dspy.configure(
        lm=create_enhanced_dspy_lm(config.models, api_key), 
        adapter=FunctionCallAdapter()
    )

    trainset, testset = prepare_data(config)

    logger.info("Compiling baseline agent...")
    agent = WebReActAgent()

    params = config.as_flat_dict()
    safe_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
    mlflow.log_params(safe_params)

    # Log source snapshot
    if os.path.exists("./src"):
        mlflow.log_artifacts("./src", artifact_path="source_code")
    if os.path.exists(__file__):
        mlflow.log_artifact(__file__, artifact_path="source_code")

    # Baseline safety panel at step=0
    evaluate_safety_panels(agent, testset, config, step=0)

    # Stop if optimization is disabled
    if config.optimization.run_optimization is False:
        logger.info("Skipping optimization as per config.")
        return
    
    # ---- Optimization ----
    logger.info("Optimizing agent...")
    optimized_agent = optimize_agent(agent, trainset, config, api_key)

    # Post-optimization safety panel at step=1
    evaluate_safety_panels(optimized_agent, testset, config, step=1)

    logger.info("Run complete")

if __name__ == "__main__":
    main()