"""CLI entrypoint for causal sufficiency / necessity analysis.

Example:
    python -m experiments.causal_eval --config src/config/config.yaml \
        --max-prompts 50 --interventions drop_instruction,mask_step

Relies on an existing MLflow experiment populated by prior runs.
"""
from __future__ import annotations


import argparse
import logging
import os
from dotenv import load_dotenv

from src.config import load_config
from src.causal import run_causal_analysis


# Ensure logs are shown on the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run causal prompt analysis.")
    p.add_argument("--config", default="src/config/config_causal_min.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv(override=True)
    cfg = load_config(args.config)
    # Lightweight overrides
    if getattr(cfg, "causal", None) is None:
        cfg.causal = {}

    exp_name = f"{cfg.experiment.name}_{cfg.data.task_name}_{cfg.data.split}_detail_{cfg.data.detailed_behaviors}_hint_{cfg.data.hint_included}"  # reuse naming scheme subset
    run_causal_analysis(cfg, experiment_name=exp_name)


if __name__ == "__main__":
    main()
