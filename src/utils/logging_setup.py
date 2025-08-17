"""Logging and optional MLflow setup utilities."""
from __future__ import annotations

import logging
from typing import Optional
import mlflow
import subprocess

def setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return  # already configured
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def mlflow_setup(
    uri: str,
    experiment_name: str,
    *,
    log_traces: bool = True,
    log_compiles: bool = True,
    log_evals: bool = True,
    log_traces_from_compile: bool = True,
    log_traces_from_eval: bool = True,
) -> bool:
    """Configure MLflow tracking and enable DSPy autologging.

    Returns True if configuration was applied, False otherwise (e.g., mlflow not
    installed).
    """
    # Set tracking destination and experiment first so subsequent autolog / callbacks
    # have the correct context for artifacts and params.
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    mlflow.dspy.autolog(
        log_traces=log_traces,
        log_compiles=log_compiles,
        log_evals=log_evals,
        log_traces_from_compile=log_traces_from_compile,
        log_traces_from_eval=log_traces_from_eval,
    )

    return True

def mlflow_log_metrics(metrics: dict, step: Optional[int] = None, run_id: Optional[str] = None) -> None:
    # Filter serializable metrics
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric:
        mlflow.log_metrics(numeric, step=step, run_id=run_id)

def get_git_commit_hash(short: bool = True) -> str:
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd.insert(2, "--short")
    result = subprocess.check_output(cmd)
    return result.decode('utf-8').strip()

__all__ = ["setup_logging", "mlflow_log_metrics", "get_git_commit_hash"]
