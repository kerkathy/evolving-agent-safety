"""Logging and optional MLflow setup utilities."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional
import mlflow

def setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return  # already configured
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# @contextmanager
# def mlflow_run_context(uri: str, experiment_name: str, run_name: str, params: dict):
#     """Context manager that activates MLflow run  is installed.
#     """
#     try:
#         import mlflow  # type: ignore
#     except ImportError:  # pragma: no cover - optional dependency
#         yield None
#         return

#     with mlflow.start_run(run_name=run_name) as run:
#         # Log params once
#         safe_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
#         mlflow.log_params(safe_params)
#         yield run


def mlflow_setup(
    uri: str,
    experiment_name: str,
    *,
    log_traces: bool = True,
    log_compiles: bool = False,
    log_evals: bool = False,
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

    # Enable DSPy autologging if available. Import via importlib to avoid static
    # attribute access assumptions on the mlflow package.
    try:
        import importlib

        mlflow_dspy = importlib.import_module("mlflow.dspy")
    except Exception:
        mlflow_dspy = None

    if mlflow_dspy is not None and hasattr(mlflow_dspy, "autolog"):
        mlflow_dspy.autolog(
            log_traces=log_traces,
            log_compiles=log_compiles,
            log_evals=log_evals,
            log_traces_from_compile=log_traces_from_compile,
            log_traces_from_eval=log_traces_from_eval,
        )

    return True


def mlflow_log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    try:
        import mlflow  # type: ignore
    except ImportError:  # pragma: no cover
        return
    # Filter serializable metrics
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric:
        import mlflow  # type: ignore  # re-import for linters
        mlflow.log_metrics(numeric, step=step)


__all__ = ["setup_logging", "mlflow_run_context", "mlflow_log_metrics"]
