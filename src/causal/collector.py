"""Collection utilities for retrieving prompt/result pairs from MLflow.

Uses MLflowClient to:
  - Discover runs (either by explicit run IDs or recent runs in an experiment)
  - Fetch artifacts (e.g., traces) that contain prompts + model outputs
  - Normalize them into PromptRecord objects for downstream analysis

Assumptions / Conventions:
  - Artifacts under a subdirectory like 'traces/' or any user-provided pattern
  - Each trace file is JSON or JSONL with fields that allow recovering the
    original prompt and result. We keep this flexible; parsing is pluggable.

If direct MLflow access is unavailable, future fallback could read from the
local `mlartifacts/` mirror. For now we rely on the tracking server.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging

from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptRecord:
    run_id: str
    prompt_id: str
    prompt_text: str
    result_text: str | None
    label: float | None
    raw: dict  # Original parsed structure for additional introspection


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def collect_prompts(
    experiment_name: str,
    run_name: str | None = None,
    model_lm_name: str | None = None,
    param_key: str = "WebReActAgent.react.predict.signature.instructions",
    child_prefix: str = "eval_full_",
    limit: int | None = None,
) -> list[PromptRecord]:
    """Collect instruction strings as prompts from evaluation child runs.

    Parameters
    ----------
    experiment_name: str
        MLflow experiment name.
    run_name: str | None
        Optional explicit parent run name (tag ``mlflow.runName``). If absent,
        ``model_lm_name`` must be supplied.
    model_lm_name: str | None
        Parent selection fallback using parameter ``model_lm_name``.
    param_key: str
        Parameter key holding instruction text.
    child_prefix: str
        Child run name prefix (default ``eval_full_``) to filter evaluation runs.
    limit: int | None
        Optional cap on number of instruction prompts returned.

    Returns
    -------
    list[PromptRecord]
        Each record's ``prompt_text`` is the instruction string. ``result_text``
        and ``label`` are left ``None``.
    """
    logger.info("[COLLECT] Collecting prompts from experiment '%s' (run_name=%s, model_lm_name=%s)", experiment_name, run_name, model_lm_name)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Identify parent runs
    # Build base filter depending on selector type
    if run_name:
        base_filter = f"tags.mlflow.runName = '{run_name}'"
    else:
        if not model_lm_name:
            raise ValueError("Either run_name or model_lm_name must be provided")
        base_filter = f"params.model_lm_name = '{model_lm_name}'"

    parent_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=base_filter,
    )
    logger.info("Searched %d parent runs with filter: %s", len(parent_runs), base_filter)
    if not parent_runs:
        raise ValueError(f"No parent runs found by filter: {base_filter}")

    out: list[PromptRecord] = []
    seen: set[str] = set()

    for parent in parent_runs:
        parent_id = parent.info.run_id
        logger.info("Using parent run %s (name=%s)", parent_id, parent.data.tags.get("mlflow.runName", ""))

        child_runs = client.search_runs(
            [parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
            max_results=500,
        )
        logger.info("Found %d child runs under parent %s", len(child_runs), parent_id)

        for run in child_runs:
            child_name = run.data.tags.get("mlflow.runName", "")
            if not child_name.startswith(child_prefix):
                continue
            if param_key not in run.data.params:
                continue
            instructions = run.data.params[param_key]
            prompt_id = _hash_text(instructions)
            if prompt_id in seen:
                continue
            seen.add(prompt_id)
            out.append(
                PromptRecord(
                    run_id=run.info.run_id,
                    prompt_id=prompt_id,
                    prompt_text=instructions,
                    result_text=None,
                    label=None,
                    raw={
                        "source": "eval_instructions",
                        "child_run_name": child_name,
                        "param_key": param_key,
                        "parent_run_id": parent_id,
                        "source_metrics": run.data.metrics,
                    },
                )
            )
            if limit and len(out) >= limit:
                return out

    return out
