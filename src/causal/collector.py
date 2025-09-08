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
from typing import Iterable, Sequence, Callable
import hashlib
import json
import logging
import os

from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

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


def default_trace_parser(content: str) -> Iterable[PromptRecord]:
    """Parse a trace artifact content into PromptRecord objects.

    Supports JSON (list/dict) or JSONL lines. Tries to infer field names:
      - prompt => any key containing 'prompt'
      - output/response => key containing 'response' or 'output'
      - score/label => key containing 'score' or 'label'
    This is intentionally heuristic; adapt as tracing schema evolves.
    """
    def yield_from_obj(obj):
        if not isinstance(obj, dict):
            return
        lower_keys = {k.lower(): k for k in obj.keys()}
        # Guess prompt
        prompt_key = next((k for lk, k in lower_keys.items() if "prompt" in lk), None)
        output_key = next((k for lk, k in lower_keys.items() if any(x in lk for x in ("response", "output", "answer"))), None)
        score_key = next((k for lk, k in lower_keys.items() if any(x in lk for x in ("score", "label"))), None)
        if not prompt_key:
            return
        prompt_text = obj.get(prompt_key, "")
        result_text = obj.get(output_key) if output_key else None
        label = obj.get(score_key) if score_key else None
        yield PromptRecord(
            run_id="",  # Filled in by caller
            prompt_id=_hash_text(str(prompt_text)),
            prompt_text=str(prompt_text),
            result_text=str(result_text) if result_text is not None else None,
            label=float(label) if isinstance(label, (int, float)) else None,
            raw=obj,
        )

    # Attempt JSON / JSONL parsing
    try:
        if "\n" in content:
            # Try JSONL first
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            parsed_any = False
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                parsed_any = True
                yield from yield_from_obj(obj)
            if parsed_any:
                return
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                yield from yield_from_obj(item)
        else:
            yield from yield_from_obj(data)
    except Exception:
        logger.debug("Failed to parse trace content heuristically.", exc_info=True)
        return


def collect_prompts(
    experiment_name: str,
    run_name: str | None = None,
    model_lm_name: str | None = None,
    param_key: str = "WebReActAgent.react.predict.signature.instructions",
    child_prefix: str = "eval_full_",
    limit: int | None = None,
) -> list[PromptRecord]:
    """Collect instruction strings as prompts from evaluation child runs.

    This REPLACES the previous artifact-based prompt harvesting. Now a "prompt"
    corresponds to the instructions parameter captured in evaluation child
    runs (those whose runName starts with ``child_prefix``), matching the logic
    in `test_load_artifact.py`.

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
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Identify parent run
    if run_name:
        parent_runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"""
            tags.mlflow.runName = '{run_name}'
                AND params.optim_run_optimization = 'True'
            """,
        )
    else:
        if not model_lm_name:
            raise ValueError("Either run_name or model_lm_name must be provided")
        parent_runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"""
            params.model_lm_name = '{model_lm_name}'
                AND params.optim_run_optimization = 'True'
            """,
        )
    if not parent_runs:
        raise ValueError("No matching parent run found")
    parent = parent_runs[0]
    parent_id = parent.info.run_id
    logger.info("Using parent run %s (name=%s)", parent_id, parent.data.tags.get("mlflow.runName", ""))

    child_runs = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        max_results=500,
    )
    logger.info("Found %d child runs under parent", len(child_runs))

    out: list[PromptRecord] = []
    seen: set[str] = set()
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
            break
    return out
