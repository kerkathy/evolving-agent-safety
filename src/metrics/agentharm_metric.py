"""AgentHarm scoring / metric logic extracted from the prior `main.py`.

Provides a synchronous metric function usable by DSPy optimizers while keeping
the async evaluation primitives (grading + refusal detection) encapsulated.
"""
from __future__ import annotations

import os, json, tempfile
from typing import Any, Dict, List

import numpy as np
from inspect_ai.model import ChatMessage, ChatMessageAssistant, get_model
import mlflow

from external.inspect_evals.src.inspect_evals.agentharm.scorer import RefusalJudgeLLM
from external.inspect_evals.src.inspect_evals.agentharm.benchmark import harmful_grading_functions, benign_grading_functions
import logging
import threading
from src.utils.logging_setup import mlflow_log_metrics
from src.utils.sync_wrappers import asyncio_run_wrapper, get_sync_grading_function


def trajectory_to_messages(trajectory: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Convert DSPy trajectory entries into inspect_ai ChatMessages."""
    messages: List[ChatMessage] = []
    for step in trajectory:
        if "selected_fn" in step and "args" in step:
            content = f"I'll call {step['selected_fn']} with arguments: {step['args']}"
            if step.get("reasoning"):
                content = f"{step['reasoning']}\n\n{content}"
            messages.append(ChatMessageAssistant(content=content))
        if "return_value" in step and step["return_value"]:
            messages.append(ChatMessageAssistant(content=str(step["return_value"])))
    return messages


class AgentHarmMetricFactory:
    """
    Factory that prepares metric functions bound to a given task name.
    We create/jit the heavy model clients once and re-use them across calls.
    """

    def __init__(self, task_name: str, refusal_judge_model: str, semantic_judge_model: str):
        self.task_name = task_name
        self.refusal_judge_model = RefusalJudgeLLM(judge_model=refusal_judge_model)
        self.semantic_judge_model = get_model(semantic_judge_model)
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._current_run_id: str | None = None  # Track MLflow run ID for logging
        
        # Create synchronous wrappers for async operations
        self._sync_refusal_check = asyncio_run_wrapper(self.refusal_judge_model.check_refusal)

    def _score_sync(self, example, pred, trace=None) -> float:
        """Synchronous scoring function for DSPy compatibility."""
        grading_function_name = getattr(example, "grading_function", "")
        logging.getLogger(__name__).debug("Scoring example with grading function: %s", grading_function_name)
        trajectory = getattr(pred, "trajectory", [])
        messages = trajectory_to_messages(trajectory)
        metadata = dict(getattr(example, "metadata", {}) or {})
        metadata["semantic_judge"] = self.semantic_judge_model

        # Capture run ID on first call
        if self._current_run_id is None and mlflow.active_run():
            self._current_run_id = mlflow.active_run().info.run_id

        # Use sync grading function
        try:
            sync_grading_func = get_sync_grading_function(self.task_name, grading_function_name)
            score_dict = sync_grading_func(metadata, messages)
            task_score = float(np.mean(list(score_dict.values()))) if score_dict else 0.0
        except (AttributeError, Exception) as e:
            self.logger.warning("Failed to get grading function %s: %s", grading_function_name, e)
            score_dict = {}
            task_score = 0.0

        # Log score_dict 
        # mlflow_log_metrics(score_dict) # TODO

        # Refusal detection (binary) - now synchronous
        refusal_score = 0.0
        for message in messages:
            content = getattr(message, "content", None)
            if not content:
                continue
            refusal, _ = self._sync_refusal_check(str(content))
            if refusal:
                refusal_score = 1.0
                break

        # Log summary metrics
        record = {
            "grading_function": grading_function_name,
            "task_score": task_score,
            "refusal": refusal_score,
            "task_name": self.task_name,
            "metadata": metadata,
            "score_dict": score_dict,
        }
        with self._lock:
            self._records.append(record)
        self.logger.debug("Metric record: %s", record)
        
        # mlflow_log_metrics({"task_score": task_score, "refusal": refusal_score})
        return task_score

    def metric(self, example, pred, trace=None) -> float:
        """Synchronous metric for use with DSPy."""
        return float(self._score_sync(example, pred, trace))

    def _calc_and_log_summary(self, records, label, step=None):
        task_scores = [r["task_score"] for r in records]
        refusals = [r["refusal"] for r in records]
        summary = {
            "label": label,
            "n": len(records),
            "task_score_mean": float(np.mean(task_scores)),
            "task_score_std": float(np.std(task_scores)) if len(task_scores) > 1 else 0.0,
            "refusal_rate": float(np.mean(refusals)),
            "refusal_std": float(np.std(refusals)) if len(refusals) > 1 else 0.0,
        }
        mlflow_log_metrics(
            {
                f"{label}_task_score_mean": summary["task_score_mean"],
                f"{label}_task_score_std": summary["task_score_std"],
                f"{label}_refusal_rate": summary["refusal_rate"],
                f"{label}_refusal_std": summary["refusal_std"],
            },
            run_id=self._current_run_id,
            step=step
        )
        self.logger.info(
            "Summary [%s]: n=%d task_score=%.4f±%.4f refusal_rate=%.3f±%.4f",
            label,
            summary["n"],
            summary["task_score_mean"],
            summary["task_score_std"],
            summary["refusal_rate"],
            summary["refusal_std"],
        )
        return summary

    def summarize_and_log(self, label: str, reset: bool = False, num_records_per_step: int = -1) -> dict:
        with self._lock:
            if not self._records:
                return {}
            if num_records_per_step != -1:
                summaries = {}
                total_steps = len(self._records) // num_records_per_step
                for step_idx in range(total_steps):
                    start = step_idx * num_records_per_step
                    end = start + num_records_per_step
                    step_records = self._records[start:end]
                    step_label = f"{label}_step{step_idx+1}"
                    summary = self._calc_and_log_summary(step_records, step_label, step=step_idx)
                    summaries[step_label] = summary
            else:
                summary = self._calc_and_log_summary(self._records, label)

            if reset:
                self._records.clear()
                self._current_run_id = None
    

    def log_detailed_results(self, artifact_name: str = "detailed_results", reset: bool = False):
        """Log all detailed results as a single artifact"""
        with self._lock:
            if not self._records:
                return
                
            # Create comprehensive results
            detailed_results = {
                "task_name": self.task_name,
                "total_examples": len(self._records),
                "examples": self._records.copy()
            }
            
            # Log as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(detailed_results, f, indent=2, default=str)
                temp_path = f.name
            os.rename(temp_path, f"{artifact_name}.json")  # Rename for clarity

            try:
                mlflow.log_artifact(f"{artifact_name}.json", artifact_path="results", run_id=self._current_run_id)
            finally:
                os.unlink(f"{artifact_name}.json")

            if reset:
                self.reset()
                self._current_run_id = None


    def reset(self) -> None:
        with self._lock:
            self._records.clear()


__all__ = [
    "AgentHarmMetricFactory",
    "trajectory_to_messages",
]
