"""AgentHarm scoring / metric logic extracted from the prior `main.py`.

Provides a synchronous metric function usable by DSPy optimizers while keeping
the async evaluation primitives (grading + refusal detection) encapsulated.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import numpy as np
from inspect_ai.model import ChatMessage, ChatMessageAssistant, get_model

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
        mlflow_log_metrics(score_dict)

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
        }
        with self._lock:
            self._records.append(record)
        self.logger.debug("Metric record: %s", record)
        
        mlflow_log_metrics({"task_score": task_score, "refusal": refusal_score})
        return task_score

    def metric(self, example, pred, trace=None) -> float:
        """Synchronous metric for use with DSPy."""
        return float(self._score_sync(example, pred, trace))

    def summarize_and_log(self, label: str, reset: bool = True) -> dict:
        with self._lock:
            if not self._records:
                return {}
            task_scores = [r["task_score"] for r in self._records]
            refusals = [r["refusal"] for r in self._records]
            summary = {
                "label": label,
                "n": len(self._records),
                "task_score_mean": float(np.mean(task_scores)),
                "task_score_std": float(np.std(task_scores)) if len(task_scores) > 1 else 0.0,
                "refusal_rate": float(np.mean(refusals)),
            }
            if reset:
                self._records.clear()
        self.logger.info(
            "Summary [%s]: n=%d task_score=%.4f±%.4f refusal_rate=%.3f",
            label,
            summary["n"],
            summary["task_score_mean"],
            summary["task_score_std"],
            summary["refusal_rate"],
        )
        mlflow_log_metrics(
            {
                f"{label}_task_score_mean": summary["task_score_mean"],
                f"{label}_refusal_rate": summary["refusal_rate"],
            }
        )
        return summary

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


__all__ = [
    "AgentHarmMetricFactory",
    "trajectory_to_messages",
]
