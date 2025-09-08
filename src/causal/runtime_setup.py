"""Runtime setup helpers for causal pipeline.

Separates environment configuration and agent/metric construction from
`pipeline.py` to keep orchestration logic concise.
"""
from __future__ import annotations

import logging
import dspy

from src.utils.enhanced_dspy import create_enhanced_dspy_lm
from src.adapter import FunctionCallAdapter
from src.agent import WebReActAgent
from src.metrics import AgentHarmMetricFactory

logger = logging.getLogger(__name__)


def configure_dspy(models_cfg, api_key: str) -> None:
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set; cannot configure DSPy.")
    dspy.configure(
        lm=create_enhanced_dspy_lm(models_cfg, api_key),
        adapter=FunctionCallAdapter(),
    )
    logger.info("[SETUP] DSPy language model configured.")


def build_agent_and_metric(config) -> tuple[WebReActAgent, AgentHarmMetricFactory]:
    agent = WebReActAgent()
    metric_factory = AgentHarmMetricFactory(
        task_name=config.data.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    logger.info("[SETUP] Created agent and metric factory (task=%s)", config.data.task_name)
    return agent, metric_factory


__all__ = ["configure_dspy", "build_agent_and_metric"]
