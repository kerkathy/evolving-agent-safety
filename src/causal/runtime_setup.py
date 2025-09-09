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


def configure_dspy(models_cfg, api_key: str) -> dspy.BaseLM:
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set; cannot configure DSPy.")
    main_lm = create_enhanced_dspy_lm(models_cfg, api_key)
    dspy.configure(
        lm=main_lm,
        adapter=FunctionCallAdapter(),
    )
    logger.info("[SETUP] DSPy language model configured. LM: %s", models_cfg)
    return main_lm


def build_agent_and_metric(config) -> tuple[WebReActAgent, AgentHarmMetricFactory]:
    agent = WebReActAgent()
    metric_factory = AgentHarmMetricFactory(
        task_name=config.data.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    logger.info("[SETUP] Created agent and metric factory (task=%s, refusal_model=%s, semantic_model=%s)", config.data.task_name, config.models.refusal_judge_model, config.models.semantic_judge_model)
    return agent, metric_factory


def build_agent_and_dual_metric(config) -> tuple[WebReActAgent, AgentHarmMetricFactory, AgentHarmMetricFactory]:
    agent = WebReActAgent()
    benign_metric_factory = AgentHarmMetricFactory(
        task_name="benign",
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    harmful_metric_factory = AgentHarmMetricFactory(
        task_name="harmful",
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    logger.info("[SETUP] Created agent and dual metric factories (benign_model=%s, harmful_model=%s)", config.models.refusal_judge_model, config.models.semantic_judge_model)
    return agent, benign_metric_factory, harmful_metric_factory


__all__ = ["configure_dspy", "build_agent_and_metric", "build_agent_and_dual_metric"]
