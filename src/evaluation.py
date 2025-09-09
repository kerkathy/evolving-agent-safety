from typing import Literal, cast
import logging
from src.data import load_agentharm_data, build_dspy_examples
from src.metrics import AgentHarmMetricFactory
import mlflow
import dspy

logger = logging.getLogger("evaluation")

# Cache for safety panel datasets and metric factories to avoid repeated heavy instantiation
safety_cache: dict = {"devsets": {}, "factories": {}}

def _load_dev_examples(config, task_name: Literal["harmful", "benign", "chat"]):
    data_cfg = config.data
    raw = load_agentharm_data(
        behavior_ids=list(data_cfg.behavior_ids or []),
        sample_ids=list(data_cfg.sample_ids or []),
        task_name=cast(Literal["harmful", "benign", "chat"], task_name),
        split=data_cfg.split,
        detailed_behaviors=data_cfg.detailed_behaviors,
        hint_included=data_cfg.hint_included,
        n_irrelevant_tools=data_cfg.n_irrelevant_tools,
    )
    examples = build_dspy_examples(raw)
    # Use all examples directly as a dev/eval set (no split needed here)
    return examples

def evaluate_safety_panels(current_agent, config, step: int):
    """Evaluate and log benign task score and harmful refusal.

    Uses cached datasets and metric factories (created on first call) to minimize
    memory churn and redundant model client construction.
    Metrics logged: 'benign_task_score' (mean task score) and 'harmful_refusal' (refusal rate).
    """
    panels = [
        {
            "task": "benign",
            "metric_name": "benign_task_score",
            "metric_attr": "task_score_mean",
            "metric_fn_selector": "metric",  # task score
            "summary_phase": "benign_eval",
        },
        {
            "task": "harmful",
            "metric_name": "harmful_refusal",
            "metric_attr": "refusal_rate",
            "metric_fn_selector": "refusal_metric",  # refusal rate
            "summary_phase": "harmful_eval",
        },
    ]

    for panel in panels:
        task = panel["task"]
        try:
            # Dataset caching
            if task not in safety_cache["devsets"]:
                devset = _load_dev_examples(config, cast(Literal["harmful", "benign", "chat"], task)) or []
                safety_cache["devsets"][task] = devset
            devset = safety_cache["devsets"].get(task, [])
            if not devset:
                logger.warning("[Safety Panel] %s dataset empty; skipping %s", task, panel["metric_name"])
                continue

            # Metric factory caching
            if task not in safety_cache["factories"]:
                safety_cache["factories"][task] = AgentHarmMetricFactory(
                    task_name=task,
                    refusal_judge_model=config.models.refusal_judge_model,
                    semantic_judge_model=config.models.semantic_judge_model,
                )
            factory: AgentHarmMetricFactory = safety_cache["factories"][task]
            # Reset records before fresh evaluation (retain heavy model objects)
            factory.reset()

            metric_fn = getattr(factory, panel["metric_fn_selector"])
            eval_runner = dspy.Evaluate(
                devset=devset,
                metric=metric_fn,
                num_threads=1,
                display_progress=False,
                display_table=0,
                max_errors=999,
                provide_traceback=False,
            )
            logger.info(
                "[Safety Panel] Evaluating %s via %s (n=%d)",
                panel["metric_name"],
                panel["metric_fn_selector"],
                len(devset),
            )
            eval_runner(current_agent)
            summary = factory.summarize_and_log(panel["summary_phase"], reset=True)
            if summary:
                mlflow.log_metric(panel["metric_name"], float(summary.get(panel["metric_attr"], 0.0)), step=step)
        except Exception as e:
            logger.warning(
                "[Safety Panel] %s evaluation failed: %s", panel["metric_name"], e, exc_info=True
            )
