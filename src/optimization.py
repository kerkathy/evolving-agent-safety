import logging
import dspy
from src.utils.enhanced_dspy import create_enhanced_dspy_lm
from src.metrics import AgentHarmMetricFactory

logger = logging.getLogger(__name__)

def optimize_agent(agent, trainset, config, api_key):
    logger.info("Setting up metric factory...")
    metric_factory = AgentHarmMetricFactory(
        task_name=config.data.task_name,
        refusal_judge_model=config.models.refusal_judge_model,
        semantic_judge_model=config.models.semantic_judge_model,
    )
    # Select metric depending on optimization target (task success vs refusal)
    optimize_refusal = getattr(config.optimization, "optimize_refusal", False)
    algo = config.optimization.algorithm
    if optimize_refusal:
        if algo == "gepa":
            metric_fn = metric_factory.refusal_metric_with_feedback
        elif algo in ["mipro", "copro"]:
            metric_fn = metric_factory.refusal_metric
        else:
            raise ValueError(f"Unknown optimization algorithm: {algo}")
        logger.info("Optimization target: REFUSAL (maximize refusal rate)")
    else:
        if algo == "gepa":
            metric_fn = metric_factory.metric_with_feedback
        elif algo in ["mipro", "copro"]:
            metric_fn = metric_factory.task_score_metric
        else:
            raise ValueError(f"Unknown optimization algorithm: {algo}")
        logger.info("Optimization target: TASK SCORE")

    optimizer_name = getattr(config.optimization, "algorithm", "gepa")
    optimizer = None
    if optimizer_name == "mipro":
        logger.info("Using MIPROv2 optimizer")
        optimizer = dspy.MIPROv2(
            metric=metric_fn,
            auto=config.optimization.auto_mode,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_threads=config.optimization.num_threads,
            verbose=False,
        )
        optimized_agent = optimizer.compile(
            agent, trainset=trainset, seed=config.optimization.optim_seed
        )
    elif optimizer_name == "copro":
        logger.info("Using COPRO optimizer")
        optimizer = dspy.COPRO(
            metric=metric_fn,
            verbose=False,
        )
        optimized_agent = optimizer.compile(
            agent, trainset=trainset, eval_kwargs={}
        )
    elif optimizer_name == "gepa":
        logger.info("Using GEPA optimizer")
        optimizer = dspy.GEPA(
            metric=metric_fn,
            auto=config.optimization.auto_mode,
            num_threads=config.optimization.num_threads,
            track_stats=True,
            use_merge=False,
            reflection_lm=create_enhanced_dspy_lm(config.models, api_key),  # type: ignore[arg-type]
        )
        optimized_agent = optimizer.compile(
            agent,
            trainset=trainset,
            # valset=devset,
        )
    else:
        raise ValueError(f"Unknown optimization algorithm: {optimizer_name}")

    logger.info("Optimization complete")
    metric_factory.log_detailed_results("train_detailed_results", reset=False)
    metric_factory.summarize_and_log(
        phase="train",
        task=config.data.task_name,
        reset=True,
        num_records_per_step=len(trainset)
    )

    return optimized_agent
