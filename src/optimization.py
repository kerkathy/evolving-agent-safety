import logging
import dspy
from src.utils.enhanced_dspy import create_enhanced_dspy_lm

logger = logging.getLogger("optimization")

def optimize_agent(agent, trainset, config, metric_fn, api_key):
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

    return optimized_agent
