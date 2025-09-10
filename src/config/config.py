"""Unified configuration module.
The public API now exposes:

    - AgentHarmDataConfig
    - ModelConfig
    - OptimizationConfig
    - ExperimentConfig
    - Config (top-level aggregate)
    - load_config() -> Config
"""
from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass, asdict
from typing import Literal, Sequence


@dataclass(slots=True)
class AgentHarmDataConfig:
    task_name: Literal["harmful", "benign", "chat"] = "harmful"
    split: Literal["val", "test_private", "test_public"] = "val"
    behavior_ids: Sequence[str] | None = None
    sample_ids: Sequence[str] | None = None
    detailed_behaviors: bool | None = True
    hint_included: bool | None = False
    n_irrelevant_tools: int = 0
    shuffle_seed: int = 0
    train_fraction: float = 0.2  # portion of total examples used for train

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1 (exclusive)")

@dataclass(slots=True)
class ModelConfig:
    lm_name: str = "openai/gpt-5-nano"
    lm_temperature: float = 1.0
    max_tokens: int = 2048
    refusal_judge_model: str = "openai/gpt-5-nano"
    semantic_judge_model: str = "openai/gpt-5-nano"
    api_base: str | None = None
    headers: dict | None = None
    seed: int | None = None

@dataclass(slots=True)
class OptimizationConfig:
    optim_seed: int = 6793115
    auto_mode: Literal["light", "medium", "heavy"] = "light"
    num_threads: int = 4
    # Which optimization algorithm to use. Supported: 'mipro', 'copro', 'gepa'
    algorithm: Literal["mipro", "copro", "gepa"] = "gepa"
    # Whether to run the optimization stage after baseline evaluation
    run_optimization: bool = True
    # If true, use refusal-oriented metrics (refusal rate) instead of task success score during optimization
    optimize_refusal: bool = False

@dataclass(slots=True)
class ExperimentConfig:
    uri: str = "http://127.0.0.1:5000"
    name: str = "agentharm_experiment"

@dataclass(slots=True)
class CausalConfig:
    enabled: bool = False
    run_name: str | None = None          # Explicit mlflow run name
    param_key: str = "WebReActAgent.react.predict.signature.instructions"
    child_prefix: str = "eval_full_"
    max_collected_prompts: int = 200  # passed as limit
    seed: int = 42
    output_dir: str = "results/causal"  # base directory; timestamped subdir created per run
    optimization: CausalOptimizationConfig | None = None       # Nested config for instruction optimization

@dataclass(slots=True)
class CausalOptimizationConfig:
    """Nested optimization settings specific to causal instruction search.

    Distinct from top-level OptimizationConfig used in baseline/batch runs.
    Includes evaluation sampling controls (e.g., max_data_size) and
    mutation / segmentation model configuration.
    """
    frontier_size: int = 16
    max_generations: int = 10
    random_seed: int = 42
    target_completion: float = 1.0
    target_refusal: float = 1.0
    max_candidates_evaluated: int | None = 500
    max_data_size: int | None = None     # Clamp Max Dataset size (for debugging)
    minibatch_size: int = 5            # How many examples to sample per eval step (<= max_data_size)
    segment_lm_name: str = "openai/gpt-4o-mini"
    segment_model_api_base: str = "https://api.openai.com/v1"

    def clamp(self) -> None:
        if self.frontier_size <= 0:
            self.frontier_size = 1
        if self.max_generations <= 0:
            self.max_generations = 1
        if self.minibatch_size <= 0:
            self.minibatch_size = 1

@dataclass(slots=True)
class Config:
    data: AgentHarmDataConfig
    models: ModelConfig
    optimization: OptimizationConfig
    experiment: ExperimentConfig
    # Optional causal sub-config (duck-typed; not required by existing runs)
    causal: CausalConfig | None = None

    def as_flat_dict(self) -> dict:
        flat = {}
        flat.update(asdict(self.data))
        flat.update({f"model_{k}": v for k, v in asdict(self.models).items()})
        flat.update({f"optim_{k}": v for k, v in asdict(self.optimization).items()})
        flat.update({f"exp_{k}": v for k, v in asdict(self.experiment).items()})
        return flat


def _mask_sensitive(obj):
    """Recursively mask common sensitive keys/values in nested structures."""
    sensitive_keys = {"key", "token", "secret", "password", "authorization", "auth"}
    if isinstance(obj, dict):
        masked = {}
        for k, v in obj.items():
            if any(s in str(k).lower() for s in sensitive_keys):
                masked[k] = "***"
            else:
                masked[k] = _mask_sensitive(v)
        return masked
    if isinstance(obj, (list, tuple)):
        return type(obj)(_mask_sensitive(v) for v in obj)
    return obj


def _config_safe_for_logging(cfg: Config) -> dict:
    """Return a flat config dict with sensitive fields masked for safe logging."""
    flat = cfg.as_flat_dict()
    # Special-case: mask any header values entirely if present
    headers = flat.get("model_headers")
    if isinstance(headers, dict):
        flat["model_headers"] = {k: "***" for k in headers.keys()}
    return _mask_sensitive(flat)

def load_config(config_path: str = "src/config/config.yaml") -> Config:
    with open(config_path, "r") as file:
        raw = yaml.safe_load(file)

    # Handle sample_ids as comma-separated string or list
    data_dict = dict(raw["data"])
    sample_ids = data_dict.get("sample_ids", None)
    if isinstance(sample_ids, str):
        # Split by comma and strip whitespace
        data_dict["sample_ids"] = [s.strip() for s in sample_ids.split(",") if s.strip()]
    data = AgentHarmDataConfig(**data_dict)
    models_dict = dict(raw["models"])
    # Optionally convert seed to int if present and not None
    if "seed" in models_dict and models_dict["seed"] is not None:
        try:
            models_dict["seed"] = int(models_dict["seed"])
        except Exception:
            pass
    models = ModelConfig(**models_dict)
    # Normalize optimization values (e.g., algorithm case)
    optimization_dict = dict(raw["optimization"])
    algo = optimization_dict.get("algorithm")
    if isinstance(algo, str):
        optimization_dict["algorithm"] = algo.lower()
    optimization = OptimizationConfig(**optimization_dict)
    experiment = ExperimentConfig(**raw["experiment"])
    causal_cfg = None
    if "causal" in raw:
        causal_raw = dict(raw["causal"])
        causal_opt_raw = dict(causal_raw.pop("optimization", None))
        causal_cfg = CausalConfig(**causal_raw)
        if causal_opt_raw:
            causal_opt = CausalOptimizationConfig(**causal_opt_raw)  # type: ignore[arg-type]
            causal_opt.clamp()
            causal_cfg.optimization = causal_opt

    cfg = Config(
        data=data, models=models, optimization=optimization, experiment=experiment, causal=causal_cfg
    )

    # Log the loaded configuration at INFO level (with sensitive fields masked)
    logger = logging.getLogger(__name__)
    try:
        logger.info("Loaded configuration: %s", _config_safe_for_logging(cfg))
    except Exception:
        # Don't fail loading if logging/serialization has issues
        logger.debug("Failed to log configuration safely", exc_info=True)

    return cfg

__all__ = [
    "AgentHarmDataConfig",
    "ModelConfig",
    "OptimizationConfig",
    "ExperimentConfig",
    "CausalOptimizationConfig",
    "CausalConfig",
    "Config",
    "load_config",
]
