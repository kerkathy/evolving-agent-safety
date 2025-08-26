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

@dataclass(slots=True)
class ExperimentConfig:
    uri: str = "http://127.0.0.1:5000"
    name: str = "agentharm_experiment"

@dataclass(slots=True)
class Config:
    data: AgentHarmDataConfig
    models: ModelConfig
    optimization: OptimizationConfig
    experiment: ExperimentConfig

    def as_flat_dict(self) -> dict:
        flat = {}
        flat.update(asdict(self.data))
        flat.update({f"model_{k}": v for k, v in asdict(self.models).items()})
        flat.update({f"optim_{k}": v for k, v in asdict(self.optimization).items()})
        flat.update({f"exp_{k}": v for k, v in asdict(self.experiment).items()})
        return flat

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
    optimization = OptimizationConfig(**raw["optimization"])
    experiment = ExperimentConfig(**raw["experiment"])

    # data.validate()
    return Config(
        data=data, models=models, optimization=optimization, experiment=experiment
    )

__all__ = [
    "AgentHarmDataConfig",
    "ModelConfig",
    "OptimizationConfig",
    "ExperimentConfig",
    "Config",
    "load_config",
]
