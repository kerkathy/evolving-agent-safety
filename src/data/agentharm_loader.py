"""Utilities for loading and preparing AgentHarm data for DSPy.

This was extracted from the previous monolithic `main.py` to separate data
concerns (IO, transformation) from training / optimization logic.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional

import dspy

# External package imports (agentharm + inspect ai) are retained here.
from external.inspect_evals.src.inspect_evals.agentharm.utils import load_dataset, filter_dataset, get_benchmark_tools
from external.inspect_evals.src.inspect_evals.agentharm.benchmark import harmful_tools, benign_tools
from src.utils.sync_wrappers import convert_tools_to_sync


def load_agentharm_data(
    behavior_ids: List[str] | None = None,
    sample_ids: List[str] | None = None,
    task_name: Literal["harmful", "benign", "chat"] = "harmful",
    split: Literal["val", "test_private", "test_public"] = "test_public",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
) -> List[dict]:
    """Load AgentHarm dataset and convert to DSPy friendly list of dicts.

    Returns a list where each element has keys:
    question, answer, functions, metadata, id, category, detailed_prompt,
    hint_included, grading_function, target_functions.
    """

    behavior_ids = behavior_ids or []
    sample_ids = sample_ids or []
    dataset = load_dataset(task_name, split)
    dataset = filter_dataset(dataset, behavior_ids, detailed_behaviors, hint_included)
    if sample_ids:
        # Only keep samples whose id is in sample_ids
        dataset = [s for s in dataset if getattr(s, "id", None) in sample_ids]
    all_tool_names = get_benchmark_tools(task_name) if task_name != "chat" else []

    data: List[Dict[str, Any]] = []
    for sample in dataset:
        metadata = getattr(sample, "metadata", None) or {}
        target_list = metadata.get("target_functions") or []

        # Target tools
        target_tools_dict: Dict[str, Any] = {}
        if target_list:
            tools_module = harmful_tools if task_name == "harmful" else benign_tools
            for tool_name in target_list:
                try:
                    tool_func = getattr(tools_module, tool_name)()
                    target_tools_dict[tool_name] = tool_func
                except AttributeError:
                    print(f"Warning: Tool {tool_name} not found")
                    continue

        # Irrelevant tools
        tools_dict = target_tools_dict.copy()
        if n_irrelevant_tools > 0 and target_list:
            tools_module = harmful_tools if task_name == "harmful" else benign_tools
            irrelevant_tool_names = [
                tool_name for tool_name in all_tool_names if tool_name not in target_list
            ]
            if irrelevant_tool_names:
                k_irrelevant_tool_names = random.sample(
                    irrelevant_tool_names,
                    min(n_irrelevant_tools, len(irrelevant_tool_names)),
                )
                for tool_name in k_irrelevant_tool_names:
                    try:
                        irrelevant_tool = getattr(tools_module, tool_name)()
                        tools_dict[tool_name] = irrelevant_tool
                    except AttributeError:
                        print(f"Warning: Irrelevant tool {tool_name} not found")
                        continue

        # Convert async tools to sync versions for DSPy compatibility
        tools_dict = convert_tools_to_sync(tools_dict)

        datapoint = {
            "question": sample.input,
            "answer": "",  # No ground-truth textual answer in AgentHarm
            "functions": tools_dict,
            "metadata": sample.metadata or {},
            "id": getattr(sample, "id", None),
            "category": metadata.get("category", ""),
            "detailed_prompt": metadata.get("detailed_prompt", False),
            "hint_included": metadata.get("hint_included", False),
            "grading_function": metadata.get("grading_function", ""),
            "target_functions": target_list,
        }
        data.append(datapoint)

    return data


def build_dspy_examples(raw_data: List[dict]) -> list[dspy.Example]:
    """Convert raw datapoint dictionaries to DSPy Examples.

    Adds a synthetic `finish` tool to each example mirroring the previous
    inline definition in the monolithic script.
    """

    examples: list[dspy.Example] = []

    def finish(answer: str):  # noqa: D401 - simple wrapper
        """Conclude the trajectory and return the final answer."""
        return answer

    for datapoint in raw_data:
        datapoint["functions"]["finish"] = finish
        example = dspy.Example(
            question=datapoint["question"],
            answer=datapoint["answer"],
            functions=datapoint["functions"],
            metadata=datapoint["metadata"],
            sample_id=datapoint["id"],
            category=datapoint["category"],
            detailed_prompt=datapoint["detailed_prompt"],
            hint_included=datapoint["hint_included"],
            grading_function=datapoint["grading_function"],
            target_functions=datapoint["target_functions"],
        )
        examples.append(example.with_inputs("question", "functions"))
    return examples


def split_examples(
    examples: list[dspy.Example], train_fraction: float, seed: int
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split examples into train/dev/test (50/50 split of the non-train remainder)."""
    rnd = random.Random(seed)
    shuffled = examples[:]  # copy
    rnd.shuffle(shuffled)
    total = len(shuffled)
    train_size = int(train_fraction * total)
    train = shuffled[:train_size]
    dev = shuffled[train_size:]
    return train, dev


__all__ = [
    "load_agentharm_data",
    "build_dspy_examples",
    "split_examples",
]
