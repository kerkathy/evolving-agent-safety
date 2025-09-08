"""Intervention generation utilities.

Each intervention is a controlled transformation of the original prompt text
intended to test aspects of causal sufficiency / necessity.

We define a registry of simple string-level operations. More advanced semantic
operations can be added later (e.g., removing tool descriptions via parsing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import random


@dataclass(slots=True)
class Intervention:
    kind: str
    variant_text: str


def drop_first_sentence(text: str) -> str:
    parts = text.split(".")
    if len(parts) <= 1:
        return text
    return ".".join(parts[1:]).strip()


def shuffle_sentences(text: str, seed: int | None = None) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) < 2:
        return text
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return ". ".join(sentences)


def mask_middle(text: str) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) < 3:
        return text
    mid = len(sentences) // 2
    sentences[mid] = "[MASKED]"
    return ". ".join(sentences)


REGISTRY: dict[str, Callable[[str, int | None], str]] = {
    "drop_instruction": lambda t, _: drop_first_sentence(t),
    "shuffle_order": shuffle_sentences,
    "mask_step": lambda t, _: mask_middle(t),
    # TODO add more interventions I want
}


def generate_interventions(
    prompt_text: str,
    kinds: Iterable[str],
    seed: int | None = None,
) -> list[Intervention]:
    out: list[Intervention] = []
    for k in kinds:
        fn = REGISTRY.get(k)
        if not fn:
            continue
        out.append(Intervention(kind=k, variant_text=fn(prompt_text, seed)))
    return out
