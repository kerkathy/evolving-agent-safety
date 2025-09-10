"""Shared utilities for writing optimization artifacts to disk.

This avoids duplication between pipeline and optimization modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json


def _serialize_candidate(obj: Any) -> dict:
    """Convert a candidate-like object into a JSON-serializable dict.

    Supports either a mapping with the expected keys or an object with
    attributes: text, refusal, completion, meta, and hash() method.
    """
    if isinstance(obj, Mapping):
        # Assume already serialized or mapping-like
        d = dict(obj)
        # Ensure required keys; provide fallbacks if absent
        return {
            "text": d.get("text", ""),
            "refusal": d.get("refusal", 0.0),
            "completion": d.get("completion", 0.0),
            "meta": d.get("meta", {}),
            "hash": d.get("hash") or d.get("id") or "",
        }
    # Object with attributes
    text = getattr(obj, "text", "")
    refusal = getattr(obj, "refusal", 0.0)
    completion = getattr(obj, "completion", 0.0)
    meta = getattr(obj, "meta", {}) or {}
    # hash can be either method or attribute
    h = None
    if hasattr(obj, "hash") and callable(getattr(obj, "hash")):
        try:
            h = obj.hash()
        except Exception:  # noqa: BLE001
            h = None
    if not h:
        h = getattr(obj, "hash", None) or ""
    return {
        "text": text,
        "refusal": float(refusal),
        "completion": float(completion),
        "meta": meta,
        "hash": h,
    }


def write_optimization_outputs(
    out_dir: str | Path,
    frontier: Iterable[Any],
    population: Iterable[Any],
    summary: dict,
    per_generation_full_eval: list[dict] | None = None,
) -> None:
    """Write standard optimization artifacts to ``out_dir``.

    Files written:
      - optimization_frontier.json
      - optimization_population.json
      - summary.json
      - frontier_full_eval_per_generation.json (optional if provided)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frontier_json = [_serialize_candidate(c) for c in frontier]
    population_json = [_serialize_candidate(c) for c in population]

    with open(out_path / "optimization_frontier.json", "w") as f:
        json.dump(frontier_json, f, indent=2)
    with open(out_path / "optimization_population.json", "w") as f:
        json.dump(population_json, f, indent=2)
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if per_generation_full_eval:
        with open(out_path / "frontier_full_eval_per_generation.json", "w") as f:
            json.dump(per_generation_full_eval, f, indent=2)
