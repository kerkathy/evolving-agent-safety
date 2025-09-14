"""Evaluate per-generation frontiers on the test set and write summaries.

Usage:
    python -m experiments.eval_frontiers \
        --config src/config/config_causal_min.yaml \
        --results-dir results/causal_optimization_no_necessity_fromscratch_20250914_101855

This script scans the given results directory for gen_*/checkpoints folders,
loads the frontier candidates at each generation, evaluates them on the test
split using the dual-metric evaluation (refusal on harmful, completion on
benign), and saves aggregated results into gen_00x/checkpoints/summary.json.

Notes:
  - Requires OPENAI_API_KEY in the environment for model-backed evaluation.
  - Expects the config to include a `causal` section so evaluation examples
    can be prepared consistently with the causal pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.config import load_config
from src.causal.eval_utils import load_eval_examples, build_full_eval_fn
from src.causal.runtime_setup import configure_dspy
from src.metrics import AgentHarmMetricFactory
from src.agent import WebReActAgent


logger = logging.getLogger("eval_frontiers")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate gen_* frontiers on test set and write summaries.")
    ap.add_argument("--config", required=True, help="Path to config YAML used for the run (must include causal section).")
    ap.add_argument("--results-dir", required=True, help="Path to the results folder containing gen_*/checkpoints subfolders.")
    ap.add_argument("--overwrite", action="store_true", help="If set, recompute and overwrite any existing test_results in summary.json.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return ap.parse_args()


def _load_frontier_texts(checkpoints_dir: Path) -> List[str]:
    frontier_file = checkpoints_dir / "optimization_frontier.json"
    if not frontier_file.exists():
        raise FileNotFoundError(f"Missing frontier file: {frontier_file}")
    with open(frontier_file, "r") as f:
        data = json.load(f)
    texts: List[str] = []
    for item in data:
        t = item.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t)
    return texts


def _load_existing_summary(checkpoints_dir: Path) -> Dict[str, Any]:
    summary_file = checkpoints_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to read existing summary at %s; will recreate.", summary_file)
    return {}


def _write_summary(checkpoints_dir: Path, summary: Dict[str, Any]) -> None:
    out = checkpoints_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to %s", out)


def main() -> int:
    load_dotenv(override=True)
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if getattr(cfg, "causal", None) is None or getattr(cfg, "causal").optimization is None:  # type: ignore[union-attr]
        raise ValueError("Config must include a 'causal' section with 'optimization' settings.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run evaluation.")

    # Configure runtime (DSPy + agent + dual metric factories)
    configure_dspy(cfg.models, api_key)
    agent = WebReActAgent()
    benign_full_metric_factory = AgentHarmMetricFactory(
        task_name="benign",
        refusal_judge_model=cfg.models.refusal_judge_model,
        semantic_judge_model=cfg.models.semantic_judge_model,
    )
    harmful_full_metric_factory = AgentHarmMetricFactory(
        task_name="harmful",
        refusal_judge_model=cfg.models.refusal_judge_model,
        semantic_judge_model=cfg.models.semantic_judge_model,
    )
    examples_by_task = load_eval_examples(cfg, full=True)
    if not examples_by_task.get("benign") or not examples_by_task.get("harmful"):
        raise ValueError("Missing benign or harmful examples; cannot proceed with evaluation.")
    test_eval_fn = build_full_eval_fn(
        examples_by_task,
        benign_metric_factory=benign_full_metric_factory,
        harmful_metric_factory=harmful_full_metric_factory,
        agent=agent,
        split="test",
    )

    results_root = Path(args.results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    # Find gen_* folders and process in numeric order
    gens = [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("gen_")]
    gens.sort(key=lambda p: int(p.name.split("_")[-1]), reverse=True)
    if not gens:
        logger.warning("No gen_* folders found under %s", results_root)

    for gen_dir in gens:
        checkpoints_dir = gen_dir / "checkpoints"
        if not checkpoints_dir.exists():
            logger.info("Skipping %s (no checkpoints dir)", gen_dir)
            continue

        logger.info("Evaluating frontier for %s...", gen_dir.name)
        try:
            frontier_texts = _load_frontier_texts(checkpoints_dir)
        except Exception as e:
            logger.warning("Failed to load frontier for %s: %s", gen_dir, e)
            continue

        # Read existing summary to preserve checkpoint info, and respect overwrite flag
        summary = _load_existing_summary(checkpoints_dir)
        if (not args.overwrite) and summary.get("test_results"):
            logger.info("Test results already present in %s; skipping (use --overwrite to recompute).", checkpoints_dir)
            continue

        test_results: List[Dict[str, Any]] = []
        for idx, text in enumerate(frontier_texts, start=1):
            try:
                refusal, completion, extra = test_eval_fn(text)
            except Exception as e:
                logger.warning("[%s] Evaluation failed for candidate %d/%d: %s", gen_dir.name, idx, len(frontier_texts), e)
                refusal, completion, extra = 0.0, 0.0, {"error": str(e)}
            test_results.append({
                "refusal_test": float(refusal),
                "completion_test": float(completion),
                "extra": extra,
            })

        test_summary = {
            "avg_refusal_test": sum(r["refusal_test"] for r in test_results) / (len(test_results) or 1),
            "avg_completion_test": sum(r["completion_test"] for r in test_results) / (len(test_results) or 1),
            "best_refusal_test": max((r["refusal_test"] for r in test_results), default=0.0),
            "best_completion_test": max((r["completion_test"] for r in test_results), default=0.0),
            "count": len(test_results),
            "test_results": test_results,  # full list of per-candidate results
        }

        # Merge into summary and write back
        summary.update({
            "test_results": test_results,
            "test_summary": test_summary,
        })
        _write_summary(checkpoints_dir, summary)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
