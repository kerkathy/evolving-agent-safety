"""Instruction optimization via multi-objective (refusal, completion) search.

Core idea: explore mutated variants of instruction strings and
retain ONLY the Pareto frontier (non-dominated) under two objectives:

* refusal  : Safety / appropriate refusal rate  (higher is better)
* completion: Task completion performance       (higher is better)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Optional, Dict, Any
import hashlib
import logging
import random
import json
import ast
import os
import csv
from pathlib import Path
from tqdm import tqdm

import dspy 
from src.config.config import CausalOptimizationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


EvalFn = Callable[[str], tuple[float, float, dict]]  # (refusal, completion, extra)


@dataclass(slots=True)
class Candidate:
    text: str
    refusal: float
    completion: float
    meta: dict = field(default_factory=dict)

    def hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def _dominates(a: Candidate, b: Candidate) -> bool:
    return (a.refusal >= b.refusal and a.completion >= b.completion) and (
        a.refusal > b.refusal or a.completion > b.completion
    )


def pareto_frontier(cands: Sequence[Candidate]) -> list[Candidate]:
    front: list[Candidate] = []
    for c in cands:
        dominated = False
        remove_idx: list[int] = []
        for i, f in enumerate(front):
            if _dominates(f, c):
                dominated = True
                break
            if _dominates(c, f):
                remove_idx.append(i)
        if dominated:
            continue
        # Remove those we dominate
        for i in reversed(remove_idx):
            del front[i]
        front.append(c)
    # Stable sort for reproducibility (descending metrics, then hash)
    front.sort(key=lambda x: (-x.refusal, -x.completion, x.hash()))
    return front

# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OptimizationResult:
    frontier: list[Candidate]
    population: list[Candidate]
    generations: int
    num_evaluations: int
    segment_effects: Dict[str, Dict[str, float]] | None = None  # aggregated effects per segment key
    # Per-generation full-eval metrics for the frontier (optional extra evaluation set)
    per_generation_full_eval: list[dict] = field(default_factory=list)


def optimize_instructions(
    initial_texts: Iterable[str],
    main_lm: dspy.BaseLM,
    config: CausalOptimizationConfig,
    train_eval_fn: EvalFn,
    dev_eval_fn: EvalFn,
    checkpoint_dir: str | os.PathLike[str],
    initial_population: list[Candidate] | None = None,
    initial_frontier: list[Candidate] | None = None,
    start_generation: int = 0,
    initial_segment_effects: Dict[str, Dict[str, float]] | None = None,
) -> OptimizationResult:
    """
    Optimize instructions via multi-objective search over refusal and completion.
    Args:
        initial_texts: Seed instruction texts to initialize the population.
        main_lm: The main language model to use for generating mutations.
        config: CausalOptimizationConfig with optimization settings.
    """
    cfg = config
    rng = random.Random(cfg.random_seed)
    train_eval_fn = train_eval_fn  # Real eval should be injected by caller.
    segments_cache: Dict[str, Dict[str, str]] = {}
    segment_effect_sums: Dict[str, Dict[str, float]] = {}
    low_info_alternatives: Dict[str, str] = {
        "metadata_context": "",
        "system_role": "You are an AI assistant.",
        "explicit_constraints": "Follow instructions.",
        "examples_demos": "",
        "format_specifiers": "Provide a clear answer.",
        "temperature_verbosity_hints": "Be neutral.",
    }
    intervention_keys = list(low_info_alternatives.keys())

    SEGMENT_PROMPT_TEMPLATE = (
        f"Segment the instruction below into several semantically-meaningful chunks: {', '.join(intervention_keys)}.\n"
        "Output a dictionary in JSON code block ONLY. Keys MUST be the segment names.\n\n"
        "Instruction to be segmented:\n{instruction}"
    )

    _lm_cache: Dict[str, Optional[dspy.LM]] = {}

    def _get_lm(lm_name: str) -> Optional[dspy.LM]:
        lm = _lm_cache.get(lm_name)
        if lm is not None:
            return lm
        kwargs: Dict[str, Any] = {
            "model": lm_name,
            "temperature": 0.0,
            "seed": cfg.random_seed,
            "api_key": os.getenv("OPENAI_API_KEY"), # TODO pass in
            "base_url": cfg.segment_model_api_base,
        }
        try:
            lm_obj = dspy.LM(**kwargs)  # type: ignore[arg-type]
            _lm_cache[lm_name] = lm_obj
            return lm_obj
        except Exception as e:  # noqa: BLE001
            logger.warning("[OPT] Failed to initialize dspy.LM (model=%s): %s", lm_name, e)
            _lm_cache[lm_name] = None
            return None

    def llm_call(lm, prompt: str) -> str:
        # lm = _get_lm(cfg.segment_lm_name)
        if lm is None:
            raise ValueError(f"No LM available for model '{cfg.segment_lm_name}'")
        try:
            return str(lm(prompt)[0]).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("[OPT] dspy.LM call failed: %s", e)
            return ""

    def _parse_segments(raw_text: str) -> Dict[str, str]:
        text = raw_text.strip()
        if not text:
            return {}
        # Parse out code block if present
        if "```json" in text:
            text = text.split("```json", 1)[-1]
            text = text.strip("`").strip()
        elif text.startswith("```") and text.endswith("```"):
            text = text.strip("`").strip()
        # Parse JSON 
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
        except Exception:  # noqa: BLE001
            pass
        # Python literal fallback
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
        except Exception:  # noqa: BLE001
            pass
        return {}

    def segment_instruction(instr: str) -> Dict[str, str]:
        k = hashlib.sha256(instr.encode("utf-8")).hexdigest()[:16]
        if k in segments_cache:
            return segments_cache[k]
        prompt = SEGMENT_PROMPT_TEMPLATE.format(instruction=instr)
        try:
            lm = _get_lm(cfg.segment_lm_name)
            raw = llm_call(lm, prompt)
            mapping = _parse_segments(raw)
            expected_any = any(
                key in mapping
                for key in intervention_keys
            )
            if not expected_any:
                logger.warning("[OPT] No expected keys found in segmentation output: %s", raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("[OPT] Segmentation failed (%s); using fallback.", e)
            mapping = {"full": instr}
        mapping = {k2: v for k2, v in mapping.items() if isinstance(v, str) and v.strip()}
        segments_cache[k] = mapping if mapping else {"full": instr}
        return segments_cache[k]

    # Mutation spec: dict with keys text, intervention_key, mutation_kind, parent_hash

    def sufficiency_mutations(parent: Candidate) -> list[dict[str, Any]]:
        """Generate mutations attempting to enhance completion OR safety for each segment.

        Two variants per segment (when possible):
          - completion-oriented improvement
          - safety-oriented improvement
        """
        segs = segment_instruction(parent.text)
        segs_json = json.dumps(segs, ensure_ascii=False, indent=2)
        out: list[dict[str, Any]] = []
        for key in segs.keys():
            for target in ("completion", "safety"):
                intent_desc = (
                    "Improve task success (higher completion) without reducing safety" if target == "completion" else
                    "Improve safety / appropriate refusal while preserving legitimate task success"
                )
                prompt = (
                    f"You are optimizing ONE segment ('{key}') of an instruction. {intent_desc}.\n"
                    "Keep all OTHER segments identical. Return a json object with the same keys.\n"
                    f"Current segments (JSON):\n{segs_json}\n"
                )
                try:
                    raw = llm_call(main_lm, prompt)
                    segs_new = _parse_segments(raw)
                    # If parsing failed, fall back to treating raw as plain text (legacy behavior)
                    if segs_new:
                        candidate_text = " ".join(v for v in segs_new.values() if v).strip()
                    else:
                        candidate_text = raw
                    if candidate_text and candidate_text != parent.text and len(candidate_text.split()) > 3:
                        out.append({
                            "text": candidate_text,
                            "intervention_key": key,
                            "mutation_kind": f"sufficiency_improve_{target}",
                            "parent_hash": parent.hash(),
                        })
                        logger.info("[OPT] Sufficiency mutation generated for segment '%s' targeting %s: %s", key, target, candidate_text)
                except Exception as e:  # noqa: BLE001
                    logger.debug("[OPT] Sufficiency mutation failed: %s", e)
        return out

    def necessity_mutations(parent: Candidate) -> list[dict[str, Any]]:
        """Generate necessity probing mutations.

        For each segment produce up to two variants:
          - removal (segment cleared)
          - low-info replacement (generic, less informative)
        Additionally we classify target axis as completion or safety removal heuristically based on key name.
        """
        segs = segment_instruction(parent.text)
        out: list[dict[str, Any]] = []
        for key in segs.keys():
            removed_text = " ".join(v for k, v in segs.items() if k != key and v)
            if removed_text and removed_text != parent.text and len(removed_text.split()) > 3:
                out.append({
                    "text": removed_text,
                    "intervention_key": key,
                    "mutation_kind": "necessity_remove",
                    "parent_hash": parent.hash(),
                })
            lowinfo_segs = segs.copy()
            lowinfo_segs[key] = low_info_alternatives.get(key, "")
            lowinfo_text = " ".join(v for v in lowinfo_segs.values() if v)
            if lowinfo_text and lowinfo_text != parent.text and len(lowinfo_text.split()) > 3:
                out.append({
                    "text": lowinfo_text,
                    "intervention_key": key,
                    "mutation_kind": "necessity_lowinfo",
                    "parent_hash": parent.hash(),
                })
        return out

    # key -> candidate mapping
    seen_cache: dict[str, Candidate] = {}

    def _update_effects(intervention_key: str, parent: Candidate, child: Candidate):
        rec = segment_effect_sums.setdefault(
            intervention_key,
            {"count": 0.0, "effect_on_refusal_sum": 0.0, "effect_on_completion_sum": 0.0},
        )
        rec["count"] = float(rec.get("count", 0.0) or 0.0) + 1.0
        rec["effect_on_refusal_sum"] = float(rec.get("effect_on_refusal_sum", 0.0) or 0.0) + (child.refusal - parent.refusal)
        rec["effect_on_completion_sum"] = float(rec.get("effect_on_completion_sum", 0.0) or 0.0) + (child.completion - parent.completion)

    def score_and_register_candidate(t: str, reason: str, mutation_meta: Optional[Dict[str, Any]] = None):
        k = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
        if k in seen_cache:
            return None
        refusal, completion, extra = train_eval_fn(t)
        meta = {"reason": reason, **extra}
        if mutation_meta:
            meta.update(mutation_meta)
        cand = Candidate(text=t, refusal=refusal, completion=completion, meta=meta)
        seen_cache[k] = cand
        return cand

    # Enforce max frontier size
    try:
        max_frontier = int(getattr(cfg, "frontier_size", 0) or 0)
    except Exception:
        max_frontier = 0

    # Initialize population with seeds or resume from previous state
    if initial_population is not None and initial_frontier is not None:
        population = initial_population
        frontier = initial_frontier
        for cand in population:
            seen_cache[cand.hash()] = cand
        if initial_segment_effects:
            # Assume effects were converted by the resume loader to sum-based format
            segment_effect_sums = initial_segment_effects
        n_seeds = 0  # Since we're resuming, seeds are already in population
        num_evaluations = len(population)
        if max_frontier > 0 and len(frontier) > max_frontier:
            frontier = frontier[:max_frontier]
        logger.info("[OPT] Resumed from generation %d: population=%d frontier=%d", start_generation, len(population), len(frontier))
    else:
        seeds_list = list(initial_texts)
        for t in seeds_list:
            score_and_register_candidate(t, "seed")
            # No effect update for seeds
            if cfg.max_candidates_evaluated and len(seen_cache) >= cfg.max_candidates_evaluated:
                break
        n_seeds = len(seeds_list)
        population = list(seen_cache.values())
        frontier = pareto_frontier(population)
        num_evaluations = len(population)
        if max_frontier > 0 and len(frontier) > max_frontier:
            frontier = frontier[:max_frontier]
        logger.info("[OPT] Seed: population=%d frontier=%d", len(population), len(frontier))

    # Store per-generation full frontier evaluations (if dev_eval_fn provided)
    per_generation_full_eval: list[dict] = []
    
    # Lazy import to avoid cycles
    from .io_utils import write_optimization_outputs  # type: ignore

    last_generation = start_generation
    for gen in tqdm(range(start_generation + 1, cfg.max_generations + 1), desc="Generations", unit="gen"):
        last_generation = gen
        # If train_eval_fn supports per-generation resampling (minibatch stochastic fitness) invoke it now.
        try:
            if hasattr(train_eval_fn, "resample") and callable(getattr(train_eval_fn, "resample")):
                getattr(train_eval_fn, "resample")(gen)
        except Exception as e:  # noqa: BLE001
            logger.warning("[OPT] Minibatch resample failed at generation %d (%s)", gen, e)
        # Generate mutations from frontier
        for parent_cand in tqdm(list(frontier), desc=f"Gen {gen} Candidates", unit="cand"):
            # stop if we've reached max evals
            if cfg.max_candidates_evaluated and num_evaluations >= cfg.max_candidates_evaluated:
                break

            logger.info("[OPT] Gen %d: processing candidate r=%.2f c=%.2f text=%s", gen, parent_cand.refusal, parent_cand.completion, parent_cand.text)
            segment_instruction(parent_cand.text)

            # Generate mutations
            new_specs: list[Dict[str, Any]] = []
            if parent_cand.completion < cfg.target_completion or parent_cand.refusal < cfg.target_refusal:
                new_specs.extend(sufficiency_mutations(parent_cand))
            # Optionally include necessity mutations
            try:
                include_necessity = bool(getattr(cfg, "include_necessity_mutations", True))
            except Exception:
                include_necessity = True
            if include_necessity:
                new_specs.extend(necessity_mutations(parent_cand))
            if not new_specs:
                logger.info("[OPT] No new mutations at generation %d; stopping early.", gen)
                break
            rng.shuffle(new_specs)

            logger.info("[OPT] Gen %d: generated %d mutations", gen, len(new_specs))

            # Generate new candidates from specs
            new_children = []
            children_dir = Path(checkpoint_dir) / f"gen_{gen:03d}" / "children"
            children_dir.mkdir(parents=True, exist_ok=True)
            parent_hash = parent_cand.hash()
            children_file = children_dir / "children.csv"
            with open(children_file, "a", newline='') as f:
                fieldnames = ["text", "refusal", "completion", "meta", "hash", "parent_hash", "generation"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            for spec in tqdm(new_specs, desc="Mutations", unit="mut"):
                if cfg.max_candidates_evaluated and num_evaluations >= cfg.max_candidates_evaluated:
                    break
                child_cand = score_and_register_candidate(
                    spec["text"], 
                    f"gen{gen}", 
                    mutation_meta={k: v for k, v in spec.items() if k != "text"}
                )
                if child_cand:
                    new_children.append(child_cand)
                    if spec.get("intervention_key"):
                        _update_effects(spec["intervention_key"], parent_cand, child_cand)
                    with open(children_file, "a", newline='') as f:
                        fieldnames = ["text", "refusal", "completion", "meta", "hash", "parent_hash", "generation"]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow({
                            "text": child_cand.text,
                            "refusal": child_cand.refusal,
                            "completion": child_cand.completion,
                            "meta": json.dumps(child_cand.meta),
                            "hash": child_cand.hash(),
                            "parent_hash": parent_hash,
                            "generation": gen
                        })
                num_evaluations += 1
            logger.info("[OPT] Gen %d: evaluated %d new candidates from parent %s", gen, len(new_children), parent_hash)

        # Refresh population and frontier
        population = list(seen_cache.values())
        frontier = pareto_frontier(population)
        # Enforce max frontier size each generation
        if max_frontier > 0 and len(frontier) > max_frontier:
            frontier = frontier[:max_frontier]
        logger.info(
            "[OPT] Gen %d: population=%d frontier=%d evals=%d best(max r=%.2f c=%.2f)",
            gen,
            len(population),
            len(frontier),
            num_evaluations,
            max(c.refusal for c in frontier),
            max(c.completion for c in frontier),
        )

        # Checkpoint population, frontier, and segment effects after generation finishes
        base_dir = Path(checkpoint_dir) / f"gen_{gen:03d}" / "checkpoints"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)

            # Compute partial segment effects at this point in time
            segment_effects_ckpt: Dict[str, Dict[str, float]] = {}
            for key, rec in segment_effect_sums.items():
                count = rec.get("count", 0.0) or 1.0
                segment_effects_ckpt[key] = {
                    "count": rec.get("count", 0.0),
                    "avg_effect_on_refusal": rec.get("effect_on_refusal_sum", 0.0) / count,
                    "avg_effect_on_completion": rec.get("effect_on_completion_sum", 0.0) / count,
                }

            ckpt_summary = {
                "generation": gen,
                "population_size": len(population),
                "frontier_size": len(frontier),
                "evaluations": num_evaluations,
                "n_seeds": n_seeds,
                "best_refusal": max(c.refusal for c in frontier) if frontier else 0.0,
                "best_completion": max(c.completion for c in frontier) if frontier else 0.0,
                "avg_refusal": sum(c.refusal for c in population) / (len(population) or 1),
                "avg_completion": sum(c.completion for c in population) / (len(population) or 1),
                "segment_effects": segment_effects_ckpt or None,
            }

            write_optimization_outputs(
                base_dir,
                frontier=frontier,
                population=population,
                summary=ckpt_summary,
            )
            logger.info("[OPT][CKPT] Wrote generation %d checkpoint (pre-full-eval) to %s", gen, base_dir)
        except Exception as e:
            logger.warning("[OPT][CKPT] Failed to write pre-full-eval checkpoint for generation %d: %s", gen, e)

        # After each generation, evaluate current frontier on full eval set (if provided)
        gen_full_metrics: list[dict[str, Any]] = []
        logger.info("[OPT][FULL] Evaluating frontier on full eval set (gen=%d, size=%d)", gen, len(frontier))
        for cand in frontier:
            try:
                full_r, full_c, extra_full = dev_eval_fn(cand.text)
            except Exception as e:  # noqa: BLE001
                logger.warning("[OPT][FULL] Full evaluation failed for candidate %s (%s)", cand.hash(), e)
                full_r, full_c, extra_full = 0.0, 0.0, {"error": str(e)}
            # Attach latest full metrics to candidate meta (namespaced by generation)
            cand.meta.setdefault("full_eval", {})[f"gen_{gen}"] = {
                "refusal": full_r,
                "completion": full_c,
                **extra_full,
            }
            gen_full_metrics.append({
                "gen": gen,
                "hash": cand.hash(),
                "text": cand.text,
                "refusal_full": full_r,
                "completion_full": full_c,
                "refusal_train": cand.refusal,
                "completion_train": cand.completion,
                "meta_extra": extra_full,
            })
        per_generation_full_eval.append({
            "generation": gen,
            "frontier_size": len(frontier),
            "candidates": gen_full_metrics,
            "summary": {
                "best_refusal_full": max((m["refusal_full"] for m in gen_full_metrics), default=0.0),
                "best_completion_full": max((m["completion_full"] for m in gen_full_metrics), default=0.0),
                "avg_refusal_full": sum((m["refusal_full"] for m in gen_full_metrics)) / (len(gen_full_metrics) or 1),
                "avg_completion_full": sum((m["completion_full"] for m in gen_full_metrics)) / (len(gen_full_metrics) or 1),
            }
        })

        # Save the full eval result after running the whole eval
        try:
            full_eval_file = base_dir / "frontier_full_eval_per_generation.json"
            with open(full_eval_file, "w") as f:
                json.dump(per_generation_full_eval, f, indent=2)
            logger.info("[OPT][FULL] Saved full eval for generation %d to %s", gen, full_eval_file)
        except Exception as e:
            logger.warning("[OPT][FULL] Failed to save full eval for generation %d: %s", gen, e)

        if all(c.refusal >= cfg.target_refusal and c.completion >= cfg.target_completion for c in frontier):
            logger.info("[OPT] Targets reached by entire frontier at generation %d.", gen)
            break

    # Aggregate averages
    segment_effects: Dict[str, Dict[str, float]] = {}
    for key, rec in segment_effect_sums.items():
        count = rec.get("count", 0.0) or 1.0
        segment_effects[key] = {
            "count": rec.get("count", 0.0),
            "avg_effect_on_refusal": rec.get("effect_on_refusal_sum", 0.0) / count,
            "avg_effect_on_completion": rec.get("effect_on_completion_sum", 0.0) / count,
        }

    return OptimizationResult(
        frontier=frontier,
        population=population,
        generations=last_generation,
        num_evaluations=num_evaluations,
        segment_effects=segment_effects or None,
    )


__all__ = [
    "OptimizationResult",
    "optimize_instructions",
    "pareto_frontier",
]
