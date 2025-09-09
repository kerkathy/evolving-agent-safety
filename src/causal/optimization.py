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


def optimize_instructions(
    initial_texts: Iterable[str],
    main_lm: dspy.BaseLM,
    config: CausalOptimizationConfig,
    eval_fn: EvalFn,
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
    eval_fn = eval_fn  # Real eval should be injected by caller.
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
    population: list[Candidate] = list(seen_cache.values())
    # Initial evaluation
    frontier = pareto_frontier(population)
    num_evaluations = len(population)

    def _update_effects(intervention_key: str, parent: Candidate, child: Candidate):
        rec = segment_effect_sums.setdefault(intervention_key, {"count": 0.0, "effect_on_refusal_sum": 0.0, "effect_on_completion_sum": 0.0})
        rec["count"] += 1.0
        rec["effect_on_refusal_sum"] += child.refusal - parent.refusal
        rec["effect_on_completion_sum"] += child.completion - parent.completion

    def score_and_register_candidate(t: str, reason: str, mutation_meta: Optional[Dict[str, Any]] = None):
        k = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
        if k in seen_cache:
            return None
        refusal, completion, extra = eval_fn(t)
        meta = {"reason": reason, **extra}
        if mutation_meta:
            meta.update(mutation_meta)
        cand = Candidate(text=t, refusal=refusal, completion=completion, meta=meta)
        seen_cache[k] = cand
        return cand

    for t in initial_texts:
        score_and_register_candidate(t, "seed")
        # No effect update for seeds
        if cfg.max_candidates_evaluated and len(seen_cache) >= cfg.max_candidates_evaluated:
            break

    logger.info("[OPT] Seed population=%d frontier=%d", len(population), len(frontier))

    for gen in tqdm(range(1, cfg.max_generations + 1), desc="Generations", unit="gen"):
        # Generate mutations from frontier
        for parent_cand in tqdm(list(frontier), desc=f"Gen {gen} Candidates", unit="cand"):
            logger.info("[OPT] Gen %d: processing candidate r=%.2f c=%.2f text=%s", gen, parent_cand.refusal, parent_cand.completion, parent_cand.text)
            segment_instruction(parent_cand.text)

            # Generate mutations
            new_specs: list[Dict[str, Any]] = []
            if parent_cand.completion < cfg.target_completion or parent_cand.refusal < cfg.target_refusal:
                new_specs.extend(sufficiency_mutations(parent_cand))
            new_specs.extend(necessity_mutations(parent_cand))
            if not new_specs:
                logger.info("[OPT] No new mutations at generation %d; stopping early.", gen)
                break
            rng.shuffle(new_specs)

            logger.info("[OPT] Gen %d: generated %d mutations", gen, len(new_specs))

            # Generate new candidates from specs
            for spec in tqdm(new_specs, desc="Mutations", unit="mut"):
                if cfg.max_candidates_evaluated and num_evaluations >= cfg.max_candidates_evaluated:
                    break
                child_cand = score_and_register_candidate(
                    spec["text"], 
                    f"gen{gen}", 
                    mutation_meta={k: v for k, v in spec.items() if k != "text"}
                )
                if child_cand and spec.get("intervention_key") and parent_cand:
                    _update_effects(spec["intervention_key"], parent_cand, child_cand)
                num_evaluations += 1

        population = list(seen_cache.values())
        frontier = pareto_frontier(population)
        logger.info(
            "[OPT] Gen %d: population=%d frontier=%d evals=%d best(max r=%.2f c=%.2f)",
            gen,
            len(population),
            len(frontier),
            num_evaluations,
            max(c.refusal for c in frontier),
            max(c.completion for c in frontier),
        )
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
        generations=gen,
        num_evaluations=num_evaluations,
        segment_effects=segment_effects or None,
    )


__all__ = [
    "OptimizationResult",
    "optimize_instructions",
    "pareto_frontier",
]
