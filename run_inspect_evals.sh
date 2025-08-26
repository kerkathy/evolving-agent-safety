#!/usr/bin/env bash

# Inspect Evals provider docs: https://inspect.aisi.org.uk/providers.html#openai-api
# This script lets you switch between the default OpenAI provider and an OpenAI-compatible alias
# by changing a single variable (PROVIDER_PREFIX) or setting INSPECT_PROVIDER_PREFIX in the env.

set -euo pipefail

# Provider prefix (one-line switch):
#   - Default OpenAI provider:    openai
#   - Compatible alias example:   openai-api/ge
# You can also override this via env: export INSPECT_PROVIDER_PREFIX="openai-api/ge"
PROVIDER_PREFIX=${INSPECT_PROVIDER_PREFIX:-openai}

# Base model name used for agent and judges (override as needed)
BASE_MODEL_AGENT=${BASE_MODEL_AGENT:-gpt-4o-mini}
BASE_MODEL_JUDGE=${BASE_MODEL_JUDGE:-gpt-4o-mini}

# Derived fully-qualified model strings
MODEL="${PROVIDER_PREFIX}/${BASE_MODEL_AGENT}"
REFUSAL_JUDGE_MODEL="${PROVIDER_PREFIX}/${BASE_MODEL_JUDGE}"
SEMANTIC_JUDGE_MODEL="${PROVIDER_PREFIX}/${BASE_MODEL_JUDGE}"

# Example overrides for versioned judges (uncomment to pin):
# REFUSAL_JUDGE_MODEL="${PROVIDER_PREFIX}/gpt-4o-2024-08-06"
# SEMANTIC_JUDGE_MODEL="${PROVIDER_PREFIX}/gpt-4o-2024-08-06"

# Data sampling (use --sample-shuffle to set seed for data sampling)
SEED=${SEED:-42}

# Create a list of strings as sample_id
sample_ids=${SAMPLE_IDS:-"1-1,1-2,1-3,1-4,10-1,10-2,10-3,10-4"}

# Quick examples:
# inspect eval inspect_evals/agentharm --model openai/gpt-4o-mini --limit 1
# INSPECT_PROVIDER_PREFIX=openai-api/ge inspect eval inspect_evals/agentharm --model "${PROVIDER_PREFIX}/gpt-4o-mini" --limit 10 --sample-shuffle 0

inspect eval inspect_evals/agentharm \
	--model "${MODEL}" \
	--seed "${SEED}" \
	-T split=test_public \
	-T refusal_judge="${REFUSAL_JUDGE_MODEL}" \
	-T semantic_judge="${SEMANTIC_JUDGE_MODEL}" \
	-T agent_kwargs="{\
	'prompt_technique': 'standard',
}"

# Add optional flags as needed, e.g.:
#	--sample-id "${sample_ids}" \
#   --limit 10
#   --sample-shuffle 0
#   -T agent_kwargs="{ 'tool_choice': 'auto' }"