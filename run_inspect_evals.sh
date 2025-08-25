# https://inspect.aisi.org.uk/providers.html#openai-api
# follow that to set the openai compatible apis
# use sample-shuffle to set seed for data sampling

# inspect eval inspect_evals/agentharm --model openai/gpt-4o-mini --limit 1 \

# inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini \
# inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini --limit 10 --sample-shuffle 0 \


# -T refusal_judge="openai-api/ge/gpt-4o-2024-08-06" \
# -T semantic_judge="openai-api/ge/gpt-4o-2024-08-06" \


# create a list of strings as sample_id
sample_ids="1-1,1-2,1-3,1-4,10-1,10-2,10-3,10-4"

inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini --seed 42 --sample-id "$sample_ids" \
 -T split=test_public \
 -T refusal_judge="openai-api/ge/gpt-4o-mini" \
 -T semantic_judge="openai-api/ge/gpt-4o-mini" \
 -T agent_kwargs="{\
 'prompt_technique': 'standard',
}"

# 'tool_choice': 'auto'