# https://inspect.aisi.org.uk/providers.html#openai-api
# follow that to set the openai compatible apis

# inspect eval inspect_evals/agentharm --model openai/gpt-4o-mini --limit 1 \

# inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini \

inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini --limit 5 \
-T split=val \
-T refusal_judge="openai-api/ge/gpt-4o-2024-08-06" \
-T semantic_judge="openai-api/ge/gpt-4o-2024-08-06" \
-T agent_kwargs="{\
'prompt_technique': 'standard',
}"

# 'tool_choice': 'auto'