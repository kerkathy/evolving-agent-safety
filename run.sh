inspect eval inspect_evals/agentharm --model openai/gpt-5-nano --limit 1 \
-T split=val \
-T refusal_judge="openai/gpt-5-nano" \
-T semantic_judge="openai/gpt-5-nano" \
-T agent_kwargs="{
'prompt_technique': 'standard',
'tool_choice': 'forced_first'
}"
