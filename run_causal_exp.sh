uv run -m experiments.causal_eval --config src/config/config_causal_gpt4.yaml > log_gpt4.txt 2>&1
echo "Finished causal eval gpt4"

uv run -m experiments.causal_eval --config src/config/config_causal_gemini.yaml > log_gemini.txt 2>&1
echo "Finished causal eval gemini"

# uv run -m experiments.causal_eval --config src/config/config_causal_qwen.yaml > log_qwen.txt 2>&1
# echo "Finished causal eval qwen"

# uv run -m experiments.causal_eval --config src/config/config_causal_claude.yaml > log_claude.txt 2>&1
# echo "Finished causal eval claude"