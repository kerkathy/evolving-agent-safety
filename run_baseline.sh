# Define arrays for prefixes, behaviors, and models
prefixes=("" "gepa_")
behaviors=("harmful_refusal_" "benign_completion_")
models=("gpt4" "gemini")

# Nested loops to generate all combinations
for prefix in "${prefixes[@]}"; do
    for behavior in "${behaviors[@]}"; do
        for model in "${models[@]}"; do
            config="${prefix}${behavior}${model}"
            echo "Starting $config"
            uv run main.py --config src/config/config_${config}.yaml > output_baseline_${config}.txt 2>&1
            echo "Finished $config"
        done
    done
done


# uv run main.py --config src/config/config_harmful_refusal_qwen7b.yaml > output_baseline_harmful_refusal_qwen7b.txt 2>&1
# echo "Finished harmful refusal qwen7b"
# uv run main.py --config src/config/config_benign_completion_qwen7b.yaml > output_baseline_benign_completion_qwen7b.txt 2>&1
# echo "Finished benign completion qwen7b"

# uv run main.py --config src/config/config_harmful_refusal_claude.yaml > output_baseline_harmful_refusal_claude.txt 2>&1
# echo "Finished harmful refusal claude"
# uv run main.py --config src/config/config_benign_completion_claude.yaml > output_baseline_benign_completion_claude.txt 2>&1
# echo "Finished benign completion claude"
