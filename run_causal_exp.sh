echo "Running causal eval gpt4 no necessity fromscratch..."
uv run -m experiments.causal_eval --config src/config/config_causal_gpt4_no_necessity_fromscratch.yaml > log_gpt4_no_necessity_fromscratch.txt 2>&1
echo "Finished causal eval gpt4 no necessity fromscratch"
