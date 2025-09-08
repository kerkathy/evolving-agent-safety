uv run -m src.causal.verify_instruction_scores \
--experiment agentharm_harmful_val_detail_None_hint_None \
--run-name grandiose-grub-989 \
--metric-key eval \
--abs-tol 1e-6 --rel-tol 1e-4 \
--tracking-uri http://127.0.0.1:5000 \
--json-out verify_report.json \
--child-prefix eval \
# --run-is-optim \