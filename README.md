# Evolving Agent Safety

Investigate the safety of self-evolving LLM agents using the AgentHarm benchmark with DSPy for agent logic and MLflow for tracking.

## Requirements

- Linux or macOS (tested on Linux)
- Python 3.12+
- OpenAI API key (set in a .env file or environment variable)

Project dependencies are defined in `pyproject.toml` and include: `dspy`, `inspect-ai`, `mlflow`, `pyyaml`, `aiohttp`, `ipywidgets`. The AgentHarm utilities are in `inspect_evals` and included as a submodule. 

## Quick start
0) Fetch the submodule `inspect_evals`
```bash
git submodule update --init --recursive
```

1) Create a virtual environment and install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Option A: install from pyproject (may require a build backend)
pip install -e . || true

# Ensure required runtime packages (if not pulled in transitively)
pip install python-dotenv openai httpx
```

2) Set your OpenAI API key

Create a `.env` file at the project root or export it in your shell.

`.env` (recommended):

```
OPENAI_API_KEY=sk-...
```

Or via shell (temporary):

```bash
export OPENAI_API_KEY=sk-...
```

3) Start the MLflow tracking server (in another terminal)

The default tracking URI is `http://127.0.0.1:5000` (see `src/config/config.yaml`). You can use the provided helper script:

```bash
bash setup_tracker.sh
```

This runs MLflow with a local SQLite backend (`mydb.sqlite`). Alternatively:

```bash
mlflow server --backend-store-uri sqlite:///mydb.sqlite --host 127.0.0.1 --port 5000
```

If `mlflow` isn’t found, install it inside the venv:

```bash
pip install mlflow
```

4) Run the main script

By default, the script uses `./src/config/config.yaml`:

```bash
python main.py --config ./src/config/config.yaml
```

You’ll see progress logs in the terminal. Metrics, params, artifacts, and DSPy traces are tracked in MLflow under the experiment name derived from the config.

## Configuration

The YAML at `src/config/config.yaml` controls datasets, models, and tracking. Key fields:

- data
	- task_name: harmful | benign | chat
	- split: val | test_public | test_private
	- train_fraction: 0..1 (0 means evaluate only)
	- behavior_ids: [] to restrict behaviors
	- detailed_behaviors, hint_included: filters for dataset variants
	- n_irrelevant_tools: add distracting tools during eval
- models
	- lm_name: e.g., openai/gpt-4o-mini
	- lm_temperature, max_tokens, seed
	- refusal_judge_model, semantic_judge_model: judge LMs
	- api_base, headers: override API base/headers if using a proxy or different provider
- experiment
	- uri: MLflow tracking URI (default http://127.0.0.1:5000)
	- name: experiment name prefix used in runs

Notes:

- The script requires `OPENAI_API_KEY`. It’s loaded via `python-dotenv` if present in `.env`.
- The current `config.yaml` includes an `api_base` pointing to `https://run.v36.cm/v1/` with custom headers. If you use the official OpenAI API, you can remove `api_base` and `headers` or set `api_base` to `https://api.openai.com/v1`.

## What happens when you run it

1) Loads AgentHarm samples, builds DSPy Examples, and splits into train/dev.
2) Initializes a WebReAct-style DSPy agent and an AgentHarm metric.
3) Runs evaluation on the dev split (baseline). Optimization is scaffolded but commented out.
4) Logs parameters, metrics, detailed results, and a source snapshot to MLflow.

Artifacts and logs:

- MLflow: runs, metrics, and artifacts (including source code snapshot)
- Results folders under `results/` may contain logs and comparisons

## Troubleshooting

- OPENAI_API_KEY not set: The script will raise `ValueError`. Create `.env` or export the variable.
- 401/403 from model API: Ensure your key is valid and `models.api_base`/`headers` match your provider.
- MLflow not showing runs: Confirm the server is running on `http://127.0.0.1:5000` and matches `experiment.uri`.
- Python version errors: Ensure you’re on Python 3.12+ (`requires-python = ">=3.12"`).
- Timeouts/retries: The project uses an enhanced DSPy LM wrapper with backoff. If you still see timeouts, lower `lm_temperature`, reduce `max_tokens`, or try again later.

## Run Inspect Evals (CLI)

This repo includes a helper script `run_inspect_evals.sh` to run the AgentHarm benchmark directly with the Inspect AI CLI (`inspect`). It targets the `openai-api/ge/...` provider alias, which expects credentials via environment variables.

1) Ensure dependencies are available in your venv

```bash
pip install inspect-ai
```

2) Configure `.env` for Inspect Evals

Add the following to `.env` (or export them in your shell):

```
# Credentials for the named provider alias "openai-api/ge/..."
GE_API_KEY=sk-...
GE_BASE_URL=https://api.gpt.ge/v1/

# Optional: logging and runtime tuning for Inspect AI
INSPECT_LOG_DIR=./results/logs
INSPECT_LOG_LEVEL=warning
INSPECT_EVAL_MAX_RETRIES=5
INSPECT_EVAL_MAX_CONNECTIONS=20
```

Notes:
- If you are using a different OpenAI-compatible endpoint, set `GE_BASE_URL` to that base URL (e.g., `https://run.v36.cm/v1/`).
- You can keep `OPENAI_API_KEY` for running `main.py`; `GE_*` variables are used only when calling models as `openai-api/ge/...` via Inspect.
- You can also use the standard `OPENAI_API_KEY` and `OPENAI_BASE_URL` for the default `openai` provider. See Inspect AI docs for provider specifics.

3) Run the script

```bash
chmod +x run_inspect_evals.sh
source .env
bash run_inspect_evals.sh
```

The script demonstrates setting explicit `sample_ids` and uses:

```
inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini --seed 42 --sample-id "1-1,1-2,..." \
	-T split=test_public \
	-T refusal_judge="openai-api/ge/gpt-4o-mini" \
	-T semantic_judge="openai-api/ge/gpt-4o-mini" \
	-T agent_kwargs="{ 'prompt_technique': 'standard' }"
```

Common tweaks:
- Limit items: add `--limit 10`
- Shuffle seed: add `--sample-shuffle 0`
- Change split: update `-T split=val|test_public|test_private`

## Optional: Using uv

If you use [uv](https://github.com/astral-sh/uv), you can install and run faster:

```bash
# install deps
uv sync

# run (inside the env uv creates)
uv run python main.py --config ./src/config/config.yaml
```

## License

See `LICENSE` in this repository.

