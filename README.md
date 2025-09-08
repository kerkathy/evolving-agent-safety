# Evolving Agent Safety

Investigate the safety of self-evolving LLM agents using the [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) benchmark with [DSPy](https://dspy.ai/) for prompt optimization and agent logic, and [MLflow](https://mlflow.org/docs/latest/) for auto tracking.

## Requirements

- Python 3.12+
- OpenAI API key (set in a .env file or environment variable)

Project dependencies are defined in `pyproject.toml`. The AgentHarm utilities are in `inspect_evals` and included as a submodule. 

## Prerequisite
1) Fetch the submodule `inspect_evals`
```bash
git submodule update --init --recursive
```

2) Set your OpenAI API key

In `.env` (recommended):

```
OPENAI_API_KEY=sk-...
```

Or via shell (temporary):

```bash
export OPENAI_API_KEY=sk-...
```

## Set Up Environment and run
### Option 1 (Recommended): Using uv

If you use [uv](https://github.com/astral-sh/uv), you can install and run faster:

```bash
# install deps
uv sync

# Setup mlflow client in another bash session
source .venv/bin/activate
bash setup_tracker.sh

# run (inside the env uv creates)
uv run main.py --config ./src/config/config.yaml
```

### Option 2 (Alternative): Using venv

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

2) Start the MLflow tracking server (in another terminal)

The default tracking URI is `http://127.0.0.1:5000` (see `src/config/config.yaml`). You can use the provided helper script:

```bash
bash setup_tracker.sh
```
This runs MLflow with a local SQLite backend (`mydb.sqlite`).

If `mlflow` isn’t found, install it inside the venv:

```bash
pip install mlflow
```

3) Run the main script

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
	- lm_name: e.g., gpt-4o-mini (if using official OpenAI endpoint), openai/gpt-4o-mini (if using OpenAI compatible endpoint)
	- lm_temperature, max_tokens, seed
	- refusal_judge_model, semantic_judge_model: judge LMs
	- api_base, headers: override API base/headers if using a proxy or different provider
- experiment
	- uri: MLflow tracking URI (default http://127.0.0.1:5000)
	- name: experiment name prefix used in runs

Notes:

- The script requires `OPENAI_API_KEY`. It’s loaded via `python-dotenv` if present in `.env`.
- The current `config.yaml` includes an `api_base` pointing to `https://run.v36.cm/v1/` with custom headers, and thus each model_name comes with a `openai/` prefix for litellm to recognize it as an OpenAI compatible endpoint. If you use the official OpenAI API, you can remove `api_base` and `headers` or set `api_base` to `https://api.openai.com/v1`, also remove the `openai/` prefix in model_name.

## What happens when you run it

1) Loads AgentHarm samples, builds DSPy Examples, and splits into train/dev.
2) Initializes a WebReAct-style DSPy agent and an AgentHarm metric.
3) Runs evaluation on the dev split (baseline). Optimization is scaffolded but commented out.
4) Logs parameters, metrics, detailed results, and a source snapshot to MLflow.

To see the results, open a browser and open your mlflow client (usually at `http://127.0.0.1:5000`.) Model metrics, traces for each sample, and other detailed logs can be seen there.

## Troubleshooting

- OPENAI_API_KEY not set: The script will raise `ValueError`. Create `.env` or export the variable.
- 401/403 from model API: Ensure your key is valid and `models.api_base`/`headers` match your provider.
- MLflow not showing runs: Confirm the server is running on `http://127.0.0.1:5000` and matches `experiment.uri`.
- Python version errors: Ensure you’re on Python 3.12+ (`requires-python = ">=3.12"`).
- Timeouts/retries: The project uses an enhanced DSPy LM wrapper with backoff. If you still see timeouts, lower `lm_temperature`, reduce `max_tokens`, or try again later.

## Run Original AgentHarm Benchmark Using Inspect Evals (CLI)
This repo includes a helper script `run_inspect_evals.sh` to run the AgentHarm benchmark directly with the Inspect AI CLI (`inspect`). This is for reproducing the results in the [AgentHarm paper](https://arxiv.org/abs/2410.09024) and ensuring our inital evaluation has the correct numbers. 
<!-- It targets the `openai-api/ge/...` provider alias, which expects credentials via environment variables. -->

1) (Optional) If you haven't followed either [Option 1](#option-1-recommended-using-uv) or [Option 2](#option-2-alternative-using-venv) to installed packages, ensure these dependencies are available in your venv.

```bash
pip install inspect-ai
pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
```

2) Create and configure `.env` for Inspect Evals

If you want to use the default OpenAI provider (recommended):

```
# Default OpenAI provider
OPENAI_API_KEY=sk-...
# (Optional) Explicitly set the default base URL if needed
# OPENAI_BASE_URL=https://api.openai.com/v1
```

If you want to use other OpenAI-compatible endpoints:

```
# Use a custom named provider alias (example: "openai-api/ge/...")
GE_API_KEY=sk-...
GE_BASE_URL=https://api.gpt.ge/v1/
```

Note: You have to keep `run_inspect_evals.sh` compatible with the variables in `.env`: 
- For the default provider, use `--model openai/gpt-4o-mini` and related `-T ..._judge` values.
- For compatible endpoints using a named provider alias, you must pass the alias in CLI args: use `--model openai-api/ge/gpt-4o-mini` instead of `--model openai/gpt-4o-mini` (apply the same pattern for `-T refusal_judge` and `-T semantic_judge`). 
- Keep `OPENAI_API_KEY` for running `main.py`; `GE_*` variables are used only when calling models as `openai-api/ge/...` via Inspect.
- In `run_inspect_evals.sh`, you can switch providers by setting `INSPECT_PROVIDER_PREFIX=openai` or `INSPECT_PROVIDER_PREFIX=openai-api/ge`.
- For more detailed usage please refer to [the API usage instruction by Inspect AI](https://inspect.aisi.org.uk/providers.html#openai-api).

Optional: specify other Inspect AI options (see [here](https://inspect.aisi.org.uk/options.html) for a complete list)

```
INSPECT_LOG_DIR=./results/logs
INSPECT_LOG_LEVEL=warning
INSPECT_EVAL_MAX_RETRIES=5
INSPECT_EVAL_MAX_CONNECTIONS=20
```

Also important: set the exact model strings you want in the Bash script (`run_inspect_evals.sh`) so the CLI uses the intended models for both the agent and judges.

3) Run the script

```bash
chmod +x run_inspect_evals.sh
source .env
bash run_inspect_evals.sh
```

The script uses:

```
inspect eval inspect_evals/agentharm --model openai-api/ge/gpt-4o-mini --seed 42 \
	-T split=test_public \
	-T refusal_judge="openai-api/ge/gpt-4o-mini" \
	-T semantic_judge="openai-api/ge/gpt-4o-mini" \
	-T agent_kwargs="{ 'prompt_technique': 'standard' }"
```

Common tweaks:
- Run items with certain ids: add `--sample_ids "1-1,1-2,1-3"`
- Limit number of items: add `--limit 10`
- Shuffle seed: add `--sample-shuffle 0`
- Change split: update `-T split=val|test_public|test_private`

After the run finishes, you can view the results directly via Inspect AI's VSCode extension, or use `insepct view` to open up a browser window to see results. Refer to their websites for details on [intepreting the logs](https://inspect.aisi.org.uk/log-viewer.html) and [their VSCode extension](https://inspect.aisi.org.uk/vscode.html).

## License

See `LICENSE` in this repository.

