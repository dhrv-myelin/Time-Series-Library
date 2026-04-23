# AGENTS.md

## Fast orientation

- Primary entrypoint is `run.py`; all training/inference flows go through `python -u run.py ...`.
- Task dispatch is by `--task_name` in `run.py`: `long_term_forecast`, `short_term_forecast`, `imputation`, `anomaly_detection`, `classification`, `zero_shot_forecast`.
- Runtime pipelines live in `exp/`; data loading is centralized in `data_provider/data_factory.py`; models are in `models/`; reusable blocks are in `layers/`.
- Repro configs are script-first: use `scripts/**/**/*.sh` as source of truth for per-task arguments.

## Environment and dependency quirks

- This fork includes `pyproject.toml` + `uv.lock` (Python 3.13+ metadata), but many upstream scripts/docs were tuned around Python 3.11 + Torch 2.5.1.
- If environment behavior diverges from scripts, use `README.md`/`Dockerfile` as the fallback known-good runtime (`pip install -r requirements.txt` after installing matching Torch).
- Optional model deps are real runtime requirements:
  - `models/Mamba.py` needs `mamba_ssm` (Linux/CUDA-specific wheel in README/Dockerfile).
  - `models/Moirai.py` needs `uni2ts --no-deps`.

## Data behavior that affects runs

- Many loaders auto-fetch from Hugging Face (`thuml/Time-Series-Library`) when local files are missing (ETT/PSM/SWAT/UEA/M4 and others in `data_provider/data_loader.py` and `data_provider/m4.py`).
- This means missing local datasets may silently trigger network downloads; offline runs fail unless data is pre-populated.
- Expected local convention in scripts is `./dataset/...` (not `./data/...`).

## Critical CLI gotchas

- GPU is effectively on by default (`--use_gpu` defaults true). To force CPU, use `--no_use_gpu` (not `--use_gpu False`).
- Most upstream scripts hardcode `export CUDA_VISIBLE_DEVICES=<id>`; adjust or remove before running.
- For UEA classification, `UEAloader` resolves dataset files using `args.model_id` as dataset name (`<model_id>_TRAIN.ts` / `<model_id>_TEST.ts`). Keep `--model_id` aligned with dataset folder/name.
- `Exp_Basic` auto-discovers every `models/*.py` file; adding a model file is usually enough for `--model <FileName>` lookup.

## Focused verification (no formal unit-test suite)

- There is no repo-configured `pytest`/lint/typecheck/CI workflow in-tree; verification is usually a targeted `run.py` smoke run.
- Minimal smoke pattern (1 epoch) is documented in `README.md` under "Quick Test"; use one command for the task you changed.
- Outputs are written to `checkpoints/`, `results/`, `test_results/`, and `result_*txt` files at repo root.
