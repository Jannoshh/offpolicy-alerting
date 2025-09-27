# Repository Guidelines

## Project Layout
All Python entry points live at the repository root for quick discovery: `build_offpolicy_dataset.py` generates supervision data, `train_alert_model.py` handles fine-tuning, and shared helpers sit in `data_utils.py` and `constants.py`. Keep generated corpora under `data/` (for example `data/generated/train.jsonl`) so they remain easy to diff and archive. If you add supporting notebooks or notebooks, place them alongside the scripts with descriptive names so the workflow stays linear.

## Fast Iteration Workflow
1. Install deps with `uv venv && source .venv/bin/activate` followed by `uv pip install -r requirements.txt` (cached wheels make reinstalls cheap).
2. Generate a fresh slice: `python build_offpolicy_dataset.py --math_split train[:5] --output_train data/generated/train.jsonl`. By default this calls `google/gemma-3-4b-it` for the on-policy trace and `Qwen/Qwen3-0.5B-Instruct` for the injected sentence; adjust `--policy_model` or `--injection_model` if you want other anchors. Inspect the output immediately with `python -m json.tool data/generated/train.jsonl | head` or open the file in your editor to confirm each record looks right.
3. Once satisfied, fine-tune with `python train_alert_model.py --train_file data/generated/train.jsonl --max_steps 40 --logging_steps 5 --kl_weight 0.05 --use_lora` (Gemma 3 4B is the default base; override `--model_name` or `--kl_reference_model` if you need another anchor). Monitor the console loss and saved `trainer_state.json` to spot regressions quickly.
4. Re-run step 2 with tweaked injection prompts or partial ratios whenever the alert behaviour drifts, then repeat fine-tuning. Keep old JSONL slices in `data/archive/` if you need to compare behaviours.

## Data Handling Notes
Use `data/generated/` for working sets and reserve `data/samples/` or `data/archive/` for curated examples you want to keep. Always strip sensitive content before writing to disk, and record the model names you used for policy and injection generation in a README next to the dataset so future runs are reproducible.
