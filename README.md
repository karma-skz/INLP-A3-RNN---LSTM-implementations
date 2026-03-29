[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tTL6Bg44)
# Assignment3_boilerPlate
Deadline: 27 March 2026

## Implemented entrypoints

The following subcommands are now implemented from `main.py`:

- `task1_rnn`
- `task1_lstm`
- `task2_ssm`
- `task2_bilstm`
- `task3_ssm`
- `task3_bilstm`

## Project layout additions

- `src/common/` shared data, metrics, custom layers, artifact helpers
- `src/task1/` char-level decryption training/eval (RNN, LSTM)
- `src/task2/` language models (SSM NWP, Bi-LSTM MLM)
- `src/task3/` decryption + LM correction pipeline
- `config/task*/` default YAML configs for each command

## Quick start

1. Create/sync environment and install dependencies.
2. Add your tokens in `.env` (`HF_TOKEN_READ`, `HF_TOKEN_PUSH`, `WANDB_API_KEY`).
3. Train Task 1 models:

```bash
uv run main.py task1_rnn --mode train
uv run main.py task1_lstm --mode train
```

4. Train Task 2 models:

```bash
uv run main.py task2_ssm --mode train
uv run main.py task2_bilstm --mode train
```

5. Run Task 3 pipelines:

```bash
uv run main.py task3_ssm --mode evaluate
uv run main.py task3_bilstm --mode evaluate
```

All outputs are written in `outputs/results/` and checkpoints in `outputs/checkpoints/`.
W&B run files are stored in `outputs/wandb/`, and each run also writes a small `*_wandb.json`
file into `outputs/results/` containing the direct run URL.

## Notes

- RNN/LSTM/Bi-LSTM implementations are manual recurrent computations (no `nn.RNN`/`nn.LSTM`).
- Task 3 supports loading models from local or Hugging Face through config.
- When `checkpoint.source: local`, loading now auto-falls back to Hugging Face if the local checkpoint is missing and `repo_id`/`hf_filename` are configured.
- Default training configs now auto-push checkpoints to `karma-skz/inlp-assignment3-models` after a successful train save.
- If you want to disable Hub uploads for a run, set `checkpoint.push_to_hf: false` in that config.
