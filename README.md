# ASMA2

ASMA2 collects code for training and evaluating agents on a custom BipedalWalker variant (flip-focused curriculum).

This workspace includes:

- `custom_bipedal.py` — modified BipedalWalker environment with custom reward shaping and curriculum wrappers.
- `BipedalWalker.ipynb` — training and evaluation notebook (stage-based PPO training, wrappers, and evaluation code).
- `play_checkpoints.py` — play saved checkpoints interactively and print episode rewards.
- `plot_results.py` — offline TensorBoard event reader and plotting utilities (reads `rollout/ep_rew_mean`, `flip/*` tags).
- `checkpoints/` — saved model checkpoints for each training stage.
- `ppo_bipedal_curriculum/` and `tb_logs/` — TensorBoard log directories.

## Quick setup

1. Create and activate a Python virtual environment.

Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Recommended Python: 3.8–3.11.

## How to run

- Launch the training notebook: open `BipedalWalker.ipynb` and run the Stage 1/2/3 cells. Training uses Stable Baselines 3 PPO and writes TensorBoard logs to `./ppo_bipedal_curriculum/`.
- View TensorBoard while training:
```bash
tensorboard --logdir ./ppo_bipedal_curriculum/ --port 6006
```
- Evaluate a saved model (stage 3) using the notebook cell or run `play_checkpoints.py` to step through episodes and print totals.
- Plot offline results from TensorBoard event files:
```bash
python plot_results.py --logdir ./ppo_bipedal_curriculum/
```

## What is logged

- Stable Baselines emits rollout scalars such as `rollout/ep_rew_mean` and `rollout/ep_len_mean`.
- The notebook also records custom tags (e.g. `flip/success_rate`, `flip/max_rotation_deg`) and a `train/episode_reward` scalar for per-episode totals.

## Notes

- The main training flow is implemented in `BipedalWalker.ipynb` and depends on `custom_bipedal.py` wrappers.
- Checkpoint saving and TensorBoard logging are enabled; see `checkpoints/` and `ppo_bipedal_curriculum/` for outputs.
- If any module import fails, make sure the virtual environment is activated and the packages from `requirements.txt` are installed.

If you want, I can also:

- add a short `train.py` script that reproduces the notebook training cells,
- or update `requirements.txt` to pin exact package versions used locally.
