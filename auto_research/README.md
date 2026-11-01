# auto_research

Claude runs this line on its own: TTT on the Shell Game DiT using the native FFN weights as fast weights. Nothing here is imported into or copied to `world_model/` (the hand-written line). Read LOG.md for results and pitfalls.

| File | Purpose |
|---|---|
| `wm.py` | Model: per-chunk functional forward, `State` (KV window + per-sample fast W1/W2), `FastWriter` (dot/mse writes), `e2e_update`, `BallProbe` |
| `train.py` | Teacher-forced unroll + BPTT fine-tuning; every design option is a CLI flag |
| `evaluate.py` | tf/free accuracy, soft metric p_true (σ=1 single-step prediction), base-model evaluation |
| `check.py` | Correctness checks; rerun after changing `wm.py` |
| `summary.py` | One summary table across all runs; `--curve <run>` shows one run's eval curve |
| `train.slurm` / `eval.slurm` | Single-GPU Slurm wrappers. `CODE=snapshots/wN` runs that wave's frozen code |
| `snapshots/wN/` | Code frozen per wave, so that later edits don't change runs still queued |
| `ref/` | Snapshot of the base code from world_model (used by `check.py` only) |

Commands:

```
python train.py --run dot_both_all --ttt-layers all --ttt-params both --ttt-loss dot --gpu 0
sbatch -J NAME --export=ALL,CODE=$PWD/snapshots/w2 train.slurm --run NAME --ttt-layers last3 --ttt-loss e2e --ttt-lr 10 --ttt-lr-lr 1e-2
python evaluate.py --ckpt /project/sds-rise/ethan/Finetune/world/auto_research/<run>/ckpt_006000.pt
python summary.py
```

Run outputs go to `/project/sds-rise/ethan/Finetune/world/auto_research/<run>/` and logs to `.../auto_research/logs/`. The wandb project is `world-auto`.
