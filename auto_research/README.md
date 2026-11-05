# auto_research

Claude runs this line on its own: TTT on the Shell Game DiT using the native FFN weights as fast weights. Nothing here is imported into or copied to `world_model/` (the hand-written line). Read LOG.md for results and pitfalls.

Research directions. Each one has its own LOG.md and README.md. A new direction gets its own self-contained subfolder, with its own code, waves and snapshots, so that directions never break each other:

| Direction | Folder | Task |
|---|---|---|
| 1. state tracking | this folder (root) | Shell Game: hidden ball position under swaps |
| 2. visual memory | `visual/` | Rotating cube: random face patterns recalled after a controlled gap |

| File | Purpose |
|---|---|
| `wm.py` | Model: per-chunk functional forward, `State` (KV window + per-sample fast W1/W2), `FastWriter` (dot/mse writes), `e2e_update`, `BallProbe` |
| `train.py` | Teacher-forced unroll + BPTT fine-tuning; every design option is a CLI flag |
| `evaluate.py` | tf/free accuracy, soft metric p_true (σ=1 single-step prediction), base-model evaluation |
| `check.py` | Correctness checks; rerun after changing `wm.py` |
| `summary.py` | One summary table across all runs; `--curve <run>` shows one run's eval curve |
| `train.slurm` / `eval.slurm` | Single-GPU Slurm wrappers. `CODE=snapshots/wN` runs that wave's frozen code |
| `snapshots/wN/` | Code frozen per wave, so that later edits don't change runs still queued |
| `ref/` | Snapshot of the base code from world_model (used by `check.py` and `train_base.py`) |
| `train_base.py` / `train_base.slurm` | Multi-ball base: continue the single-ball base on Dataset/ShellGameMulti (diffusion forcing, full attention, DDP) |
| `datagen/gen_multi.py` / `render_multi.html` | Multi-ball data: k = 1..5 colored balls; `render_multi.slurm` / `encode_multi.slurm` |
| `waves/local_queue.py` | Feeds wave MC cells onto local GPUs (`--gpus 6 7`, `--slots 4`) when a slot and ≥ 24 GB are free. Started cells (with a last.pt) go first, and train.py resumes them. Skips cells that are evaluated or running locally / on Slurm, and takes pending cells from Slurm. `--dry` prints the order only; `--stop` stops the feeder, then every local run when it is not writing a .pt, and logs the stop steps. Since 2026-11-03 17:22 wave MC is Slurm only (LOG.md, Wave MC) |
| `waves/mc.sh` | Wave MC grid (k balls × fast q rows r × time), `MODE=local GPU=6` or `MODE=slurm`; tables via `python summary.py --cap mbc` |

Commands:

```
python train.py --run dot_both_all --ttt-layers all --ttt-params both --ttt-loss dot --gpu 0
sbatch -J NAME --export=ALL,CODE=$PWD/snapshots/w2 train.slurm --run NAME --ttt-layers last3 --ttt-loss e2e --ttt-lr 10 --ttt-lr-lr 1e-2
python evaluate.py --ckpt /project/sds-rise/ethan/Finetune/world/auto_research/<run>/ckpt_006000.pt
python summary.py
CUDA_VISIBLE_DEVICES=1,6 torchrun --standalone --nproc_per_node 2 train_base.py --run mb_base
python train.py --run mb_q_k3_s0 --data multi --balls 3 --base .../mb_base/ema_XXXXXX.pt --max-ctx 0 --curriculum 1 --ttt-layers all --ttt-loss mse --ttt-params none --ttt-attn q --ttt-fast-frac 0.25 --ttt-heads 4
python evaluate.py --ckpt .../mb_q_k3_s0/ckpt_024000.pt --balls 3
```

Run outputs go to `/project/sds-rise/ethan/Finetune/world/auto_research/<run>/` and logs to `.../auto_research/logs/`. The wandb project is `world-auto`.
