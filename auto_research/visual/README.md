# auto_research / visual: direction 2, visual memory

The question: can native fast weights (attention q/k/v, or FFN down, written by TTT) hold **concrete visual content**, and recall it after the content has been out of view for a controlled time? Direction 1 (the auto_research root, Shell Game) asks the same of a hidden state.

The folder is self-contained: it has its own model, data, training, evaluation, waves and snapshots, and imports nothing from the root. Direction 1's files are untouched. Like the rest of auto_research, nothing here goes into `world_model/`. Read `LOG.md` for the design, results and pitfalls.

## Task: rotating cube (`datagen/cube.py`)

- One cube fills most of the 256×192 frame. It rotates about the vertical axis; camera and light are fixed.
- Each of the 4 side faces gets an independent random 4×4 grid of 8 palette colors, resampled per episode. Top and bottom faces are plain.
- The VAE round trip is lossless at the cell level: cell accuracy 1.00, PSNR 38 dB, SSIM .988, LPIPS .0007.
- Chunk timing is the Shell Game's: 17 frames, then 20 frames per chunk, which is 5 latent frames each. A turn is 90° over 16 frames followed by a 4-frame hold, so every chunk ends on a still, axis-aligned pose.
- Actions: 0 hold, 1 frontal face +1, 2 frontal face −1.

Episode with gap g (all `g` even):

| chunks | phase | input at eval |
|---|---|---|
| 0 | hold, start face frontal | GT |
| 1–3 | observe: 3 turns, same direction; every face is frontal once, and the last one is the **target** (uniform) | GT |
| 4 | leave: the target turns to the side (just invisible) | generated |
| 5 … 4+g | gap: the target stays in the back range (side, back, other side) and every chunk still turns 90°, so speed and path stay the same | generated |
| 5+g | return: turn back, the target ends frontal → evaluate the last frame | generated |

- gap is counted in chunks, with frames (= 20g + 5) and online writes (= g) recorded alongside.
- Train gaps are U{0, 2, 4, 6}. Test gaps are 0, 2, 4, 6, 8, 12, 16, 24, 32, with 64 episodes each, a separate seed, and the same set for every config.
- `--cubes n --gap-mode hold` also renders the multi-cube variant, where other cubes turn while the target cube waits. It is not used yet.

## Metrics (`metrics.py`)

The frontal face is rectified in face coordinates (inner 4×4 area only, no border) and compared with the exact GT render:

- `cell_acc`: nearest-palette color of each cell vs the original pattern. Chance is .125.
- `mse`, `psnr`, `ssim`, `lpips`
- Pose errors, kept separate:
  - `wrong_face`: another face's pattern matches clearly better.
  - `yaw_err`: the silhouette fit in degrees.
- Short-gap level: `vae_*` is the VAE ceiling and g = 0 is the shortest gap.

## Files

| File | Purpose |
|---|---|
| `datagen/cube.py` | Episode sampler + torch ray-cast renderer (CPU/GPU, 4×4 SSAA), exact face masks |
| `datagen/build.py` / `encode.slurm` | meta.jsonl for all splits; render + Wan2.1 VAE encode, sharded with claims (local GPUs and Slurm together); `--check` VAE round trip |
| `datagen/rubik.py` | Task 2 (user, 2026-11-03): 2×2 Rubik's cube (any n), random 3×3 color blocks per sticker, display → scramble → display, 19 actions; renderer, episode sampler, `cell_acc` per visible cell; `--sample` writes a small set (meta + latents + previews) |
| `datagen/preview.py` | Look at stored episodes: the dataset keeps only latents + meta.jsonl, so this re-renders them with a phase bar, next to the VAE decode of the stored latent (mp4 + GIF + per-chunk end-frame sheet). Output: `Dataset/CubeMemory/preview/` |
| `wm.py` | Model: configurable DiT (`200m` / `50m`), `forward_seq` (base training) and per-chunk `forward` with `State` (KV window + per-sample ΔW); `AttnMem` (q/k/v), `FFNMem` (down), `DeltaMem` (delta/dot, raw/norm key) |
| `train_base.py` / `.slurm` | Base from scratch: diffusion forcing, full chunk-causal attention, NorMuon, DDP |
| `train.py` / `.slurm` | Memory TTT fine-tuning: max_ctx 0, unroll + BPTT; every option is a `--mem-*` flag; final EMA eval runs automatically |
| `evaluate.py` / `eval.slurm` | free / tf / σ=1 evaluation over all gaps, per-episode JSON, side-by-side videos |
| `check.py` | Correctness checks (rerun after changing `wm.py`) |
| `summary.py` | Tables and curves across runs |
| `waves/rN.sh`, `snapshots/rN/` | Submission scripts, and the code frozen per wave |
| `waves/local_queue.py` | Local queue on this direction's 2 GPUs: one training run per GPU (packing gives no throughput, see root LOG #20) plus one eval slot; reads `rN_queue.tsv` every minute (editable live), takes runs that are still pending on Slurm |

## Commands

```
python datagen/cube.py --preview 4 --gap 4 --out /tmp/cube_preview
python datagen/preview.py --split test/g02 test/g06 test/g12 --num 1 --decode 1
python datagen/rubik.py --sample 6 --moves 0 2 4 8 12 16 --out /project/sds-rise/ethan/Dataset/RubikMemory/sample
python train.py --run v_h4 --mem-proj v --gpu 0
python evaluate.py --ckpt /project/sds-rise/ethan/Finetune/world/auto_research/visual/r1_v/ckpt_012000.pt
python summary.py
```

Data lives in `/project/sds-rise/ethan/Dataset/CubeMemory`. Runs go to `/project/sds-rise/ethan/Finetune/world/auto_research/visual/<run>/`, and logs to `.../visual/logs/`. The wandb project is `world-auto-visual`.
