# Auto research, direction 2: visual memory in native fast weights (rotating cube)

## Goal and plan (user, 2026-11-02)

- **Question:** can native fast weights hold concrete visual content and recall it after a controlled time out of view? Direction 1 (Shell Game) tested state tracking; this direction tests a visual cue.
- **Round 1 (this file):** find a TTT that works for visual memory. Axes:
  - which projection: q / k / v / combinations
  - how many fast slots
  - how many blocks
  - possibly a different loss
- **Round 2 (later):** the grid gap × model size (200M + one smaller) × outer training budget.
  - Budget is counted in latent tokens, with steps and GPU hours alongside.
  - Several checkpoints of the same run give the learning progress.
  - Compare, on the same backbone with the same history access: no cross-time memory, an extra memory module (LaCT-style branch, still to implement), and native-weight memory.
- **Constraints from the user:**
  - Random patterns with no prior.
  - Large cube, so the ROI metrics are clear.
  - Multiple cubes are allowed, as another way to stretch the gap.
  - Direction 1 is still being written in the auto_research root, so nothing there was moved or edited; this direction lives in `visual/`.

## Task design (`datagen/cube.py`, `datagen/build.py`)

**Renderer:** a torch ray-caster (CPU/GPU) instead of the Three.js/Chrome pipeline of direction 1.
- Exact per-pixel face ids and (u, v) give exact ROIs.
- There is no video codec in the loop: frames go straight to the VAE.
- A 197-frame episode renders in 0.1 s on an H200 at 4×4 SSAA.

**Scene:**
- One cube, front face about 157×145 px of the 256×192 frame.
- Camera 15° above the cube; fixed Lambert lighting, so the frontal face is shaded ×0.90.
- Plain top face, dark face borders, slate background.

**Patterns:** a 4×4 grid per side face, 8 palette colors, i.i.d. per cell and per episode.
- That is 48 bits per face and 192 bits per episode.
- The palette makes a crisp cell-accuracy metric possible (chance .125); MSE/SSIM/LPIPS are computed as well.

**Timing:** the same chunking as the Shell Game, and every turn takes the same time.
- Chunk 0 is a 17-frame hold. Every later chunk is 20 frames: a 90° turn over 16 frames, then 4 still frames.
- So the last latent frame of each chunk is a still, axis-aligned pose.

**Episode:**
1. chunk 0 hold
2. chunks 1–3 observe: 3 turns, after which every face has been frontal once and the last one is the target
3. leave: one turn
4. gap: g chunks
5. return: one turn the other way, after which the target is frontal again

**Gap mechanism (single cube):** during the gap the target face walks the back range on the path side ↔ back ↔ other side.
- Each gap chunk still turns 90°.
- The leave and return turns are mirror images and identical for every g.
- The walk is sampled uniformly over all closed walks of length g, so g is even.
- g = 0 (leave, then come straight back) is the short-gap level.

**Bookkeeping per test gap:**
- gap_frames = 20g + 5 (the last 5 frames of the leave chunk are already edge-on).
- writes during the gap = g.

**Multi-cube variant (the user's idea):** `--cubes n --gap-mode hold` keeps the target cube still in its back pose while the other cubes turn.
- The renderer and sampler support it.
- It is not used in round 1, because one cube keeps the face as large as possible.

**Actions:** 0 hold, 1 frontal +1, 2 frontal −1 (3 embeddings).
- The model gets only actions, never the pose, so it must also track which face is in front.
- A wrong face shows up as `wrong_face`, not as a texture error.

**Data: `Dataset/CubeMemory`.** All configs share the same test samples and action seeds.

| split | episodes | gap | chunks |
|---|---|---|---|
| train | 60,000 | uniform over {0, 2, 4, 6} | 6–12 |
| test/gXX | 64 per gap, separate seed | 0, 2, 4, 6, 8, 12, 16, 24, 32 | up to 38 (757 frames) |

Latents are Wan2.1 VAE, float16 [16, 5K, 24, 32], with the same normalization as the Shell Game.

**VAE round trip (`build.py --check 8`):** every chunk end (frontal face) was decoded and compared with the exact render.

| | cell_acc | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| full-sequence decode | 1.000 | 38.1 | .988 | .0007 |
| last chunk decoded alone (what eval does) | 1.000 | 37.3 | .986 | .0010 |

So the patterns are cleanly reconstructible at this resolution; the VAE is not a bottleneck.

## Metrics (`metrics.py`)

**ROI:** the frontal face is rectified with `grid_sample` on its 4×4 area: face coordinates [.05, .95]², 64×64 samples, border excluded. The geometry is the same for every face at the frontal pose.

**Texture metrics against the exact GT render:**
- `mse`, `psnr`, `ssim` (Gaussian 11×11), `lpips` (AlexNet). LPIPS was added to the `world` env: `lpips 0.1.4` + `scipy 1.16.2` (exdif's version), with `--no-deps`.
- `cell_acc`: the mean color of the central half of each cell, snapped to the nearest frontal-shaded palette color, compared with the original pattern.

**Pose errors (kept separate):**
- `ident` / `wrong_face`: which of the 4 faces the generated face matches. `wrong_face` = a different face matches ≥ 50% of the cells.
- `yaw_err`: the |δ| whose silhouette (rendered at the frontal pose + δ, for δ ∈ [−45°, 45°]) best matches the generated silhouette, by IoU.

**Modes:**
- `free` = the main protocol: 4 GT observation chunks, then everything is generated, and no GT of the returning face is ever fed.
- `tf` = GT up to the return chunk; the target face is invisible during the gap, so it is still never fed.
- `tf1` = the σ=1 one-step x0, i.e. what the memory alone predicts.
- `vae` = the GT latent decoded, i.e. the ceiling.

## Model and training (`wm.py`, `train_base.py`, `train.py`)

- **Model:** same architecture and parameter names as `world_model/model.py`, with size presets (`200m` = 201.2M; `50m` for round 2).
  - `forward_seq` (full chunk-causal attention) trains the base.
  - The per-chunk functional `forward` with `State` (KV window + per-sample ΔW) runs TTT and inference.
- **Memory = per-sample ΔW on native weights** (zero init, so it starts exactly as the base).
  - Writes happen at the σ=0 commit, write-then-read; denoise only reads.
  - `AttnMem` covers any of q/k/v: the output channels of the last `fast_heads` heads are fast (slots), and the first heads are fallback. Each projection has its target t = T·a (zero init) and per-token step.
  - `FFNMem` is the direction-1 recipe: down, 1/4 of the hidden units fast, 8 blocks.
  - `delta`: ℓ = ½‖ΔW·k − t‖², which can overwrite. `dot`: adds t every time.
  - `key raw`: the layer input with a softplus η, plus renorm (direction 1).
  - `key norm`: unit keys per block and β = 2·sigmoid ∈ (0, 2). Then ΔW ← ΔW(I − β·k̂k̂ᵀ) + β·t·k̂ᵀ has eigenvalues in (−1, 1], so the memory cannot blow up however many writes it gets. This is the candidate fix for direction 1's pitfall 18 (fast v diverges beyond the training length).
- **Changes vs direction 1's wm.py:**
  - The state is ΔW, not slow + ΔW, so a bf16 read never rounds a small increment away against the large slow weight (direction 1's pitfall 13).
  - The write runs in fp32 with autocast off.
- **`check.py` results:**
  1. Per-chunk forward vs `forward_seq`: 7.3e-6 apart.
  2. Zero-init memory vs no memory: exactly 0 for q / v / qkv (no fallback) / kv norm / ffn / v+ffn.
  3. Hand-written write vs autograd: 1.5e-8 to 6e-8 for delta/dot × raw/norm.
  4. BPTT reaches T_q and T_v. The step gradient is 0 at init, which is expected: the error is 0 while T = 0.
- **Base (from scratch):** diffusion forcing (per-chunk σ, shift 3, clean prefix with p = .5), full chunk-causal attention over the episode, NorMuon 2e-4 (warmup 1000), EMA .999, global batch 16 on 2 GPUs, 20000 steps, grad checkpointing.
  - Smoke test: 1.8 s/step at batch 4 on a shared H200, 19 GB.
  - Every 2500 steps the EMA is evaluated with the KV cache fully kept (gaps 0/2/6, tf + free). At the same time the data loader is rebuilt, so episodes still being encoded get in.
- **TTT (round 1):** from base ema_020000.
  - max_ctx 0: no history KV and no re-injected first frame, so the memory is the only path between chunks.
  - Teacher-forced unroll + BPTT, B = 8, NorMuon 1e-4 + AdamW for the step parameters (lr 1e-3), EMA .999, 12000 steps.
  - Quick eval every 1000 steps: raw weights, tf, gaps 0/2/6/12, 32 episodes, 20 steps.
  - The final EMA eval runs automatically: all gaps, tf + free, 64 episodes, 50 steps, with videos.
  - A base gate makes a run exit immediately if the base's own g00 tf cell_acc (KV kept) is below .5.
  - Logged: loss per phase (observe / leave / gap / return), ‖ΔW‖/‖W‖ per memory, tokens, GPU hours, writable fast weights and state size per sample.

## Round 1 (`waves/r1.sh`, code frozen in `snapshots/r1`, submitted 2026-11-02 22:55)

- **Jobs:** base 22357148 (2 GPUs, 20000 steps), then two base evals (KV fully kept / ctx 0), then 17 runs, all afterok on the base. Job ids are in `Finetune/world/auto_research/visual/r1_jobs.txt`.
- **Defaults:** fast_heads 4 (288 rows), blocks 4, delta, raw key, all 12 layers.

| group | runs | question |
|---|---|---|
| A projection | ctrl, v, q, k, qk, qv, kv, qkv, ffn | which native weight can carry content; ctrl = max_ctx 0 without memory |
| B write rule | v_norm, qkv_norm (β init .5), v_dot | does the bounded write fix long gaps; is pure accumulation enough for store-once recall |
| C slots | v with fast heads 1 / 8 / 16 (72 / 576 / 1152 rows; 16 = no fallback) | capacity |
| D blocks | v with blocks 1 / 16 | block-diagonal split |

Fast weights per sample for the default v: 1.0M writable (block-diagonal nonzeros), 4.0M state (16 MB fp32) over 12 layers.

**Predictions (before results):**
1. **Content needs a content path.** v, qkv and ffn learn the return. q-only and k-only (routing inside the current chunk, with no other chunk to attend to) stay near ctrl, or learn only coarse colors. This would reverse direction 1, where q beat v for a 2.3-bit state.
2. **Within the training range** (g ≤ 6), v and ffn approach the base with KV fully kept. Beyond it, raw-key v degrades (direction 1's pitfall 18) and `v_norm` holds up better.
3. **Slots:** 1 head (72 rows, 0.2M writable) is not enough for 192 bits plus the pose; ≥ 4 heads saturate.
4. **dot** works better here than in direction 1, because the content is written once rather than updated, but it degrades at long gaps as writes pile up.
5. **Pose tracking (Z4) is easy for every memory that works at all.** `wrong_face` should be rare; failures show up as blurry or averaged textures, not as wrong faces.

## Status log

- **2026-11-02 23:00:**
  - Data encode: Slurm array 22355254 (16 × 1 GPU; only 2 running because of priority), plus 2 local workers on GPU 1 of udc-an25-1. Workers claim shards. Encode is VAE-bound: H200 1.67 episodes/s, A6000 0.3 episodes/s.
  - Base and round 1 are pending in the Slurm queue.
- **2026-11-03 12:20:**
  - Data complete: 60,000 train + 9 × 64 test, 64 GB. Three Slurm encode tasks on A6000s hit the 4 h limit mid-shard (shards 38/40/41, 792 episodes). I released their claims and finished them locally on GPU 5 in 7 minutes.
  - The Slurm base (22357148) never started in 13 h because of priority, so it and its 19 dependents were cancelled.
  - New local GPU quota for this direction (user): 2 local GPUs, +1 for quick data jobs while one is free; fine-tuning is capped at 2 local GPUs.
  - The base now runs locally on GPUs 2 and 3 (`waves/r1_local.sh`): DDP with batch 8 per GPU, 1.6 s/step, 36 GB/GPU, so 20,000 steps take about 9 h.
  - When the base finishes, the same script runs `LOCAL=... SKIP_BASE=1 bash waves/r1.sh`. GPU 2 runs r1_v then r1_qkv, GPU 3 runs r1_q then r1_ctrl, and the other 13 runs plus the 2 base evals go to Slurm.
  - **Pitfall:** `torchrun ... train_base.py --run X` fails, because torchrun's argparse reads `--run` as an abbreviation of its own `--run-path`. Put `--` before the script: `torchrun ... -- train_base.py --run X`.
- **2026-11-03 16:10:**
  - **Base (full KV kept) solves the task by step 5000.** At steps 5000 and 7500 the quick eval (32 episodes, gaps 0/2/6, 20 sampling steps) gives cell_acc 1.00 in tf, tf1 and free, with PSNR 37.0–37.4 dB against a VAE ceiling of 37.3–37.5. So the reference ceiling with history is the VAE ceiling, and round 1 has the full range from ctrl (chance .125) to 1.00 to spread out over. Loss at step 7700: 0.026 (episodes with K=6 sit higher at 0.035, because a larger share of their chunks are unpredictable new faces). Base gate passed.
  - Pace: 1.66 s/step averaged over evals and saves (69 min per 2,500 steps), so step 20,000 lands around 21:30.
  - **Slurm is jammed:** fairshare 0.08 for jqf8qm, 94 of our jobs pending, the oldest (ttt_a1_*) for 28 h, and only 8 running. The 13 r1 runs sent there would likely wait days.
  - **Change:** `LOCAL=...` in `waves/r1.sh` now means a local queue (`waves/local_queue.py`) instead of fixed lanes of 2 runs per GPU. It runs one training run per GPU, because packing gives no throughput (root LOG #20), plus one eval slot per GPU. The order is: base evals ctx-1 / ctx0, then r1_v, r1_q, r1_qkv, r1_ctrl, then the rest in r1 order. The other 13 runs are still submitted to Slurm as a fallback; when the queue reaches a run that is still pending there, it cancels it and runs it locally. The queue file is `Finetune/.../visual/r1_queue.tsv`, re-read every minute, so priorities can change live once the first results come in. `r1_local.sh` itself was not edited (bash is still reading it); it calls `r1.sh` after the base.
  - At about 8.5 h per run, the 17 runs need about 3 days on 2 GPUs. The first 4 results should arrive tomorrow morning; then I reorder or prune the rest.
  - `datagen/preview.py` (the user wanted to see what the videos look like; the dataset stores only latents) re-renders stored test episodes with a phase bar, next to the VAE decode of the stored latent. Output is in `Dataset/CubeMemory/preview/` (g02/g06/g12, mp4 + GIF + per-chunk end-frame sheet). The decoded latent matches the render; the only visible VAE artifacts are thin slivers on side faces seen at grazing angles.

## Task 2: 2×2 Rubik's cube (`datagen/rubik.py`, user 2026-11-03 16:30)

The user's request: "把任务变成魔方 … 魔方上每一个小面上的图案都是随机出来的纹理 … 动作就是跟魔方操作一样 每一个面怎么转 … 最后展示就是固定的6个面的展示 几个整体转的额外动作". The rotating cube is superseded. Its base keeps running on GPUs 2/3 until the Rubik data needs them; the Slurm part of round 1 is switched off (`SLURM_FALLBACK=1` re-enables it).

Design choices (the user picked each from options, the recommended one every time):

| Choice | Picked | Alternatives offered |
|---|---|---|
| Order | **2×2** first (8 cubies, 24 stickers); 3×3 later as a harder level. The code works for any n | 3×3; both |
| Sticker texture | **random 3×3 color blocks**, 8 colors, independent per sticker and episode (24 × 9 × 3 = 648 bits) | procedural textures; one random color per sticker |
| Episode | **display → scramble → display**: chunk 0 hold, chunks 1–3 whole-cube `y y x` (back, left, down come into view: every sticker seen), N random face turns, the same `y y x` again; evaluation on the end frames of the last scramble chunk and the 3 display chunks | rotations mixed into the scramble; no initial display |
| Actions | **19**: hold, 12 quarter face turns `U U' D D' L L' R R' F F' B B'` (named by current position), 6 rotations `x x' y y' z z'` | + 6 half turns (25); only U F R turns (13) |

Rendering details:
- 3/4 view: azimuth 45°, elevation 35°, distance 9. U, F and R are visible, shaded .94 / .76 / .65.
- The frame fits the cube's circumscribed sphere, so a layer stays in frame halfway through a turn. Black plastic; the palette replaces black with orange.
- Every cubie is a box with its own cumulative rotation. A turning layer gets an extra rotation for that frame. The sticker (u, v) frame rides on the cubie, so stickers rotate with it, the way real stickers do.
- Timing is the same as the rotating cube: 16 frames turning (cosine easing) plus 4 still frames.
- Scramble: uniform over the 12 face turns; the next turn is never the inverse of the previous one.
- Episode length K = 7 + N chunks. Proposal: train N ~ U{0..8}, test N ∈ {0, 2, 4, 8, 12, 16, 24}.

**Sample** (`Dataset/RubikMemory/sample/`: meta.jsonl + latents + preview/, N = 0, 2, 4, 8, 12, 16):
- Every evaluation frame shows 108 cells (3 faces × 4 stickers × 9).
- The VAE round trip keeps all of them: cell_acc 1.000 on all 24 evaluation frames. Each cell's interior is averaged and classified as the nearest shaded palette color.
- PSNR is 32.0–32.6 dB, lower than the rotating cube's 37–38 dB because there are many more edges. Fast-moving layers mid-turn decode slightly blurred.
- The cube is about 150 px tall; cells are about 11–12 px.
- Cost: rendering is 4.3 ms/frame (SSAA 4×4, H200 shared with the base), about 0.9 s for an 11-chunk episode. With VAE encode, a full set of 60K episodes is about 12–13 h on 2 GPUs.

### Full data and base (user 2026-11-03 17:00: "都对 直接开始做 … 先做数据 然后数据做好了就开始用一个比较promising的setting开始从头train")

- **17:01:**
  - Stopped the rotating-cube base at about step 10,250 (`base/ema_010000.pt` is kept), together with its launcher, so round 1 never starts. GPUs 2/3 are free.
  - `datagen/build_rubik.py --meta`: train 60,000 episodes (N ~ U{0..8}, K = 7–15 chunks), plus `test/n00 n02 n04 n08 n12 n16 n24` with 64 each (K = 7–31).
  - Encoding: 4 local workers, 2 each on GPUs 2 and 3, claiming 128 shards. Throughput is about 1.64 episodes/s in total, so it should finish around 03:30.
  - No Slurm array this time: a slow A6000 task that claims one of the last shards would hold up the end while the local GPUs sit idle.
- **Code is task-aware** (the rotating-cube code is frozen in `snapshots/r1`):
  - `data.TASKS` gives each task its root, action count, bucket key and splits.
  - `train_base.py` and `train.py` take `--task rubik` (now the default); `evaluate.py` reads the task from the checkpoint.
  - `rubik_eval.RubikEvaluator`:
    - Evaluation frames: the end of the last scramble chunk plus the end of the 3 display chunks; the last display frame shows only stickers that were on the back during the scramble.
    - Per-cell accuracy is split into `hid`/`vis` by whether the sticker faced away at the end of the scramble.
    - Modes: free (GT chunks 0–3, then generated), tf (GT up to the display), vae.
- `waves/rubik_base.sh` (launched on GPUs 2/3):
  - waits for the encode workers to exit;
  - sweeps for missing latents;
  - freezes the code into `snapshots/rubik`;
  - starts `train_base.py --task rubik --run rubik_base --steps 40000` (DDP, full chunk-causal attention).
- **Plan after the base:** memory fine-tuning at max_ctx 0 with one promising setting plus a control:
  - Promising setting: `qkv` + normalized keys (β∈(0,2)). q/k handle routing and permutation tracking (direction 1's best), v carries the texture content, and bounded writes guard against direction 1's long-horizon v blow-up.
  - Control: no memory.
- **17:45:**
  - **Smoke tests pass** (scratch runs on GPU 3, next to the encoders):
    - `train_base.py --task rubik`: 20 steps plus the quick eval. VAE ceiling is 1.000 on every eval frame (32 dB); the untrained model is at chance (≈.125).
    - `train.py --task rubik --mem-proj qkv --mem-key norm`: 10 steps, then the final eval over all 7 test N with videos. Fast weights per sample: 3.0M writable, 11.9M state (46 MB fp32); 12.0M new parameters.
    - Fixed: with N = 0 the "scramble end" frame is the last GT display chunk, so `scr` is now scored only when it is generated.
  - **Why Slurm can't speed this up:**
    - The GPU partitions are saturated by other users: syw3ev has 1018 GPU jobs pending, dqr2ye 513 on a6000, jrj6wm 421 on a100-80; most GPU nodes are in the `mix-` (planned) state.
    - jqf8qm's fairshare is .08. Cancelling our pending jobs doesn't raise priority, and a fresh submission starts with zero age.
    - Levers I gave the user: lend GPUs 6/7 (data about 4 h sooner, base 2× faster); the paid account or the `sds` partition (needs their OK); stopping the base once its eval saturates (my default).

### Remote package (user 2026-11-03 18:00: run everything on a remote 8x H100 node from a zip plus an instruction note, modeled on `position_geo_obs/remote/`; hand-back through public `riselab` repos on Hugging Face)

- `remote/` (maintainer README, `package/`, `tools/build_package.py`, `dist/rubik_memory_remote.zip`):
  - **The run:** one command, `./run.sh pipeline 0,...,7`: data (2 workers per GPU), then the base (8-GPU DDP, 40K steps), then the sweep, then the summary. The hand-back is `./run.sh handback`.
  - **The sweep:** 8 fine-tunes from `ema_040000`, one per GPU: ctrl, qkv_norm, qkv, v_norm, v, q, k, ffn. Next to them run the base evaluations ctx -1 and ctx 0. Est. ~24 h in total.
  - **Hand-back:** public HF model repo `riselab/rubik_memory`, folder `rubik_r1_<time>`: base EMAs 20K/40K, `ckpt_012000.pt` per run, eval JSONs, video tars, logs, summary and a sha256 MANIFEST. The dataset can optionally go to `riselab/rubik_memory_data`.
- Code changes for it:
  - `paths.py` reads `VM_DATA_DIR` / `VM_OUT_DIR` / `VM_VAE` from the environment, with the local defaults; data, evaluate, train_base, summary and the datagen builders use it.
  - `summary.py --task rubik` prints the per-run × N table.
  - The build swaps in a strict `paths.py` and refuses any local path, host or user name.
- Env lock = the dependency closure of the local `world` env, read with `PYTHONNOUSERSITE=1`. Without that flag a `~/.local` copy (annotated-types 0.7.0) shadowed the env's 0.8.0.
- **Pitfall in the local test:** `CUDA_VISIBLE_DEVICES=3 ./run.sh smoke 0` does not stay on GPU 3, because run.sh sets `CUDA_VISIBLE_DEVICES` per process from its own list. I killed it before it touched GPU 0 (another user's). GPU lists given to run.sh are physical indices.
- **`pkill -f "<pattern>"` from a tool shell can kill that shell itself** when the pattern appears in its own command line (exit 144). Kill by PID.
- **Scope change (user, 19:00): "不能有evaluation 只能有dataset creation training 和inference eval这些我们拿回来跑".** The remote side now does dataset creation, training and inference only:
  - **Evaluation off:** the base runs with `--eval-every 0`; the fine-tunes with `--eval-every 0 --final-eval 0 --base-gate 0`. Evaluation modules are imported only when an evaluation is requested.
  - **New stage `infer`** (`infer.py`): rollouts of the 7 test splits × 64 episodes, free + tf, for 27 checkpoints (base 40K ctx-1/ctx0, base 20K ctx-1, every fine-tune at 4K/8K/12K). It saves fp16 latents (`free.npy` whole episodes, `tf.npy` last 4 chunks), ~1.1 GB per job.
  - **Hand-back:** ~45 GB to `riselab/rubik_memory`.
  - **The package ships only** wm/train_base/train/infer/sampling/data/normuon + datagen rubik/build_rubik/vae_io. The build refuses top-level imports of evaluation modules and dry-imports the dist with them blocked.
- Refactor for it, local behavior unchanged:
  - `datagen/vae_io.py`: latent paths, VAE load/encode, shard claims, moved out of `build.py`.
  - `datagen/rubik_sample.py`: the sample CLI, moved out of `rubik.py`.
  - `sampling.py`: σ schedule, noise, chunk sampler, rollout, `build_from_ckpt`, moved out of `evaluate.py`.
  - `rubik_eval.py`: `score()` is shared by the live `run()` and the new `run_saved()`. `python rubik_eval.py --saved <infer out> --out <json>` scores handed-back latents.
- **Check:** live evaluation and infer + saved evaluation of the same checkpoint agree within 0.0012 on all 51 metrics (fp16 storage).
- Local half of the hand-back: `remote/tools/handback.py fetch` (HF download, sha256 against MANIFEST.json), then `handback.py eval` (one `rubik_eval --saved` per inference job, into `<run>/eval/`, then the summary table).
- **19:28, final zip** `remote/dist/rubik_memory_remote.zip` (60 KB, 20 files):
  - End-to-end test passed: unzip, `check` READY, `smoke 3` PASSED in 10.5 min (data 4.5, base 1, fine-tunes 1, inference 3), plus the hand-back dry run.
  - The hand-back now picks checkpoints by step: the base's final and half-final EMA, each run's final ckpt. The smoke test gathers them too (110 files, 4.7 GB).
  - Local scoring of the smoke run's inference outputs (`rubik_eval.py --saved`, against the local dataset) works: the toy test episodes equal the first two of the real splits.
  - The final build differs from the tested copy only in INSTRUCTIONS.md.
- **20:45:** the user took GPUs 2/3 back for a more urgent task ("我们在remote跑 不要让后面再把这个卡拿回来 不用守护了"):
  - I stopped my monitor and the launcher `waves/rubik_base.sh` (pid 6863), so the local base will never start.
  - The 4 encode workers were left for the user's process to stop. Local dataset: about 22.6K of 60K train, 46 of 64 per test split at 20:40.
  - Direction 2 now runs only on the remote node (the zip). Nothing local is armed.
  - Evaluating the hand-back needs the complete local test splits: re-encode only the test splits (small) when the GPUs are ours again.
