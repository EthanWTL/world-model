# Runbook: visual memory in native fast weights, 2x2 Rubik's cube (dataset creation, training, inference)

This file is the complete task; section 0 says what is finished and what remains. Everything runs on one node with 8
GPUs (8x H100 80 GB; A100 80 GB / H200 work too) from one command, `./run.sh pipeline 0,1,2,3,4,5,6,7`, which runs the
first four stages in order, resumes where it stopped, and skips what is finished. The remote side creates the dataset,
trains and runs inference; it computes no metrics. Evaluation happens at the project, on the handed-back generated
latents.

| # | stage | command | GPUs | est. time on 8x H100 | writes |
| --- | --- | --- | --- | --- | --- |
| 1 | data: render + VAE-encode the dataset | `./run.sh data 0,1,2,3,4,5,6,7` | 8, two workers each | ~3.5 h | `$VM_DATA_DIR/RubikMemory/` (~75 GB) |
| 2 | base: world model from scratch | `./run.sh base 0,1,2,3,4,5,6,7` | 8, one DDP job | ~7 h | `$VM_OUT_DIR/rubik_base/` (~15 GB) |
| 3 | sweep: 8 memory fine-tunes | `./run.sh sweep 0,1,2,3,4,5,6,7` | 8, one run each | ~10 h | `$VM_OUT_DIR/rk_*/` (~70 GB) |
| 4 | infer: test-split rollouts of 27 checkpoints | `./run.sh infer 0,1,2,3,4,5,6,7` | 8, one job each at a time | ~1.5 h | `$VM_OUT_DIR/inference/` (~30 GB) |
| 5 | hand-back: upload | `HF_TOKEN=... ./run.sh handback` | none (internet) | ~1-2 h | public Hugging Face repo `riselab/rubik_memory` (~45 GB) |

Stage 5 is separate from `pipeline`. The times are estimates: data was measured at the project (0.8 episodes/s per
H200), the base, the fine-tunes and the inference only at toy size. Section 1 says what runs, section 2 is the setup,
section 3 the runs and the hand-back, section 4 the fixed settings, section 5 the reference.

## 0. Status (2026-11-03): finished and remaining

Nothing has run on the remote side yet. Remaining, in order: setup (2.1-2.6), smoke test (3.1), full run (3.2),
hand-back (3.4). The same dataset is also being generated at the project; the remote side generates its own copy
(stage 1 is deterministic, seed 0: the same episodes).

## 1. What runs

### 1.1 Task and question

- Video world model of a 2x2 Rubik's cube seen from a fixed 3/4 view (top, front and right faces visible). Each of the
  24 stickers carries its own random 3x3 pattern of 8 colors, drawn independently per episode: there is no prior to
  learn, only content to remember. 256x192, 16 fps.
- An episode is a sequence of chunks (chunk 0: 17 frames, every later chunk: 20 frames = 5 latent frames); one action per
  chunk, 19 actions: 0 hold; 1-12 the quarter face turns `U U' D D' L L' R R' F F' B B'` (named by the current position:
  `U` is the layer on top now); 13-18 the whole-cube rotations `x x' y y' z z'`. A turn takes 16 frames, then 4 still
  frames, so every chunk ends on a still pose.
- Every episode: chunk 0 hold; chunks 1-3 the display `y y x` (back, left and bottom faces come into view: every sticker
  is seen once); then N random face turns (never the inverse of the previous turn); then the same display `y y x`.
  K = N + 7 chunks. The final display shows stickers that were on the hidden faces while they were being moved: drawing
  them needs both their patterns and the permutation the turns applied.
- Question (answered at the project from the hand-back): can native fast weights (attention q / k / v or the FFN,
  written at test time by a delta rule) carry this visual content across chunks when the model attends to no past chunk?

### 1.2 Data (stage 1)

- `train`: 60,000 episodes, N uniform in 0..8 (7-15 chunks). `test/n00 n02 n04 n08 n12 n16 n24`: 64 episodes each
  (N = 12, 16, 24 are longer than any training episode).
- Each episode: one line of `meta.jsonl` (textures, actions, ...) and one latent file `latents/XX/XXXXXX.npy`
  (float16 `[16, 5K, 24, 32]`, Wan2.1 VAE mode, normalized). Videos are not stored.
- Rendering: a torch ray caster (every cubie is a box with its own rotation; 4x4 supersampling) on the GPU, then the
  Wan2.1 VAE encoder. The meta is deterministic (seed 0).

### 1.3 Training (stages 2-3)

- A 200M-parameter diffusion transformer (Wan architecture: width 1152, 16 heads, 12 layers), causal over chunks, trained
  from scratch on the latents, with one action embedding per chunk. No evaluation runs during training (`--eval-every
  0`, `--final-eval 0`): the logs hold the training losses only.
- Stage 2, `rubik_base`: diffusion forcing over whole episodes with full chunk-causal attention (every past chunk is
  visible): the reference with complete history and the starting point of stage 3.
- Stage 3, 8 fine-tunes from `rubik_base/ema_040000.pt`, each with `max_ctx 0`: attention sees only the current chunk,
  so information crosses chunks only through per-episode fast-weight changes (dW) on native weights, written after every
  chunk and read while generating the next. Training unrolls whole episodes (teacher forcing) and backpropagates through
  every write.

  | run | memory |
  | --- | --- |
  | `rk_ctrl` | none (no way across chunks) |
  | `rk_qkv_norm` | q, k, v of 4 of 16 heads; unit keys, write strength in (0, 2) |
  | `rk_qkv` | q, k, v; raw keys |
  | `rk_v_norm` | v only, unit keys |
  | `rk_v` | v only, raw keys |
  | `rk_q` | q only |
  | `rk_k` | k only |
  | `rk_ffn` | FFN down projection, last 1/4 of the hidden units |

### 1.4 Inference (stage 4)

- 27 jobs, one checkpoint each, on all 7 test splits (64 episodes each), under two protocols:
  - `free`: the ground truth of chunks 0-3 (hold + initial display) is given, every later chunk is generated;
  - `tf`: the ground truth is given up to the final display, only its 3 chunks are generated.
- Checkpoints: `rubik_base/ema_040000.pt` with every past chunk in the KV cache (`ctx-1`) and with none (`ctx0`),
  `rubik_base/ema_020000.pt` (`ctx-1`); every fine-tune's `ckpt_004000.pt`, `ckpt_008000.pt`, `ckpt_012000.pt` (EMA
  weights, `max_ctx 0` as trained).
- Output per job, `$VM_OUT_DIR/inference/<run>/<checkpoint>/`: `nXX/free.npy` (float16 `[64, 16, 5K, 24, 32]`, the
  whole episode, chunks 0-3 = the given ground truth), `nXX/tf.npy` (float16 `[64, 16, 20, 24, 32]`, the last 4
  chunks: the scramble's last chunk = ground truth, then the 3 generated display chunks), `nXX/ids.json` (episode order),
  `done.json` (checkpoint, step, max_ctx, sampler settings, shapes, time). ~1.1 GB per job.
- Sampler: Euler on the flow, 50 steps, shift 3; the noise of an episode comes from its own generator seeded with its
  id, so the project can reproduce any rollout exactly.
- No metrics, no decoding: the project decodes the latents and scores them.

## 2. Setup

### 2.1 Sources

| part | source | how to get it |
| --- | --- | --- |
| Code: this folder (`run.sh`, `code/`, ...) | `rubik_memory_remote.zip` | `unzip rubik_memory_remote.zip && cd rubik_memory_remote` |
| Environment: Python 3.11, torch 2.8.0+cu128, 39 pinned packages | conda-forge (Python), `download.pytorch.org` (torch), PyPI (`env/requirements.lock.txt`) | 2.3: `env/setup_env.sh` |
| Wan2.1 VAE (diffusers layout, the `vae/` folder only, 0.5 GB) | public Hugging Face repo `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, commit `0fad780a534b6463e45facd96134c9f345acfa5b` | 2.4: `./download.sh` (checks both files' sha256) |
| Data | generated on the node (stage 1) | nothing to download |

- Internet is needed for 2.3, 2.4 and the hand-back (3.4) only, and for wandb in online mode. The runs read local files.
- If a script reports "permission denied": `chmod +x *.sh *.py env/*.sh`, or call it with `bash` / `python`.

### 2.2 Locations

A run touches these places and nothing else:

| what | where | size | contents |
| --- | --- | --- | --- |
| code, environment, VAE, caches | this folder: `code/`, `env/.env/`, `models/`, `.cache/` | ~7 GB | this package |
| dataset | `$VM_DATA_DIR/RubikMemory/` | ~75 GB | `train/`, `test/nXX/`: `meta.jsonl`, `latents/`; `logs/claims/` |
| outputs | `$VM_OUT_DIR/` | ~120 GB at peak | `rubik_base/`, `rk_*/`, `inference/`, `logs/`, `handback/`, `wandb/`, `tmp/` |

- The two locations are set in `config.env` (or as environment variables of the same names, which win). They are
  independent of each other and of this folder; fast local disk helps the data stage and the DataLoaders.
- `run.sh` exports every cache, config and temp location before Python starts; nothing goes to `~/.cache`, `~/.config`,
  `~/.triton`, `~/.nv` or `/tmp` (apart from the short `TMPDIR` link below):
  - into `<this folder>/.cache/`: `HF_HOME`, `TORCH_HOME`, `PYTORCH_KERNEL_CACHE_PATH`, `TRITON_HOME`, `TRITON_CACHE_DIR`,
    `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `XDG_{CACHE,CONFIG,DATA,STATE}_HOME`, `MPLCONFIGDIR`,
    `WANDB_{CACHE,CONFIG,DATA,ARTIFACT}_DIR`;
  - into `$VM_OUT_DIR/`: `WANDB_DIR`, `TMPDIR=$VM_OUT_DIR/tmp`, `logs/`. When that path is longer than 60 characters,
    `TMPDIR` is a short link `/tmp/rkm-<uid>-<number>` to it (Unix sockets of the DataLoader workers allow 107
    characters); the files still land in `$VM_OUT_DIR/tmp`;
  - offline and isolated: `HF_HUB_OFFLINE=1` (nothing is downloaded at run time), `PYTHONNOUSERSITE=1` (packages in
    `~/.local` are ignored), `PYTHONDONTWRITEBYTECODE=1`.
- Still read from outside: `WANDB_API_KEY` (optional), conda from `PATH` (`setup_env.sh` only), the NVIDIA driver,
  shared memory (`/dev/shm`) for NCCL and the DataLoaders.

### 2.3 Environment

```bash
env/setup_env.sh                                          # -> env/.env; needs conda on PATH (CONDA=mamba works too)
ENV_PREFIX=/path/to/envs/rubik_memory env/setup_env.sh    # elsewhere (then PYTHON=/path/to/envs/rubik_memory/bin/python ./run.sh ...)
```

| step | what | command |
| --- | --- | --- |
| 1 | Python 3.11 (conda-forge) | `conda create -p env/.env -c conda-forge --override-channels python=3.11 pip` |
| 2 | torch 2.8.0+cu128 | `pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128` |
| 3 | everything else, exact versions | `pip install -r env/requirements.lock.txt` |

- torch 2.8.0+cu128 bundles CUDA 12.8 and has kernels for A100 / H100 / H200 (sm_80 / sm_90) and B200 / RTX PRO 6000
  (sm_100 / sm_120). NVIDIA driver >= 570 recommended (>= 525 works through CUDA minor-version compatibility);
  `TORCH_CUDA=cu126 env/setup_env.sh` for older drivers (no Blackwell).
- No flash-attn (attention runs on torch SDPA), no model repositories (the VAE class comes from diffusers).
- `setup_env.sh` checks every version at the end; `run.sh` uses `env/.env/bin/python` when it exists, else
  `PYTHON=...`.

### 2.4 Download

```bash
./download.sh                                             # -> models/Wan2.1-T2V-1.3B-Diffusers (the default VM_VAE)
VAE_DIR=/path/to/models/Wan2.1-T2V-1.3B-Diffusers ./download.sh   # elsewhere; then export VM_VAE=<that folder>
```

Two files (`vae/config.json`, `vae/diffusion_pytorch_model.safetensors`) at the pinned commit; their sha256 must match
(the script stops otherwise). Its Hugging Face cache and temp files stay in `.cache/download/` (deletable afterwards).

### 2.5 Config

- `config.env`: the two `SET ME` paths `VM_DATA_DIR` and `VM_OUT_DIR` (or export them).
- wandb (optional): `export WANDB_API_KEY=...` before `run.sh` makes every training run log online to project
  `world-auto-visual`; without a key wandb is off. Every logged value is in `<run>/log.jsonl` in both cases.
- `WORKERS` (DataLoader workers per process, default 6): lower it on nodes with fewer than ~64 CPU cores (stage 3 runs
  8 processes).

### 2.6 Check

```bash
./run.sh check
```

About a minute, CPU (it loads the VAE once on the CPU); prints READY when everything is in place. It verifies Python and
every package version against the lock file, the visible GPUs (count, memory >= 40 GB each, kernels for their
architecture), ffmpeg, the VAE files and that they load offline, both locations (set, writable, free space: ~80 GB data,
~120 GB outputs), the dataset's state (before stage 1: "no meta.jsonl yet", a warning) and the wandb key. Fix every FAIL
item before section 3.

## 3. Runs

GPU lists are physical indices as `nvidia-smi` numbers them (`run.sh` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES` per process itself; an outer `CUDA_VISIBLE_DEVICES` is overridden). Only one `run.sh` at a time
per `VM_DATA_DIR` / `VM_OUT_DIR`.

### 3.1 Smoke test (~15 minutes, ~13 GB)

```bash
./run.sh smoke 0            # one GPU is enough; 0,1,2,3,4,5,6,7 also works
rm -rf "$VM_OUT_DIR/smoke"  # afterwards (VM_OUT_DIR as in config.env)
```

Every stage at toy size under `$VM_OUT_DIR/smoke/` (its own data and outputs; the real locations stay untouched): 96
train + 14 test episodes, a base of 20 steps, `rk_ctrl` and `rk_qkv_norm` for 10 steps each, inference of 4 checkpoints
(2 episodes per split, 5 sampling steps), and a dry run of the hand-back (measured at the project: 11 minutes on one
H200 that other jobs shared). Expected:

- `data: render + encode ...`, then `READY: the dataset is complete`;
- `run rubik_base | task rubik | 200m 201.2M | 1 GPU x batch 2 | moves [0, 1, 2, 3, 4, 5, 6, 7, 8] | start step 0` and
  `step 5 | loss 2.3...` (the loss starts at ~2.3); no eval lines;
- in `logs/rk_qkv_norm.log`: `run rk_qkv_norm | MemConfig(proj='qkv', ... key='norm', lr=0.5 ...) | max_ctx 0 | new params
  11.99M | fast weights / sample: writable 2.99M, state 11.94M (46 MB fp32)` and step lines with `dW_q`, `dW_k`, `dW_v`
  growing from ~1e-5;
- `infer: 4 jobs ...`, per job `test/n00: 2 episodes, K 7, free, tf saved` ... `test/n24: 2 episodes, K 31, ...`, then
  `infer: all 4 jobs finished`;
- the hand-back dry run: `gathered 110 files, 4.7 GB -> .../handback/stage/rubik_r1_<time>` (the toy checkpoints and
  latents) and `dry run: nothing uploaded`;
- last line `smoke: PASSED`.

### 3.2 Full run

```bash
tmux new -d -s rubik "./run.sh pipeline 0,1,2,3,4,5,6,7"        # or a single-node 8-GPU job with the same command
```

- `pipeline` = data -> base -> sweep -> infer. A rerun of the same command continues where it stopped: the data stage
  skips every latent already written (stale shard claims are cleared at its start); the base resumes from
  `rubik_base/last.pt` (saved every 500 steps) and is skipped once `ema_040000.pt` exists; every fine-tune resumes from
  its `last.pt` and is skipped once `ckpt_012000.pt` exists; every inference job is skipped once its `done.json`
  exists, and an interrupted job redoes only its unfinished splits.
- The stages also run one by one (the table at the top), e.g. after a failure in one of them.
- The base needs 1, 2, 4 or 8 GPUs (global batch 16). A resume may use another GPU count.
- With fewer than 8 GPUs, stages 3 and 4 run in lanes (GPU i: items i, i + #GPUs, ...).

### 3.3 Monitoring

- Logs: `$VM_OUT_DIR/logs/` (`data_gpu<g><a|b>.log`, `data_sweep.log`, `base_<time>.log`, `rk_<config>.log`,
  `infer_<run>_<checkpoint>.log`); every training run also writes `$VM_OUT_DIR/<run>/log.jsonl` (all numbers) and, with
  a key, wandb.
- Data: `[shard i/128] <n> episodes in <m> min (<r>/s)` per finished shard (~470 episodes); 128 shards in total; at the
  end `READY: the dataset is complete`.
- Base (`base_*.log`): `step N | loss L | gn G | S s/step | M GB | T tok` every 50 steps; the loss falls from ~2.3 within
  the first thousand steps and then slowly; `ema_<step>.pt` every 2500 steps.
- Fine-tunes (`rk_*.log`): `step N | loss | obs .. ret .. | gn | dW_q .. dW_k .. dW_v .. | S s/step | M GB` every 50
  steps (`obs` = the loss on the initial display chunks, `ret` = on the last chunk, whose stickers were all hidden;
  `dW_*` = relative size of the written fast weights; `rk_ctrl` has none). `ckpt_004000.pt`, `ckpt_008000.pt`,
  `ckpt_012000.pt` along the way.
- Inference (`infer_*.log`): one line per split; `done.json` at the end of a job.
- A loss that turns `nan`, or a run that stops with a traceback: rerun the same stage (it resumes); if it recurs, hand
  back what exists (3.4) together with the log.

### 3.4 Outputs and hand-back

- Outputs: `rubik_base/`: `log.jsonl`, `ema_002500.pt` ... `ema_040000.pt` (EMA weights, 0.8 GB each), `last.pt`
  (resume state). `rk_<config>/`: `log.jsonl`, `info.json` (memory size), `ckpt_004000.pt`, `ckpt_008000.pt`,
  `ckpt_012000.pt` (model + EMA, 1.7 GB each), `last.pt`. `inference/<run>/<checkpoint>/` (1.4).
- Hand-back, after `pipeline`:

  ```bash
  HF_TOKEN=hf_... ./run.sh handback                    # ~45 GB -> riselab/rubik_memory (public model repo)
  HF_TOKEN=hf_... UPLOAD_DATA=1 ./run.sh handback      # also the dataset -> riselab/rubik_memory_data (public dataset repo)
  ./run.sh handback --dry-run                          # gather and list only
  ```

  It gathers into `$VM_OUT_DIR/handback/stage/<folder>/` (hard links, no copies): `README.md`, `MANIFEST.json` (every
  file with size and sha256), `logs/`, `rubik_base/` (`log.jsonl`, `ema_020000.pt`, `ema_040000.pt`), each `rk_*/`
  (`log.jsonl`, `info.json`, `ckpt_012000.pt`), `inference/` (every job), `data/` (the dataset's `info.json`, sha256 of
  every `meta.jsonl`, latent counts); then it uploads that folder, resumably (rerun the same command after an
  interruption or an HTTP 429). The folder name is `rubik_r1_<YYYYmmddHHMMSS>` of the first hand-back, kept in
  `$VM_OUT_DIR/handback/FOLDER`.
- The repos must stay public (the project's account limits private storage); they are created public when missing. The
  token needs write access to the `riselab` organization.
- The dataset upload (`UPLOAD_DATA=1`) is optional: the project has the same dataset. It packs every `latents/XX/`
  folder into a tar first (as much free space under `$VM_OUT_DIR/handback/` as the dataset, ~75 GB).
- After the upload, report the folder name. Keep `$VM_OUT_DIR` until the project confirms the hand-back; the `last.pt`
  files can go once the stages are finished.

## 4. Fixed settings

Part of the experiments; keep them:

- Data: seed 0; 60,000 train episodes with N uniform in 0..8; 64 per test N in {0, 2, 4, 8, 12, 16, 24}; 2x2 cube, 3x3
  cells per sticker, 8 colors; 256x192, 16 fps; 4x4 supersampling; Wan2.1 VAE mode latents.
- Base (`rubik_base`): 200M (`--size 200m`); global batch 16 (16 / #GPUs per GPU); NorMuon, learning rate 2e-4, 1000
  warmup steps, constant after; 40,000 steps; EMA 0.999; diffusion forcing: one noise level per chunk (shift 3), with
  probability 0.5 a clean prefix of 1..K-1 chunks (no loss on it); bf16 autocast; gradient checkpointing; full
  chunk-causal attention; EMA checkpoint every 2500 steps; no evaluation.
- Fine-tunes (`rk_*`): from `rubik_base/ema_040000.pt`; `max_ctx 0`; all 12 layers; batch 8 on one GPU; NorMuon 1e-4
  (the write-strength parameters: AdamW 1e-3), 200 warmup steps; 12,000 steps; EMA 0.999; checkpoints every 4000 steps;
  no evaluation; memory defaults: 4 fast heads of 16, 4 diagonal blocks, delta rule, write then read at every chunk;
  `_norm`: unit keys per block, write strength in (0, 2) starting at 0.5; `ffn`: the last 1/4 of the FFN hidden units,
  8 blocks.
- Inference: the 27 checkpoints of 1.4; EMA weights; all 64 episodes of each test split; protocols `free` and `tf`;
  Euler, 50 steps, shift 3; episode noise seeded with the episode id (seed offset 0); batch 16 (the base with full
  history: 16 x 12 / K).

## 5. Reference

### Adjustable settings

- `VM_DATA_DIR`, `VM_OUT_DIR`, `VM_VAE`, `PYTHON`, `WORKERS`, the wandb key;
- the launcher (tmux, Slurm, k8s, ...), keeping `run.sh` in the chain (it sets the cache locations);
- for a node with fewer GPUs: the GPU lists (3.2).

### How the code works

| file | role |
| --- | --- |
| `run.sh` | entry point: locations, caches, stages (`check`, `smoke`, `data`, `base`, `sweep`, `infer`, `pipeline`, `handback`); the fixed run and checkpoint lists of stages 3-4 |
| `config.env` | the two locations |
| `check_setup.py`, `download.sh`, `upload.py` | check (2.6), VAE download (2.4), hand-back (3.4) |
| `code/datagen/rubik.py` | the cube: moves as rotations of cubies, the ray-cast renderer, the episode sampler |
| `code/datagen/build_rubik.py` | the dataset: `--meta` (every split's `meta.jsonl`), `--encode [--claim]` (render + VAE-encode; workers claim shards) |
| `code/datagen/vae_io.py` | latent file paths, VAE loading and encoding, shard claims |
| `code/wm.py` | the model: DiT blocks, per-chunk forward with a state (KV window + per-sample dW), the fast-weight memories (`AttnMem` for q/k/v, `FFNMem`, `DeltaMem`) |
| `code/train_base.py` | stage 2: diffusion forcing, DDP, NorMuon, EMA |
| `code/train.py` | stage 3: the unrolled fine-tune with backpropagation through the memory writes; every option is a `--mem-*` flag |
| `code/infer.py`, `code/sampling.py` | stage 4: the rollouts and the saved latents; the sampler, the rollout and model loading |
| `code/data.py`, `code/paths.py` | the latent dataset and its batching (one episode length per batch); the locations from the environment |
| `code/normuon.py` | the optimizer |

`train.py` and `train_base.py` contain hooks for evaluation modules that are not part of this package; the fixed
settings switch them off (`--eval-every 0`, `--final-eval 0`), so those modules are never imported.

### Troubleshooting

| symptom | action |
| --- | --- |
| `./run.sh check`: versions differ | rebuild the environment with `env/setup_env.sh` (remove `env/.env` first); do not upgrade single packages |
| `./run.sh check`: VAE files missing / load fails | `./download.sh`; with the VAE elsewhere: `export VM_VAE=<folder that contains vae/>` |
| `no kernel image is available for execution on the device` | the GPUs are newer than the torch build: `TORCH_CUDA=cu128 env/setup_env.sh` |
| data stage ends with `NOT READY: the dataset is incomplete` | rerun `./run.sh data <gpus>` (it continues); see `logs/data_gpu*.log` for the error |
| `ModuleNotFoundError: rubik_eval` / `evaluate` | an evaluation was switched on: keep `--eval-every 0 --final-eval 0` (run.sh passes them) |
| CUDA out of memory | another process holds GPU memory (`nvidia-smi`); give the stage GPUs of its own |
| killed / out of host RAM, or DataLoader errors | lower `WORKERS` (e.g. 3) |
| NCCL error or hang at the start of the base | `nvidia-smi topo -m`; `NCCL_DEBUG=INFO ./run.sh base ...` (single node, no network fabric involved) |
| `infer: no checkpoint ...` | the stage before it is not finished: `./run.sh sweep <gpus>` (or `base`) first |
| `OSError: AF_UNIX path too long` | `TMPDIR` is too long: `run.sh` shortens it through `/tmp/rkm-*`; check that `/tmp` is writable |
| hand-back: 401 / 403 | the token lacks write access to `riselab`; HTTP 429 or a broken connection: rerun the same command (it resumes) |
