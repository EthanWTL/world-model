# Auto research log: FFN fast weights (TTT) on the Shell Game world model

## ★ Wave MC (started 2026-11-02 evening): state capacity vs parameter capacity

**⏸ Local GPUs released (2026-11-03 17:22). Since then this task runs on Slurm only.**
The user needs GPUs 6/7 for the paper. First they asked for a record of where every run stood, then said the paper side could kill our runs at any time. Then: "把所有的守护进程的都取消掉 slurm的不用取消 这样我们这个任务就完全不去用本地gpu了".
- **17:22–17:23:** the paper side killed the 8 local runs (k1 s0 on GPU 6, k5 s0 on GPU 7). It started its own encode on 6/7 at ~17:25.
- **17:24:48:** I stopped the feeder `waves/local_queue.py` (pid 36693). Its log shows no launch after 12:20, so nothing restarted on 6/7.
- **17:30 check:** no process of this task is left on udc-an25-1 (ps, nvidia-smi). Users here cannot use crontab. Slurm jobs cannot land on udc-an25-1, because it is outside Slurm.
- **No checkpoint was damaged.** All 23 .pt files of the 8 runs open as zips, have a readable data.pkl with the stored step, and pass a full CRC check. For example, k5 r1152 last.pt has 942/942 members matching, with 2.61 GB actually read.

| run | last logged step | last.pt = resume point | kept ckpts | latest quick eval n05 / n10 (step) | T_learn |
|---|---|---|---|---|---|
| k1 r16 | 15350 | 15000 | 4K / 8K / 12K | .97 / 1.00 (15K) | 12000 |
| k1 r72 | 15350 | 15000 | 4K / 8K / 12K | .28 / .19 (15K) | – |
| k1 r288 | 15050 | 15000 | 4K / 8K / 12K | 1.00 / .97 (15K) | 9000 |
| k1 r1152 | 14150 | 14000 | 4K / 8K / 12K | 1.00 / .88 (14K) | 10000 |
| k5 r16 | 4350 | 4000 | 4K | .03 / .00 (4K) | – |
| k5 r72 | 4300 | 4000 | 4K | .78 / .62 (4K), jumping | – |
| k5 r288 | 4200 | 4000 | 4K | 1.00 / .94 (4K) | 3000 |
| k5 r1152 | 4000 | 3500 | – | .06 / .00 (3K) | – |

- k5 r1152 died inside its step-4000 quick eval, before the 4000 save. So it resumes from 3500 and has no ckpt_004000.
- **Resubmitted to Slurm (17:29):** the same 8 cells, via `MODE=slurm KS="1 5" SEEDS=0 bash waves/mc.sh`.
  - Train jobs: 22424375–22424389, odd IDs. Each eval job is train ID + 1, chained afterok.
  - The IDs are appended to `mc_jobs.txt`.
  - train.py resumes each cell from its last.pt in the same run dir.
- **Now:** 32 train + 32 eval jobs on Slurm, nothing local. The event watcher (scratchpad, reads logs only, no GPU) keeps running for the Slurm results.
- **Local GPUs again only when the user says so,** and after checking nvidia-smi:
  - To start: `nohup python waves/local_queue.py --gpus 6 7 >> /project/sds-rise/ethan/Finetune/world/auto_research/logs/local_queue.log 2>&1 &`
    - Started cells go first, i.e. every cell with a last.pt.
    - It skips cells that are evaluated or running locally / on Slurm, and takes pending cells from Slurm.
    - `--dry` prints the plan only.
  - To stop: `python waves/local_queue.py --stop`
    - It stops the feeder first, because otherwise the feeder refills freed slots within 5 min.
    - Then it stops each run when that run is not writing a .pt, and logs the stop steps to `logs/local_queue.log`.
- **Why resuming is safe:**
  - train.py reloads last.pt: model, EMA, NorMuon state and step. The data are re-seeded with seed + step.
  - A cell already at 20000 leaves the loop at once and goes straight to the eval.
  - `log.jsonl` is appended. `mc.sh` now also appends (`>>`) to `logs/<run>_local.log`; before this change, a restart would have wiped it.

**Question (user):** does the memory layer need more parameter capacity to remember more state? Three sub-questions, all on one grid fine-tuned from ema_200000:
1. **Time:** with the same state count and the same capacity, is it just a matter of training longer? A curve that is still rising means time-limited; a flat curve means capacity-limited or stuck.
2. **Capacity:** with the same state count, does a different capacity change the learning time?
3. **Minimum capacity:** with the same time budget, does the minimum capacity grow with the state count?

**No window anywhere (user):** max_ctx 0 in training and evaluation, so the memory is the only path between chunks. The full-attention multi-ball base stage I first proposed is dropped (`train_base.py` is kept but unused). The ema_200000 weights were trained with full attention, but nothing in this experiment can use a window.

**Data: `Dataset/ShellGameMulti`** (`datagen/gen_multi.py` + `render_multi.html`):
- k balls with fixed colors: ball b is always PALETTE[b] = red, green, blue, yellow, magenta, each under a different cup. k = 1 is exactly the old red ball.
- The scene, actions and chunking are unchanged. Labels: `balls[c][b]` = slot of ball b at the end of chunk c. `ball/start/final` = ball 0, so the old probe still works.
- `k{K}/train`: curriculum N = 0..5, count per N = min(120/1000/3000/3000/3000/3000, unique episodes) → 9.8K / 11.5K / 12.9K / 13.4K / 13.4K episodes for k = 1..5.
- `k{K}/test/n05, n10, n20, n30`: 64 each, disjoint from train. 61K episodes and 8.0M frames in total (seed 2).

**Detector (`detect.ball_slots`)**: HSV hue windows per color, red [330,20], green [95,160], blue [190,245], yellow [40,70], magenta [265,320], each with S/V minimums.
- The table sits at hue 31–34 with S ≈ .40 (shadows up to ~.55), only ~15° from yellow, so yellow needs S > .65.
- Calibration on rendered frames and on Wan VAE round-trip frames (encode/decode on CPU): every ball correct, 130–180 px in its slot vs ≤1 px anywhere else.
- On old single-ball GT frames (test n05/n10/n30), `ball_slots(f, 1)` agrees with the paper's red detector on 384/384.

**Metrics:**
- `acc` = all k balls in the right slot (chance 1/5, 1/20, 1/60, 1/120, 1/120).
- `ball` = per-ball accuracy.
- `tf_p` = each ball's color mass at its true slot in the σ=1 one-step x0, averaged over balls.

**Grid (`waves/mc.sh`, runs `mbc_k{k}_r{r}_s{seed}`, code frozen in `snapshots/mc`):**

| axis | values |
|---|---|
| state | k = 1 / 2 / 3 / 5 balls → 5 / 20 / 60 / 120 states (2.3 / 4.3 / 5.9 / 6.9 bits); k = 4 also has 120 states, so it is skipped |
| parameter capacity | r = 16 / 72 / 288 / 1152 fast q rows per layer, 4 blocks → 4.6K / 20.7K / 82.9K / 331.8K numbers per layer (12 layers: 55K / 0.25M / 1.0M / 4.0M per sample); r = 288 is the wh_q recipe |
| time | quick eval every 1000 steps (k{K}/curr3, n05, n10 tf, 32 episodes, raw weights), up to 20000 steps |

- Everything else is wh_q: delta rule on q, all 12 layers, curriculum N = 0..5, NorMuon 1e-4, B = 8, EMA .999.
- Seeds 0 and 1. In a1, the jump timing turned out to be seed-random.
- When a run finishes, its 20000-step EMA gets the full eval: N5/10/20/30, tf + free, 64 episodes, 50 steps.

**Readouts (`python summary.py --cap mbc`):**
- T_learn = the first quick-eval step where n05 tf all-correct ≥ .9. Runs that never reach it are censored at 20000.
- Minimum r per budget (4K / 8K / 12K / 16K / 20K), for each k.
- 20000-step EMA: all-correct and per-ball at N5/10/20/30. This is the "more swaps" axis.
- Stage 2: cells not learned by 20000 continue to 40000, to separate "needs more time" from "cannot".

**Caveats:**
- r = 1152 makes all 16 heads fast and leaves no fallback heads, so it changes structure as well as size. Every r ≤ 288 keeps ≥ 12 fallback heads.
- For k ≥ 2 the model also has to learn to render the new colors, because the base only knew red. In this design that cost is part of "more state". If k = 2..5 lag k = 1 by the same offset at every r, that offset is the color cost, not capacity.
- Storage is nowhere near binding, even at r = 16 (55K numbers vs 6.9 bits). A minimum r that grows with k would therefore be a learnability or precision effect, not a storage one.

**Predictions (before results):**
- r ≥ 72 learns every k eventually, and T_learn grows with k.
- r = 16 is uncertain. It may fail for every k, k = 1 included (in a1 one seed of 72 rows had still not learned at 10K); then the minimum r lies between 16 and 72 for all k, and the grid cannot resolve a k effect at that resolution.
- Null for accuracy vs k (no capacity limit): per-ball accuracy stays at the k = 1 level and all-correct ≈ acc₁(N)^k. A swap moves 2k/5 balls on average (0.4 for k = 1, 2.0 for k = 5).

**Status (2026-11-03 ~00:40):**
- **Data:** all 60955 episodes rendered and encoded.
  - Render: Slurm CPU arrays (one 32-core task + 76 small tasks + reruns).
  - Encode: `encode_latents.py --follow` encodes while rendering. It uses per-episode lock files, so local and Slurm workers can share the work. Workers: local H200 × 2 at ~2.9 episodes/s each, Slurm A6000 × 8 at ~0.7 each.
- **Runs:**
  - k1 s0 (4 cells) run locally on GPU 6. GPU 1 was left alone because the user's `smoke_qv` test was running there.
  - The other 28 runs are on Slurm; IDs are in `Finetune/world/auto_research/mc_jobs.txt`. Each run chains the full EMA eval.
- **Step 1000, k1 s0, all r:** at chance (n05 .12–.16, p .20), i.e. still on the pre-jump plateau.

**Interim 2026-11-03 12:15 (k1 seed 0, local GPU 6, step ~11000; quick eval, raw weights, 32 episodes):**

| r (per-layer numbers) | n05 all-correct by step (1K, 2K, …) | first ≥ .9 | latest n05 / n10 |
|---|---|---|---|
| 16 (4.6K) | flat ≈ .2 until 9K, then .59 (10K), .75 (11K); p .47 → .73 → .85 | not yet (jumping now) | .75 / .59 |
| 72 (20.7K) | ≈ .2 through 11K; p .18–.28 | no | .22 / .19 |
| 288 (82.9K, the recipe) | jump at 5K (.59), .91 at 9K, .97 at 10K | 9000 | .97 / .88 |
| 1152 (331.8K, no fallback) | jump at 5–7K, .91 at 10K | 10000 | .91 / .59 |

- **Even r = 16 learns k = 1**: 16 fast q channels of one head, 4.6K numbers per layer. So for 5 states, the minimum capacity is ≤ 16 rows.
- **Learning time is not monotonic in r** (16 jumped before 72). With one seed this is jump-time noise, as in a1. The seed-1 runs are needed before T_learn vs r can be read.

**Scheduling change (12:15):** none of the 28 Slurm runs started in 12 h (all Priority; the user's a2 jobs hold the allocation). k5 s0 (4 cells) was moved to local GPU 7, the second allowed local GPU, and its Slurm copies were cancelled. k1 vs k5 at seed 0 will therefore come first.
- The user confirmed: at most 2 local GPUs for this task. `waves/local_queue.py` (nohup, log `logs/local_queue.log`) feeds GPUs 6/7 in order k2 s0, k3 s0, k1 s1, k5 s1, k2 s1, k3 s1.
  - Each GPU has 4 slots for our runs. When one of ours finishes and the GPU has ≥ 24 GB free, it starts the next cell. Other users' jobs stacked onto 6/7 do not make us yield (user: "别人叠上来的 就不要给他们让开").
  - Per cell, it cancels the pending Slurm copy, or skips the cell if Slurm is already running or has finished it.

**Interim 2026-11-03 16:00 (seed 0; quick eval, raw weights, 32 episodes; k1 at step 14000, k5 at step 3000):**

| r | k1 (5 states): first ≥ .9 | k1 latest n05 / n10 | k5 (120 states): n05 all-correct at 1K / 2K / 3K (per-ball) | k5: first ≥ .9 |
|---|---|---|---|---|
| 16 | 12000 | .97 / .97 (14K) | .03 / .03 / .00 (.21 / .20 / .19) | not yet |
| 72 | no; still ≈ .22 at 14K (curr3 .53) | .22 / .19 | .03 / .03 / .03 (.21 / .25 / .24) | not yet |
| 288 | 9000 | 1.00 / .94 (13K) | .00 / .09 / **.94** (.18 / .39 / .98) | **3000** |
| 1152 | 10000 | 1.00 / .84 (13K) | .00 / .03 / – (.17 / .25 / –) | not yet |

- **k5 r288 learned at step 3000, 3× sooner than k1 r288 (9000).** At 3000 it has curr3 / n05 / n10 all-correct .97 / .94 / .78, per-ball .99 / .98 / .91, tf_p 1.00 / .99 / .97.
  - The finished-video detector and the one-step x0 color mass agree, so this is not a detector artifact.
  - n10 < n05 (errors pile up with length), so it is memory tracking, not a leak through the swap frames. A leak would be flat in N and would also lift r16 / r72, which are still at chance.
  - The jump happened between 2K and 3K. At 2K, curr3 was already .16 with p .73.
- **This contradicts the prediction "T_learn grows with k"** for r = 288 at seed 0. Three readings, not yet separable:
  1. **Denser supervision with more balls.** Every k5 reveal grades 5 balls, and every swap moves 2 balls (0.4 on average for k1, where 60% of swaps are no-ops for the ball). So each episode carries more gradient about "apply this swap to the state".
  2. **Seed-random jump time.** One seed per cell so far.
  3. **The base prior.** k1 is the base's own task, which was learned with a window. The memory may first have to stop relying on the missing window, while k5 builds a new path anyway.
- **The color cost was small.** At step 0, k5 per-ball is .03, because the base draws no new colors. By 1K it is .17–.21, the level of drawing all 5 colors in random slots. So learning the colors took < 1000 steps, not a large offset.
- **Question 3 (minimum r vs k) is still open.** k1 learned at r = 16, so if k5 r16 also learns, 120 states need no more rows than 5 states in this range.

**Pitfalls hit while launching:**
- **19. `sh -c "..."` + a multi-line argument string.** The newline ends the command, so every flag after the line break was silently dropped. The first local launch ran heads=1, fast_frac=1.0, 6000 steps for all four cells. It was caught from the printed TTTConfig and killed after ~1 min. Fix in `waves/mc.sh`: `T=$(echo $T)` before `sh -c`. Slurm mode was not affected, because `sbatch ... $T` word-splits.
- **20. Packing several runs onto one H200 does not raise throughput.** 4 q-memory runs together take 4.2 s/step each, about the same total as 1 run alone (~1 s/step). One run already saturates the GPU (~1.5e14 FLOP per step). The whole grid (32 × 20K steps) is about 180 H200-hours.
- **21. Small render tasks (8 cores, 4 Chrome workers) are ~4× slower per worker than 32-core tasks**, and 30 of 76 hit the 1 h limit. Reruns skip finished videos, so nothing was lost.

## ★ Update 2026-11-01 (read first): memory in attention q beats the FFN

Config `wh_q` / `wi_q_s1`:
- Delta rule on the **q projection only**: `--ttt-params none --ttt-attn q`, so the FFN is not updated.
- `--ttt-fast-frac 0.25 --ttt-heads 4`: the fast part is the q channels of the last 4 attention heads, block-diagonal over those 4 heads.
- All 12 layers, max_ctx 0, curriculum, 12000 steps, EMA.

| | N5 | N10 | N20 | N30 |
|---|---|---|---|---|
| **q only, seed 0 / seed 1 (free)** | .98 / 1.00 | 1.00 / 1.00 | .97 / .94 | **.91 / .95** |
| k only, seed 0 (free) | .97 | 1.00 | .97 | .88 |
| FFN best below (6000 steps, 3-seed mean, free) | 1.00 | .99 | .88 | .71 |
| paper's LaCT8 (free) | 1.00 | 1.00 | .98 | .69 |

Caveats and rules learned (details in Wave H / H2 / I):
- The attention runs had 12000 steps vs 6000 for the FFN. The equal-budget FFN run (wi_ffn_12k) is running.
- **Never make v fast** (pitfall 18): x0 error blows up 30–1000× beyond the training length.
- **dot and e2e learn nothing**, in every setting tried.
- Attention runs jump late, around step 5000–6000 (pitfall 17), so evaluate the EMA only after about 12k steps.
- A learned write key (`--ttt-key proj`) does not help.

## ★ Previous conclusion (2026-10-31): FFN memory

**Native-FFN TTT can make the world model track the hidden state well beyond the training length.**

Best config `wd_mse_down_c0_f025_h8`:
- **Write rule:** delta rule (mse on the fast part), **updating only down**, all 12 layers.
- **Fallback + heads:** only 25% of FFN hidden units fast, the other 75% a frozen fallback; 8 heads on the fast part.
- **Context:** **no cross-chunk attention at all** (max_ctx=0).
- **Training:** curriculum N~U{0..5}, fine-tuned from ema_160000 for 6000 steps (B=8), evaluated with the **EMA weights**.

64 episodes, 50 steps:

| | N5 | N10 | N20 | N30 |
|---|---|---|---|---|
| **free (paper protocol: 2 GT chunks, then generate everything)** | **1.00** | **1.00** | **.88** | **.78** |
| tf (only the last chunk generated) | 1.00 | .98 | .94 | .72 |
| same run, raw weights (no EMA), free | .67 | .55 | .25 | .17 |
| base model SWA7 (no fine-tuning), free | 1.00 | .20 | .22 | .22 |
| paper's LaCT8 (dedicated fast-weight branch, from scratch, 200K steps) | 1.00 | 1.00 | .98 | .69 |

The key ingredients, each established by a comparison:
1. **Delta rule, not dot.** dot never extrapolates (Analysis A: dot down has no self-path).
2. **Curriculum.** With only N=5, nothing learns, not even full attention.
3. **Train without cross-chunk attention.** Under SWA1 the model takes the KV-relay shortcut and never learns memory (pitfall 15).
4. **Fallback + heads.**
   - All units fast (wc_mse_down_c0): N10 only .28–.50, N20 at chance.
   - f0.25+h8: N10 1.00, N20 .88.
5. **EMA.** The raw weights swing a lot (pitfall 14).
6. **Layers: update all layers.** In the world model, all layers > last 4 > first 4 (free N20 .88/.64/.22). This is the opposite of the toy's "early layers are better", so don't carry that over.
7. **★ Final recipe: first train memory with no cross-chunk attention, then switch on SWA1 and fine-tune 3000 steps.**
   - free N5/10/20/30, 3-seed mean = **1.00/.99/.89/.72** (seed 0 is 1.00/1.00/.94/.89).
   - Compared with memory alone (3-seed mean 1.00/.99/.88/.71): **on par**. Its value is keeping attention without losing the memory.
   - Against the paper's LaCT8 (1.00/1.00/.98/.69): comparable at N30, lower at N20. Starting directly from base with SWA1 (even with dropout) is much worse (.66/.39/.20/.23). Once the memory is learned, adding attention does not switch the model to the shortcut. See the "Thorough eval" section at the end.


The goal is to find out early which designs work and where the pitfalls are, so the hand-written version can skip them.
Everything lives in this folder. Outputs go to `/project/sds-rise/ethan/Finetune/world/auto_research/<run>/` (`log.jsonl`, `last.pt`, `ckpt_*.pt`, `eval/`); logs go to `.../auto_research/logs/`; wandb project is `world-auto`.

## Setup (shared by every run)

- **Base model:** `shellgame_normuon/ema_160000.pt` (full causal attention over 8 chunks; n05 is 1.00, beyond the window it is at chance).
- **Attention window:** max_ctx=1 (SWA1): each chunk sees itself plus the previous chunk's KV. The ball is visible only in chunks 0–1, so from the second swap chunk on, the ball position must come from memory.
- **Training:** teacher forcing, unrolled exactly like inference. For c=0..7: denoise noisy chunk c (reads the state, loss for c≥1), then commit clean chunk c (σ=0 forward that writes the KV window plus the fast weights). The outer loss backpropagates through all commits (BPTT). Each block uses gradient checkpointing.
- **Optimizer:** NorMuon lr 1e-4 (warmup 200) + AdamW; the inner-lr parameters get their own AdamW group (`--ttt-lr-lr`). B=8, 6000 steps, EMA 0.999.
- **Evaluation:** `evaluate.py`, using the Wan2.1 VAE to decode only the last chunk.
  - `tf`: GT for every chunk except the last; generate the last chunk (20 steps during training, 50 at the end), then `ball_slot`.
  - `free`: the paper protocol (2 GT chunks, then generate everything).
  - `tf_p`: the x0 predicted in one step at σ=1 (pure noise). Its red mass at the 5 cup slots gives the share at the correct slot. 0.2 = guessing, 1 = certain; this measures only what memory/context knows.
  - `tf_mse1`: latent MSE of that prediction.

### TTT variants (`wm.py`)

FFN = up W1 (1152→4608) → GELU → down W2 (4608→1152). The fast weights selected in a TTT layer are per sample; `None` = still the slow weight.

| Flag | Options |
|---|---|
| `--ttt-layers` | all / last3 / first3 / any list |
| `--ttt-params` | up / down / both |
| `--ttt-loss` | **dot**: ℓ=-(1/T)Σ η_t⟨f(u_t), v_t⟩, v=Wv·u (Wv zero-init) — LaCT / In-Place<br>**mse**: ℓ=½(1/T)Σ η_t‖(f−f_slow)(u_t) − v_t‖² — delta rule, only the fast part<br>**e2e**: inner loss = this chunk's own diffusion loss (backprop through the whole network) — TTT-E2E |
| `--ttt-value` | self: v from the same token; next: v from the same position in the next latent frame (In-Place's NTP-aligned target) |
| `--ttt-order` | wr: write then read (LaCT); rw: read then write (In-Place) |
| `--ttt-norm` | after each update, restore each row's norm to the slow weight's (LaCT) |
| inner lr | dot/mse: per-token η=softplus(W·u+b), starting at `--ttt-lr`; e2e: one learnable scalar per layer per matrix |

`check.py` checks three things:
1. Per-chunk forward == the reference model.py forward over all chunks at once (4.5e-6, fp32).
2. Zero-init TTT == no TTT (exactly 0).
3. Hand-written inner gradient == autograd (3e-7).

## Pitfalls hit so far

1. **bf16 + MSE (delta rule): the residual is all noise.**
   - The writer needs y_fast − y_slow. Computing it as the difference of two bf16 matmuls leaves rounding noise of ~4e-3 relative, and the writer stores that noise in the weights.
   - Symptom: after 5 steps, dW was already 3e-3 (dot was 2e-7), and at n10, mse1 rose from 0.11 to 0.42.
   - Fix: compute it incrementally: z = z0 + ΔW1·u, and y−y_slow = W2(h−h0) + ΔW2·h0.
2. **e2e inner lr scale.**
   - At init, the gradient of the native FFN wrt the per-sample diffusion loss is tiny (‖G‖/‖W‖≈8e-4). At lr 1e-3 the relative update per commit is only 8e-7.
   - The lr sits inside a softplus, so an AdamW at 1e-4 would need tens of thousands of steps to raise it.
   - Fix: init the lr at ≈10 (≈1% relative update per commit), and give the lr parameters a separate AdamW group with lr 1e-2.
3. **e2e second-order gradient passes through the KV cache.**
   - The previous chunk's K/V at layers above the first TTT layer were computed by the post-update σ=0 commit pass with the same fast weights. So the inner gradient flows back into the previous chunk's flash attention, which has no double backward.
   - Fix: in the inner forward, treat the TTT layers' window K/V as constants, so the inner gradient flows only through the current chunk. This is also the correct semantics: at eval the cache has no graph, so detaching keeps train and eval consistent.
4. **Memory.**
   - Without checkpointing, B=4 already exceeds 135 GB: every sample's fast weights plus every pass's intermediates.
   - With per-block checkpointing, dot/all layers at B=8 = 44 GB, 3.5 s/step (H200).
   - e2e/last3 at B=8 = 120 GB (the double-backward graph of math attention), so it uses `--accum 2` (62 GB).
5. **e2e eval:** checkpointing the inner forward → recompute mismatch. Inner blocks don't checkpoint.
6. **Red-pixel metric:** the table's r−max(g,b) is about 0.10–0.15 and the ball's is >0.25, so the soft-metric threshold is 0.2. Decoding only the last chunk gives ball_slot 100% correct on GT.
7. **(Hypothesis, not yet verified) The SWA1 KV cache can carry state on its own.**
   - Chunk c's K/V at layer l comes from layer l's input, which already attended to chunk c−1's cache at layer l−1. Each hop forward in chunks costs one layer, so 12 layers can relay at most ~11 chunks.
   - So at N=5, the no-TTT SWA1 control could in principle still learn to track; n10/n20 are the real memory test.
8. **Implications for the hand-written plan (Rollout writes in place + BPTT).**
   - BPTT through 8 chunks requires per-block checkpointing.
   - Checkpointing requires that the state (KV + fast weights) is input and output of a pure function. If `cache['k'] = ...` mutates in place inside a block, the recompute during checkpointing writes a second time.
   - So either make the state functional (`wm.py` does this), or skip checkpointing and use a small batch, which costs a lot of memory.
9. **The base model under SWA1 (no fine-tuning) is at chance at n05: tf 0.22 / p 0.20.**
   - It relies on attention reaching all the way back to the reveal chunk; nothing it has learned relays the state from chunk to chunk.
   - So every run has to learn memory from scratch.
10. **Toy (`toy.py`): if training uses only fixed N=5, nothing can learn it, not even full attention.**
    - Setup: 5 tokens per chunk; 4 layers of full attention, or SWA1 + any kind of TTT.
    - With N fixed at 5 for 8000 steps, the loss stays at log5=1.609, flat the whole time.
    - With a curriculum (each step samples N~U{0..5}), the same full-attention model reaches N1 0.79 / N3 0.43 after just 1000 steps.
    - This matches the base model's ~80K-step plateau: when every episode needs the full 5-swap composition before any reward, the gradient is almost entirely noise.
    - **For the world model:** if TTT memory has to be learned from scratch on N=5 data, expect a long plateau. A curriculum would need newly rendered episodes with N=0..4 (generate.py supports any N).
11. **e2e's inner lr is being pushed down in the first few hundred steps** (9.83→9.63, dW_up 1.2e-3→3e-4).
    - Before the memory is useful, every inner update is noise to the outer loss.
    - The meta-gradient first learns to "write less", so TTT can degenerate into no TTT. Keep watching.
12. **Slicing bug when trimming the KV window: `l[len(l) - max_ctx:]`.**
    - When len(l) < max_ctx this is a negative index, so it keeps only the last few chunks. With max_ctx=7 the window was stuck at 3.
    - Symptom: the base model's SWA7 eval came out 0.52 (it should be 1.0). The reference rollout on the same episodes gave 1.0 while mine gave 0.19.
    - check.py [1] originally used max_ctx=None and missed it. It now uses 7.
    - Every training run uses max_ctx 0/1, which this bug does not affect (verified case by case). Fix: `l[max(len(l)-max_ctx, 0):]`.
    - **The user's rollout.py uses `c["k"][-self.max_ctx:]`, which is correct.**
13. **(Hypothesis, not yet verified) bf16 reads quantize small fast-weight increments.**
14. **With a constant learning rate, the raw weights swing a lot in extrapolation ability.**
    - In `wd_mse_down_c0_f025_h8`, from step 3000 to 6000:
      - n10: .69/.59/.56/.97/.62/.81/.59
      - n20: .31/.38/.31/.53/.44/.75/.34
    - Part of this is noise from 32 eval episodes (±0.08); part is real drift in the weights.
    - Training-length accuracy (n05) is stable, but how far it extrapolates depends on which step's weights you take.
    - **For the hand-written line: evaluate with EMA, or decay the lr at the end, and use ≥64 eval episodes.** Don't judge from a single eval at a single step.
   - The read casts W_slow+ΔW to bf16. The bf16 ulp of |w|≈0.02 is about 1.2e-4, so element-wise increments smaller than that are mostly rounded away (an unbiased dither, but noisy).
   - The forward barely sees the small updates early in training, while the gradient still passes straight through to Wv.

## Analysis A: which updates can track a state that keeps changing (predictions, before any results)

The ball position has to be updated recurrently at every chunk: p_c = swap_c(p_{c−1}).

1. **Within one layer, the write never sees that layer's own read.**
   - The write's k/v come from the FFN input u_l. That is upstream of this layer's FFN, so it contains neither "this layer's read before the write" nor "after the write".
   - The only way the old fast weights can enter the update is through the nonlinear dependence of the inner gradient on the old W, which I call a **self-path**.
2. **Paths across layers and chunks climb.**
   - p_{c−1} first appears in commit c's residual stream at layers above the lowest layer that holds it, so the new state can only be written into higher layers.
   - The KV relay is the same: from chunk c−1 at layer l−1 to chunk c at layer l.
   - Hence the lowest layer that holds the state rises by at least 1 per chunk, so it survives at most ~12 chunks.
3. **Composing swaps after the fact is not enough.**
   - Tracking one ball under transpositions is the S5 word problem (NC1-complete). Storing only the record of each swap and composing them in one fixed-depth read cannot work at long N.
   - So extrapolating to N ≫ 12 requires an update with a self-path.
4. **Which variants have a self-path:**

| Update | Self-path | Prediction |
|---|---|---|
| dot + down only: ΔW2 = η·v·hᵀ, h = gelu(W1_slow·u) | **None** (pure linear-attention store; the same as In-Place TTT) | Can fit n05 at best, should fail at n20 |
| dot + up: ΔW1 = η·(W2ᵀv ⊙ gelu′(W1_old·u))·uᵀ | Gated by gelu′(W1_old·u) | Possible |
| dot + both | Above, plus W2_oldᵀv (a bilinear recurrence across matrices) | Richer |
| mse (delta rule) | The error subtracts the old fast output, so it can **overwrite** | In principle best suited |
| e2e | The diffusion loss's gradient depends on the old W in complicated ways | Possible |
| ctrl (no TTT) | KV relay only | At most n05/n10, n20 should fail |

This lines up with the paper (LaCT ablations):
- "W1 only" (= updating only the readout) gives N=5 1.00 but N≥10 at chance.
- "freeze W1" (update only the input side) works.

## Wave 1 (submitted 2026-10-30 23:xx, 6000 steps each)

| run | what's different | where |
|---|---|---|
| ctrl | no TTT (SWA1 control) | slurm |
| dot_both_all | dot, up+down, all 12 layers (main config) | local GPU0 |
| dot_up_all / dot_down_all | update only up / only down | slurm |
| dot_both_last3 / dot_both_first3 | layer placement | slurm |
| mse_both_all | delta rule | slurm |
| dot_both_all_next | In-Place's next-frame value | slurm |
| dot_both_all_rw | read then write | slurm |
| dot_both_all_nonorm | no row-norm restore | slurm |
| dot_both_all_lw6 | last chunk's loss weight 6 (≈50%) | slurm |
| e2e_both_last3 | TTT-E2E, lr init 10, accum 2 | local GPU4 |

Base references (no fine-tuning): `base_eval/ema_160000_ctx{7,1}.json` (64 episodes, 50 steps, after the window bug fix).

| Base | n05 tf/free | n10 | n20 | n30 |
|---|---|---|---|---|
| SWA7 (its training setting) | **1.00/1.00** | .20/.20 | .22/.22 | .19/.22 |
| SWA1 | .27/.27 | .17/.14 | .23/.22 | .23/.23 |

As expected: full attention tracks perfectly at the training length and falls to chance once the reveal leaves the window (N≥10).

## Results

### Toy (`toy.py`, 4 layers, dim 128, curriculum N~U{0..5}, 4000 steps, B=128; 2048 test episodes each)

**Columns:**
- w1 = SWA1 (sees the previous chunk); none = no cross-chunk attention at all (everything goes through the fast weights); full = full attention.
- Every TTT variant updates all layers (FFN 128→512→128 with GELU), lr0 1e-2, renorm.

| Config | N1 | N3 | N5 | N10 | N20 | N30 |
|---|---|---|---|---|---|---|
| full, no TTT | 1.00 | 0.76 | 0.46 | 0.22 | 0.21 | 0.20 |
| w1, no TTT | 0.62 | 0.19 | 0.20 | 0.21 | 0.20 | 0.20 |
| w1 dot_down | 1.00 | 0.99 | 0.66 | 0.22 | 0.18 | 0.18 |
| w1 dot_up | 1.00 | 0.28 | 0.22 | 0.19 | 0.19 | 0.20 |
| w1 dot_both | 1.00 | 0.98 | 0.49 | 0.20 | 0.19 | 0.20 |
| **w1 mse_both (delta rule)** | **1.00** | **1.00** | **1.00** | **1.00** | **0.93** | **0.75** |
| none, no TTT | 0.19 | 0.19 | 0.21 | 0.20 | 0.21 | 0.20 |
| none dot_down | 1.00 | 0.97 | 0.61 | 0.21 | 0.19 | 0.21 |
| none dot_up | 1.00 | 1.00 | 0.97 | 0.80 | 0.44 | 0.29 |
| none dot_both | 1.00 | 1.00 | 0.98 | 0.43 | 0.20 | 0.19 |

**Full table (the "read" variant: order=rw, v = Wv·u + Wr·y_old, i.e. what is written depends on what this layer read with its old weights; h4 = block-diagonal update with 4 heads).** Cells are N10 / N20 / N30.

| | dot_down | dot_up | dot_both | dot_both_h4 | **mse_down** | mse_both | read_down | read_both |
|---|---|---|---|---|---|---|---|---|
| SWA1 | .22/.18/.18 | .19/.19/.20 | .20/.19/.20 | .23/.21/.20 (N5 .83) | **1.00/.98/.89** | 1.00/.95/.78 (seed 0: .93/.75) | 1.00/.84/.32 | .91/.35/.21 |
| none | .21/.19/.21 | .80/.44/.29 | .43/.20/.19 | – | **1.00/1.00/.98** | ✗ (N1 .61, single seed) | 1.00/.98/.76 | 1.00/.89/.53 |

**Overall conclusion: whether an update rule extrapolates depends on whether it has a self-path** (Analysis A confirmed).
- Updates that extrapolate:
  - The delta rule: its error = the old memory's output − v, so it can overwrite.
  - read: what is written explicitly depends on what this layer read from its old memory.
- The purely additive dot rule doesn't extrapolate. With dot_up, the gelu′ gating gives it a little.
- Updating only down beats updating both (both is harder to optimize, and none + mse_both did not learn at all).
- Multiple heads (h4) help dot at the training length (N5 0.49→0.83) but do not give extrapolation.

Earlier per-config conclusions:
1. **The native FFN fast weights can carry and update the state on their own.** With no context at all, dot_both/dot_up reach 0.97–0.98 at N5. The relay limit for 4 layers is ~4 chunks, so a self-path must be at work.
2. **Updating only down (dot) never extrapolates** (N10 at chance in both settings), as Analysis A predicts: ΔW2=η·v·hᵀ has no dependence on the old W2, so it is a pure accumulator (= In-Place TTT / the paper's "W1 only").
3. **The delta rule (mse, error = fast part's output − v) is by far the best.** It is the only one that stays ≥0.75 out to N30.
   - Reason: its update subtracts the old memory's output, so it can **overwrite** "where the ball is". The dot rule can only add, and the old position lingers.
   - This is the opposite of the paper's "MSE fails". The paper's MSE pulled the whole fast network toward v. Ours is a delta rule on "the fast part's increment relative to the slow FFN", and it is zero at init.
4. **dot extrapolates better with up only than with both** (none setting: N10 0.80 vs 0.43).
5. **With dot, SWA1 does worse than no context.** Attention provides a short-range shortcut that drags the solution toward the relay. With mse, SWA1 is perfectly fine.

**Multiple seeds:**
- w1 mse_both is stable across 3 seeds: N20 0.93/0.95/0.96, N30 0.75/0.78/0.78.
- none mse_both fails across 3 seeds: N1 0.61/0.20/0.56. This is a real failure, not seed noise.
- L12 (12 layers): ctrl can't learn even N1 in 4000 steps (deeper trains slower). dot_down N5 0.50, no extrapolation.

### New axis (user, 2026-10-31): keep some slots as a fallback + split the rest into heads

`--ttt-fast-frac f`:
- Only the last f·F of the FFN hidden units ("slots") are fast. The first (1−f)·F always use the slow weights (the **fallback**: no matter what TTT writes, that part of the base model's function is unaffected; this is the TTT-E2E idea of a "static MLP", but carved out of the native FFN).
- The state stores only the fast sub-matrices W1[fast rows] / W2[:, fast columns], so per-sample memory and compute shrink by a factor of f.
- renorm restores only the norm of each row's fast segment. The slow segment is never touched.

`--ttt-heads H` applies only within the fast part:
- The update keeps only the block diagonal of input dims D/H × fast hidden units Ff/H. W2 likewise: fast hidden units Ff/H × output dims D/H.

In the world model, f=0.5, H=8 gives fast blocks of exactly input 144 × hidden 288 per head. That is the same shape as LaCT8 (d=144, r=2), but carved out of the native FFN.

`check.py [4]`: with partial fast units + heads, the hand-written gradient matches autograd at 3e-7.

Toy results (SWA1, 4 layers; columns are N10 / N20 / N30):

| mse_down | Result |
|---|---|
| f=1 (all fast) | 1.00/.98/.89 |
| f=0.5 | 1.00/.97/.89 |
| **f=0.25 (75% fallback)** | **1.00/1.00/.99** |
| f=1, 4 heads | 1.00/.99/.95 |

- dot_both f=0.25 + 4 heads: N5 .79, N10 .24. The fallback and heads do not rescue dot extrapolation.
- More dot seeds:
  - w1 dot_down N5 .66/.73/.74, N10 at chance in all 3.
  - w1 dot_both: N5 .49/.54/1.00, one seed reaching N10 .92, but N20 at chance in all.
  - none dot_up varies a lot across seeds: N30 .29/.45/.77.
- **Conclusion: the fallback (only a small fraction of units fast) and the heads both make the delta rule extrapolate better.**
  - Fewer fast units means the base function is disturbed less, and the memory is more "dedicated".
  - With dot, neither helps.

More results (columns are N10 / N20 / N30):

| Setting | Config | Result |
|---|---|---|
| w1 | mse_down f0.5+h4 | 1.00/.98/.93 |
| w1 | mse_down f0.25+h4 | 1.00/.98/.93 |
| w1 | mse_both f0.5 | ✗ (N1 .86) |
| w1 | mse_both h4 | .93/.65/.39 |
| none | **mse_both f0.5** | **1.00/.89/.59** (with every unit fast it failed in all 3 seeds; **the fallback rescued it**) |
| none | mse_both f0.5+h4 | 1.00/.94/.68 |
| none | mse_down f0.5+h4 | 1.00/.99/.94 |
| none | mse_up | ✗ (N1 .54) |

- **mse_down is stable in every combination** (f / h / context), N30 ≥ 0.89.
- **mse_both is fragile:** it succeeds or fails depending on the setting, and adding a fallback can either rescue it or break it. The delta rule over two matrices together is hard to optimize.
- **mse_up doesn't learn.**

**Layer combinations (w1 mse_down, 4 layers in total; columns are N10 / N20 / N30):**

| Which layers are TTT | Result |
|---|---|
| all 4 layers | 1.00/.98/.89 |
| **first 2 layers** | **1.00/.99/.98** |
| first layer only | .99/.86/.77 |
| last 2 layers | ✗ (N1 .28) |
| last layer only | ✗ (N1 .20) |

- **Memory works in the early layers and completely fails in the last layers.** This is the opposite of TTT-E2E's "update the last 1/4 of layers".
- For state tracking, the old state has to be read early, so that the following layers can combine it with the swap and compute the new state. With memory in the last layer, no layers are left to compute with.
- **For the hand-written line: put the TTT layers at the front, or update every layer; don't copy TTT-E2E's last few layers.**

**toy6 (robustness + combinations; w1 mse_down unless noted; columns are N10 / N20 / N30):**

| Config | Result |
|---|---|
| f0.25, 3 seeds | N30 .99/1.00/.93 (N20 all 1.00) |
| first2, 3 seeds | N30 .98/.99/.84 |
| f0.125 | 1.00/1.00/.98 (even 1/8 of units fast is enough) |
| first2+f0.25 | 1.00/1.00/.98 |
| first2+f0.25+h4 | 1.00/1.00/.98 |
| only1 (layer 1 alone) | ✗. Of the single-layer runs, only layer 0 learns (.99/.86/.77) |
| **dot_both first2** | .95/.60/.31. With all layers it was .20/.19/.20, so **putting dot in the early layers also helps a lot** |
| **12-layer toy**: mse_down all layers | 1.00/1.00/.97 |
| **12-layer toy**: mse_down first4 | 1.00/.99/.96 |

In the 12-layer toy, ctrl and dot_down can't even learn N1–N5, but mse_down works.
- World-model check: `wd_mse_down_c1_f025_first4` vs `_last4`.

→ World-model wave D (`waves/wd.sh`):
- mse_down + SWA1 with f0.25, f0.25+h8, f1+h8.
- mse_down + no context with f0.25+h8.

The toy screen (toy_configs5a/b) is running:
- Fallback ratio × heads × {mse_down, mse_both, dot_both}.
- Layer combinations (first/last half, a single layer).
- mse_up.

## World model, wave C (curriculum) progress

### ★ Breakthrough (2026-10-31 02:07): `wc_mse_down_c0` jumped at step 3000

**Config:**
- max_ctx=0 (**no cross-chunk attention at all**)
- delta rule (mse), updating only down (W2), all 12 layers
- curriculum N~U{0..5}, starting from ema_160000, B=8, NorMuon 1e-4
- In the toy this exact config was the best (none mse_down: N30 0.98).

| step | curr3 acc / p | n05 acc / p (held-out test) | n10 p | n20 p |
|---|---|---|---|---|
| 0–2000 | ≈.2 / .20 | ≈.2 / .20 | .20 | .20 |
| 2500 | .28 / .25 | .31 / .29 | .15 | .17 |
| **3000** | **.88 / .88** | **.72 / .75** | .35 | .16 |

- A classic plateau-then-jump.
- With no attention path at all, the only thing that can carry "where the ball is" across chunks is the native FFN's down-projection fast weights. **This is direct evidence that native-FFN TTT memory can track a hidden state that keeps changing.**
- At the same steps the SWA1 control (KV relay) reaches only curr3 .50 (step 4500), with n05 still at chance.
- step 3500: curr3 .81/.85, n05 .69/.71, n10 .16/.20, n20 .09/.19. Stable at the training length, but **no extrapolation**.
- The step-3000 raw weights are saved as `wc_mse_down_c0/snap_003000.pt`.

### ★★ The user's design (keep slots as a fallback + split into heads) extrapolates: `wd_mse_down_c0_f025_h8`

**Config:** same as above (max_ctx=0, delta rule, down only, all 12 layers, curriculum), plus **only 25% of the FFN hidden units fast (the other 75% are a frozen fallback) and 8 heads on the fast part** (per head: 144 input × 144 fast hidden).

| step | curr3 acc / p | n05 acc / p | **n10 acc / p** | n20 acc / p |
|---|---|---|---|---|
| 2000 | .16 / .23 | .19 / .20 | – / .20 | – / .20 |
| 2500 | .50 / .71 | .28 / .63 | .09 / .19 | .09 / .19 |
| **3000** | **.88 / .91** | **.78 / .81** | **.69 / .75** | .31 / .27 |

- **Trained only up to N=5, it tracks at N=10 (12 chunks)**, and N=20 is above chance.
- The all-units-fast version without heads (wc_mse_down_c0) reaches n10 of only .16/.20 at step 3500.
- It also jumped earlier: step 2500 vs 3000.
- **This matches the toy prediction exactly:** the fallback and heads make the delta rule extrapolate better.
- step 3500: .94/.91, .66/.72, .59/.72, .38/.34.
- step 4000: 1.00/1.00, .88/.96, .56/.69, .31/.28.
- **step 4500: curr3 1.00/1.00, n05 .97/1.00, n10 .97/.97, n20 .53/.47**:
  - N10 (twice the training length) is solved.
  - N20 (4×) is clearly above chance (.2).
  - For reference, the paper's LaCT8 (a dedicated fast-weight branch, trained from scratch for 200K steps): N5/10/20/30 = 1.00/1.00/.98/.69.
  - Here: native-FFN fast weights only, fine-tuned for 4500 steps.
  - Step-4500 weights: `snap_004500.pt`, thorough eval Slurm 22214061.
- step 5000: .97, .78, .62, .44. The 32-episode eval fluctuates, so this is a dip.
- **step 5500: curr3 1.00, n05 .97, n10 .81, n20 .75** (p .99/1.00/.84/.72): **N20, 4× the training length, is at 75% accuracy.**
- Thorough eval (n05–n30, tf + free, 64 episodes, 50 steps): Slurm 22213891 → `wd_mse_down_c0_f025_h8/eval/snap_003000_model.json`.
- Thorough eval (n05–n30, tf + free, 64 episodes, 50 steps): Slurm 22213822, result in `wc_mse_down_c0/eval/snap_003000_model.json`.

**First positive signal (2026-10-31, step 3000): `wc_dot_up_c0`** (max_ctx=0 = no cross-chunk attention at all; dot loss, up only).
- The probe (reading the ball position from the last-layer features in the denoise pass), N=0..5:
  - step 550: .80/.48/.43/.34/.31/.30
  - step 3000: 1.00/.79/.61/.54/.45/.41
- **Caveat:** the probe averages over chunk 1 (the lowering chunk, where the ball is still visible in the input). Reading only chunk 1 from the input would give a baseline of about .60/.47/.40/.33 for N=1/2/3/5. Step 3000 is above that throughout.
- With no attention path, that extra information can only come from the fast weights. It is now on par with the SWA1 control (KV relay; `wc_ctrl_c1` at step 1300: 1.00/.81/.61/.51/.44/.41).
- The eval's curr3 p rose from .20 to .29, and the σ=1 x0 MSE dropped from .0053 to .0044.
- **This is the first evidence in the world model that native-FFN fast weights can carry the hidden state.**
- Step 4000, continuing:
  - Probe N=0..5: 1.00/.84/.67/.56/.47/.44.
  - Eval p: curr3 .39, n05 .26, **n10 .26**, n20 .22.
  - At the same step the SWA1 control has n05/n10 p .19/.20 (at chance).
  - n10 means remembering across 12 chunks, and this run has no attention path at all, so it can only come from the fast weights.
  - Consistent with the toy's none dot_up (N10 .80–1.00).
  - **Correction (step 4500): n10 p fell back to .20.** With 32 eval episodes, p fluctuates by about ±0.05, so step 4000's n10 .26 was most likely noise. n05 p .26→.28 is holding.
  - Conclusion so far: at the training length it does use the fast weights; extrapolation is unconfirmed. Wait for the final eval (64 episodes, 50 steps).

`wc_ctrl_c1` (SWA1, no TTT): curr3 accuracy .25→.38→.41 (step 1000→3000), p .19→.28, but n05/n10/n20 are still at chance. The KV relay has learned about 3 swaps. Final (step 6000): curr3 .69/.59, n05+ at chance.

**Pitfall 15 (important): with SWA1, the world model takes the KV-relay shortcut and does not learn the TTT memory.**

| SWA1 delta-rule run | step | curr3 | n05 |
|---|---|---|---|
| wc_mse_down_c1 | 4500 | .50 | chance |
| wd_mse_down_c1_f025_first4 | 3000 | .44 | chance |
| wd_c1_f025 / f025_h8 / h8 | 2000–2500 | chance | chance |

- All of these stall at the level of the no-TTT relay control.
- The no-context (c0) versions jump at step 2500–3000 and learn memory.
- The toy didn't show this for the delta rule: with only 4 layers the toy's relay was hard to learn. In the world model the relay is easy to learn (the control reaches curr3 .69 by itself), so it competes with memory.
- **For the hand-written line: if you keep attention to the previous chunk, the memory may never get learned.** Options:
  - Train first with no cross-chunk attention, then add attention.
  - Context dropout.
- Wave E (`waves/we.sh`) tests these:
  - we_c1_f025h8_cd05: SWA1 + 50% context dropout, from the base.
  - we_warm_c1: start from the c0 memory model, then SWA1.
  - we_warm_c1_cd05: start from the c0 memory model, then SWA1 + dropout.
  - Even right after the warm start, SWA1 eval gives curr3 .75 / p .43 (8 episodes).

## World model, wave 1 (N=5 only, no curriculum)

ctrl, dot_up_all, dot_down_all are still running on Slurm as the "no curriculum" reference. Up to step 1500 every eval is at chance. dot_both_all and e2e_both_last3 ran locally to step ~1100, then I stopped them to free the GPUs. The remaining runs were cancelled before they started, because the toy showed that without a curriculum they would likely stay on the plateau.

## Overnight results (2026-10-31, all runs finished; quick eval at the last step: raw weights, 32 episodes; acc / p)

| run | setting | curr3 | n05 | n10 | n20 |
|---|---|---|---|---|---|
| **we_warm_c1_cd05** | start from the c0 memory model → SWA1 + 50% context dropout, 3000 steps | 1.00/1.00 | .97/1.00 | .84/.88 | **.69/.67** |
| we_warm_c1 | start from the c0 memory model → SWA1, 3000 steps | .97/.99 | .94/1.00 | .84/.88 | .47/.59 |
| **we_c1_f025h8_cd05** | **from base**, SWA1 + 50% context dropout | .97/.97 | .69/.70 | .72/.71 | .34/.25 |
| wd_c0_f025h8_first4 | c0, only the first 4 layers TTT | 1.00/1.00 | .94/1.00 | .84/.84 | .50/.41 |
| wd_c0_f025h8_last4 | c0, only the last 4 layers TTT | .97/1.00 | .78/.87 | .88/.88 | .47/.56 |
| wc_mse_both_c0 | c0, delta rule updating both matrices, all units fast | .94/.97 | .97/.94 | .66/.53 | .19/.17 |
| all of wc/wd *_c1 (no dropout) | SWA1 | .47–.72 | chance | chance | chance |

1. **Context dropout solves pitfall 15.**
   - Starting from base with SWA1, dropping all cross-chunk attention on 50% of steps is enough for the memory to get learned: n10 .72.
   - Without dropout, every SWA1 run stays at the relay level.
2. **Warm start + dropout gives the best SWA1 model:** train memory without attention first, then add attention + dropout. n20 .69, and the model keeps attention, so it can serve as the final world model.
3. **Layer placement: in the world model, the last 4 layers also work** (first4 ≈ last4, both near all layers). This differs from the toy, where only the last 1–2 of 4 layers failed.
   - My reading: the toy failed because there were not enough layers after the memory to compute with. With 12 layers, the last 4 still leave 3 layers after them.
   - So the toy's "early layers only" conclusion **does not carry over directly to the world model**. What matters is leaving a few layers after the memory.
4. In the world model, mse_both + c0 also learned (the toy failed in 3 seeds). It extrapolates worse than down only.

Thorough eval (EMA, 64 episodes, 50 steps) of the 6 runs above: running on local GPU 0/4 (`logs/final_evals_gpu{0,4}.log`).

All-units-fast `wc_mse_down_c0` (EMA, thorough): tf 1.00/.77/.11/.12, free .97/.56/.17/.14. The f0.25+h8 version reaches free 1.00/1.00/.88/.78 → this confirms that the fallback + heads are the source of the extrapolation.

### Thorough eval (EMA, 64 episodes, 50 steps; free = paper protocol) — 2026-10-31 afternoon

| run | free N5/10/20/30 | tf N5/10/20/30 |
|---|---|---|
| **we_warm_c1** (start from the c0 memory model → SWA1 3000 steps, **no dropout**) | **1.00/1.00/.94/.89** | .98/1.00/.91/.88 |
| we_warm_c1_cd05 (start from c0 → SWA1 + 50% dropout) | 1.00/1.00/.88/.72 | .98/1.00/.95/.69 |
| wd_mse_down_c0_f025_h8 (c0, all layers; the earlier best) | 1.00/1.00/.88/.78 | 1.00/.98/.94/.72 |
| wd_c0_f025h8_last4 (c0, last 4 layers only) | 1.00/1.00/.64/.41 | .98/.97/.59/.28 |
| wd_c0_f025h8_first4 (c0, first 4 layers only) | .91/.42/.22/.34 | .84/.52/.34/.22 |
| wc_mse_both_c0 (c0, both matrices, all units fast) | 1.00/.89/.47/.36 | .98/.92/.44/.36 |
| wc_mse_down_c0 (c0, down only, all units fast) | .97/.56/.17/.14 | 1.00/.77/.11/.12 |
| we_c1_f025h8_cd05 (from base, SWA1 + 50% dropout) | .66/.39/.20/.23 | .64/.44/.38/.17 |
| paper LaCT8 | 1.00/1.00/.98/.69 | – |

1. **Best model overall: we_warm_c1.**
   - Recipe: first train memory with no cross-chunk attention, then switch on SWA1 and fine-tune 3000 steps (no dropout).
   - Result: N20 .94, N30 .89, better than memory alone (.88/.78). N30 beats LaCT8 (.69).
   - Attention handles local dynamics; the memory handles the ball, and the two complement each other.
   - Once the memory is learned, adding attention **does not** make the model switch to the relay shortcut, so no dropout is needed (dropout was actually worse at N30: .72).
2. **Starting directly from base with SWA1 + dropout is much worse** (.66/.39/.20/.23); the quick eval (raw weights, 32 episodes) overestimated it.
   - Conclusion: **first force the memory to be learned without attention, then add attention** — this order matters.
3. **Correction on layer placement: in the world model, all layers > last 4 > first 4** (free N20 .88 / .64 / .22).
   - This is the opposite of the toy (where the last 1–2 of 4 layers didn't learn). The quick eval's "first4 ≈ last4" was wrong.
   - The toy's "early layers" conclusion does not transfer to the world model; with 12 layers the features in later layers are probably richer.
   - **Recommendation: update all layers.**
4. Among all-units-fast runs, both (N20 .47) beats down (.17). But down only + f0.25 + h8 (.88) is still far better than both.

### Wave F: window size during warm start (start from the c0 memory model, train 3000 steps, EMA, 64 episodes, 50 steps)

| Window during warm start | free N5/10/20/30 | tf N5/10/20/30 |
|---|---|---|
| **SWA1 (sees 1 chunk; we_warm_c1)** | **1.00/1.00/.94/.89** | .98/1.00/.91/.88 |
| SWA3 (wf_warm_c3) | 1.00/.95/.75/.72 | .98/.98/.78/.62 |
| SWA7 (wf_warm_c7, the base's original window) | 1.00/.59/.22/.25 | 1.00/.56/.30/.19 |

**Pitfall 16: the bigger the window, the worse the memory.**
- SWA7 covers the whole training episode (N≤5 → ≤8 chunks), so the final reveal can always see the reveal chunk directly through attention.
- As a result the memory gets no training signal and degrades. After warm start, N20 falls from .94 to chance.
- During training, the quick eval (raw weights) bounced around (N10 .44→.62→.44, N20 .16→.38→.19), but the thorough eval confirms it got worse.
- **For the hand-written line:** if you want a large window (for image quality), either:
  - train on **episodes longer than the window**, so the memory is still needed; or
  - use context dropout.
  Otherwise the memory gets "shorted out" by attention.

Multi-seed (in progress): wf_c0_s1 / wf_c0_s2 (the c0 stage) also jump around step 2500.
- At step 3500: s1 N5/10/20 = .91/.75/.44, s2 = .69/.53/.34.
- The original seed at the same stage: .66/.59/.38.
- **The recipe reproduces across seeds.**

**Warm start: longer is no better.** wf_warm_c1_long (SWA1, 6000 steps):
- EMA, thorough: free .98/1.00/.94/.89, tf .98/1.00/.98/.88.
- 3000 steps (we_warm_c1): free 1.00/1.00/.94/.89, tf .98/1.00/.91/.88.
- Free is identical; only tf N20 improves a bit. **3000 steps of warm start is enough.**
- The quick eval's N20 .97 at step 6000 was a high point of the raw weights; the thorough eval is the reference.

**Multi-seed (c0 stage, EMA, 64 episodes, 50 steps; free N5/10/20/30):**

| seed | free | tf |
|---|---|---|
| 0 (wd_mse_down_c0_f025_h8) | 1.00/1.00/.88/.78 | 1.00/.98/.94/.72 |
| 1 (wf_c0_s1) | 1.00/.98/.86/.70 | .98/.98/.83/.66 |
| 2 (wf_c0_s2) | 1.00/.98/.89/.64 | .97/1.00/.91/.62 |

- **Stable across 3 seeds:** N10 ≥ .98, N20 .86–.89, N30 .64–.78 (mean ≈ 1.00/.99/.88/.71).
- All three jump at around step 2500.
- The warm-start stage for the seeds (wf_warm_c1_s1/s2) is still being evaluated.

**Both matrices vs down only (c0, f0.25+h8, EMA, thorough):**

| config | free N5/10/20/30 | tf N5/10/20/30 |
|---|---|---|
| wf_c0_both_f025h8 | 1.00/.98/.92/.73 | 1.00/.98/.88/.75 |
| down only, 3 seeds | 1.00/.98–1.00/.86–.89/.64–.78 | – |

- Both falls **within the seed range of down only**; no clear difference.
- Down only is simpler and cheaper: W1 doesn't need a per-sample copy. **Recommend down only.**
- Contrast:
  - With all units fast, both is clearly better than down (N20 .47 vs .17).
  - With fallback + heads added, the two are equal.
  - So **fallback + heads is the main factor**; the choice of matrix matters little once it's in place.

**Multi-seed (warm-start stage: c0 → SWA1, 3000 steps, EMA, thorough; free N5/10/20/30):**

| seed | after c0 | **after warm start** |
|---|---|---|
| 0 (we_warm_c1) | 1.00/1.00/.88/.78 | 1.00/1.00/.94/.89 |
| 1 (wf_warm_c1_s1) | 1.00/.98/.86/.70 | 1.00/.98/.89/.73 |
| 2 (wf_warm_c1_s2) | 1.00/.98/.89/.64 | 1.00/1.00/.84/.55 |
| **mean** | 1.00/.99/.88/.71 | **1.00/.99/.89/.72** |

**Correction:**
- My earlier claim that "adding attention beats memory alone" came mainly from seed 0.
- Averaged over 3 seeds, before and after the warm start are **essentially the same**.
- What the warm start buys is **keeping attention while the memory loses nothing**; it does not make tracking stronger.
- Against LaCT8 (1.00/1.00/.98/.69): our 3-seed mean is comparable at N30 (.72) and lower at N20 (.89).

## Wave G (submitted 2026-10-31 night): the three papers' recipes inside our training recipe + loss-only comparison

All runs: c0 + curriculum, 6000 steps from ema_160000, followed automatically by the thorough eval (EMA, 64 episodes, 50 steps, n05–n30, tf + free). Code frozen in `snapshots/wg`; commands in `waves/wg.sh`.

| run | what it tests |
|---|---|
| wg_inplace | In-Place TTT as published: dot, down only, next-frame value, read-then-write, no renorm, every 6th layer (5, 11) |
| wg_inplace_all | same, but all 12 layers |
| wg_lact | LaCT8 shape carved out of the native FFN: dot, both, f0.5 + 8 heads (144×288 per head), renorm, write-then-read |
| wg_e2e | TTT-E2E: diffusion loss as the inner loss, both, last 3 layers (1/4), half the units frozen (= its static MLP), no renorm |
| wg_dot_down_f025h8 | **loss only**: dot instead of delta; everything else = best config |
| wg_dot_both_f025h8 | **loss only**: dot + both (the paper's way to get a self-path) |
| wg_e2e_f025h8 | **loss only**: TTT-E2E's loss; everything else = best config (B200, accum 4) |
| wg_f025h1 | fallback without heads |
| wg_f1h8 | heads without fallback |

Why the earlier paper-style runs don't count: they all used SWA1 (wc_dot_down_c1, wc_dot_both_c1, wc_e2e_last3_c1, …) and stalled at the relay level (final quick eval curr3 ≈ .6, n05 at chance), exactly like our own delta rule under SWA1 (wc_mse_down_c1). That is pitfall 15, not the loss. `wc_dot_up_c0` (the only paper-style c0 run) ended at curr3 .41, n05 .31, n10/n20 at chance (quick eval).

### Analysis B: a swap is a reflection, and only the delta rule writes reflections

- Swapping cups i and j is the permutation P = I − (e_i − e_j)(e_i − e_j)ᵀ = I − 2k̂k̂ᵀ with k̂ = (e_i − e_j)/√2: a Householder reflection.
- If the memory M maps cup keys to "ball here?", a swap needs M_new = M_old·P = M_old − 2(M_old k̂)k̂ᵀ. The update has to **subtract what the old memory answers along the swap direction**.
- Delta rule on down: ΔW2_new = ΔW2(I − η·h hᵀ) + η·v hᵀ. With v = 0, h ∝ k̂ and η‖h‖² = 2, this is exactly M_old·P. One write = one swap.
- Dot: ΔW2_new = ΔW2 + η·v hᵀ. The old memory is multiplied by I, so it can only accumulate. To swap, v would have to equal −(2/η)·M_old k̂, which needs a read of the old memory. Our `read` variant does exactly that (toy none read_down: 1.00/.98/.76), and LaCT gets it through the input-side update, whose gradient passes through the current W and activation. The paper's "W1 only" (readout only, 1.00/.14/.30/.16) and "1-layer MLP" (1.00/.23/.23/.13) have no such path and fail.
- This matches the linear-RNN literature: DeltaNet's transition I − βkkᵀ with β up to 2 has negative eigenvalues, which is what makes state tracking possible.
- The paper's MSE (.22/.16/.22/.22) fits the whole fast output to v, so the error is nonzero from the start. Ours fits only the increment (fast − slow), so the memory starts empty and the error term is exactly "old memory's answer − v".
- Not yet checked: whether the learned η‖h‖² is near 2 at swap chunks.
- **The paper's own baselines confirm Analysis B.** Standard GDN/KDA (delta rule with β ∈ [0,1], so eigenvalues 1−β ∈ [0,1]: they can only shrink the old value, never flip it) fit N=5 and fall to chance at N≥10. GDN-neg/KDA-neg (β ∈ [0,2], eigenvalues in [−1,1]) extrapolate; **GDN-neg reaches .95 at N30, the best model in the paper** (LaCT8 .69). The additive scratchpad control also fails by N20 (its memory norm keeps growing).
- Our η is an unbounded softplus and the keys are not normalized, so the effective β along a direction, (1/T)Σ_t η_t (h_t·k̂)², can exceed 1: our delta rule is in the "-neg" regime. Bounding β to [0,2] per token would *not* reproduce GDN-neg here, because we average one gradient step over the 960 tokens of a chunk (a sum of rank-1 terms, not a product of per-token Householders); a reflection along k̂ would then need every token aligned with k̂.
- So the dividing line is not "delta vs dot" or "in-place vs separate branch" but **whether the update can express a swap**: delta with β up to 2 (direct), nonlinear fast weights (LaCT, indirect), or a value that reads the old memory (our `read` variant).

### e2e (the general loss): why it has not been tested yet, and how wave G tests it
- The two earlier e2e runs can't separate "the loss is bad" from "the setting is bad":
  - e2e_both_last3 (wave 1): N=5 only, no curriculum, SWA1, stopped at ~1100 steps. Without a curriculum nothing learns (ctrl and dot_up_all stayed at chance for 6000 steps).
  - wc_e2e_last3_c1: curriculum, but SWA1 → the relay shortcut (curr3 .59, n05 at chance), the same as wc_mse_down_c1 (curr3 .50, n05 at chance).
- e2e never got c0 + curriculum + fallback/heads + all layers, which is what made the delta rule work.

| run | layers | params | structure | inner σ | question |
|---|---|---|---|---|---|
| wg_e2e | last3 | both | f0.5, 1 head, no renorm | random (shift 3, median .75) | TTT-E2E as published |
| wg_e2e_s1 | last3 | both | same | 1.0 | does a memory-only inner loss help? |
| wg_e2e_f025h8 | all | down | f0.25+h8 (= best config) | random | loss only |
| wg_e2e_both_f025h8 | all | both | f0.25+h8 | random | e2e needs both? |

**Hypothesis (to test):**
- On a swap chunk the cups are identical, the ball is hidden and the motion is given by the action. So the chunk can be predicted without any memory, and the inner diffusion loss does not depend on the ball. The e2e gradient therefore lives in "visible error" directions.
- e2e on down only adds visible-error values and cannot move a stored ball value. Updating up changes the keys, which can re-address what is stored.
- The delta rule avoids this problem: it multiplies the old memory by (I − η·h hᵀ), so a swap defined by visible keys is applied blindly to whatever is stored, including the hidden ball.
- Prediction: e2e down fails at N≥10; e2e both has a chance; σ=1 makes the inner loss as memory-dependent as it can be.

## Wave H (submitted 2026-11-01 early): memory in the attention q/k/v projections

Code: `AttnWriter` in wm.py; `--ttt-attn {q,k,v,qkv}`, and `--ttt-params none` keeps the FFN slow. check.py [2] (zero init is exactly 0 for qkv / v, also combined with the FFN) and [5] (hand-written gradient vs autograd, 1e-7).
- **Fast part:** the q/k/v output channels of the last f·16 attention heads (f0.25 = heads 12–15). The first heads are the fallback. `--ttt-heads 4` = block-diagonal inside the fast part, aligned to attention heads (each fast head writes from 1/4 of the input).
- **Target per projection:** t_P = T_P·a (zero-init) = "where this chunk wants P to move"; ΔW_P·a = how far TTT has already moved P for this input.
  - delta: ℓ = ½‖ΔW_P·a − t_P‖² → ΔW_P ← ΔW_P(I − η·a aᵀ) + η·t_P aᵀ. This is the same Householder form as the FFN delta rule, so it can express swaps.
  - dot: ΔW_P ← ΔW_P + η·t_P aᵀ (only accumulates).
- **e2e on q/k:** second-order gradients through RMSNorm + softmax are far spikier than on the FFN. At inner lr 10, gn reached 3e6 by step 4; with init 1 it stays at ≈2–20 (clipped at 1). lr is still learnable.
- **Pre-existing, not a bug:** FFN partial-fast mse configs differ from no-TTT by ~2.5e-4 at zero init.
  - Cause: ffn_apply sums the slow and fast halves of down separately (summation order), and the difference is amplified through 12 layers × 7 chunks. The old code gives the same number.
  - The attention path concatenates channels instead, so its difference is exactly 0.

| run | projection | loss | structure | question |
|---|---|---|---|---|
| wh_v / wh_k / wh_q / wh_qkv | v / k / q / qkv | delta | f0.25 + h4 | which projection can hold the state |
| wh_v_dot / wh_qkv_dot | v / qkv | dot | f0.25 + h4 | loss |
| wh_qkv_e2e | qkv | e2e (inner lr 1, accum 4) | f0.25 + h4 | loss |
| wh_v_f1 / wh_qkv_f1 | v / qkv | delta | all heads fast, no mask | fallback + heads |

All runs: c0 + curriculum, all 12 layers, 6000 steps, chained thorough eval.

Predictions:
- v + delta ≈ FFN down + delta. Same algebra (a linear associative memory keyed by the attention input), and the content enters the output directly.
- q/k + delta: the memory only changes routing inside the chunk (c0), so attention has to turn "which slot" into content. Expect it to be slower or weaker.
- dot fails, as on the FFN.
- f1 is worse than f0.25 + h4.
- e2e is uncertain.

Wave G interim (quick eval, raw weights): wg_inplace at step 5000 reaches curr3 .56 / n05 .47 / n10 .16 / n20 .00. It learns the training length and does not extrapolate, as predicted.

### Wave G results (EMA at step 6000, 64 episodes, 50 steps)

| run | free N5/10/20/30 | tf N5/10/20/30 |
|---|---|---|
| **best (delta, down, f0.25+h8), 3 seeds** | **1.00/.99/.88/.71** | – |
| wg_inplace (In-Place as published, layers 5,11) | .38/.14/.22/.03 | .38/.20/.00/.00 |
| wg_inplace_all | .23/.19/.16/.11 | .23/.09/.22/.19 |
| wg_lact (LaCT8 shape) | .23/.11/.22/.12 | .28/.00/.22/.16 |
| wg_e2e (TTT-E2E as published) | .09/.25/.22/.17 | .19/.27/.23/.25 |
| wg_e2e_s1 (σ=1) | .17/.25/.19/.25 | .23/.16/.17/.27 |
| wg_dot_down_f025h8 (loss only) | .22/.17/.22/.25 | .23/.22/.17/.20 |
| wg_dot_both_f025h8 (loss only) | .27/.17/.14/.11 | .23/.23/.22/.17 |
| wg_e2e_f025h8 (loss only) | .19/.14/.22/.19 | .20/.20/.17/.20 |
| wg_e2e_both_f025h8 | .17/.16/.22/.20 | .23/.19/.17/.19 |
| wg_f025h1 (fallback, no heads) | 1.00/.95/.33/.22 | 1.00/.78/.19/.20 |
| wg_f1h8 (heads, no fallback) | 1.00/.97/.09/.08 | 1.00/.88/.14/.19 |
| wc_mse_down_c0 (neither) | .97/.56/.17/.14 | 1.00/.77/.11/.12 |

1. **Every paper recipe fails in our setting, mostly already at N5.** In-Place as published is the only one above chance at N5 (.38).
2. **The loss alone decides it.** With the best structure held fixed, dot (down or both) and e2e (down or both) are at chance even at N5. Only the delta rule learns.
3. **e2e is flat at exactly p = .20 in all 5 variants for 6000 steps.** The inner lr did not collapse (it ends at 5–10, or 0.5 for qkv), so e2e writes something, just nothing useful. This is consistent with the observability hypothesis: the inner loss of a swap chunk does not depend on the ball.
4. **Fallback vs heads (delta, down):** either one alone fixes N10 (.56 → .95 / .97), but N20+ needs both (.33 / .09 alone vs .88 together).
5. The dot runs were still rising slowly at 6000 (curr3 p .3–.5, no jump), so they are continued to 12000 for a fair comparison (wave H2).

### Wave H results at step 6000 — the attention memory learns, but late

| run | EMA free N5/10/20/30 | raw weights at 6000, quick eval (curr3 / n05) |
|---|---|---|
| wh_v | .25/.12/.09/.17 | .94 / .69 |
| wh_k | .38/.19/.19/.30 | .91 / .84 |
| wh_q | .36/.19/.19/.23 | .78 / .72 |
| wh_qkv | .75/.44/.06/.16 | .94 / .91 |
| wh_v_f1 | .69/.23/.09/.20 | 1.00 / 1.00 |
| wh_qkv_f1 | .28/.14/.22/.16 | .47 / .22 |
| wh_v_dot / wh_qkv_dot | .25 / .30 at N5 | no jump (curr3 p .34 at most) |
| wh_qkv_e2e | chance | chance |

- **Pitfall 17:** every attention delta run jumps at step 5000–6000, much later than the FFN (2500). EMA 0.999 averages over ~1000 steps, so at step 6000 it still mostly holds pre-jump weights, and the EMA eval badly understates these runs.
- Fix: continue the same runs to 12000 steps (`waves/wh2.sh`, resumed from last.pt with a constant lr) and evaluate ckpt_012000.
- Even q alone or k alone (routing only) learns the training length.

### Wave H2: attention memory at 12000 steps (EMA, thorough) — q / k extrapolate further than anything so far

| run | free N5/10/20/30 | tf N5/10/20/30 |
|---|---|---|
| **wh_q** (delta, q only, f0.25+h4) | **.98/1.00/.97/.91** | 1.00/1.00/.97/.92 |
| **wh_k** (delta, k only) | **.97/1.00/.97/.88** | 1.00/1.00/.98/.83 |
| wh_v (delta, v only) | 1.00/.97/.11/.02 | 1.00/.97/.19/.11 |
| FFN best (delta, down, f0.25+h8), 6000 steps, 3 seeds | 1.00/.99/.88/.71 | – |
| paper LaCT8 (from scratch, 200K) | 1.00/1.00/.98/.69 | – |

- **Memory in the routing (q or k alone) beats memory in the content (v, FFN down) by a wide margin at N30.** These are single seeds, with twice the FFN's step budget.
- Hypothesis (unverified): q and k pass through RMSNorm and then softmax, so a read is normalized and selective (a soft argmax over tokens), and small write errors get cleaned up at every step. v and FFN down add content linearly, so errors accumulate over many swaps.
- The other 12k evals (qkv, f1, dot) are running locally on GPUs 6/7 (Slurm duplicates cancelled).

## Wave I (submitted 2026-11-01 evening)

| run | what |
|---|---|
| wi_ffn_key | FFN best + `--ttt-key proj` (write key k = u + W_k·u, W_k zero-init, read unchanged), 6000 steps |
| wi_q_key | attention q + `--ttt-key proj`, 12000 steps |
| wi_ffn_12k | FFN best continued 6000 → 12000 (equal budget to the attention runs) |
| wi_q_s1 / wi_q_s2 / wi_k_s1 | seeds for the q / k results |

**Why `--ttt-key proj` (the user's question about LaCT's "k predicts v"):**
- Our writes have always used k = the layer's own input (u or a), with v = a zero-init projection. LaCT uses separate learned k / q / v projections (with RoPE): write with k, read with q.
- In our setting, writes see the clean chunk at σ=0, while reads see the noisy next chunk (different σ, and a different time-embedding scale/shift on u). With k = u, write and read addresses match only to the extent that clean and noisy features happen to agree. A learned k can move the write address toward what the next chunk's queries will look like.
- The read path (the native forward) is untouched; only where the memory writes changes. Swapping the read input as well would change the native computation, so that was not done.
- check.py [2]: zero init still matches key=self (exactly 0 on the attention path). [6]: hand-written gradient vs autograd agree to 1e-7.

### Wave H2, more 12k results: any fast v blows up beyond the training length

| run | free N5/10/20/30 | tf N5/10/20/30 | σ=1 x0 MSE (tf_mse1) N5/10/20/30 |
|---|---|---|---|
| wh_q | .98/1.00/.97/.91 | 1.00/1.00/.97/.92 | .0003/.0016/.0008/.0023 |
| wh_k | .97/1.00/.97/.88 | 1.00/1.00/.98/.83 | .0003/.0014/.0008/.0029 |
| wh_v | 1.00/.97/.11/.02 | 1.00/.97/.19/.11 | .0001/.0014/**.0927/.1297** |
| wh_qkv | 1.00/.98/.00/.16 | 1.00/1.00/.02/.00 | .0003/.0013/**.0355/.0529** |
| wh_qkv_f1 | 1.00/.94/.22/.12 | 1.00/.94/.12/.19 | .0002/.0019/**.2035/1.114** |
| FFN best (6000) | 1.00/1.00/.88/.78 | 1.00/.98/.94/.72 | .0003/.0011/.0013/.0029 |

**Pitfall 18: making v fast breaks generation beyond the training length.**
- Every run with a fast v keeps a normal x0 error up to N10, then it jumps 30–1000× at N20/N30. The frames break and the ball can't be detected, which is why some accuracies come out as 0.
- q / k pass through RMSNorm (over all 1152 channels) before the softmax, so whatever the memory writes is rescaled to a bounded size. v goes straight into the output with no normalization, so its drift grows without bound.
- Adding v to q/k (wh_qkv) inherits v's failure.
- **Recommendation: put the memory in q or k only.**

Caveat on the fallback for q/k: that RMSNorm is over the whole 1152-dim q (or k), not per head. The fast heads therefore change each token's RMS and rescale the fallback heads' q/k by a common factor, so the q/k fallback is not perfectly isolated (v's fallback is).

More 12k results (EMA, thorough; free N5/10/20/30, x0 MSE at N10/N20/N30):
- wh_v_f1 (v, all heads fast): free 1.00/.25/.19/.09. The x0 MSE is already 1.06 at N10: without the fallback, a fast v blows up even sooner (pitfall 18).
- wh_v_dot: .28/.14/.12/.00, and wh_qkv_dot: .39/.16/.05/.00. Even 12k steps of dot never learns, and a fast v drifts here too (x0 MSE .09–.25 at N20/N30).
- wg_dot_down_f025h8 (FFN dot): .31/.12/.17/.22. Still at chance at 12k, but stable (x0 MSE ≤ .008). **dot is dead in every setting.**

Wave I scheduling: wi_q_s1 and wi_q_key are trained locally on GPUs 6/7 (`scratchpad run_local.sh`, code `snapshots/wi`, train then eval). Their Slurm copies were cancelled. The other four wave I runs stay on Slurm.

### Wave I, first results (local GPUs, 12000 steps, EMA, thorough: 64 episodes, 50 steps)

| run | free N5/10/20/30 | tf N5/10/20/30 | σ=1 x0 MSE N5/10/20/30 |
|---|---|---|---|
| wh_q (seed 0, from H2) | .98/1.00/.97/.91 | 1.00/1.00/.97/.92 | .0003/.0016/.0008/.0023 |
| **wi_q_s1** (seed 1) | **1.00/1.00/.94/.95** | 1.00/1.00/.97/.95 | .0003/.0016/.0010/.0024 |
| wi_q_key (q + `--ttt-key proj`) | 1.00/1.00/.94/.81 | 1.00/1.00/.97/.81 | .0002/.0016/.0012/.0031 |

- **q-only memory reproduces on a second seed.** N30 free goes .91 → .95; the two-seed mean is .99/1.00/.955/.93. Generation stays stable (x0 MSE ≤ .003 everywhere).
- **A learned write key does not help.** N20 is the same, N30 drops to .81.
  - This is a single seed. At p≈.9, 64 episodes give SE ≈ .04, so the drop is about 2 SE: likely worse, not certain.
  - The key-proj run's write size keeps growing during training (dW q .24 at 12k, vs .09–.16 with key=self). Hypothesis, unverified: a learned key inflates ‖k‖, so the effective step η‖k‖² grows, and bigger writes drift more over 30 swaps.
  - Answer to the "LaCT k → v" question: in our in-place design, a separately learned write address (LaCT-style k) buys nothing. k = the layer's own input is already the right address.
- Training probes (raw weights, N0–5) look alike for all three runs. Only the long-horizon eval separates them.

Scheduling: GPU 6 was taken by another user after wi_q_s1 finished. wi_ffn_12k was moved to local GPU 7 (Slurm copy cancelled), resuming from the copied `wd_mse_down_c0_f025_h8` last.pt. wi_ffn_key, wi_q_s2 and wi_k_s1 stay on Slurm, pending behind the user's own pgo jobs.
