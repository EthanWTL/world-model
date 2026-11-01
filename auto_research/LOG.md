# Auto research log: FFN fast weights (TTT) on the Shell Game world model

## ★ Current conclusion (2026-10-31, read this first)

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
