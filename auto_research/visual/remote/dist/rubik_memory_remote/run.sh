#!/usr/bin/env bash
# The one entry point of the package: sets every cache / config / temp location (below), then runs
#
#   ./run.sh check                          environment, GPUs, VAE files, locations, free space (CPU, about a minute)
#   ./run.sh smoke 0,1,2,3,4,5,6,7          a tiny end-to-end run of every stage under $VM_OUT_DIR/smoke (~15 min)
#   ./run.sh pipeline 0,1,2,3,4,5,6,7       data -> base -> sweep -> infer; a rerun resumes
#   ./run.sh data 0,1,2,3,4,5,6,7           the dataset: meta.jsonl, then render + VAE-encode, 2 workers per GPU
#   ./run.sh base 0,1,2,3,4,5,6,7           the base world model from scratch, DDP over the listed GPUs (1, 2, 4 or 8)
#   ./run.sh sweep 0,1,2,3,4,5,6,7          the 8 memory fine-tunes (one per GPU; with fewer GPUs they queue up)
#   ./run.sh infer 0,1,2,3,4,5,6,7          rollouts of the test splits for 27 checkpoints (one job per GPU at a time):
#                                           generated latents only, no metrics (the project scores them)
#   ./run.sh handback                       uploads the results to a public Hugging Face repo (HF_TOKEN, HANDBACK_REPO)
#   tmux new -d -s rubik "./run.sh pipeline 0,1,2,3,4,5,6,7"                     kept running after logout
#
# Locations: config.env (VM_DATA_DIR, VM_OUT_DIR); environment variables of the same names override it.
# Env: PYTHON (default ./env/.env/bin/python from env/setup_env.sh if present, else python), VM_VAE (default
# ./models/Wan2.1-T2V-1.3B-Diffusers from download.sh), WORKERS (DataLoader workers per process, default 6),
# WANDB_API_KEY (optional: wandb online, project world-auto-visual; without it wandb is off, log.jsonl has every value),
# HF_TOKEN / HANDBACK_REPO / HANDBACK_FOLDER / UPLOAD_DATA (handback only).
#
# What a run touches: this folder (code, env/.env, models/, .cache/), $VM_DATA_DIR/RubikMemory (the dataset) and
# $VM_OUT_DIR (everything else it writes). Nothing goes to ~/.cache, ~/.config, ~/.triton, ~/.nv or /tmp (apart from the
# short TMPDIR link below):
#   <this folder>/.cache/   HF_HOME, TORCH_HOME, TRITON_HOME / TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, CUDA_CACHE_PATH,
#                           PYTORCH_KERNEL_CACHE_PATH, XDG_{CACHE,CONFIG,DATA,STATE}_HOME, MPLCONFIGDIR, WANDB_{CACHE,CONFIG,DATA,ARTIFACT}_DIR
#   $VM_OUT_DIR/            WANDB_DIR, TMPDIR=$VM_OUT_DIR/tmp (through a short link /tmp/rkm-<uid>-<n> when the path is
#                           longer than 60 characters: Unix sockets of the DataLoader workers allow 107), logs/
# HF_HUB_OFFLINE=1: nothing is downloaded at run time (download.sh does that beforehand), except by `handback`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$HERE/code"
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
MODE="${1:-}"
case "$MODE" in check|smoke|pipeline|data|base|sweep|infer|handback) ;; *) usage ;; esac
shift
GPUS=""
case "$MODE" in smoke|pipeline|data|base|sweep|infer)
  [[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "$MODE needs a GPU list such as 0,1,2,3,4,5,6,7"; exit 2; }
  GPUS="$1"; shift ;;
esac

# locations: the environment wins over config.env
_DATA="${VM_DATA_DIR:-}"; _OUT="${VM_OUT_DIR:-}"
# shellcheck disable=SC1091
source "$HERE/config.env"
VM_DATA_DIR="${_DATA:-$VM_DATA_DIR}"; VM_OUT_DIR="${_OUT:-$VM_OUT_DIR}"
for v in VM_DATA_DIR VM_OUT_DIR; do
  [[ -n "${!v}" && "${!v}" != /path/to* ]] || { echo "$v is not set: edit config.env (SET ME) or export $v"; exit 2; }
done
mkdir -p "$VM_DATA_DIR" "$VM_OUT_DIR/logs"
VM_DATA_DIR="$(cd "$VM_DATA_DIR" && pwd)"; VM_OUT_DIR="$(cd "$VM_OUT_DIR" && pwd)"
export VM_DATA_DIR VM_OUT_DIR
export VM_VAE="${VM_VAE:-$HERE/models/Wan2.1-T2V-1.3B-Diffusers}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HERE/env/.env/bin/python" ]]; then PYTHON="$HERE/env/.env/bin/python"; else PYTHON=python; fi
fi
WORKERS="${WORKERS:-6}"
WANDB_FLAG="--wandb 0"
[[ -n "${WANDB_API_KEY:-}" ]] && WANDB_FLAG="--wandb 1"
CACHE="${CACHE_DIR:-$HERE/.cache}"

short_dir() {  # $1 = a folder; prints it, or a short link to it in /tmp when its path is longer than 60 characters
  local real="$1" link
  mkdir -p "$real"
  if (( ${#real} <= 60 )); then echo "$real"; return; fi
  link="/tmp/rkm-$(id -u)-$(printf '%s' "$real" | cksum | cut -d' ' -f1)"
  ln -sfn "$real" "$link"
  echo "$link"
}

isolate() {
  export HF_HOME="$CACHE/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
  export TORCH_HOME="$CACHE/torch" PYTORCH_KERNEL_CACHE_PATH="$CACHE/torch/kernels"
  export TRITON_HOME="$CACHE/triton" TRITON_CACHE_DIR="$CACHE/triton/cache" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
  export CUDA_CACHE_PATH="$CACHE/nv/ComputeCache"
  export XDG_CACHE_HOME="$CACHE/xdg/cache" XDG_CONFIG_HOME="$CACHE/xdg/config" XDG_DATA_HOME="$CACHE/xdg/data" XDG_STATE_HOME="$CACHE/xdg/state"
  export MPLCONFIGDIR="$CACHE/matplotlib"
  export WANDB_CACHE_DIR="$CACHE/wandb/cache" WANDB_CONFIG_DIR="$CACHE/wandb/config" WANDB_DATA_DIR="$CACHE/wandb/data"
  export WANDB_ARTIFACT_DIR="$CACHE/wandb/artifacts" WANDB_DIR="$VM_OUT_DIR"
  TMPDIR="$(short_dir "$VM_OUT_DIR/tmp")"
  export TMPDIR
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" CUDA_DEVICE_ORDER=PCI_BUS_ID
  mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
    "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
}

stamp() { date +%Y%m%d_%H%M%S; }
say() { echo "$(date '+%F %T') $*"; }
IFS=',' read -r -a GPU_LIST <<<"${GPUS:-0}"
NGPU="${#GPU_LIST[@]}"

# ---- fixed settings of the experiments (INSTRUCTIONS section 4) ----
BASE_RUN=rubik_base
BASE_STEPS=40000
BASE_CKPT="$VM_OUT_DIR/$BASE_RUN/ema_$(printf '%06d' $BASE_STEPS).pt"
FT_STEPS=12000
FT_INFER="4000 8000 12000"          # fine-tune checkpoints that get inference (ckpt_XXXXXX.pt, EMA weights)
BASE_INFER="40000:-1 40000:0 20000:-1"  # base checkpoints that get inference, step:max_ctx (-1 = every past chunk, 0 = none)
RUNS=(  # name|flags; one memory configuration per run, all from the same base, max_ctx 0
  "rk_ctrl|"
  "rk_qkv_norm|--mem-proj qkv --mem-key norm --mem-lr 0.5"
  "rk_qkv|--mem-proj qkv"
  "rk_v_norm|--mem-proj v --mem-key norm --mem-lr 0.5"
  "rk_v|--mem-proj v"
  "rk_q|--mem-proj q"
  "rk_k|--mem-proj k"
  "rk_ffn|--mem-ffn 1"
)

data() {  # [extra build_rubik.py --meta args]
  local root="$VM_DATA_DIR/RubikMemory" pids=() failed=0 g w
  if [[ ! -f "$root/train/meta.jsonl" ]]; then
    say "data: writing meta.jsonl of every split"
    (cd "$CODE/datagen" && "$PYTHON" build_rubik.py --meta "$@")
  fi
  rm -rf "$root/logs/claims"  # claims of earlier, stopped workers: the new workers skip finished latents anyway
  mkdir -p "$root/logs"
  say "data: render + encode with 2 workers on each of GPUs $GPUS (logs $VM_OUT_DIR/logs/data_gpu*.log)"
  for g in "${GPU_LIST[@]}"; do for w in a b; do
    (cd "$CODE/datagen" && CUDA_VISIBLE_DEVICES="$g" "$PYTHON" -u build_rubik.py --encode --claim ${NUM_SHARDS:+--num-shards $NUM_SHARDS} \
      >>"$VM_OUT_DIR/logs/data_gpu$g$w.log" 2>&1) &
    pids+=("$!")
  done; done
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  (( failed == 0 )) || say "data: a worker failed (see $VM_OUT_DIR/logs/data_gpu*.log); the sweep below fills its gaps"
  say "data: sweep for missing latents"
  (cd "$CODE/datagen" && CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON" -u build_rubik.py --encode ${NUM_SHARDS:+--num-shards $NUM_SHARDS} \
    >>"$VM_OUT_DIR/logs/data_sweep.log" 2>&1)
  "$PYTHON" "$HERE/check_setup.py" --data-only
}

base() {  # [extra train_base.py args]
  if [[ -f "$BASE_CKPT" ]]; then say "base: $BASE_CKPT exists, skipped"; return 0; fi
  local log="$VM_OUT_DIR/logs/base_$(stamp).log"
  say "base: $NGPU GPUs ($GPUS), log $log"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" -- \
    train_base.py --task rubik --run "$BASE_RUN" --steps "$BASE_STEPS" --eval-every 0 --workers "$WORKERS" $WANDB_FLAG "$@" \
    2>&1 | tee -a "$log")
  [[ -f "$BASE_CKPT" ]] || { say "base: no $BASE_CKPT after training (see $log)"; return 1; }
}

finetune() {  # $1 = GPU, $2 = "name|flags", then extra train.py args
  local g="$1" name="${2%%|*}" flags="${2#*|}"
  shift 2
  if [[ -f "$VM_OUT_DIR/$name/ckpt_$(printf '%06d' $FT_STEPS).pt" ]]; then say "$name: done already"; return 0; fi
  say "$name: GPU $g"
  # shellcheck disable=SC2086
  (cd "$CODE" && CUDA_VISIBLE_DEVICES="$g" "$PYTHON" -u train.py --task rubik --run "$name" --base "$BASE_CKPT" \
    --steps "$FT_STEPS" --keep-every 4000 --eval-every 0 --final-eval 0 --base-gate 0 --workers "$WORKERS" $WANDB_FLAG \
    $flags "$@" --gpu 0 >>"$VM_OUT_DIR/logs/$name.log" 2>&1)
}

lanes() {  # $1 = function, then the items: item k runs on GPU k mod #GPUs, the items of one GPU one after another
  local fn="$1" pids=() i k
  shift
  local items=("$@")
  for ((i = 0; i < NGPU; i++)); do
    ( for ((k = i; k < ${#items[@]}; k += NGPU)); do "$fn" "${GPU_LIST[$i]}" "${items[$k]}" || echo "FAILED ${items[$k]}"; done ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

sweep() {
  [[ -f "$BASE_CKPT" ]] || { say "sweep: no base checkpoint $BASE_CKPT (run the base stage first)"; return 1; }
  say "sweep: ${#RUNS[@]} runs on $NGPU GPUs ($GPUS), logs $VM_OUT_DIR/logs/<run>.log"
  lanes finetune "${RUNS[@]}"
  local missing=() r
  for r in "${RUNS[@]}"; do
    [[ -f "$VM_OUT_DIR/${r%%|*}/ckpt_$(printf '%06d' $FT_STEPS).pt" ]] || missing+=("${r%%|*}")
  done
  if (( ${#missing[@]} )); then say "sweep: not finished: ${missing[*]} (see their logs)"; return 1; fi
  say "sweep: all ${#RUNS[@]} runs finished"
}

infer_job() {  # $1 = GPU, $2 = "<checkpoint path>|<max_ctx or empty>|<output folder>"
  local g="$1" ckpt out ctx
  IFS='|' read -r ckpt ctx out <<<"$2"
  if [[ -f "$out/done.json" ]]; then say "infer $(basename "$out"): done already"; return 0; fi
  [[ -f "$ckpt" ]] || { say "infer: no checkpoint $ckpt"; return 1; }
  say "infer $out: GPU $g"
  # shellcheck disable=SC2086
  (cd "$CODE" && CUDA_VISIBLE_DEVICES="$g" "$PYTHON" -u infer.py --ckpt "$ckpt" ${ctx:+--max-ctx $ctx} --out "$out" \
    ${INFER_ARGS:-} --gpu 0 >>"$VM_OUT_DIR/logs/infer_$(basename "$(dirname "$out")")_$(basename "$out").log" 2>&1)
}

infer() {
  local jobs=() x s c r
  for x in $BASE_INFER; do
    s="${x%%:*}"; c="${x#*:}"
    jobs+=("$VM_OUT_DIR/$BASE_RUN/ema_$(printf '%06d' "$s").pt|$c|$VM_OUT_DIR/inference/$BASE_RUN/ema_$(printf '%06d' "$s")_ctx$c")
  done
  for r in "${RUNS[@]}"; do for s in $FT_INFER; do
    jobs+=("$VM_OUT_DIR/${r%%|*}/ckpt_$(printf '%06d' "$s").pt||$VM_OUT_DIR/inference/${r%%|*}/ckpt_$(printf '%06d' "$s")")
  done; done
  say "infer: ${#jobs[@]} jobs on $NGPU GPUs ($GPUS), outputs $VM_OUT_DIR/inference/<run>/<checkpoint>/, logs $VM_OUT_DIR/logs/infer_*.log"
  lanes infer_job "${jobs[@]}"
  local missing=() j
  for j in "${jobs[@]}"; do [[ -f "${j##*|}/done.json" ]] || missing+=("${j##*|}"); done
  if (( ${#missing[@]} )); then say "infer: not finished: ${missing[*]}"; return 1; fi
  say "infer: all ${#jobs[@]} jobs finished"
}

smoke() {  # every stage at toy size in $VM_OUT_DIR/smoke/{data,out}; the real locations are not touched
  local S="$VM_OUT_DIR/smoke"
  rm -rf "$S"
  export VM_DATA_DIR="$S/data" VM_OUT_DIR="$S/out"
  mkdir -p "$VM_DATA_DIR" "$VM_OUT_DIR/logs"
  BASE_STEPS=20; BASE_CKPT="$VM_OUT_DIR/$BASE_RUN/ema_000020.pt"; FT_STEPS=10
  FT_INFER="10"; BASE_INFER="20:-1 20:0"; INFER_ARGS="--num 2 --steps 5"
  RUNS=("rk_ctrl|" "rk_qkv_norm|--mem-proj qkv --mem-key norm --mem-lr 0.5")
  NUM_SHARDS=4 data --train-num 96 --test-num 2
  base --batch "$(( 2 * NGPU ))" --save-every 10 --keep-every 10 --log-every 5
  sweep_args=(--batch 2 --log-every 5)
  local pids=() i
  for ((i = 0; i < ${#RUNS[@]}; i++)); do finetune "${GPU_LIST[$((i % NGPU))]}" "${RUNS[$i]}" "${sweep_args[@]}" & pids+=("$!"); done
  for p in "${pids[@]}"; do wait "$p"; done
  infer
  "$PYTHON" "$HERE/upload.py" --dry-run
  local ok=1 f
  for f in "$VM_OUT_DIR/$BASE_RUN/ema_000020.pt" "$VM_OUT_DIR/rk_ctrl/ckpt_000010.pt" "$VM_OUT_DIR/rk_qkv_norm/ckpt_000010.pt" \
           "$VM_OUT_DIR/inference/$BASE_RUN/ema_000020_ctx-1/done.json" "$VM_OUT_DIR/inference/$BASE_RUN/ema_000020_ctx0/done.json" \
           "$VM_OUT_DIR/inference/rk_ctrl/ckpt_000010/done.json" "$VM_OUT_DIR/inference/rk_qkv_norm/ckpt_000010/n24/free.npy"; do
    [[ -f "$f" ]] || { echo "smoke: MISSING $f"; ok=0; }
  done
  if (( ok )); then say "smoke: PASSED (outputs in $S; delete it: rm -rf $S)"; else say "smoke: FAILED (logs in $VM_OUT_DIR/logs/)"; exit 1; fi
}

isolate
case "$MODE" in
check)    exec "$PYTHON" "$HERE/check_setup.py" "$@" ;;
smoke)    smoke ;;
data)     data "$@" ;;
base)     base "$@" ;;
sweep)    sweep ;;
infer)    infer ;;
handback) unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; exec "$PYTHON" "$HERE/upload.py" "$@" ;;
pipeline)
  data || { say "pipeline: data stage failed"; exit 1; }
  base || { say "pipeline: base stage failed"; exit 1; }
  sweep || say "pipeline: sweep incomplete (see above); inference runs for what finished"
  infer || { say "pipeline: inference incomplete (see above)"; exit 1; }
  say "pipeline done: $VM_OUT_DIR (next: ./run.sh handback)"
  ;;
esac
