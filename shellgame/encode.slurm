#!/bin/bash
# Encode the rendered Shell Game videos to Wan2.1 VAE latents as a Slurm job array, one GPU per task.
# Every task runs encode_latents.py --claim, i.e. takes the next shard nobody has claimed yet, so the array can run
# alongside local GPUs doing the same (CUDA_VISIBLE_DEVICES=i python shellgame/encode_latents.py --claim);
# tasks that start after every shard is claimed exit right away. To redo shards after a crash: once no worker is
# left running, rm -r <root>/logs/encode_claims and submit again (finished latents are skipped).
# The world env (torch 2.8 + cu128) runs on every GPU of the gpu partition (V100 ... B200); the gpu QOS allows 32 GPUs per user.
#   sbatch shellgame/encode.slurm
#SBATCH --job-name=shellgame-encode
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/sds-rise/ethan/Dataset/ShellGame/logs/encode_%A_%a.log

export HF_HUB_OFFLINE=1  # the Wan2.1 VAE is already in ~/.cache/huggingface
export PYTHONNOUSERSITE=1  # ~/.local/lib/python3.11 would otherwise shadow the env's packages (cv2, ...)

echo "$(date '+%F %T') $(hostname) task $SLURM_ARRAY_TASK_ID, $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /project/sds-rise/ethan/world_model
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u shellgame/encode_latents.py --claim
