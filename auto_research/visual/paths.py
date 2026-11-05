# 数据和输出放哪：环境变量优先（远端包的 run.sh 会设 VM_DATA_DIR / VM_OUT_DIR），没有就用本地的默认位置
#   VM_DATA_DIR：数据集的上一层（里面是 RubikMemory/、CubeMemory/）
#   VM_OUT_DIR ：所有 run 的输出（<run>/ckpt、log.jsonl、eval/），logs/
#   VM_VAE     ：Wan2.1 VAE（diffusers 格式，里面有 vae/）：HF 的 repo id 或者本地文件夹
import os

DATA_DIR = os.environ.get("VM_DATA_DIR", "/project/sds-rise/ethan/Dataset")
OUT_DIR = os.environ.get("VM_OUT_DIR", "/project/sds-rise/ethan/Finetune/world/auto_research/visual")
VAE = os.environ.get("VM_VAE", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
