# remote (direction 2, Rubik's cube)

The package for a remote 8-GPU node: dataset creation, base from scratch, the memory sweep and inference, from one
command (`./run.sh pipeline 0,1,2,3,4,5,6,7`), then the hand-back (`./run.sh handback`). **No evaluation on the remote
side** (user, 2026-11-03: "不能有evaluation 只能有dataset creation training 和inference eval这些我们拿回来跑"): it hands
back checkpoints, training logs and the generated latents of the test splits; scoring happens here
(`tools/handback.py eval`). This file is the maintainer side and is never shipped; `package/INSTRUCTIONS.md` is the
runbook that goes with the zip (impersonal English, for the remote side and its agent). Modeled on
`position_geo_obs/remote/` (same layout, same isolation of caches, same leak check at build time).

| what | where |
| --- | --- |
| package source | `package/`: INSTRUCTIONS.md, run.sh (stages data / base / sweep / infer + the fixed run and checkpoint lists), config.env (`/path/to` placeholders), check_setup.py, download.sh (the Wan2.1 VAE at a pinned commit, sha256-checked), upload.py (hand-back), env/ (setup_env.sh, requirements(.lock).txt pinned to the local `world` env, read without `~/.local`) |
| code shipped with it | `../{wm,train_base,train,infer,sampling,data,normuon}.py`, `../datagen/{rubik,build_rubik,vae_io}.py`: the same files the local runs use; `paths.py` is replaced by a strict version (locations only from the environment). Not shipped: `evaluate`, `rubik_eval`, `metrics`, `summary`, `check`, `cube`, `build`, `rubik_sample` |
| build | `PYTHONDONTWRITEBYTECODE=1 python tools/build_package.py` -> `dist/rubik_memory_remote/` + `dist/rubik_memory_remote.zip`; refuses when a shipped file contains a local path, host or user name, imports an evaluation module at top level, or does not import from the dist folder alone |
| end-to-end test | unzip the build into a scratch folder, `./download.sh`, `./run.sh check`, `./run.sh smoke <gpu>`; the GPU list is physical: never wrap it in an outer `CUDA_VISIBLE_DEVICES` (run.sh overrides it per process) |
| inference format | `inference/<run>/<ckpt>[_ctxC]/nXX/{free.npy [64,16,5K,24,32], tf.npy [64,16,20,24,32], ids.json}`, `done.json`; seeds = episode ids, identical to `rubik_eval.py` live evaluation (checked 2026-11-03: live vs saved differ by <= 0.0012 over 51 metrics, fp16 storage) |
| hand-back | public Hugging Face repos of the `riselab` organization (user: "数据和model都可以走这里的public 不要private会有限制"): results `riselab/rubik_memory` (model repo, one folder `rubik_r1_<time>`, ~45 GB), optional dataset `riselab/rubik_memory_data` |
| fetch + evaluate | `tools/handback.py fetch --folder <f>` -> `$VM_OUT_DIR/remote/<f>/` (local: Finetune/world/auto_research/visual/remote/), checked against MANIFEST.json; `tools/handback.py eval --folder <f> --gpu <ours>` -> `<run>/eval/<ckpt>_ema.json` per inference job + `summary_rubik.txt` |
| local paths | `../paths.py`: `VM_DATA_DIR`, `VM_OUT_DIR`, `VM_VAE` with the local defaults (Dataset/, Finetune/world/auto_research/visual, the HF repo id) |

## Rebuild / test

```bash
cd /project/sds-rise/ethan/world_model/auto_research/visual/remote
PY=/project/sds-rise/ethan/miniforge3/envs/world/bin/python
PYTHONDONTWRITEBYTECODE=1 $PY tools/build_package.py      # after any change in package/ or the shipped code
T=<scratch>; rm -rf $T && mkdir -p $T && cd $T && unzip -q .../dist/rubik_memory_remote.zip && cd rubik_memory_remote
PYTHON=$PY ./download.sh
VM_DATA_DIR=$T/data VM_OUT_DIR=$T/out PYTHON=$PY ./run.sh check
VM_DATA_DIR=$T/data VM_OUT_DIR=$T/out PYTHON=$PY WORKERS=2 ./run.sh smoke <a GPU of ours>   # ~20 min
```
