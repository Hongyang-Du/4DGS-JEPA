# Environment for the controlled protocol-B benchmark (HyperNeRF ×6, D-NeRF ×8)

- Image: `4dgs-jepa:real` (containers 4dgs-jepa-{fair,real,dnerf,g3,g4,g6,g7}, one GPU each)
- Python 3.12.3, torch 2.7.0a0+79aa17489c.nv25.04, CUDA 12.9, cuDNN 90900
- GPU: NVIDIA A100-SXM4-80GB, driver 535.129.03
- `pip freeze` of the image: `results/env/pip_freeze.txt`
- ODE-GS (Stage 1 + baseline model/loss): https://github.com/preacherwhite/ODE-GS.git @ bc7f1e0c604480117cf3524e5b6b1af9923d9c53, no local changes
- Data: `/data/HyperNeRF/{interp,misc,vrig}/<scene>`, `/data/D-NeRF/<scene>`

## Code version
- HyperNeRF runs: 2026-10-25/26, working tree between 5c43c7c and 06b19b1.
- D-NeRF runs: 2026-10-26/27, code as committed in 06b19b1.
- Edits after the HyperNeRF runs (from the session log, not a recorded hash) touch only `plan()` horizon flags, baselines (group-rigid added), dynamic-mask keys (name@fid) and
  the finalize script (fa9b1ff); HyperNeRF baselines were recomputed and the pre-existing numbers were unchanged.
  Stage-2 training code (`train_odegs`, `run_jepa`, `gsjepa.real.jepa`) was not edited in between.

## Per-run records (committed)
`results/<ds>/<scene>/fair/`: `configs/*.json`, `runs/*/{metrics.json,train_log.txt,curve.csv}`, `baselines/metrics.json`,
`report.md`, `tables.md`, `summary.json`, `level2.json`, curves. `stage1/cfg_args` per scene.
Not committed (DGX only, `/home/junjie.ye/workspace/4DGS-JEPA/results`): `last.pt`, `masks.pt`, Stage-1 models, videos.
