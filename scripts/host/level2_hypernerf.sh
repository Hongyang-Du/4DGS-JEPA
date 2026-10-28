#!/bin/bash
# host-side: group-rigid baselines + Level II diagnostics for the HyperNeRF scenes (one container per list)
cd /home/junjie.ye/workspace/4DGS-JEPA
declare -A SRC=([cut-lemon1]=/data/HyperNeRF/interp/cut-lemon1 [chickchicken]=/data/HyperNeRF/interp/chickchicken [keyboard]=/data/HyperNeRF/misc/keyboard [split-cookie]=/data/HyperNeRF/misc/split-cookie [vrig-3dprinter]=/data/HyperNeRF/vrig/3dprinter [slice-banana]=/data/HyperNeRF/interp/slice-banana)
run() {  # container scenes...
  c=$1; shift
  for s in "$@"; do
    A="--source ${SRC[$s]} --root results/hypernerf/$s --time_split 0.9 --inner_cut 0.8"
    docker exec -w /workspace/4DGS-JEPA -e PYTHONPATH=src:third_party/ODE-GS -e WANDB_MODE=disabled -e USER=user $c bash -c \
      "python -m gsjepa.real.fair baselines $A --out results/hypernerf/$s/fair/baselines > logs/fair/baselines_rigid_$s.log 2>&1 && \
       python -m gsjepa.real.consistency scene $A > logs/fair/level2_$s.log 2>&1" && echo "$(date +%T) $s ok" || echo "$(date +%T) $s FAIL"
  done
}
run 4dgs-jepa-g3 cut-lemon1 keyboard &
run 4dgs-jepa-g6 chickchicken split-cookie &
run 4dgs-jepa-g7 vrig-3dprinter &
wait
echo done
