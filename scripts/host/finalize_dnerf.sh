#!/bin/bash
# host-side: videos + report + Level II for finished D-NeRF scenes (one container per list)
cd /home/junjie.ye/workspace/4DGS-JEPA
run() {  # container scenes...
  c=$1; shift
  for s in "$@"; do
    docker exec -w /workspace/4DGS-JEPA -e DS=dnerf -e TS=0.8 $c scripts/fair_finalize_scene.sh $s /data/D-NeRF/$s > logs/fair/finalize_$s.log 2>&1 && \
    docker exec -w /workspace/4DGS-JEPA -e PYTHONPATH=src:third_party/ODE-GS -e WANDB_MODE=disabled -e USER=user $c \
      python -m gsjepa.real.consistency scene --source /data/D-NeRF/$s --root results/dnerf/$s --time_split 0.8 --inner_cut 0.64 \
      > logs/fair/level2_$s.log 2>&1 && echo "$(date +%T) $s ok" || echo "$(date +%T) $s FAIL"
  done
}
run 4dgs-jepa-real bouncingballs hellwarrior mutant &
run 4dgs-jepa-dnerf hook jumpingjacks &
run 4dgs-jepa-g7 lego standup &
wait
echo done
