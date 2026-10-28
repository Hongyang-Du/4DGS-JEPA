#!/bin/bash
# host-side watcher: run fair_prep_scene.sh in the scene's container once its Stage 1 has finished
cd /home/junjie.ye/workspace/4DGS-JEPA
declare -A C=([cut-lemon1]=4dgs-jepa-fair [chickchicken]=4dgs-jepa-g4 [keyboard]=4dgs-jepa-g6 [split-cookie]=4dgs-jepa-g7 [vrig-3dprinter]=4dgs-jepa-g4)
declare -A SRC=([cut-lemon1]=/data/HyperNeRF/interp/cut-lemon1 [chickchicken]=/data/HyperNeRF/interp/chickchicken [keyboard]=/data/HyperNeRF/misc/keyboard [split-cookie]=/data/HyperNeRF/misc/split-cookie [vrig-3dprinter]=/data/HyperNeRF/vrig/3dprinter)
todo="cut-lemon1 chickchicken keyboard split-cookie vrig-3dprinter"
while [ -n "$todo" ]; do
  next=""
  for s in $todo; do
    if grep -q "Training complete" logs/stage1_$s.log; then
      echo "$(date +%T) prep $s"; docker exec ${C[$s]} scripts/fair_prep_scene.sh $s ${SRC[$s]} > logs/fair/prep_$s.log 2>&1 && echo "$(date +%T) prep $s ok" || echo "$(date +%T) prep $s FAIL" &
    else next="$next $s"; fi
  done
  todo=$(echo $next); sleep 30
done
wait
