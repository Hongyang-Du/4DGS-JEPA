#!/bin/bash
# host-side watcher: once a D-NeRF scene's Stage 1 has finished, run fair_prep_scene.sh and enqueue its 4 runs;
# touches results/dnerf/_queue/.closed when every scene is queued (workers then exit on an empty queue)
cd /home/junjie.ye/workspace/4DGS-JEPA
declare -A C=([bouncingballs]=4dgs-jepa-g3 [hellwarrior]=4dgs-jepa-g6 [hook]=4dgs-jepa-g7 [jumpingjacks]=4dgs-jepa-g3
              [lego]=4dgs-jepa-g6 [standup]=4dgs-jepa-g7 [trex]=4dgs-jepa-g3)
todo="bouncingballs hellwarrior hook jumpingjacks lego standup trex"
while [ -n "$todo" ]; do
  next=""
  for s in $todo; do
    if grep -q "Training complete" logs/stage1_$s.log; then
      echo "$(date +%T) prep $s"
      (docker exec -e DS=dnerf -e TS=0.8 -e IC=0.64 -e PLAN_ARGS="--horizon 0.2" ${C[$s]} \
         scripts/fair_prep_scene.sh $s /data/D-NeRF/$s 1e-3 > logs/fair/prep_$s.log 2>&1 \
       && DS=dnerf TS=0.8 IC=0.64 bash scripts/fair_enqueue_scene.sh $s /data/D-NeRF/$s > /dev/null \
       && echo "$(date +%T) prep+enqueue $s ok" || echo "$(date +%T) prep $s FAIL") &
    else next="$next $s"; fi
  done
  todo=$(echo $next); sleep 30
done
wait
touch results/dnerf/_queue/.closed
echo "$(date +%T) all scenes queued"
