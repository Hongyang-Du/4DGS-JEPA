#!/bin/bash
# Claim job scripts from results/dnerf/<scene>/fair/queue (atomic mv) and run them one by one.
# Several workers (one or two per GPU/container) can share the queue. Exits when the queue is empty.
cd /workspace/4DGS-JEPA
Q=${QDIR:-results/dnerf/${SCENE:-mutant}/fair}
export PYTHONPATH=src:third_party/ODE-GS WANDB_MODE=disabled USER=user
while true; do
  j=$(ls $Q/queue/*.sh 2>/dev/null | sort | head -1)
  [ -z "$j" ] && break
  b=$(basename "$j")
  mv "$j" "$Q/running/$b" 2>/dev/null || continue
  echo "$(date +%T) [$(hostname)] start $b"
  bash "$Q/running/$b" > "logs/fair/${b%.sh}.log" 2>&1 && st=ok || st=FAIL
  mv "$Q/running/$b" "$Q/done/$b"
  echo "$(date +%T) [$(hostname)] $st $b"
done
