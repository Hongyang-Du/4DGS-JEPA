#!/bin/bash
# Enqueue the 4 controlled stage-2 runs of one scene into the shared queue of results/$DS (default hypernerf):
#   scripts/fair_enqueue_scene.sh <scene> <source> [steps]
# priority 1: full-prefix runs (headline, scored on t > 0.9); priority 2: inner-split runs (held-out validation curves)
S=$1; SRC=$2; STEPS=${3:-90000}; R=results/${DS:-hypernerf}/$S; Q=results/${DS:-hypernerf}/_queue
mkdir -p $Q/queue $Q/running $Q/done
A="--source $SRC --root $R --time_split ${TS:-0.9} --inner_cut ${IC:-0.8} --seed 0 --steps $STEPS"
for m in odegs jepa; do
  echo "python -m gsjepa.real.fair $m --split full $A --cfg @$R/fair/configs/$m.json --out $R/fair/runs/${m}_full" > $Q/queue/1_${S}_${m}_full.sh
  echo "python -m gsjepa.real.fair $m --split inner $A --cfg @$R/fair/configs/$m.json --out $R/fair/runs/${m}_inner" > $Q/queue/2_${S}_${m}_inner.sh
done
ls $Q/queue
