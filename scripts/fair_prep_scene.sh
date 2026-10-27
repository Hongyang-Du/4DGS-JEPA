#!/bin/bash
# Per-scene preparation after Stage 1 (run inside a container):
#   scripts/fair_prep_scene.sh <scene> <source> [lr]
# writes results/$DS/<scene>/fair/{configs/*.json, masks.pt, baselines/metrics.json}
# DS=hypernerf (time_split 0.9, inner_cut 0.8) by default; D-NeRF: DS=dnerf TS=0.8 IC=0.64 PLAN_ARGS="--horizon 0.2"
set -e
cd /workspace/4DGS-JEPA
export PYTHONPATH=src:third_party/ODE-GS WANDB_MODE=disabled USER=user
S=$1; SRC=$2; LR=${3:-3e-4}; R=results/${DS:-hypernerf}/$S
A="--source $SRC --root $R --time_split ${TS:-0.9} --inner_cut ${IC:-0.8}"
python -m gsjepa.real.fair plan $A --lr $LR $PLAN_ARGS
python -m gsjepa.real.fair masks $A
python -m gsjepa.real.fair baselines $A --out $R/fair/baselines
