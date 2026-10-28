#!/bin/bash
# Videos + report for one scene (DS=hypernerf by default, TS = time_split) once its full-prefix runs exist (run inside a container):
#   scripts/fair_finalize_scene.sh <scene> <source> [report-only]
set -e
cd /workspace/4DGS-JEPA
export PYTHONPATH=src:third_party/ODE-GS WANDB_MODE=disabled USER=user
S=$1; SRC=$2; R=results/${DS:-hypernerf}/$S; F=$R/fair
K=$(python -c "import json; print(json.load(open('$F/baselines/metrics.json'))['selected_k'])")
NE=$(python -c "import json; print(json.load(open('$F/configs/scene.json'))['eval_cams'])")
LABELS='{"jepa_full":"4DGS-JEPA","jepa_inner":"4DGS-JEPA","odegs_full":"ODE-GS","odegs_inner":"ODE-GS"}'
DATA=$SRC; [ -f $SRC/dataset.json ] || DATA=$(dirname $SRC)  # videos.py joins <data>/<scene> for D-NeRF
if [ -z "$3" ]; then
  python -m gsjepa.real.videos render --scene $S --root $R --time_split ${TS:-0.9} --data $DATA --sweep_cam $((NE / 2)) \
    --methods gt deform static cv$K odegs_fair jepa_fair --jepa_ckpt $F/runs/jepa_full/last.pt --odegs_ckpt $F/runs/odegs_full/last.pt
  python -m gsjepa.real.videos compose --scene $S --root $R --prefix $S --cols 3 --methods gt deform static cv$K odegs_fair jepa_fair \
    --labels "{\"gt\":\"GT\",\"cv$K\":\"CV k=$K\",\"odegs_fair\":\"ODE-GS\",\"jepa_fair\":\"JEPA (ours)\",\"deform\":\"Deform-GS\",\"static\":\"Static\"}"
fi
python scripts/fair_report.py --root $R --write_report --labels "$LABELS" > /dev/null
echo "finalized $S"
