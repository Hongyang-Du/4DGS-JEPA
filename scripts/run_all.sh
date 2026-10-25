#!/usr/bin/env bash
# Reproduce all three paper experiments. Every (variant, seed) run of Exp1/Exp2 is an
# independent process (the models are tiny, so they share one GPU), then tables are aggregated.
#   bash scripts/run_all.sh            # inside the container, from the repo root
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

python -m gsjepa.run configs/exp3.yaml > logs/exp3.log 2>&1 &

for exp in exp1 exp2; do
  mapfile -t variants < <(python -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['variants']))" "configs/${exp}.yaml")
  for variant in "${variants[@]}"; do
    for seed in 0 1 2; do
      tag="$(echo "$variant" | tr -c 'A-Za-z0-9' '_')seed${seed}"
      python -m gsjepa.run "configs/${exp}.yaml" --variants "$variant" --seeds "$seed" \
        > "logs/${exp}_${tag}.log" 2>&1 &
    done
  done
done
wait

python -m gsjepa.run configs/exp1.yaml --aggregate > logs/exp1_aggregate.log 2>&1
python -m gsjepa.run configs/exp2.yaml --aggregate > logs/exp2_aggregate.log 2>&1
echo "done: see results/exp{1,2,3}/report.md"
