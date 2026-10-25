# 4DGS-JEPA

Clean, tested re-implementation of the controlled experiments in
**4DGS-JEPA: Temporally Compositional Joint-Embedding Prediction for Dynamic Gaussian Splatting**
([arXiv:2609.25036](https://arxiv.org/abs/2609.25036)).

The original Colab notebooks are kept in [`notebooks/`](notebooks/). The code in `src/gsjepa/` is a refactor of
those notebooks into a small package, verified to be numerically equivalent to them (see
[Equivalence with the notebooks](#equivalence-with-the-original-notebooks)), and reproduces every number in
Tables 2, 3, 4, 7, 8, 10 and 11 of the paper.

## Code structure

```
4DGS-JEPA/
├── configs/                  # one YAML per experiment = the paper's exact settings
│   ├── exp1.yaml             #   Exp1: path supervision + temporal composition (Table 2, 7)
│   ├── exp2.yaml             #   Exp2: geometry-level composition (Table 3, 8)
│   └── exp3.yaml             #   Exp3: hybrid correspondence stress test (Table 4, 10, 11)
├── src/gsjepa/
│   ├── config.py             # dataclass configs + YAML loader with key=value overrides
│   ├── geometry.py           # SO(3)/SE(3) maps, SE(3) distance, log-Cholesky, affine-invariant SPD distance
│   ├── data.py               # procedural dynamic Gaussian worlds (eq. 64-65, Table 5) + Dataset
│   ├── model.py              # causal features, hierarchical encoder, transition Φ, geometry decoder Q, EMA
│   ├── losses.py             # D_hier, L_comp, l_geo / D_match, VICReg-style L_rep
│   ├── rollout.py            # causal history slicing, direct prediction, composed rollout
│   ├── train.py              # shared training loop for Exp1/Exp2 (one objective, toggled per variant)
│   ├── correspondence.py     # index / count-aware OT / hybrid matching + perturbations + metrics
│   ├── experiments/
│   │   ├── exp1.py           #   Exp1 evaluation (FDE, Δroll, CompErr, GeoCompErr, held-out partitions)
│   │   ├── exp2.py           #   Exp2 evaluation (cov err, E_T^target, E_T^dc, D_match)
│   │   └── exp3.py           #   Exp3 stress test, bootstrap CIs, ID-dropout
│   ├── paper_numbers.py      # numbers reported in the paper, used in the comparison reports
│   └── run.py                # single CLI entry point
├── tests/                    # pytest: geometry, causality/data, training step, correspondence
├── scripts/
│   ├── run_all.sh            # reproduce all three experiments (21 training runs in parallel)
│   └── check_equivalence.py  # compare against the original notebook code
├── docker/                   # Dockerfile + run_docker.sh
├── notebooks/                # original Colab notebooks (incl. exp4, which is not in the paper)
└── results/                  # reproduction outputs: report.md + summary CSVs per experiment
```

How the paper maps to the code:

| Paper | Code |
|---|---|
| Hierarchical state Z = (scene, groups, Gaussians), eq. 9 | `model.HierEncoder` (token layout `[1 + G + N, D]`) |
| Horizon-conditioned transition Φ(Z, Δ), eq. 12 | `model.Transition` |
| Selective geometry decoder Q, eqs. 28-32 | `model.GeometryDecoder` |
| EMA teacher, eq. 10 | `model.make_teacher`, `model.ema_update` |
| D_hier, eq. 15 | `losses.hier_distance` |
| L_end / L_path / L_comp / L_geo / L_gcomp / L_rep, eqs. 16, 21, 23, 39, 43, 48 | `train.compute_loss` |
| Composed rollout + SE(3) composition, eqs. 20, 36, 40 | `rollout.rollout` |
| Hybrid correspondence, eqs. 56-61 | `correspondence.hybrid_plan`, `correspondence.ot_plan` |

Each experiment variant is a set of on/off switches in its YAML file, e.g.
`Endpoint + Path: {path: true}` or `Full geometry composition: {comp: true, gcomp_se3: true, gcomp_match: true}`.

## Setup

```bash
GPU=0 bash docker/run_docker.sh       # builds the image (nvcr.io/nvidia/pytorch:25.04-py3) and starts a container
docker exec -it 4dgs-jepa bash        # repo is mounted at /workspace/4DGS-JEPA, PYTHONPATH already set
```

Without Docker: `pip install -e ".[dev]"` (needs torch, numpy, pandas, pyyaml).

## Usage

```bash
python -m pytest -q                                 # 13 tests, a few seconds on CPU
bash scripts/run_all.sh                             # everything; writes results/exp{1,2,3}/report.md

python -m gsjepa.run configs/exp1.yaml              # all variants x seeds of one experiment
python -m gsjepa.run configs/exp1.yaml --variants "Endpoint + Composition" --seeds 0
python -m gsjepa.run configs/exp1.yaml --aggregate  # rebuild tables from finished runs
python -m gsjepa.run configs/exp3.yaml              # Exp3, CPU only
python -m gsjepa.run configs/exp1.yaml --set epochs=1 train_scenes=16 out_dir=/tmp/smoke   # smoke test
```

Every (variant, seed) run writes `results/<exp>/runs/<variant>_seed<k>/` with `history.csv`, `summary.json`,
`per_sample.csv` and `best.pt` (checkpoints and per-sample CSVs are git-ignored).

## Reproduction results

Setup: one NVIDIA A100-80GB (the paper also used an A100), 3 seeds per variant, the paper's exact
hyper-parameters (Tables 5, 6, 9). Training one run takes about 8–11 min when all 21 runs share one GPU.
Each cell shows **reproduced** mean ± std over seeds, with the paper's number below it. Full reports:
[`results/exp1/report.md`](results/exp1/report.md), [`results/exp2/report.md`](results/exp2/report.md),
[`results/exp3/report.md`](results/exp3/report.md).

### Experiment 1: path supervision and temporal composition (Table 2)

| Method | Direct FDE ↓ | Rollout FDE ↓ | Δroll ↓ | CompErr ↓ | GeoCompErr ↓ | Unseen FDE ↓ |
|---|---|---|---|---|---|---|
| Endpoint | **0.03024±0.00048**<br>0.03013±0.00042 | **0.03069±0.00066**<br>0.03061±0.00065 | **0.00045±0.00021**<br>0.00048±0.00023 | **0.00247±0.00057**<br>0.00244±0.00055 | **0.00733±0.00079**<br>0.00739±0.00082 | **0.03632±0.00085**<br>0.03621±0.00080 |
| Endpoint + Path | **0.03204±0.00247**<br>0.03195±0.00243 | **0.03245±0.00248**<br>0.03228±0.00256 | **0.00042±0.00008**<br>0.00033±0.00014 | **0.00067±0.00011**<br>0.00065±0.00007 | **0.00652±0.00069**<br>0.00649±0.00069 | **0.03859±0.00294**<br>0.03838±0.00304 |
| Endpoint + Composition | **0.02909±0.00089**<br>0.02905±0.00095 | **0.02945±0.00095**<br>0.02940±0.00101 | **0.00036±0.00012**<br>0.00035±0.00014 | **0.00096±0.00011**<br>0.00096±0.00011 | **0.00666±0.00071**<br>0.00667±0.00074 | **0.03474±0.00112**<br>0.03466±0.00122 |
| Endpoint + Path + Composition | **0.03192±0.00241**<br>0.03206±0.00245 | **0.03231±0.00243**<br>0.03236±0.00255 | **0.00039±0.00005**<br>0.00029±0.00013 | **0.00060±0.00007**<br>0.00058±0.00003 | **0.00651±0.00086**<br>0.00653±0.00089 | **0.03828±0.00270**<br>0.03837±0.00283 |

Baselines (seen-horizon mean FDE): static persistence 0.14802 (paper 0.14802), constant velocity 0.02484 (paper 0.02484).
At Δ=8, Endpoint + Composition reaches direct FDE 0.05689 vs 0.07119 for constant velocity (paper: 0.05687 vs 0.07120).

Held-out temporal decompositions (Table 7):

| Method | Held-out CompErr ↓ | Held-out GeoCompErr ↓ | Unseen-seg CompErr ↓ | Unseen-seg GeoCompErr ↓ |
|---|---|---|---|---|
| Endpoint | **0.01151±0.00257**<br>0.01134±0.00244 | **0.01420±0.00149**<br>0.01428±0.00160 | **0.00239±0.00050**<br>0.00235±0.00047 | **0.00893±0.00118**<br>0.00901±0.00120 |
| Endpoint + Path | **0.00292±0.00025**<br>0.00286±0.00013 | **0.01122±0.00157**<br>0.01105±0.00146 | **0.00066±0.00009**<br>0.00064±0.00006 | **0.00786±0.00121**<br>0.00778±0.00119 |
| Endpoint + Composition | **0.00453±0.00051**<br>0.00454±0.00049 | **0.01197±0.00122**<br>0.01204±0.00125 | **0.00096±0.00011**<br>0.00096±0.00011 | **0.00801±0.00106**<br>0.00803±0.00110 |
| Endpoint + Path + Composition | **0.00269±0.00016**<br>0.00260±0.00009 | **0.01108±0.00164**<br>0.01096±0.00157 | **0.00060±0.00007**<br>0.00057±0.00002 | **0.00781±0.00138**<br>0.00782±0.00141 |

### Experiment 2: geometry-level composition at Δ=8 (Table 3)

| Method | Direct FDE ↓ | Direct cov. err. ↓ | E_T^target ↓ | E_T^dc ↓ | D_match ↓ |
|---|---|---|---|---|---|
| Latent composition only | **0.06559±0.00318**<br>0.06614±0.00271 | **0.19858±0.00434**<br>0.19897±0.00387 | **0.26045±0.01090**<br>0.26124±0.01038 | **0.00726±0.00049**<br>0.00724±0.00057 | **0.00195±0.00010**<br>0.00192±0.00015 |
| +SE(3) composition | **0.06679±0.00440**<br>0.06675±0.00408 | **0.19802±0.00295**<br>0.19758±0.00330 | **0.25705±0.00183**<br>0.25554±0.00119 | **0.00206±0.00037**<br>0.00202±0.00031 | **0.00141±0.00034**<br>0.00138±0.00036 |
| Full geometry composition | **0.06734±0.00401**<br>0.06728±0.00388 | **0.19772±0.00312**<br>0.19758±0.00290 | **0.25578±0.00136**<br>0.25582±0.00130 | **0.00216±0.00031**<br>0.00235±0.00062 | **0.00123±0.00013**<br>0.00130±0.00012 |

Baselines at Δ=8: static FDE 0.32158, constant-velocity FDE 0.07119, identity-transform E_T^target 0.34170
(paper 0.32158 / 0.07120 / 0.34170). Horizon-wise numbers (Table 8) are in `results/exp2/report.md`.

### Experiment 3: hybrid correspondence (Table 4)

Exp3 is NumPy-only and deterministic, so it reproduces the paper **exactly** (every digit of Tables 4, 10 and 11):

| Method | Permutation lineage ↑ | Split/merge lineage ↑ | Birth/death F1 ↑ | Geometry discrepancy ↓ |
|---|---|---|---|---|
| Index matching | 0.04056±0.00507 | 0.15563±0.00907 | 0.61134±0.00381 | 0.74311±0.00449 |
| OT only | 0.99789±0.00024 | 0.98100±0.00099 | 1.00000±0.00000 | 0.05010±0.00014 |
| Hybrid | 1.00000±0.00000 | 0.98715±0.00103 | 1.00000±0.00000 | 0.04906±0.00013 |

### Summary

- **Exp3**: identical to the paper.
- **Exp1 / Exp2**: 53 of the 55 reproduced means are within one paper standard deviation; the other two are
  within 1.3σ (Exp1 P+C unseen-segment CompErr 0.00060 vs 0.00057±0.00002; Exp2 +SE(3) E_T^target 0.25705 vs
  0.25554±0.00119). The data-only baselines match exactly. All qualitative conclusions hold: composition lowers CompErr without hurting FDE;
  Path + Composition gives the lowest CompErr; Endpoint + Composition beats constant velocity at Δ=8;
  SE(3) composition cuts E_T^dc by ~72%; the full objective gives the lowest D_match.
- The remaining differences come from GPU non-determinism (TF32 matmuls, CUDA kernels), not from the code:
  see the next section.

## Equivalence with the original notebooks

`scripts/check_equivalence.py` runs the original notebook cells and the refactored package on the same small
config and compares the results:

```bash
CUDA_VISIBLE_DEVICES= python scripts/check_equivalence.py exp1 "Endpoint + Path + Composition"
CUDA_VISIBLE_DEVICES= python scripts/check_equivalence.py exp2 "Full geometry composition"
python scripts/check_equivalence.py exp3
```

On CPU, the training curves and test metrics agree to ≤3e-7 (float32 round-off) for Exp1/Exp2 and exactly
for Exp3. On GPU the two implementations differ by up to ~2e-4 (relative) after two epochs, because TF32 matmuls
and CUDA kernels are not bit-deterministic; this is the source of the small differences from the paper above.

## Notes

- `notebooks/4DGS_JEPA_exp4_video_prediction.ipynb` (rendered next-frame prediction) is not part of the paper
  and is kept only as the original notebook.
- As in the paper, Exp1/Exp2 use known persistent correspondence (identity alignment), and existence is fixed to 1;
  structural priors (eqs. 44-47) are described in the paper but not used in any experiment, so they are not implemented.
