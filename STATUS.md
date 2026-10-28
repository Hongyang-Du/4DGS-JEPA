# Project: 4DGS-JEPA

## Meta
- **Phase**: Experiment
- **Target**: TBD
- **Last Updated**: 2026-10-27
- **Updated By**: dgx session

## Current Focus
Controlled protocol-B benchmark done on HyperNeRF ×6 and D-NeRF ×8 (seed 0, shared Stage 1, last checkpoint).
Summaries: `results/{hypernerf,dnerf}/fair_summary_tables.md`, `results/{hypernerf,dnerf}/level2_summary.md`;
per-scene reports and videos in `results/<ds>/<scene>/fair/report.md`, `results/<ds>/<scene>/videos/`.

## Experiment Status
| Run | Config | Status | Key Metric | Notes |
|-----|--------|--------|-----------|-------|
| mutant fair (D-NeRF) | tuning J1-J6 / O1-O3 | stopped (user) | inner: JEPA 40.41, ODE-GS 31.66, CV2 38.60 | no final runs; archived in `results/dnerf/mutant/fair_tuning_20261025/report.md` |
| HyperNeRF Stage 1 ×6 | ODE-GS train_interpolation, 40k it, time_split 0.9 | done | see per-scene reports | slice-banana t>0.9 query 18.82 vs paper Deformable-GS 21.73 |
| slice-banana 30k | lr 3e-4, 4096 Gaussians/step | done | test: JEPA 19.70, ODE-GS 18.74, Static 18.70, Deform 18.87 | superseded by 90k |
| LR check (slice-banana inner, 30k) | lr 1e-3 vs 3e-4 | done | ODE-GS inner 17.44 vs 16.90; JEPA 17.45 vs 16.98 | → lr 1e-3 for both |
| HyperNeRF ×6, 90k, lr 1e-3 | full + inner runs, seed 0 | done | test PSNR: JEPA 19.00, ODE-GS 19.00; dyn PSNR 13.41 vs 12.80 | `results/hypernerf/fair_summary.md` |
| Eval cross-check | ODE-GS official `evaluate_deform_model` on our Stage 1 | done | identical to our eval (18.56) | gap to paper is Stage 1, not eval |
| Group-rigid baseline (HyperNeRF ×6, mutant) | 32 k-means groups (= JEPA's), Kabsch over k frames, k on inner split | done | HyperNeRF test PSNR 16.36 (CV 16.20, Static 18.55); mutant 30.65 | in `fair/baselines/metrics.json` |
| Level II diagnostics (HyperNeRF ×6) | `python -m gsjepa.real.consistency` | done | inner ADE_dyn/Static: JEPA 0.999, ODE-GS 0.989, rigid 2.04, CV 2.86 | `results/hypernerf/level2_summary.md` |
| D-NeRF Stage 1 ×7 (+ mutant reused) | ODE-GS train_interpolation, 40k it, time_split 0.8, --is_blender | done | | `logs/stage1_<scene>.log` |
| D-NeRF ×8, 90k, lr 1e-3 | full + inner runs (32), seed 0, horizon 0.2 | done | test PSNR: JEPA 26.53, ODE-GS 24.03, Group-rigid 26.68, CV 26.36, Static 24.95 | `results/dnerf/fair_summary_tables.md`; 2 failure cases, see Key Results |
| Level II diagnostics (D-NeRF ×8) | `python -m gsjepa.real.consistency` | done | inner ADE_dyn/Static: JEPA 0.512, ODE-GS 0.673, rigid 0.481, CV 0.489 | `results/dnerf/level2_summary.md` |

## Key Results
Test PSNR / dyn-region PSNR, means over scenes, last checkpoint, seed 0:

| method | HyperNeRF ×6 | dyn | D-NeRF ×8 | dyn | D-NeRF ×6 (excl. bouncingballs, jumpingjacks) |
|---|---|---|---|---|---|
| 4DGS-JEPA | 19.00 | 13.41 | 26.53 | 13.51 | 27.81 |
| ODE-GS | 19.00 | 12.80 | 24.03 | 11.47 | 27.59 |
| Deformable-GS query | 18.56 | 12.74 | 24.06 | 10.53 | 23.71 |
| Static | 18.55 | 11.88 | 24.95 | 10.34 | 25.31 |
| Group-rigid | 16.36 | 11.26 | 26.68 | 13.49 | 27.61 |
| Constant velocity | 16.20 | 11.48 | 26.36 | 13.53 | 27.43 |

- HyperNeRF: JEPA ties ODE-GS on full-image PSNR, +0.6 dB on the dynamic region; simple extrapolators fall below Static.
- D-NeRF: simple extrapolators are strong (Group-rigid has the best mean). JEPA wins hellwarrior, lego, standup.
- D-NeRF failure cases (reported as is, per protocol):
  - ODE-GS bouncingballs full (6.12 dB): training collapsed at ~23k steps (pred loss 0.007 → 0.186 constant,
    latent reg ~1e-15, renders identical to the untrained model). Only 1 of 28 ODE-GS runs; the official code's
    best-test-PSNR checkpoint selection would hide it. Without both failure scenes JEPA leads ODE-GS by 0.2 dB
    (27.81 vs 27.59), i.e. a tie with Group-rigid (27.61) and CV (27.43).
  - JEPA jumpingjacks full (16.84 dB): best method for h ≤ 7 frames, then the whole body translates in the
    horizontal plane (ADE 1.84 at h = 40 vs Static 0.024); also in direct mode (17.23). Test PSNR drops with training
    (22.8 @18k → 16.8 @90k) while prefix val rises (32 → 37); the inner run is stable, so inner-split selection
    would not catch it. The group-translation bound (2·vmax·h, vmax = 7× vel_scale here) does not limit it.
    Diagnostic: `logs/_diag_jj.py`.
- Level II D-NeRF (inner): JEPA ADE_dyn/Static 0.512 (ODE-GS 0.673, rigid 0.481, CV 0.489); JEPA rollout accel/jerk
  2.4×/3.4× Stage 1 (direct 1.2×/0.9×). Composition: no divergence on D-NeRF; seg1/8 chains drift at long horizon
  on the full split (centre dist 2.1× forecast motion).

## Decisions Made
- 2026-10-25: Fair benchmark protocol (same Stage 1, inner-split tuning, last checkpoint, matched params ~4M) set up
  on D-NeRF mutant. Budget there: 90 A100-min/run.
- 2026-10-25: (user) mutant judged uninformative (near-constant-velocity motion) → wrap up early, move to HyperNeRF.
- 2026-10-25: (user) No tuning; controlled-variable comparison instead. Same steps, ~same params (JEPA 4.08M,
  ODE-GS 3.88M), AdamW + 5% warmup + cosine, same LR, 4096 Gaussian samples/step (ODE-GS batch 4096; JEPA
  4 contexts × 1024), the same 30 observation time points (JEPA context 30 frames with stride; ODE-GS
  obs_time_span = the same span), training horizon ≈ 0.098, seed 0, last checkpoint.
- 2026-10-25: Dynamic-region mask = |GT_t − Stage-1 static render at the forecast origin| > 0.2, then opening r2 and
  dilation 4 px, computed once per scene. A prefix-only Gaussian-motion mask was tried and rejected: Stage 1
  under-models hand motion, so the mask missed the hands.
- 2026-10-25: ODE-GS stage-2 windows moved to host memory; the GPU copy needed ~64 GB on slice-banana.
- 2026-10-25: (user) Raise stage 2 to 90k steps for all 6 scenes, including a rerun of slice-banana. Switch the LR
  only if the inner split shows ODE-GS clearly better at 1e-3. It did (+0.54 dB, SSIM 0.50 → 0.59), so both now use
  1e-3. The test split (ODE-GS full run: 1e-3 18.26 < 3e-4 18.74) was not consulted for this decision.
- 2026-10-25: Stage 1 checked against ODE-GS paper/code: 40k iterations, rgb/2x (interp/misc), rgb/4x (vrig), with
  the camera subset chosen by folder name. Identical code and defaults, so the 2–3 dB absolute gap to the paper is
  not a settings mismatch we can fix.

- 2026-10-26: Our eval reproduces ODE-GS's own `evaluate_deform_model` exactly on all 6 HyperNeRF Stage 1s.
  The official ODE-GS training saves the checkpoint with the best PSNR on the t > split test cameras; we keep last.
- 2026-10-26: (user) Run all 8 D-NeRF scenes, 1 seed, fully apple-to-apple with HyperNeRF: same model sizes
  (JEPA 4.08M / ODE-GS 3.88M), 90k steps, 4096 samples/step, AdamW lr 1e-3, same schedule. Only change: training
  horizon 0.2 (= D-NeRF test range and ODE-GS's own D-NeRF max_extrap_time_span) instead of 0.098; observation
  window unchanged (30 points, span ≈ 0.354). ODE-GS regulariser uses its official D-NeRF config.
  Old mutant tuning runs archived in `results/dnerf/mutant/fair_tuning_20261025`.
- 2026-10-26: Fixed dynamic-mask keys (D-NeRF reuses image names across train/test → key = name@fid) and
  excluded empty masks from dynamic averages. HyperNeRF baselines re-checked: identical numbers.

- 2026-10-26: (user) Added proposal items 1 (group-rigid baseline) and 2 (Level II consistency). Level II on
  HyperNeRF: no learned method beats Static on held-out geometry (inner split, Stage-1 states as reference);
  JEPA has the lowest error growth and keeps observed-like deformation (distortion ≈ 0.9× Stage 1, ODE-GS ≈ 0.6×).
  JEPA composition: stable for partitions ≥ 1/4 of the training horizon and random 2-6 piece partitions
  (composed ≈ direct target error), but chains of 1/8-horizon steps diverge at long horizons (3/6 scenes inner).
  Partitions are defined relative to the JEPA's max training horizon because frame counts differ per scene.
- 2026-10-27: (user) D-NeRF failure cases (ODE-GS bouncingballs collapse, JEPA jumpingjacks drift) kept as is: no
  reruns or extra seeds, no method changes; report the 8-scene means with a footnote giving the 6-scene means.
- 2026-10-27: Fixed `fair_finalize_scene.sh` for D-NeRF: pass `--scene` and the dataset parent dir to `videos.py`
  (it defaulted to mutant). Metrics unaffected.

## Open Questions
- [ ] Why is the absolute PSNR on slice-banana 2–3 dB below the paper (Stage-1 seed/variance, or their eval)?
- [ ] On slice-banana's inner window, every learned method stays below Static / CV, while on test JEPA is +1 dB over
      Static. Is the 0.8–0.9 window atypical?

## Next Steps
- [x] D-NeRF: 32 runs (16 full + 16 inner) → finalize each scene → `scripts/fair_summary.py --dataset dnerf`
- [x] D-NeRF: run `consistency scene` per scene after its runs finish, then `consistency summary --dataset dnerf`
- [ ] JEPA long-horizon drift (jumpingjacks): test PSNR falls with training; consider a tighter translation bound or
      drift regulariser, validated on the inner split only
- [ ] JEPA composition diverges for fine partitions: check whether training-time composition covers short segments
- [x] GaussianPredict / TiNeuVox: not trained (user, 2026-10-26). Baselines = ODE-GS, Deformable-GS query, Static, CV
