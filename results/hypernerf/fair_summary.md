# HyperNeRF (6 scenes): controlled stage-2 comparison, 4DGS-JEPA vs ODE-GS on a shared Stage 1

Seed 0, last checkpoint, one run per method and scene. Per-scene details (config table, curves, masks, videos):
`results/hypernerf/<scene>/fair/report.md`. Generated tables: `fair_summary_tables.md` / `fair_summary.json`
(`scripts/fair_summary.py`).

## Protocol (identical for all 6 scenes)

- **Split**: ODE-GS protocol B with time_split 0.9. Cameras are read by ODE-GS `readNerfiesCameras`, which picks
  the subset and resolution from the parent folder: interp (slice-banana, cut-lemon1, chickchicken) = ids[::4] + ids[2::4]
  at rgb/2x; misc (split-cookie, keyboard) = all ids at rgb/2x; vrig (3dprinter) = train + val ids at rgb/4x.
  They are sorted by time; the first 90% form the prefix and the rest are test cameras.
- **Shared Stage 1 per scene**: Deformable-GS trained with ODE-GS `train_interpolation.py --time_split 0.9`, with
  ODE-GS defaults (40k iterations, the same as the paper). It was checked against the paper and code: same iterations,
  resolution and camera subset, so there was nothing to align. Deformable-GS query, Static, constant velocity (CV),
  ODE-GS and JEPA all start from this one Stage 1.
- **Controlled stage 2, the same for JEPA and ODE-GS**:

  | setting | value |
  |---|---|
  | optimizer steps | 90k |
  | Gaussian samples per step | 4096 (JEPA: 4 contexts × 1024) |
  | observation | the same 30 Stage-1 time points, span ≈ 0.354 |
  | training horizon | ≈ 0.098 |
  | optimizer | AdamW, LR 1e-3, weight decay 1e-4, 5% warmup then cosine to 0.01×, grad clip 1.0 |
  | params | 4.08M vs 3.88M (5% apart) |
  | seed / checkpoint | seed 0, last checkpoint |

  The per-scene stride and horizon (in frames) are derived from that scene's frame spacing; see each report's
  config table.
- **LR decision**: made on slice-banana's inner split (stage 2 trained on t ≤ 0.8, scored on prefix cameras
  0.8 < t ≤ 0.9, 30k steps). ODE-GS gained +0.54 dB and SSIM 0.50 → 0.59 with 1e-3 over 3e-4; JEPA gained +0.47 dB.
  Both therefore use 1e-3. The test split was not consulted (it would have favoured 3e-4 for ODE-GS).
- **CV window k** was chosen per scene on its inner split from {1, 2, 4, 8, 16}.
- **Dynamic-region mask** (computed once per scene, the same for every method): pixels where max_c |GT_t − R_t0| > 0.2,
  with R_t0 the Stage-1 render of the state at the forecast origin from the same camera; then opening (radius 2)
  and dilation (4 px). It marks where the scene changed, so it penalises Static by construction.

## Results on the test cameras (t > 0.9)

### PSNR ↑ (full image)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | **19.46** | 19.74 | **18.66** | **20.35** | 18.64 | **17.15** | **19.00** |
| ODE-GS (controlled, 90k) | 19.09 | 20.36 | 18.65 | 19.42 | **19.94** | 16.55 | **19.00** |
| Deformable-GS (Stage 1 queried) | 18.87 | 19.99 | 17.78 | 18.33 | 19.45 | 16.93 | 18.56 |
| Static (last prefix state) | 18.70 | **20.73** | 17.64 | 18.97 | 18.55 | 16.74 | 18.55 |
| Constant velocity (k chosen on inner split) | 17.08 | 18.02 | 14.59 | 17.12 | 14.88 | 15.48 | 16.20 |
| _paper ref: ODE-GS (ODE-GS Table 3)_ | 21.29 | 21.69 | 20.29 | 20.72 | 21.06 | 20.22 | 20.88 |
| _paper ref: Deformable-GS (ODE-GS Table 3)_ | 21.73 | 21.36 | 19.03 | 17.84 | 19.98 | 19.89 | 19.97 |

### PSNR ↑ (dynamic-region mask)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | **14.40** | 11.81 | **12.39** | **14.33** | 13.55 | **13.98** | **13.41** |
| ODE-GS (controlled, 90k) | 13.63 | 11.84 | 11.91 | 12.89 | 14.20 | 12.34 | 12.80 |
| Deformable-GS (Stage 1 queried) | 13.83 | **12.40** | 10.86 | 11.70 | **14.34** | 13.34 | 12.74 |
| Static (last prefix state) | 12.79 | 11.95 | 10.29 | 12.01 | 12.43 | 11.81 | 11.88 |
| Constant velocity (k chosen on inner split) | 12.84 | 11.61 | 9.78 | 11.43 | 11.71 | 11.49 | 11.48 |

### SSIM ↑ (full image)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.5808 | 0.5943 | 0.6645 | 0.6306 | 0.6438 | 0.4723 | 0.5977 |
| ODE-GS (controlled, 90k) | 0.5760 | 0.6115 | 0.6597 | 0.6021 | 0.6839 | 0.4433 | 0.5961 |
| Deformable-GS (Stage 1 queried) | 0.5881 | 0.6016 | 0.6450 | 0.6006 | 0.6630 | 0.4579 | 0.5927 |
| Static (last prefix state) | 0.5884 | 0.6290 | 0.6400 | 0.6009 | 0.6361 | 0.4580 | 0.5921 |
| Constant velocity (k chosen on inner split) | 0.5509 | 0.5358 | 0.5775 | 0.5250 | 0.5118 | 0.4115 | 0.5188 |
| _paper ref: ODE-GS (ODE-GS Table 3)_ | 0.6437 | 0.6964 | 0.7254 | 0.6593 | 0.7399 | 0.6946 | 0.6932 |
| _paper ref: Deformable-GS (ODE-GS Table 3)_ | 0.6530 | 0.6950 | 0.6996 | 0.5698 | 0.6957 | 0.6775 | 0.6651 |

### LPIPS ↓ (full image)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.4788 | 0.4117 | 0.3696 | 0.2389 | 0.3041 | 0.3914 | 0.3658 |
| ODE-GS (controlled, 90k) | 0.4804 | 0.3814 | 0.3851 | 0.2737 | 0.2782 | 0.4074 | 0.3677 |
| Deformable-GS (Stage 1 queried) | 0.5119 | 0.4064 | 0.3980 | 0.2840 | 0.2972 | 0.3955 | 0.3822 |
| Static (last prefix state) | 0.4979 | 0.3800 | 0.4051 | 0.2786 | 0.3180 | 0.3934 | 0.3788 |
| Constant velocity (k chosen on inner split) | 0.5388 | 0.4964 | 0.5065 | 0.4238 | 0.5242 | 0.4624 | 0.4920 |
| _paper ref: ODE-GS (ODE-GS Table 3)_ | 0.3230 | 0.3098 | 0.3023 | 0.2406 | 0.2327 | 0.2242 | 0.2721 |
| _paper ref: Deformable-GS (ODE-GS Table 3)_ | 0.3241 | 0.3207 | 0.3167 | 0.2945 | 0.2497 | 0.2378 | 0.2906 |

### LPIPS ↓ (dynamic-region mask)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.5543 | 0.6005 | 0.4925 | 0.3557 | 0.4278 | 0.4181 | 0.4748 |
| ODE-GS (controlled, 90k) | 0.5726 | 0.5997 | 0.5300 | 0.4171 | 0.4018 | 0.4895 | 0.5018 |
| Deformable-GS (Stage 1 queried) | 0.5789 | 0.5900 | 0.5679 | 0.4671 | 0.4088 | 0.4522 | 0.5108 |
| Static (last prefix state) | 0.6103 | 0.6000 | 0.6055 | 0.4623 | 0.4836 | 0.5023 | 0.5440 |
| Constant velocity (k chosen on inner split) | 0.6289 | 0.6232 | 0.6606 | 0.5635 | 0.5955 | 0.5543 | 0.6044 |

CV window k chosen on each inner split: slice-banana 4, cut-lemon1 8, chickchicken 16, split-cookie 1, keyboard 4,
vrig-3dprinter 16.

Training cost (wall-clock GPU-minutes for 90k steps, evaluation excluded): JEPA 195–204 min, ODE-GS 196–264 min.
Up to 7 jobs shared the node, and ODE-GS's host-side window sampling is CPU-bound, so these numbers carry
contention noise.

## Reading

1. **Full image: a tie on average.** JEPA − ODE-GS PSNR per scene is +0.37 / −0.62 / +0.01 / +0.93 / −1.30 / +0.60,
   mean −0.00 dB (19.00 vs 19.00). JEPA is ahead on 3 scenes, ODE-GS on 2, and chickchicken is even. SSIM is
   also even (0.598 vs 0.596); LPIPS slightly favours JEPA (0.366 vs 0.368).
2. **Dynamic region: JEPA ahead.** JEPA leads ODE-GS on dynamic PSNR by +0.61 dB on average (13.41 vs 12.80) and
   is best on 4/6 scenes, with the largest gaps on vrig-3dprinter (+1.64) and split-cookie (+1.44). Dynamic LPIPS
   agrees (0.475 vs 0.502). ODE-GS is better only on keyboard (+0.65), and cut-lemon1 is even.
3. **Baselines.** Both learned methods beat Static by +0.45 dB on average. But Static is the best method on
   cut-lemon1 (20.73), where the remaining motion is small relative to the forecast error. JEPA beats Static on
   5/6 scenes and ODE-GS on 4/6. Querying Deformable-GS beyond its training range (18.56) is about as good as Static.
   Constant velocity is clearly worst (16.20): CV extrapolation of noisy Stage-1 trajectories diverges.
4. **Absolute level vs the paper.** Our numbers are 1.4–1.9 dB below the paper's (ODE-GS 19.00 vs 20.88;
   Deformable-GS query 18.56 vs 19.97). SSIM is ~0.1 lower for every method, including the Stage-1 query. We used
   the same code, iterations and resolution, so the gap appears to come from Stage-1 / evaluation-pipeline variance
   (and their checkpoint choice) rather than from stage 2. It affects every method in this table equally, since all
   share our Stage 1.

## Caveats

- Single seed (0) per method and scene; per-scene differences under ~0.5 dB should not be read as significant.
  Only the dynamic-region advantage of JEPA is consistent across most scenes.
- The dynamic mask comes from GT-vs-origin change, so Static and CV score badly on it by construction. It is a
  fair comparison between JEPA and ODE-GS, which see the same mask.
- The LR was set once, on slice-banana's inner split, and then used for all scenes. Neither method was tuned per
  scene.
- The inner-split ("held-out validation") curves come from the separate `*_inner` runs. See each report's
  `curve_val.png`.
