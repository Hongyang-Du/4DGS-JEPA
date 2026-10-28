## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3893) | ≤ 30 frames (= 0.2013 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 293.4 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3893) | ≤ 30 frames (= 0.2013 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 196.3 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3893 time span, 30 points | ≤ 0.2013 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 292.3 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.3893 time span, 30 points | ≤ 0.2013 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 213.1 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 30.62 | 30.47 | 0.9595 | 0.0369 | 18.82 | 0.6358 | 0.2371 |
| **ODE-GS** | 29.34 | 29.18 | 0.9554 | 0.0390 | 16.71 | 0.5661 | 0.2869 |
| Group-rigid (k=8, chosen on inner split) | 28.23 | 28.10 | 0.9522 | 0.0386 | 15.97 | 0.5371 | 0.2845 |
| Constant velocity (k=8, chosen on inner split) | 28.13 | 28.01 | 0.9517 | 0.0423 | 16.00 | 0.5382 | 0.3088 |
| Static (last prefix state) | 24.33 | 24.22 | 0.9331 | 0.0590 | 10.78 | 0.2105 | 0.5279 |
| Deformable-GS (Stage 1 queried at t) | 21.38 | 21.28 | 0.9120 | 0.0851 | 9.15 | 0.1855 | 0.6398 |
| _CV, all k (display only): k=1: 30.86, k=2: 31.19, k=4: 29.14, k=8: 28.13, k=16: 28.06_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 27.83 | 27.73 | 0.9539 | 0.0381 | 14.68 | 0.4546 | 0.3232 |
| **ODE-GS** | 29.63 | 29.48 | 0.9622 | 0.0310 | 15.97 | 0.5439 | 0.2771 |
| Group-rigid (k=8, chosen on inner split) | 30.08 | 29.98 | 0.9638 | 0.0263 | 16.84 | 0.5743 | 0.2252 |
| Constant velocity (k=8, chosen on inner split) | 30.07 | 29.98 | 0.9640 | 0.0274 | 16.92 | 0.5765 | 0.2330 |
| Static (last prefix state) | 24.50 | 24.41 | 0.9390 | 0.0518 | 10.41 | 0.1927 | 0.5264 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 46.05 | 45.77 | 0.9957 | 0.0089 | 34.14 | 0.9659 | 0.0482 |
