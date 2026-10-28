## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3893) | ≤ 30 frames (= 0.2013 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 305.6 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3893) | ≤ 30 frames (= 0.2013 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 299.8 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3893 time span, 30 points | ≤ 0.2013 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 306.2 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.3893 time span, 30 points | ≤ 0.2013 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 296.1 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 30.23 | 29.93 | 0.9666 | 0.0200 | 12.38 | 0.3390 | 0.2727 |
| **ODE-GS** | 30.57 | 30.30 | 0.9705 | 0.0196 | 12.80 | 0.3951 | 0.2648 |
| Group-rigid (k=2, chosen on inner split) | 30.65 | 30.36 | 0.9667 | 0.0191 | 12.79 | 0.3583 | 0.2566 |
| Constant velocity (k=2, chosen on inner split) | 30.58 | 30.29 | 0.9657 | 0.0209 | 12.83 | 0.3618 | 0.2696 |
| Static (last prefix state) | 26.83 | 26.51 | 0.9484 | 0.0315 | 8.78 | 0.0959 | 0.4669 |
| Deformable-GS (Stage 1 queried at t) | 24.48 | 24.22 | 0.9299 | 0.0477 | 7.75 | 0.0838 | 0.5950 |
| _CV, all k (display only): k=1: 29.01, k=2: 30.58, k=4: 33.68, k=8: 34.29, k=16: 32.08_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 39.65 | 39.52 | 0.9923 | 0.0074 | 22.18 | 0.8416 | 0.0664 |
| **ODE-GS** | 34.23 | 33.91 | 0.9837 | 0.0113 | 15.43 | 0.5909 | 0.1637 |
| Group-rigid (k=2, chosen on inner split) | 39.06 | 38.89 | 0.9911 | 0.0073 | 21.75 | 0.8100 | 0.0667 |
| Constant velocity (k=2, chosen on inner split) | 38.60 | 38.40 | 0.9903 | 0.0083 | 20.87 | 0.7838 | 0.0863 |
| Static (last prefix state) | 28.75 | 28.46 | 0.9593 | 0.0240 | 8.69 | 0.0895 | 0.4604 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 45.03 | 45.00 | 0.9967 | 0.0056 | 31.51 | 0.9714 | 0.0302 |
