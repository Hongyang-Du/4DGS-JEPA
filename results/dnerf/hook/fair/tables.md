## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 1 (span 0.2929) | ≤ 20 frames (= 0.2020 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 199.6 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 1 (span 0.2929) | ≤ 20 frames (= 0.2020 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 291.7 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.2929 time span, 30 points | ≤ 0.2020 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 217.6 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.2929 time span, 30 points | ≤ 0.2020 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 291.4 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 26.95 | 26.77 | 0.9353 | 0.0423 | 13.93 | 0.3663 | 0.3378 |
| **ODE-GS** | 27.54 | 27.35 | 0.9441 | 0.0396 | 14.24 | 0.4328 | 0.3167 |
| Group-rigid (k=1, chosen on inner split) | 27.65 | 27.47 | 0.9384 | 0.0370 | 14.66 | 0.4030 | 0.2950 |
| Constant velocity (k=1, chosen on inner split) | 27.49 | 27.31 | 0.9351 | 0.0447 | 14.57 | 0.3821 | 0.3395 |
| Static (last prefix state) | 23.97 | 23.78 | 0.9103 | 0.0586 | 10.49 | 0.1159 | 0.4682 |
| Deformable-GS (Stage 1 queried at t) | 21.77 | 21.59 | 0.8878 | 0.0829 | 9.34 | 0.1159 | 0.5357 |
| _CV, all k (display only): k=1: 27.49, k=2: 28.10, k=4: 27.86, k=8: 27.53, k=16: 26.38_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 32.29 | 32.11 | 0.9665 | 0.0228 | 19.62 | 0.7016 | 0.1667 |
| **ODE-GS** | 31.17 | 31.00 | 0.9663 | 0.0237 | 18.22 | 0.6667 | 0.1831 |
| Group-rigid (k=1, chosen on inner split) | 31.56 | 31.38 | 0.9641 | 0.0233 | 18.65 | 0.6625 | 0.1756 |
| Constant velocity (k=1, chosen on inner split) | 31.25 | 31.07 | 0.9625 | 0.0275 | 18.42 | 0.6540 | 0.2041 |
| Static (last prefix state) | 24.68 | 24.50 | 0.9132 | 0.0510 | 10.92 | 0.1017 | 0.4357 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 42.97 | 42.82 | 0.9953 | 0.0103 | 31.77 | 0.9644 | 0.0640 |
