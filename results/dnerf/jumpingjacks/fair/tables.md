## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.2915) | ≤ 40 frames (= 0.2010 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 293.9 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.2915) | ≤ 40 frames (= 0.2010 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 289.2 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.2915 time span, 30 points | ≤ 0.2010 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 345.1 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.2915 time span, 30 points | ≤ 0.2010 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 240.6 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 16.84 | 16.50 | 0.7718 | 0.1881 | 9.29 | 0.2543 | 0.5893 |
| **ODE-GS** | 20.58 | 20.27 | 0.9170 | 0.0795 | 8.45 | 0.1232 | 0.6126 |
| Group-rigid (k=1, chosen on inner split) | 22.96 | 22.71 | 0.9324 | 0.0631 | 11.89 | 0.4130 | 0.4052 |
| Constant velocity (k=1, chosen on inner split) | 22.61 | 22.37 | 0.9184 | 0.0793 | 12.18 | 0.3864 | 0.4556 |
| Static (last prefix state) | 20.41 | 20.10 | 0.9164 | 0.0795 | 8.27 | 0.1063 | 0.6186 |
| Deformable-GS (Stage 1 queried at t) | 20.54 | 20.23 | 0.9144 | 0.0819 | 8.75 | 0.1690 | 0.5888 |
| _CV, all k (display only): k=1: 22.61, k=2: 22.63, k=4: 22.64, k=8: 22.42, k=16: 22.26_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 24.58 | 24.32 | 0.9309 | 0.0669 | 12.67 | 0.3760 | 0.4346 |
| **ODE-GS** | 21.69 | 21.41 | 0.9221 | 0.0748 | 9.55 | 0.1559 | 0.5716 |
| Group-rigid (k=1, chosen on inner split) | 27.56 | 27.34 | 0.9577 | 0.0361 | 15.84 | 0.5934 | 0.2499 |
| Constant velocity (k=1, chosen on inner split) | 27.25 | 27.05 | 0.9524 | 0.0452 | 15.80 | 0.5703 | 0.3021 |
| Static (last prefix state) | 21.55 | 21.27 | 0.9205 | 0.0754 | 9.37 | 0.1337 | 0.5795 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 42.11 | 41.91 | 0.9953 | 0.0087 | 30.67 | 0.9512 | 0.0624 |
