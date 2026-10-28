## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 1 (span 0.5859) | ≤ 10 frames (= 0.2020 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 287.1 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 1 (span 0.5859) | ≤ 10 frames (= 0.2020 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 291.9 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.5859 time span, 30 points | ≤ 0.2020 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 286.4 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.5859 time span, 30 points | ≤ 0.2020 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 266.4 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 26.43 | 26.04 | 0.9376 | 0.0541 | 12.91 | 0.2725 | 0.5260 |
| **ODE-GS** | 26.42 | 26.06 | 0.9393 | 0.0517 | 12.76 | 0.2660 | 0.5215 |
| Group-rigid (k=1, chosen on inner split) | 26.28 | 25.96 | 0.9316 | 0.0520 | 13.20 | 0.2939 | 0.4979 |
| Constant velocity (k=1, chosen on inner split) | 25.69 | 25.35 | 0.9247 | 0.0706 | 12.77 | 0.2580 | 0.5396 |
| Static (last prefix state) | 25.57 | 25.16 | 0.9386 | 0.0529 | 11.36 | 0.1879 | 0.5747 |
| Deformable-GS (Stage 1 queried at t) | 23.56 | 22.97 | 0.9329 | 0.0606 | 10.09 | 0.1815 | 0.6391 |
| _CV, all k (display only): k=1: 25.69, k=2: 25.49, k=4: 25.09, k=8: 25.43, k=16: 25.14_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 30.17 | 29.72 | 0.9652 | 0.0326 | 15.90 | 0.4597 | 0.3726 |
| **ODE-GS** | 27.46 | 27.07 | 0.9493 | 0.0423 | 12.64 | 0.3055 | 0.4854 |
| Group-rigid (k=1, chosen on inner split) | 28.28 | 27.86 | 0.9506 | 0.0374 | 14.59 | 0.4160 | 0.3910 |
| Constant velocity (k=1, chosen on inner split) | 28.54 | 27.93 | 0.9524 | 0.0525 | 14.66 | 0.4221 | 0.4264 |
| Static (last prefix state) | 26.26 | 25.74 | 0.9583 | 0.0389 | 10.32 | 0.1587 | 0.5932 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 37.76 | 37.38 | 0.9897 | 0.0164 | 24.84 | 0.8876 | 0.1381 |
