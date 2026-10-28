## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.2900) | ≤ 40 frames (= 0.2000 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 199.4 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.2900) | ≤ 40 frames (= 0.2000 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 195.8 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.2900 time span, 30 points | ≤ 0.2000 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 250.0 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.2900 time span, 30 points | ≤ 0.2000 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 227.1 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 21.14 | 21.13 | 0.9116 | 0.0753 | 10.60 | 0.2844 | 0.4548 |
| **ODE-GS** | 20.73 | 20.73 | 0.9404 | 0.0500 | 7.40 | 0.0920 | 0.5401 |
| Group-rigid (k=1, chosen on inner split) | 21.50 | 21.49 | 0.9434 | 0.0524 | 9.91 | 0.3436 | 0.4202 |
| Constant velocity (k=1, chosen on inner split) | 21.35 | 21.34 | 0.9383 | 0.0616 | 10.05 | 0.3365 | 0.4646 |
| Static (last prefix state) | 20.33 | 20.33 | 0.9407 | 0.0483 | 6.97 | 0.0598 | 0.5400 |
| Deformable-GS (Stage 1 queried at t) | 20.71 | 20.71 | 0.9420 | 0.0474 | 7.52 | 0.1140 | 0.5110 |
| _CV, all k (display only): k=1: 21.35, k=2: 21.15, k=4: 21.07, k=8: 20.92, k=16: 20.79_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 24.97 | 24.96 | 0.9570 | 0.0365 | 8.47 | 0.1855 | 0.4299 |
| **ODE-GS** | 23.80 | 23.79 | 0.9542 | 0.0372 | 7.87 | 0.1533 | 0.4405 |
| Group-rigid (k=1, chosen on inner split) | 25.55 | 25.55 | 0.9585 | 0.0347 | 9.92 | 0.2990 | 0.3713 |
| Constant velocity (k=1, chosen on inner split) | 25.55 | 25.55 | 0.9579 | 0.0359 | 9.94 | 0.2956 | 0.3848 |
| Static (last prefix state) | 22.76 | 22.75 | 0.9524 | 0.0388 | 6.71 | 0.0396 | 0.5034 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 41.77 | 41.76 | 0.9956 | 0.0091 | 29.51 | 0.9737 | 0.0290 |
