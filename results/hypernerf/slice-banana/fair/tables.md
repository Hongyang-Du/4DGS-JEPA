## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3537) | ≤ 16 frames (= 0.0976 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 195.5 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3537) | ≤ 16 frames (= 0.0976 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 284.0 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3537 time span, 30 points | ≤ 0.0976 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 223.1 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.3537 time span, 30 points | ≤ 0.0976 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 218.3 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 19.46 | 19.42 | 0.5808 | 0.4788 | 14.40 | 0.3512 | 0.5543 |
| **ODE-GS** | 19.09 | 19.04 | 0.5760 | 0.4804 | 13.63 | 0.3076 | 0.5726 |
| Constant velocity (k=4, chosen on inner split) | 17.08 | 17.06 | 0.5509 | 0.5388 | 12.84 | 0.3176 | 0.6289 |
| Static (last prefix state) | 18.70 | 18.65 | 0.5884 | 0.4979 | 12.79 | 0.2891 | 0.6103 |
| Deformable-GS (Stage 1 queried at t) | 18.87 | 18.82 | 0.5881 | 0.5119 | 13.83 | 0.3542 | 0.5789 |
| _CV, all k (display only): k=1: 15.28, k=2: 16.18, k=4: 17.08, k=8: 17.15, k=16: 17.40_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 17.45 | 17.43 | 0.5837 | 0.5117 | 12.57 | 0.3060 | 0.6018 |
| **ODE-GS** | 17.60 | 17.58 | 0.5977 | 0.5122 | 12.17 | 0.2924 | 0.6134 |
| Constant velocity (k=4, chosen on inner split) | 18.41 | 18.37 | 0.5930 | 0.5093 | 13.20 | 0.3278 | 0.5943 |
| Static (last prefix state) | 18.05 | 18.02 | 0.6124 | 0.4997 | 12.36 | 0.2926 | 0.6009 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 32.54 | 32.51 | 0.8735 | 0.2913 | 29.40 | 0.8682 | 0.2222 |

## Earlier 30k-step runs and the LR check (same settings otherwise)

Rows with split = inner are scored on the inner validation cameras; these rows alone drove the LR decision (3e-4 → 1e-3). Rows with split = full are scored on the test cameras and shown for display only.

| run | split | LR | steps | PSNR | SSIM | LPIPS | dyn PSNR |
|---|---|---|---|---|---|---|---|
| jepa_full | full | 0.0003 | 30000 | 19.70 | 0.5866 | 0.4743 | 14.28 |
| jepa_inner | inner | 0.0003 | 30000 | 16.98 | 0.5748 | 0.5234 | 12.27 |
| jepa_inner_lr1e-3 | inner | 0.001 | 30000 | 17.45 | 0.5867 | 0.5124 | 12.43 |
| odegs_full | full | 0.0003 | 30000 | 18.74 | 0.5524 | 0.4936 | 13.38 |
| odegs_full_lr1e-3 | full | 0.001 | 30000 | 18.26 | 0.5669 | 0.5012 | 12.76 |
| odegs_inner | inner | 0.0003 | 30000 | 16.90 | 0.5006 | 0.5341 | 12.12 |
| odegs_inner_lr1e-3 | inner | 0.001 | 30000 | 17.44 | 0.5893 | 0.5100 | 12.12 |
