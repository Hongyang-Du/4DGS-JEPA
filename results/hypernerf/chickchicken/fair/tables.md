## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 3 (span 0.3833) | ≤ 22 frames (= 0.0969 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 202.4 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3833 time span, 30 points | ≤ 0.0969 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 256.4 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 18.66 | 18.63 | 0.6645 | 0.3696 | 12.39 | 0.4306 | 0.4925 |
| **ODE-GS** | 18.65 | 18.62 | 0.6597 | 0.3851 | 11.91 | 0.3968 | 0.5300 |
| Constant velocity (k=16, chosen on inner split) | 14.59 | 14.58 | 0.5775 | 0.5065 | 9.78 | 0.3084 | 0.6606 |
| Static (last prefix state) | 17.64 | 17.62 | 0.6400 | 0.4051 | 10.29 | 0.2951 | 0.6055 |
| Deformable-GS (Stage 1 queried at t) | 17.78 | 17.75 | 0.6450 | 0.3980 | 10.86 | 0.3470 | 0.5679 |
| _CV, all k (display only): k=1: 14.09, k=2: 16.16, k=4: 16.05, k=8: 14.80, k=16: 14.59_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| Constant velocity (k=16, chosen on inner split) | 17.14 | 17.11 | 0.6242 | 0.4492 | 10.90 | 0.3342 | 0.5929 |
| Static (last prefix state) | 18.27 | 18.24 | 0.6507 | 0.4002 | 10.91 | 0.2874 | 0.5666 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 33.13 | 33.10 | 0.9042 | 0.2404 | 28.74 | 0.8817 | 0.1931 |
