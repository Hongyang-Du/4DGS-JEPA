## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3893) | ≤ 30 frames (= 0.2013 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 202.7 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 2 (span 0.3893) | ≤ 30 frames (= 0.2013 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 201.3 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3893 time span, 30 points | ≤ 0.2013 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 210.6 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.3893 time span, 30 points | ≤ 0.2013 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 321.7 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 28.49 | 28.42 | 0.9750 | 0.0361 | 12.95 | 0.5177 | 0.3046 |
| **ODE-GS** | 6.12 | 6.11 | 0.5948 | 0.2747 | 3.89 | 0.0481 | 0.7510 |
| Group-rigid (k=1, chosen on inner split) | 24.83 | 24.75 | 0.9641 | 0.0420 | 11.97 | 0.5277 | 0.3042 |
| Constant velocity (k=1, chosen on inner split) | 23.70 | 23.64 | 0.9467 | 0.0702 | 12.26 | 0.5403 | 0.3205 |
| Static (last prefix state) | 27.31 | 27.21 | 0.9745 | 0.0334 | 10.90 | 0.3845 | 0.3626 |
| Deformable-GS (Stage 1 queried at t) | 29.70 | 29.57 | 0.9816 | 0.0230 | 15.48 | 0.6494 | 0.1964 |
| _CV, all k (display only): k=1: 23.70, k=2: 23.13, k=4: 22.61, k=8: 22.65, k=16: 23.42_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 29.84 | 29.76 | 0.9764 | 0.0296 | 14.15 | 0.5840 | 0.2720 |
| **ODE-GS** | 26.32 | 26.26 | 0.9693 | 0.0351 | 11.81 | 0.4665 | 0.3495 |
| Group-rigid (k=1, chosen on inner split) | 28.42 | 28.37 | 0.9730 | 0.0290 | 14.19 | 0.6141 | 0.2370 |
| Constant velocity (k=1, chosen on inner split) | 31.08 | 31.00 | 0.9793 | 0.0279 | 15.65 | 0.6568 | 0.2266 |
| Static (last prefix state) | 25.82 | 25.75 | 0.9688 | 0.0396 | 10.40 | 0.3952 | 0.4197 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 46.98 | 46.86 | 0.9977 | 0.0070 | 36.18 | 0.9818 | 0.0241 |
