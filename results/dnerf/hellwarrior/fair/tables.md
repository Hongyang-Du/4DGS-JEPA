## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 1 (span 0.2900) | ≤ 20 frames (= 0.2000 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 305.5 |
| jepa_inner | jepa | inner | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 1 (span 0.2900) | ≤ 20 frames (= 0.2000 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 198.9 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.2900 time span, 30 points | ≤ 0.2000 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 331.7 |
| odegs_inner | odegs | inner | 3.88M | 90000 | 4096 | 0.2900 time span, 30 points | ≤ 0.2000 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 254.5 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 31.51 | 31.25 | 0.9285 | 0.0778 | 17.16 | 0.3138 | 0.4286 |
| **ODE-GS** | 30.97 | 30.69 | 0.9266 | 0.0771 | 15.52 | 0.1515 | 0.4936 |
| Group-rigid (k=1, chosen on inner split) | 31.36 | 31.11 | 0.9313 | 0.0735 | 17.55 | 0.3606 | 0.4070 |
| Constant velocity (k=1, chosen on inner split) | 31.35 | 31.10 | 0.9286 | 0.0800 | 17.58 | 0.3516 | 0.4380 |
| Static (last prefix state) | 30.86 | 30.57 | 0.9260 | 0.0739 | 15.18 | 0.1136 | 0.4963 |
| Deformable-GS (Stage 1 queried at t) | 30.34 | 30.03 | 0.9165 | 0.0830 | 16.15 | 0.2243 | 0.4694 |
| _CV, all k (display only): k=1: 31.35, k=2: 31.13, k=4: 30.94, k=8: 29.97, k=16: 29.80_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 33.67 | 33.39 | 0.9484 | 0.0553 | 19.64 | 0.4954 | 0.3220 |
| **ODE-GS** | 31.74 | 31.45 | 0.9377 | 0.0617 | 16.84 | 0.2721 | 0.4231 |
| Group-rigid (k=1, chosen on inner split) | 34.10 | 33.84 | 0.9524 | 0.0479 | 20.61 | 0.5708 | 0.2612 |
| Constant velocity (k=1, chosen on inner split) | 34.15 | 33.90 | 0.9498 | 0.0564 | 20.81 | 0.5573 | 0.3056 |
| Static (last prefix state) | 30.27 | 29.97 | 0.9243 | 0.0738 | 15.20 | 0.1222 | 0.4955 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 48.02 | 47.84 | 0.9952 | 0.0171 | 36.84 | 0.9669 | 0.0710 |
