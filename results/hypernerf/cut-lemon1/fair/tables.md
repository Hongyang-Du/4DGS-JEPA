## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 14 (span 0.3426) | ≤ 116 frames (= 0.0979 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 204.4 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3426 time span, 30 points | ≤ 0.0979 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 233.6 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 19.74 | 19.73 | 0.5943 | 0.4117 | 11.81 | 0.3871 | 0.6005 |
| **ODE-GS** | 20.36 | 20.35 | 0.6115 | 0.3814 | 11.84 | 0.3899 | 0.5997 |
| Constant velocity (k=8, chosen on inner split) | 18.02 | 18.00 | 0.5358 | 0.4964 | 11.61 | 0.3785 | 0.6232 |
| Static (last prefix state) | 20.73 | 20.72 | 0.6290 | 0.3800 | 11.95 | 0.3822 | 0.6000 |
| Deformable-GS (Stage 1 queried at t) | 19.99 | 19.98 | 0.6016 | 0.4064 | 12.40 | 0.4152 | 0.5900 |
| _CV, all k (display only): k=1: 17.70, k=2: 17.61, k=4: 17.65, k=8: 18.02, k=16: 18.61_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| Constant velocity (k=8, chosen on inner split) | 18.21 | 18.20 | 0.4473 | 0.4821 | 13.03 | 0.3267 | 0.5995 |
| Static (last prefix state) | 20.66 | 20.65 | 0.5361 | 0.3557 | 12.41 | 0.2977 | 0.5794 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 29.33 | 29.28 | 0.8211 | 0.2410 | 23.51 | 0.7560 | 0.3066 |
