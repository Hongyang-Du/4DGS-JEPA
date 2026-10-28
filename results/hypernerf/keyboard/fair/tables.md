## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 8 (span 0.3718) | ≤ 61 frames (= 0.0978 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 199.5 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3718 time span, 30 points | ≤ 0.0978 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 264.4 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 18.64 | 18.61 | 0.6438 | 0.3041 | 13.55 | 0.3148 | 0.4278 |
| **ODE-GS** | 19.94 | 19.91 | 0.6839 | 0.2782 | 14.20 | 0.3404 | 0.4018 |
| Constant velocity (k=4, chosen on inner split) | 14.88 | 14.79 | 0.5118 | 0.5242 | 11.71 | 0.2462 | 0.5955 |
| Static (last prefix state) | 18.55 | 18.53 | 0.6361 | 0.3180 | 12.43 | 0.2103 | 0.4836 |
| Deformable-GS (Stage 1 queried at t) | 19.45 | 19.42 | 0.6630 | 0.2972 | 14.34 | 0.3548 | 0.4088 |
| _CV, all k (display only): k=1: 15.10, k=2: 15.17, k=4: 14.88, k=8: 14.16, k=16: 14.33_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| Constant velocity (k=4, chosen on inner split) | 16.04 | 15.97 | 0.5372 | 0.5041 | 12.50 | 0.3105 | 0.5746 |
| Static (last prefix state) | 17.71 | 17.68 | 0.6185 | 0.3462 | 11.88 | 0.2223 | 0.5067 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 30.78 | 30.77 | 0.9205 | 0.1584 | 26.67 | 0.8673 | 0.1552 |
