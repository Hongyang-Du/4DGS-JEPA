## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 3 (span 0.4223) | ≤ 20 frames (= 0.0971 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 201.4 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.4223 time span, 30 points | ≤ 0.0971 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 244.1 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 17.15 | 17.14 | 0.4723 | 0.3914 | 13.98 | 0.3569 | 0.4181 |
| **ODE-GS** | 16.55 | 16.54 | 0.4433 | 0.4074 | 12.34 | 0.2304 | 0.4895 |
| Constant velocity (k=16, chosen on inner split) | 15.48 | 15.47 | 0.4115 | 0.4624 | 11.49 | 0.2011 | 0.5543 |
| Static (last prefix state) | 16.74 | 16.73 | 0.4580 | 0.3934 | 11.81 | 0.1725 | 0.5023 |
| Deformable-GS (Stage 1 queried at t) | 16.93 | 16.92 | 0.4579 | 0.3955 | 13.34 | 0.2984 | 0.4522 |
| _CV, all k (display only): k=1: 13.30, k=2: 13.22, k=4: 13.50, k=8: 13.69, k=16: 15.48_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| Constant velocity (k=16, chosen on inner split) | 15.61 | 15.60 | 0.4217 | 0.4520 | 11.71 | 0.1914 | 0.5371 |
| Static (last prefix state) | 16.44 | 16.43 | 0.4506 | 0.4028 | 12.08 | 0.1721 | 0.5045 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 24.07 | 24.06 | 0.7743 | 0.2347 | 22.84 | 0.7768 | 0.2186 |
