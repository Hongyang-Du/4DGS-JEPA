## Stage-2 configuration (every variable)

| run | method | split | params | optimizer steps | Gaussian samples/step | observation window | train horizon | optimizer | LR peak | schedule | weight decay | grad clip | seed | train GPU-min (A100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa_full | jepa | full | 4.08M | 90000 | 4 ctx × 1024 = 4096 | 30 Stage-1 frames, stride 7 (span 0.3773) | ≤ 52 frames (= 0.0967 time) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 201.2 |
| odegs_full | odegs | full | 3.88M | 90000 | 4096 | 0.3773 time span, 30 points | ≤ 0.0967 time (10 pts) | AdamW | 0.001 | warmup 5% + cosine → 0.01× | 0.0001 | 1.0 | 0 | 196.0 |

## Test: protocol-B evaluation cameras (t > split), forecast from the end of the prefix

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| **4DGS-JEPA** | 20.35 | 20.34 | 0.6306 | 0.2389 | 14.33 | 0.3794 | 0.3557 |
| **ODE-GS** | 19.42 | 19.40 | 0.6021 | 0.2737 | 12.89 | 0.2725 | 0.4171 |
| Constant velocity (k=1, chosen on inner split) | 17.12 | 17.11 | 0.5250 | 0.4238 | 11.43 | 0.2106 | 0.5635 |
| Static (last prefix state) | 18.97 | 18.94 | 0.6009 | 0.2786 | 12.01 | 0.2345 | 0.4623 |
| Deformable-GS (Stage 1 queried at t) | 18.33 | 18.28 | 0.6006 | 0.2840 | 11.70 | 0.2660 | 0.4671 |
| _CV, all k (display only): k=1: 17.12, k=2: 16.79, k=4: 16.18, k=8: 15.59, k=16: 16.21_ | | | | | | | |

## Inner validation: prefix cameras inner_cut < t ≤ prefix end, forecast from inner_cut

| method | PSNR | PSNR_img | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |
|---|---|---|---|---|---|---|---|
| Constant velocity (k=1, chosen on inner split) | 18.79 | 18.77 | 0.5578 | 0.3898 | 13.23 | 0.2987 | 0.5134 |
| Static (last prefix state) | 21.16 | 21.12 | 0.6805 | 0.2000 | 12.58 | 0.2413 | 0.4273 |
| _Stage-1 fit at these times (in-sample for Stage 1, NOT a forecast)_ | 34.69 | 34.67 | 0.9570 | 0.0769 | 28.00 | 0.8981 | 0.1144 |
