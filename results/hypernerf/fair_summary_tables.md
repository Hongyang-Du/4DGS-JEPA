### PSNR ↑ (full image)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 19.46 | 19.74 | 18.66 | 20.35 | 18.64 | 17.15 | 19.00 |
| ODE-GS (controlled, 90k) | 19.09 | 20.36 | 18.65 | 19.42 | 19.94 | 16.55 | 19.00 |
| Deformable-GS (Stage 1 queried) | 18.87 | 19.99 | 17.78 | 18.33 | 19.45 | 16.93 | 18.56 |
| Static (last prefix state) | 18.70 | 20.73 | 17.64 | 18.97 | 18.55 | 16.74 | 18.55 |
| Group-rigid (k chosen on inner split) | 17.03 | 18.34 | 15.16 | 16.94 | 15.21 | 15.51 | 16.36 |
| Constant velocity (k chosen on inner split) | 17.08 | 18.02 | 14.59 | 17.12 | 14.88 | 15.48 | 16.20 |

### PSNR ↑ (dynamic-region mask)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 14.40 | 11.81 | 12.39 | 14.33 | 13.55 | 13.98 | 13.41 |
| ODE-GS (controlled, 90k) | 13.63 | 11.84 | 11.91 | 12.89 | 14.20 | 12.34 | 12.80 |
| Deformable-GS (Stage 1 queried) | 13.83 | 12.40 | 10.86 | 11.70 | 14.34 | 13.34 | 12.74 |
| Static (last prefix state) | 12.79 | 11.95 | 10.29 | 12.01 | 12.43 | 11.81 | 11.88 |
| Group-rigid (k chosen on inner split) | 12.84 | 11.07 | 9.63 | 10.98 | 11.54 | 11.49 | 11.26 |
| Constant velocity (k chosen on inner split) | 12.84 | 11.61 | 9.78 | 11.43 | 11.71 | 11.49 | 11.48 |

### SSIM ↑ (full image)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.5808 | 0.5943 | 0.6645 | 0.6306 | 0.6438 | 0.4723 | 0.5977 |
| ODE-GS (controlled, 90k) | 0.5760 | 0.6115 | 0.6597 | 0.6021 | 0.6839 | 0.4433 | 0.5961 |
| Deformable-GS (Stage 1 queried) | 0.5881 | 0.6016 | 0.6450 | 0.6006 | 0.6630 | 0.4579 | 0.5927 |
| Static (last prefix state) | 0.5884 | 0.6290 | 0.6400 | 0.6009 | 0.6361 | 0.4580 | 0.5921 |
| Group-rigid (k chosen on inner split) | 0.5672 | 0.5643 | 0.5827 | 0.5387 | 0.5430 | 0.4134 | 0.5349 |
| Constant velocity (k chosen on inner split) | 0.5509 | 0.5358 | 0.5775 | 0.5250 | 0.5118 | 0.4115 | 0.5188 |

### LPIPS ↓ (full image)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.4788 | 0.4117 | 0.3696 | 0.2389 | 0.3041 | 0.3914 | 0.3658 |
| ODE-GS (controlled, 90k) | 0.4804 | 0.3814 | 0.3851 | 0.2737 | 0.2782 | 0.4074 | 0.3677 |
| Deformable-GS (Stage 1 queried) | 0.5119 | 0.4064 | 0.3980 | 0.2840 | 0.2972 | 0.3955 | 0.3822 |
| Static (last prefix state) | 0.4979 | 0.3800 | 0.4051 | 0.2786 | 0.3180 | 0.3934 | 0.3788 |
| Group-rigid (k chosen on inner split) | 0.5349 | 0.4391 | 0.4823 | 0.3598 | 0.4594 | 0.4436 | 0.4532 |
| Constant velocity (k chosen on inner split) | 0.5388 | 0.4964 | 0.5065 | 0.4238 | 0.5242 | 0.4624 | 0.4920 |

### LPIPS ↓ (dynamic-region mask)

| method | slice-banana | cut-lemon1 | chickchicken | split-cookie | keyboard | vrig-3dprinter | mean |
|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.5543 | 0.6005 | 0.4925 | 0.3557 | 0.4278 | 0.4181 | 0.4748 |
| ODE-GS (controlled, 90k) | 0.5726 | 0.5997 | 0.5300 | 0.4171 | 0.4018 | 0.4895 | 0.5018 |
| Deformable-GS (Stage 1 queried) | 0.5789 | 0.5900 | 0.5679 | 0.4671 | 0.4088 | 0.4522 | 0.5108 |
| Static (last prefix state) | 0.6103 | 0.6000 | 0.6055 | 0.4623 | 0.4836 | 0.5023 | 0.5440 |
| Group-rigid (k chosen on inner split) | 0.6251 | 0.6141 | 0.6614 | 0.5204 | 0.5707 | 0.5334 | 0.5875 |
| Constant velocity (k chosen on inner split) | 0.6289 | 0.6232 | 0.6606 | 0.5635 | 0.5955 | 0.5543 | 0.6044 |

CV window k chosen on each scene's inner split: slice-banana: k=4, cut-lemon1: k=8, chickchicken: k=16, split-cookie: k=1, keyboard: k=4, vrig-3dprinter: k=16
Group-rigid window k (same rule): slice-banana: k=4, cut-lemon1: k=8, chickchicken: k=16, split-cookie: k=1, keyboard: k=4, vrig-3dprinter: k=16
