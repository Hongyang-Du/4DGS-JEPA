### PSNR ↑ (full image)

| method | bouncingballs | hellwarrior | hook | jumpingjacks | lego | mutant | standup | trex | mean |
|---|---|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 28.49 | 31.51 | 26.95 | 16.84 | 26.43 | 30.23 | 30.62 | 21.14 | 26.53 |
| ODE-GS (controlled, 90k) | 6.12 | 30.97 | 27.54 | 20.58 | 26.42 | 30.57 | 29.34 | 20.73 | 24.03 |
| Deformable-GS (Stage 1 queried) | 29.70 | 30.34 | 21.77 | 20.54 | 23.56 | 24.48 | 21.38 | 20.71 | 24.06 |
| Static (last prefix state) | 27.31 | 30.86 | 23.97 | 20.41 | 25.57 | 26.83 | 24.33 | 20.33 | 24.95 |
| Group-rigid (k chosen on inner split) | 24.83 | 31.36 | 27.65 | 22.96 | 26.28 | 30.65 | 28.23 | 21.50 | 26.68 |
| Constant velocity (k chosen on inner split) | 23.70 | 31.35 | 27.49 | 22.61 | 25.69 | 30.58 | 28.13 | 21.35 | 26.36 |

### PSNR ↑ (dynamic-region mask)

| method | bouncingballs | hellwarrior | hook | jumpingjacks | lego | mutant | standup | trex | mean |
|---|---|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 12.95 | 17.16 | 13.93 | 9.29 | 12.91 | 12.38 | 18.82 | 10.60 | 13.51 |
| ODE-GS (controlled, 90k) | 3.89 | 15.52 | 14.24 | 8.45 | 12.76 | 12.80 | 16.71 | 7.40 | 11.47 |
| Deformable-GS (Stage 1 queried) | 15.48 | 16.15 | 9.34 | 8.75 | 10.09 | 7.75 | 9.15 | 7.52 | 10.53 |
| Static (last prefix state) | 10.90 | 15.18 | 10.49 | 8.27 | 11.36 | 8.78 | 10.78 | 6.97 | 10.34 |
| Group-rigid (k chosen on inner split) | 11.97 | 17.55 | 14.66 | 11.89 | 13.20 | 12.79 | 15.97 | 9.91 | 13.49 |
| Constant velocity (k chosen on inner split) | 12.26 | 17.58 | 14.57 | 12.18 | 12.77 | 12.83 | 16.00 | 10.05 | 13.53 |

### SSIM ↑ (full image)

| method | bouncingballs | hellwarrior | hook | jumpingjacks | lego | mutant | standup | trex | mean |
|---|---|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.9750 | 0.9285 | 0.9353 | 0.7718 | 0.9376 | 0.9666 | 0.9595 | 0.9116 | 0.9232 |
| ODE-GS (controlled, 90k) | 0.5948 | 0.9266 | 0.9441 | 0.9170 | 0.9393 | 0.9705 | 0.9554 | 0.9404 | 0.8985 |
| Deformable-GS (Stage 1 queried) | 0.9816 | 0.9165 | 0.8878 | 0.9144 | 0.9329 | 0.9299 | 0.9120 | 0.9420 | 0.9271 |
| Static (last prefix state) | 0.9745 | 0.9260 | 0.9103 | 0.9164 | 0.9386 | 0.9484 | 0.9331 | 0.9407 | 0.9360 |
| Group-rigid (k chosen on inner split) | 0.9641 | 0.9313 | 0.9384 | 0.9324 | 0.9316 | 0.9667 | 0.9522 | 0.9434 | 0.9450 |
| Constant velocity (k chosen on inner split) | 0.9467 | 0.9286 | 0.9351 | 0.9184 | 0.9247 | 0.9657 | 0.9517 | 0.9383 | 0.9386 |

### LPIPS ↓ (full image)

| method | bouncingballs | hellwarrior | hook | jumpingjacks | lego | mutant | standup | trex | mean |
|---|---|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.0361 | 0.0778 | 0.0423 | 0.1881 | 0.0541 | 0.0200 | 0.0369 | 0.0753 | 0.0663 |
| ODE-GS (controlled, 90k) | 0.2747 | 0.0771 | 0.0396 | 0.0795 | 0.0517 | 0.0196 | 0.0390 | 0.0500 | 0.0789 |
| Deformable-GS (Stage 1 queried) | 0.0230 | 0.0830 | 0.0829 | 0.0819 | 0.0606 | 0.0477 | 0.0851 | 0.0474 | 0.0639 |
| Static (last prefix state) | 0.0334 | 0.0739 | 0.0586 | 0.0795 | 0.0529 | 0.0315 | 0.0590 | 0.0483 | 0.0546 |
| Group-rigid (k chosen on inner split) | 0.0420 | 0.0735 | 0.0370 | 0.0631 | 0.0520 | 0.0191 | 0.0386 | 0.0524 | 0.0472 |
| Constant velocity (k chosen on inner split) | 0.0702 | 0.0800 | 0.0447 | 0.0793 | 0.0706 | 0.0209 | 0.0423 | 0.0616 | 0.0587 |

### LPIPS ↓ (dynamic-region mask)

| method | bouncingballs | hellwarrior | hook | jumpingjacks | lego | mutant | standup | trex | mean |
|---|---|---|---|---|---|---|---|---|---|
| 4DGS-JEPA (controlled, 90k) | 0.3046 | 0.4286 | 0.3378 | 0.5893 | 0.5260 | 0.2727 | 0.2371 | 0.4548 | 0.3939 |
| ODE-GS (controlled, 90k) | 0.7510 | 0.4936 | 0.3167 | 0.6126 | 0.5215 | 0.2648 | 0.2869 | 0.5401 | 0.4734 |
| Deformable-GS (Stage 1 queried) | 0.1964 | 0.4694 | 0.5357 | 0.5888 | 0.6391 | 0.5950 | 0.6398 | 0.5110 | 0.5219 |
| Static (last prefix state) | 0.3626 | 0.4963 | 0.4682 | 0.6186 | 0.5747 | 0.4669 | 0.5279 | 0.5400 | 0.5069 |
| Group-rigid (k chosen on inner split) | 0.3042 | 0.4070 | 0.2950 | 0.4052 | 0.4979 | 0.2566 | 0.2845 | 0.4202 | 0.3588 |
| Constant velocity (k chosen on inner split) | 0.3205 | 0.4380 | 0.3395 | 0.4556 | 0.5396 | 0.2696 | 0.3088 | 0.4646 | 0.3920 |

CV window k chosen on each scene's inner split: bouncingballs: k=1, hellwarrior: k=1, hook: k=1, jumpingjacks: k=1, lego: k=1, mutant: k=2, standup: k=8, trex: k=1
Group-rigid window k (same rule): bouncingballs: k=1, hellwarrior: k=1, hook: k=1, jumpingjacks: k=1, lego: k=1, mutant: k=2, standup: k=8, trex: k=1
