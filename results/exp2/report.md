# exp2 reproduction report

Runs per variant: {'+SE(3) composition': 3, 'Full geometry composition': 3, 'Latent composition only': 3}. Cells: reproduced mean±std over seeds, with the paper's value below.

## Table 3 (Δ=8)

| Method | direct_fde | cov_err | transform_target | transform_dc | d_match |
|---|---|---|---|---|---|
| Latent composition only | 0.06559±0.00318<br>paper 0.06614±0.00271 | 0.19858±0.00434<br>paper 0.19897±0.00387 | 0.26045±0.01090<br>paper 0.26124±0.01038 | 0.00726±0.00049<br>paper 0.00724±0.00057 | 0.00195±0.00010<br>paper 0.00192±0.00015 |
| +SE(3) composition | 0.06679±0.00440<br>paper 0.06675±0.00408 | 0.19802±0.00295<br>paper 0.19758±0.00330 | 0.25705±0.00183<br>paper 0.25554±0.00119 | 0.00206±0.00037<br>paper 0.00202±0.00031 | 0.00141±0.00034<br>paper 0.00138±0.00036 |
| Full geometry composition | 0.06734±0.00401<br>paper 0.06728±0.00388 | 0.19772±0.00312<br>paper 0.19758±0.00290 | 0.25578±0.00136<br>paper 0.25582±0.00130 | 0.00216±0.00031<br>paper 0.00235±0.00062 | 0.00123±0.00013<br>paper 0.00130±0.00012 |

## Reference baselines at Δ=8

- static_fde: 0.32158 (paper 0.32158)
- constant_velocity_fde: 0.07119 (paper 0.07120)
- identity_transform_target: 0.34170 (paper 0.34170)

## Table 8 (horizon-wise direct–composed consistency)

| Δ | Variant | E_T^dc | D_match |
|---|---|---|---|
| 2 | +SE(3) composition | 0.00046±0.00010 | 0.00037±0.00006 |
| 4 | +SE(3) composition | 0.00102±0.00026 | 0.00070±0.00012 |
| 8 | +SE(3) composition | 0.00206±0.00037 | 0.00141±0.00034 |
| 2 | Full geometry composition | 0.00049±0.00010 | 0.00035±0.00005 |
| 4 | Full geometry composition | 0.00103±0.00023 | 0.00065±0.00008 |
| 8 | Full geometry composition | 0.00216±0.00031 | 0.00123±0.00013 |
| 2 | Latent composition only | 0.00172±0.00024 | 0.00043±0.00006 |
| 4 | Latent composition only | 0.00442±0.00074 | 0.00104±0.00018 |
| 8 | Latent composition only | 0.00726±0.00049 | 0.00195±0.00010 |
