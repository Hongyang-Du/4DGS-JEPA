# exp1 reproduction report

Runs per variant: {'Endpoint': 3, 'Endpoint + Composition': 3, 'Endpoint + Path': 3, 'Endpoint + Path + Composition': 3}. Cells: reproduced mean±std over seeds, with the paper's value below.

## Table 2

| Method | direct_fde | rollout_fde | delta_roll | comp_err | geo_comp_err | unseen_fde |
|---|---|---|---|---|---|---|
| Endpoint | 0.03024±0.00048<br>paper 0.03013±0.00042 | 0.03069±0.00066<br>paper 0.03061±0.00065 | 0.00045±0.00021<br>paper 0.00048±0.00023 | 0.00247±0.00057<br>paper 0.00244±0.00055 | 0.00733±0.00079<br>paper 0.00739±0.00082 | 0.03632±0.00085<br>paper 0.03621±0.00080 |
| Endpoint + Path | 0.03204±0.00247<br>paper 0.03195±0.00243 | 0.03245±0.00248<br>paper 0.03228±0.00256 | 0.00042±0.00008<br>paper 0.00033±0.00014 | 0.00067±0.00011<br>paper 0.00065±0.00007 | 0.00652±0.00069<br>paper 0.00649±0.00069 | 0.03859±0.00294<br>paper 0.03838±0.00304 |
| Endpoint + Composition | 0.02909±0.00089<br>paper 0.02905±0.00095 | 0.02945±0.00095<br>paper 0.02940±0.00101 | 0.00036±0.00012<br>paper 0.00035±0.00014 | 0.00096±0.00011<br>paper 0.00096±0.00011 | 0.00666±0.00071<br>paper 0.00667±0.00074 | 0.03474±0.00112<br>paper 0.03466±0.00122 |
| Endpoint + Path + Composition | 0.03192±0.00241<br>paper 0.03206±0.00245 | 0.03231±0.00243<br>paper 0.03236±0.00255 | 0.00039±0.00005<br>paper 0.00029±0.00013 | 0.00060±0.00007<br>paper 0.00058±0.00003 | 0.00651±0.00086<br>paper 0.00653±0.00089 | 0.03828±0.00270<br>paper 0.03837±0.00283 |

## Table 7 (held-out decompositions)

| Method | heldout_comp_err | heldout_geo_comp_err | unseen_seg_comp_err | unseen_seg_geo_comp_err |
|---|---|---|---|---|
| Endpoint | 0.01151±0.00257<br>paper 0.01134±0.00244 | 0.01420±0.00149<br>paper 0.01428±0.00160 | 0.00239±0.00050<br>paper 0.00235±0.00047 | 0.00893±0.00118<br>paper 0.00901±0.00120 |
| Endpoint + Path | 0.00292±0.00025<br>paper 0.00286±0.00013 | 0.01122±0.00157<br>paper 0.01105±0.00146 | 0.00066±0.00009<br>paper 0.00064±0.00006 | 0.00786±0.00121<br>paper 0.00778±0.00119 |
| Endpoint + Composition | 0.00453±0.00051<br>paper 0.00454±0.00049 | 0.01197±0.00122<br>paper 0.01204±0.00125 | 0.00096±0.00011<br>paper 0.00096±0.00011 | 0.00801±0.00106<br>paper 0.00803±0.00110 |
| Endpoint + Path + Composition | 0.00269±0.00016<br>paper 0.00260±0.00009 | 0.01108±0.00164<br>paper 0.01096±0.00157 | 0.00060±0.00007<br>paper 0.00057±0.00002 | 0.00781±0.00138<br>paper 0.00782±0.00141 |

## Causal baselines (seen-horizon mean FDE)

- Static persistence: 0.14802 (paper 0.14802); at Δ=8: 0.32158
- Constant velocity: 0.02484 (paper 0.02484); at Δ=8: 0.07119

Direct FDE at Δ=8 (paper: Endpoint + Composition 0.05687 vs constant velocity 0.07120):

- Endpoint: 0.06015
- Endpoint + Composition: 0.05689
- Endpoint + Path: 0.06292
- Endpoint + Path + Composition: 0.06311
