# exp3 reproduction report

## Table 4

| Method | Permutation lineage | Split/merge lineage | Birth/death F1 | Geometry discrepancy |
|---|---|---|---|---|
| Index | 0.04056±0.00507<br>paper 0.04056±0.00507 | 0.15563±0.00907<br>paper 0.15563±0.00907 | 0.61134±0.00381<br>paper 0.61134±0.00381 | 0.74311±0.00449<br>paper 0.74311±0.00449 |
| OT only | 0.99789±0.00024<br>paper 0.99789±0.00024 | 0.98100±0.00099<br>paper 0.98100±0.00099 | 1.00000±0.00000<br>paper 1.00000±0.00000 | 0.05010±0.00014<br>paper 0.05010±0.00014 |
| Hybrid | 1.00000±0.00000<br>paper 1.00000±0.00000 | 0.98715±0.00103<br>paper 0.98715±0.00103 | 1.00000±0.00000<br>paper 1.00000±0.00000 | 0.04906±0.00013<br>paper 0.04906±0.00013 |

## Table 10 (mixed perturbations)

```
                 lineage_mass          geometry_discrepancy          event_f1_macro         
                         mean      std                 mean      std           mean      std
severity method                                                                             
0.10     Hybrid       0.96735  0.00620              0.05541  0.00108        0.83767  0.01943
         Index        0.04167  0.00928              1.08402  0.04409        0.00000  0.00000
         OT only      0.95694  0.00752              0.05693  0.00157        0.86289  0.02349
0.25     Hybrid       0.95988  0.00082              0.06349  0.00147        0.91850  0.00730
         Index        0.03194  0.00127              1.12927  0.01950        0.00000  0.00000
         OT only      0.95148  0.00091              0.06502  0.00171        0.92244  0.00700
0.40     Hybrid       0.92605  0.00137              0.07603  0.00158        0.90918  0.00275
         Index        0.03278  0.00255              1.11603  0.00790        0.00000  0.00000
         OT only      0.91962  0.00344              0.07718  0.00183        0.90999  0.00269
```

## Table 11 (persistent-ID dropout, severity 0.25)

```
                lineage_mass          geometry_discrepancy         
method                Hybrid  OT only               Hybrid  OT only
id_dropout_rate                                                    
0.00                 0.95988  0.95148              0.06349  0.06502
0.10                 0.95916  0.95148              0.06361  0.06502
0.25                 0.95838  0.95148              0.06376  0.06502
0.50                 0.95649  0.95148              0.06409  0.06502
```
