"""Reported numbers from arXiv:2609.25036 (mean, std over 3 seeds), used to check reproduction."""

# Table 2 (Exp1): averaged over supervised horizons {1,2,4,8}; unseen = rollout FDE over {3,6}.
EXP1 = {
    "Endpoint": {
        "direct_fde": (0.03013, 0.00042), "rollout_fde": (0.03061, 0.00065), "delta_roll": (0.00048, 0.00023),
        "comp_err": (0.00244, 0.00055), "geo_comp_err": (0.00739, 0.00082), "unseen_fde": (0.03621, 0.00080),
        # Table 7
        "heldout_comp_err": (0.01134, 0.00244), "heldout_geo_comp_err": (0.01428, 0.00160),
        "unseen_seg_comp_err": (0.00235, 0.00047), "unseen_seg_geo_comp_err": (0.00901, 0.00120)},
    "Endpoint + Path": {
        "direct_fde": (0.03195, 0.00243), "rollout_fde": (0.03228, 0.00256), "delta_roll": (0.00033, 0.00014),
        "comp_err": (0.00065, 0.00007), "geo_comp_err": (0.00649, 0.00069), "unseen_fde": (0.03838, 0.00304),
        "heldout_comp_err": (0.00286, 0.00013), "heldout_geo_comp_err": (0.01105, 0.00146),
        "unseen_seg_comp_err": (0.00064, 0.00006), "unseen_seg_geo_comp_err": (0.00778, 0.00119)},
    "Endpoint + Composition": {
        "direct_fde": (0.02905, 0.00095), "rollout_fde": (0.02940, 0.00101), "delta_roll": (0.00035, 0.00014),
        "comp_err": (0.00096, 0.00011), "geo_comp_err": (0.00667, 0.00074), "unseen_fde": (0.03466, 0.00122),
        "heldout_comp_err": (0.00454, 0.00049), "heldout_geo_comp_err": (0.01204, 0.00125),
        "unseen_seg_comp_err": (0.00096, 0.00011), "unseen_seg_geo_comp_err": (0.00803, 0.00110)},
    "Endpoint + Path + Composition": {
        "direct_fde": (0.03206, 0.00245), "rollout_fde": (0.03236, 0.00255), "delta_roll": (0.00029, 0.00013),
        "comp_err": (0.00058, 0.00003), "geo_comp_err": (0.00653, 0.00089), "unseen_fde": (0.03837, 0.00283),
        "heldout_comp_err": (0.00260, 0.00009), "heldout_geo_comp_err": (0.01096, 0.00157),
        "unseen_seg_comp_err": (0.00057, 0.00002), "unseen_seg_geo_comp_err": (0.00782, 0.00141)},
}
EXP1_BASELINES = {"Static persistence": {"seen_fde": 0.14802}, "Constant velocity": {"seen_fde": 0.02484}}

# Table 3 (Exp2): at Delta = 8.
EXP2 = {
    "Latent composition only": {
        "direct_fde": (0.06614, 0.00271), "cov_err": (0.19897, 0.00387), "transform_target": (0.26124, 0.01038),
        "transform_dc": (0.00724, 0.00057), "d_match": (0.00192, 0.00015)},
    "+SE(3) composition": {
        "direct_fde": (0.06675, 0.00408), "cov_err": (0.19758, 0.00330), "transform_target": (0.25554, 0.00119),
        "transform_dc": (0.00202, 0.00031), "d_match": (0.00138, 0.00036)},
    "Full geometry composition": {
        "direct_fde": (0.06728, 0.00388), "cov_err": (0.19758, 0.00290), "transform_target": (0.25582, 0.00130),
        "transform_dc": (0.00235, 0.00062), "d_match": (0.00130, 0.00012)},
}
EXP2_BASELINES = {"static_fde": 0.32158, "constant_velocity_fde": 0.07120, "identity_transform_target": 0.34170}

# Table 4 (Exp3).
EXP3 = {
    "Index": {"Permutation lineage": (0.04056, 0.00507), "Split/merge lineage": (0.15563, 0.00907),
              "Birth/death F1": (0.61134, 0.00381), "Geometry discrepancy": (0.74311, 0.00449)},
    "OT only": {"Permutation lineage": (0.99789, 0.00024), "Split/merge lineage": (0.98100, 0.00099),
                "Birth/death F1": (1.00000, 0.00000), "Geometry discrepancy": (0.05010, 0.00014)},
    "Hybrid": {"Permutation lineage": (1.00000, 0.00000), "Split/merge lineage": (0.98715, 0.00103),
               "Birth/death F1": (1.00000, 0.00000), "Geometry discrepancy": (0.04906, 0.00013)},
}
