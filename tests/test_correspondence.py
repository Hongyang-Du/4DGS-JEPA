import numpy as np

from gsjepa.config import CorrespondenceConfig
from gsjepa.correspondence import (hide_persistent_ids, hybrid_plan, index_plan, make_source, ot_plan,
                                   perturb_source, plan_metrics)

CFG = CorrespondenceConfig()


def _perm():
    src = make_source(123, CFG)
    return src, perturb_source(src, "permutation", 0.25, np.random.default_rng(5), CFG)


def test_hybrid_recovers_permutation_index_does_not():
    src, tgt = _perm()
    assert plan_metrics(src, tgt, hybrid_plan(src, tgt, CFG))["lineage_mass"] > 0.999
    assert plan_metrics(src, tgt, index_plan(src, tgt))["lineage_mass"] < 0.3


def test_count_aware_ot_marginals():
    src, tgt = _perm()
    ns, nt = len(src["mu"]), len(tgt["mu"])
    for P in (ot_plan(src, tgt, CFG), hybrid_plan(src, tgt, CFG)):
        assert np.isfinite(P).all() and (P >= 0).all()
        assert np.allclose(P[:ns].sum(1), 1.0, atol=5e-4)       # unit mass per real source
        assert np.allclose(P[:, :nt].sum(0), 1.0, atol=5e-4)    # unit mass per real target


def test_events_and_split_lineage():
    src = make_source(7, CFG)
    tgt = perturb_source(src, "split", 0.25, np.random.default_rng(0), CFG)
    assert (tgt["slot_id"] == -1).sum() == 2 * round(0.25 * CFG.active_primitives)
    m = plan_metrics(src, tgt, hybrid_plan(src, tgt, CFG))
    assert np.isnan(m["birth_f1"]) and m["lineage_mass"] > 0.9
    tgt = perturb_source(src, "birth", 0.25, np.random.default_rng(0), CFG)
    assert plan_metrics(src, tgt, hybrid_plan(src, tgt, CFG))["birth_f1"] == 1.0


def test_hiding_ids_keeps_geometry():
    _, tgt = _perm()
    hidden = hide_persistent_ids(tgt, 0.5, np.random.default_rng(0))
    assert np.allclose(hidden["mu"], tgt["mu"]) and hidden["lineage"] == tgt["lineage"]
    assert (hidden["slot_id"] >= 0).sum() == (tgt["slot_id"] >= 0).sum() - 12
