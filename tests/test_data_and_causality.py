import numpy as np
import pytest
import torch

from gsjepa.config import load_config
from gsjepa.data import GaussianWorldDataset, generate_scene, split_seeds
from gsjepa.model import HierEncoder, causal_gaussian_features


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/exp1.yaml")


def test_splits_disjoint_and_sample_counts(cfg):
    tr, va, te = split_seeds(cfg)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    ds = GaussianWorldDataset(te[:2], cfg)
    # t in [L-1, T - max_h): 24 - 8 - 3 = 13 samples per scene (6656 / 512 in the paper)
    assert len(ds) == 2 * 13


def test_generator_follows_structured_motion(cfg):
    sc = generate_scene(30_000, cfg)
    T, N = cfg.seq_len, cfg.num_gaussians
    assert sc["mu"].shape == (T, N, 3) and sc["cov"].shape == (T, N, 3, 3)
    assert np.all(np.linalg.eigvalsh(sc["cov"]) > 0)
    # Up to the small residual r (|r| <= 0.035), Gaussians follow their group's rigid motion (eq. 64).
    R, d, g = sc["group_R"], sc["group_d"], sc["group_ids"]
    for t in (5, 15):
        rel_R = R[t, g] @ np.transpose(R[0, g], (0, 2, 1))
        pred = np.einsum("nij,nj->ni", rel_R, sc["mu"][0] - d[0, g]) + d[t, g]
        assert np.abs(pred - sc["mu"][t]).max() < 0.08
    # Even seeds: group 0 is static.
    assert np.allclose(sc["mu"][:, g == 0], sc["mu"][0, g == 0], atol=0.08)


def test_encoder_is_causal(cfg):
    ds = GaussianWorldDataset([30_001], cfg)
    s = ds[0]
    t, L = int(s["t"]), cfg.context_len
    hist = lambda mu: mu[t - L + 1:t + 1].unsqueeze(0)
    perturbed = s["mu"].clone()
    perturbed[t + 1:] += 999.0
    enc = HierEncoder(cfg).eval()
    g = s["group_ids"].unsqueeze(0)
    with torch.no_grad():
        z1 = enc(hist(s["mu"]), hist(s["cov"]), g)
        z2 = enc(hist(perturbed), hist(s["cov"]), g)
    assert torch.equal(z1, z2)
    _, motion = causal_gaussian_features(hist(s["mu"]), hist(s["cov"]), g, cfg)
    v = (s["mu"][t] - s["mu"][t - 1]) / cfg.velocity_scale
    k = 3 * (L - 1)
    assert torch.allclose(motion[0, :, k:k + 3], v, atol=1e-5)
