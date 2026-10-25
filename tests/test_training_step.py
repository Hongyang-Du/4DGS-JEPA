import pytest
import torch
from torch.utils.data import DataLoader

from gsjepa.config import load_config
from gsjepa.data import GaussianWorldDataset
from gsjepa.model import JEPA, make_teacher
from gsjepa.train import compute_loss


@pytest.mark.parametrize("exp", ["exp1", "exp2"])
def test_every_variant_gives_finite_loss_and_gradients(exp):
    cfg = load_config(f"configs/{exp}.yaml")
    batch = next(iter(DataLoader(GaussianWorldDataset([10_000, 10_001], cfg), batch_size=8)))
    for name, variant in cfg.variants.items():
        torch.manual_seed(0)
        model = JEPA(cfg)
        teacher = make_teacher(model, cfg)
        for Delta in cfg.train_horizons:
            losses = compute_loss(model, teacher, batch, Delta, cfg.train_partitions[Delta][-1], variant, cfg)
            assert torch.isfinite(losses["total"]), (name, Delta)
            for term in ("path", "comp", "gcomp_se3", "gcomp_match"):   # disabled terms contribute nothing
                if not getattr(variant, term):
                    assert losses[term] == 0, (name, term)
        losses["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
