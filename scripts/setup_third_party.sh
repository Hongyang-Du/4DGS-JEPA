#!/usr/bin/env bash
# Fetch the ODE-GS code (front-end + evaluation protocol B) at the pinned commit.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d third_party/ODE-GS ] || git clone https://github.com/preacherwhite/ODE-GS.git third_party/ODE-GS
git -C third_party/ODE-GS checkout -q bc7f1e0

# D-NeRF (baseline) at the pinned commit, patched for modern PyTorch and for evaluation splits
# that do not start at t=0 (protocol B puts only t > 0.8 cameras in the test split).
if [ ! -d third_party/D-NeRF ]; then
  git clone https://github.com/albertpumarola/D-NeRF.git third_party/D-NeRF
  git -C third_party/D-NeRF checkout -q 2137203
  sed -i 's/^from torchsearchsorted import searchsorted$/searchsorted = torch.searchsorted/' third_party/D-NeRF/run_dnerf_helpers.py
  sed -i 's/^    assert max_time == 1., "max time must be 1"$/    # assert max_time == 1., "max time must be 1"/' third_party/D-NeRF/run_dnerf.py
  sed -i 's/^        assert times\[0\] == 0, "Time must start at 0"$/        # assert times[0] == 0, "Time must start at 0"/' third_party/D-NeRF/load_blender.py
fi
