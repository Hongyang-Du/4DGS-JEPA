#!/bin/bash
export PYTHONPATH=src:third_party/ODE-GS WANDB_MODE=disabled
J='"token_dim":192,"layers":5,"heads":6,"schedule":"cosine"'
for c in "{$J,\"context_len\":4}" "{$J,\"context_len\":8,\"cv_k\":4}" "{$J,\"context_len\":16,\"cv_k\":4}" "{$J,\"context_len\":8,\"cv_k\":4,\"gaussians_per_step\":16384}"; do
  python -m gsjepa.real.fair bench --method jepa --cfg "$c" 2>&1 | grep bench
done
O='"d_model":256,"latent_dim":128,"ode_nhidden":256,"decoder_nhidden":256'
for c in "{$O}" "{$O,\"batch_size\":4096}" "{$O,\"obs_time_span\":0.3,\"obs_points\":15}" "{}"; do
  python -m gsjepa.real.fair bench --method odegs --cfg "$c" 2>&1 | grep bench
done
