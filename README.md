JAX kernel of VAIKAE (arXiv:2609.37435): an invertible encoder plus an augmentation encoder, a Koopman matrix, and a state likelihood from the latent Gaussian.

PYTHONPATH=src python -m pytest
PYTHONPATH=src python train.py

Does not reproduce the ETTm1, ECL, Weather, Solar, or Sentinel-2 numbers. Those need the authors' delay embedding and a deep flow. This checks the affine coupling inverse, the latent moment rollout, the change-of-variables likelihood, and that the combined loss descends on a toy rotation.

phi is one affine coupling, not a deep flow. sigma is the softplus of chi's last d outputs, plus a 1e-4 floor. The likelihood uses the upper-left block of K^t Sigma0 (K^t)^T plus a 1e-5 jitter on the diagonal. L_pred decodes the latent mean rather than a sample, so the training loss is deterministic.
