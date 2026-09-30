"""Affine-coupling VAIKAE kernel.

The paper (arXiv:2609.37435, section 3) writes the latent embedding as a
diagonal Gaussian and rolls it out with z_t = K^t z_0. Two choices here are
fixed by the task, not by a loose reading of the paper:

- chi's last d outputs are unconstrained log standard deviations.
  sigma = softplus(log_sigma) + 1e-4, and Sigma_0 = diag(sigma**2).
  The paper calls those outputs both variance coefficients and a diagonal
  covariance. This code treats them as log standard deviations.
- Training losses average t = 1..T of the mean rollout. The paper's sums
  start at tau = 0 and take a sample expectation. At t = 0 the mean-form
  prediction and linearity residuals are zero, so the average starts at 1.
  sample() is the one-draw path the paper uses for trajectories.

The state log-density adds COV_JITTER (1e-5) to the diagonal of the
invertible covariance block before the Gaussian logpdf. K^t can make that
block nearly singular, and the jitter keeps the Cholesky finite.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

# Diagonal jitter on the invertible covariance before the log-density.
# Documented here and in the README. Not applied to latent_moments.
COV_JITTER = 1e-5
SIGMA_FLOOR = 1e-4


def _coerce_t(t):
    """Python int time index. Raises ValueError if t < 0."""
    if isinstance(t, jax.Array):
        if getattr(t, "shape", None) != ():
            raise ValueError(f"t must be a nonnegative integer, got shape {t.shape}")
        t = int(t)
    else:
        t = int(t)
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    return t


class AffineCoupling(eqx.Module):
    """One-split affine coupling, exactly invertible.

    For even n, x = (x_a, x_b) with each half of size n/2:

        y_a = x_a
        y_b = x_b * exp(s(x_a)) + t(x_a)
        log|det d phi/dx| = sum(s(x_a))

    s and t are MLPs with one hidden layer of width 16.
    """

    s: eqx.nn.MLP
    t: eqx.nn.MLP
    n: int = eqx.field(static=True)

    def __init__(self, n, *, key):
        if n % 2 != 0:
            raise ValueError(f"n must be even, got {n}")
        k_s, k_t = jax.random.split(key)
        half = n // 2
        self.s = eqx.nn.MLP(half, half, 16, depth=1, key=k_s)
        self.t = eqx.nn.MLP(half, half, 16, depth=1, key=k_t)
        self.n = n

    def forward(self, x):
        x = jnp.asarray(x)
        half = self.n // 2
        x_a = x[:half]
        x_b = x[half:]
        scale = self.s(x_a)
        shift = self.t(x_a)
        y_b = x_b * jnp.exp(scale) + shift
        y = jnp.concatenate([x_a, y_b])
        logdet = jnp.sum(scale)
        return y, logdet

    def inverse(self, y):
        y = jnp.asarray(y)
        half = self.n // 2
        y_a = y[:half]
        y_b = y[half:]
        scale = self.s(y_a)
        shift = self.t(y_a)
        x_b = (y_b - shift) * jnp.exp(-scale)
        return jnp.concatenate([y_a, x_b])


class VAIKAE(eqx.Module):
    """Variational augmented invertible Koopman autoencoder, small kernel.

    phi: R^n -> R^n affine coupling.
    chi: R^n -> R^{p+d} MLP, width 32, one hidden layer.
    K: (d, d) Koopman matrix, initialized at I + 0.01 * normal.
    d = n + p.
    """

    phi: AffineCoupling
    chi: eqx.nn.MLP
    K: jax.Array
    n: int = eqx.field(static=True)
    p: int = eqx.field(static=True)

    def __init__(self, n, p, *, key):
        if n % 2 != 0:
            raise ValueError(f"n must be even, got {n}")
        if p < 0:
            raise ValueError(f"p must be non-negative, got {p}")
        k_phi, k_chi, k_koop = jax.random.split(key, 3)
        d = n + p
        self.phi = AffineCoupling(n, key=k_phi)
        self.chi = eqx.nn.MLP(n, p + d, 32, depth=1, key=k_chi)
        self.K = jnp.eye(d) + 0.01 * jax.random.normal(k_koop, (d, d))
        self.n = n
        self.p = p

    def embed(self, y0):
        """Return (mu0, sigma) for observation y0. sigma is the std, shape (d,)."""
        y0 = jnp.asarray(y0)
        mu_i, _ = self.phi.forward(y0)
        chi_out = self.chi(y0)
        mu_a = chi_out[: self.p]
        log_sigma = chi_out[self.p :]
        sigma = jax.nn.softplus(log_sigma) + SIGMA_FLOOR
        mu0 = jnp.concatenate([mu_i, mu_a])
        return mu0, sigma

    def latent_moments(self, mu0, sigma, t):
        """mu_t = K^t mu0, Sigma_t = K^t diag(sigma**2) (K^t)^T.

        t = 0 returns mu0 and diag(sigma**2). No jitter is added here.
        """
        t = _coerce_t(t)
        mu0 = jnp.asarray(mu0)
        sigma = jnp.asarray(sigma)
        kt = jnp.linalg.matrix_power(self.K, t)
        mu_t = kt @ mu0
        sigma0 = jnp.diag(sigma**2)
        sigma_t = kt @ sigma0 @ kt.T
        return mu_t, sigma_t

    def _state_log_prob(self, x, mu_t, sigma_t):
        """log N(phi(x); mu_t[:n], Sigma_t[:n, :n] + jitter I) + log|det d phi|."""
        n = self.n
        mu_i = mu_t[:n]
        sigma_i = sigma_t[:n, :n]
        # 1e-5 diagonal jitter: a near-singular K^t block must not NaN the logpdf.
        eye = jnp.eye(n, dtype=sigma_i.dtype)
        sigma_i = sigma_i + COV_JITTER * eye
        z, logdet = self.phi.forward(x)
        log_n = jax.scipy.stats.multivariate_normal.logpdf(z, mu_i, sigma_i)
        return log_n + logdet

    def log_prob(self, x, y0, t):
        """State log-density log p(x | y0) at integer time t. Eq. 10 plus jitter."""
        t = _coerce_t(t)
        mu0, sigma = self.embed(y0)
        mu_t, sigma_t = self.latent_moments(mu0, sigma, t)
        return self._state_log_prob(x, mu_t, sigma_t)

    def log_prob_with_sigma(self, x, y0, t, sigma):
        """Same as log_prob, but with a caller-supplied std of shape (d,)."""
        t = _coerce_t(t)
        mu0, _ = self.embed(y0)
        mu_t, sigma_t = self.latent_moments(mu0, sigma, t)
        return self._state_log_prob(x, mu_t, sigma_t)

    def decode_mean(self, y0, t):
        """phi^{-1} of the invertible part of the latent mean at time t."""
        t = _coerce_t(t)
        mu0, sigma = self.embed(y0)
        mu_t, _ = self.latent_moments(mu0, sigma, t)
        return self.phi.inverse(mu_t[: self.n])

    def sample(self, y0, t, key):
        """One latent draw: z0 ~ N(mu0, Sigma0), z_t = K^t z0, x = phi^{-1}(z_t[:n])."""
        t = _coerce_t(t)
        mu0, sigma = self.embed(y0)
        z0 = mu0 + sigma * jax.random.normal(key, mu0.shape)
        kt = jnp.linalg.matrix_power(self.K, t)
        z_t = kt @ z0
        return self.phi.inverse(z_t[: self.n])

    def _orth(self):
        d = self.K.shape[0]
        gram = self.K @ self.K.T - jnp.eye(d, dtype=self.K.dtype)
        return jnp.sum(gram**2)

    def loss(self, y, alpha=0.1, beta=0.01, gamma=0.1, *, key=None):
        """Combined loss on one trajectory y of shape (T+1, n).

        y[0] is the conditioning observation. Terms are means over t = 1..T.

        pred: mean squared error of phi^{-1}(mu_t[:n]) against y[t].
        lin: mean of ||mu_t[:n] - phi(y[t])||^2.
        orth: ||K K^T - I||_F^2.
        lkl: mean of -log p(y[t] | y[0]).
        total: pred + alpha * lin + beta * orth + gamma * lkl.

        If key is given, also returns pred_sample and lin_sample from one
        reparameterized draw of z0. The training loss ignores that draw.
        """
        y = jnp.asarray(y)
        if y.ndim < 1 or y.shape[0] < 2:
            raise ValueError(
                f"y must have a time axis of length at least 2, got shape {y.shape}"
            )
        y0 = y[0]
        mu0, sigma = self.embed(y0)
        steps = y.shape[0] - 1

        pred_terms = []
        lin_terms = []
        lkl_terms = []
        pred_s_terms = []
        lin_s_terms = []
        z0 = None
        if key is not None:
            z0 = mu0 + sigma * jax.random.normal(key, mu0.shape)

        for t in range(1, steps + 1):
            mu_t, sigma_t = self.latent_moments(mu0, sigma, t)
            decoded = self.phi.inverse(mu_t[: self.n])
            pred_terms.append(jnp.mean((decoded - y[t]) ** 2))
            phi_y, _ = self.phi.forward(y[t])
            lin_terms.append(jnp.sum((mu_t[: self.n] - phi_y) ** 2))
            lkl_terms.append(-self._state_log_prob(y[t], mu_t, sigma_t))
            if z0 is not None:
                kt = jnp.linalg.matrix_power(self.K, t)
                z_t = kt @ z0
                x_t = self.phi.inverse(z_t[: self.n])
                pred_s_terms.append(jnp.mean((x_t - y[t]) ** 2))
                lin_s_terms.append(jnp.sum((z_t[: self.n] - phi_y) ** 2))

        pred = jnp.mean(jnp.stack(pred_terms))
        lin = jnp.mean(jnp.stack(lin_terms))
        orth = self._orth()
        lkl = jnp.mean(jnp.stack(lkl_terms))
        total = pred + alpha * lin + beta * orth + gamma * lkl
        out = {
            "total": total,
            "pred": pred,
            "lin": lin,
            "orth": orth,
            "lkl": lkl,
        }
        if z0 is not None:
            out["pred_sample"] = jnp.mean(jnp.stack(pred_s_terms))
            out["lin_sample"] = jnp.mean(jnp.stack(lin_s_terms))
        return out
