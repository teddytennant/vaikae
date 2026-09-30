import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

from vaikae import AffineCoupling, VAIKAE
from vaikae.model import COV_JITTER

jax.config.update("jax_platforms", "cpu")


def _rotation(key, length, n=4, angle=0.25, noise=0.01):
    """Linear rotation in the first two coordinates, plus small noise."""
    t = jnp.arange(length)
    y = jnp.zeros((length, n))
    y = y.at[:, 0].set(jnp.cos(angle * t))
    y = y.at[:, 1].set(jnp.sin(angle * t))
    y = y.at[:, 2].set(0.4)
    y = y.at[:, 3].set(-0.2)
    y = y + noise * jax.random.normal(key, y.shape)
    return y


def test_affine_coupling_inverse_and_logdet():
    coupling = AffineCoupling(4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4,))
    y, logdet = coupling.forward(x)
    x_rec = coupling.inverse(y)
    assert jnp.allclose(x_rec, x, atol=1e-5)

    def phi(z):
        out, _ = coupling.forward(z)
        return out

    jac = jax.jacfwd(phi)(x)
    _, logabsdet = jnp.linalg.slogdet(jac)
    assert jnp.allclose(logdet, logabsdet, atol=1e-4)


def test_latent_moments_t0_and_t1():
    model = VAIKAE(4, 2, key=jax.random.PRNGKey(2))
    y0 = jax.random.normal(jax.random.PRNGKey(3), (4,))
    mu0, sigma = model.embed(y0)

    mu_0, sigma_0 = model.latent_moments(mu0, sigma, 0)
    assert jnp.allclose(mu_0, mu0, atol=1e-5)
    assert jnp.allclose(sigma_0, jnp.diag(sigma**2), atol=1e-5)

    mu_1, sigma_1 = model.latent_moments(mu0, sigma, 1)
    assert jnp.allclose(mu_1, model.K @ mu0, atol=1e-5)
    expected = model.K @ jnp.diag(sigma**2) @ model.K.T
    assert jnp.allclose(sigma_1, expected, atol=1e-5)


def _moderate_identity(model):
    """K = I, and chi's log-sigma head emits 0 so std is moderate."""
    d = model.n + model.p
    p = model.p
    weight = model.chi.layers[-1].weight
    bias = model.chi.layers[-1].bias
    weight = weight.at[p:].set(0.0)
    bias = bias.at[p:].set(0.0)
    model = eqx.tree_at(lambda m: m.chi.layers[-1].weight, model, weight)
    model = eqx.tree_at(lambda m: m.chi.layers[-1].bias, model, bias)
    eye = jnp.eye(d, dtype=model.K.dtype)
    model = eqx.tree_at(lambda m: m.K, model, eye)
    return model


def test_log_prob_at_mean_matches_hand_assembly():
    model = _moderate_identity(VAIKAE(4, 2, key=jax.random.PRNGKey(4)))
    y0 = jax.random.normal(jax.random.PRNGKey(5), (4,))
    lp = model.log_prob(y0, y0, 0)
    assert jnp.isfinite(lp)

    z, logdet = model.phi.forward(y0)
    _, sigma = model.embed(y0)
    cov = jnp.diag(sigma[:4] ** 2) + COV_JITTER * jnp.eye(4)
    expected = jax.scipy.stats.multivariate_normal.logpdf(z, z, cov) + logdet
    assert jnp.allclose(lp, expected, atol=1e-4)

    other = model.log_prob(y0 + 0.1, y0, 1)
    assert jnp.isfinite(other)


def test_orthogonality():
    model = VAIKAE(4, 2, key=jax.random.PRNGKey(6))
    y = _rotation(jax.random.PRNGKey(0), length=3)
    eye = jnp.eye(model.n + model.p, dtype=model.K.dtype)
    model_i = eqx.tree_at(lambda m: m.K, model, eye)
    orth_i = model_i.loss(y)["orth"]
    assert float(orth_i) == 0.0

    bad = jnp.ones_like(model.K)
    model_bad = eqx.tree_at(lambda m: m.K, model, bad)
    assert float(model_bad.loss(y)["orth"]) > 0.0


def test_loss_finite_and_descends():
    y = _rotation(jax.random.PRNGKey(0), length=6, n=4)
    model = VAIKAE(4, 2, key=jax.random.PRNGKey(7))
    initial_dict = model.loss(y)
    for key in ("total", "pred", "lin", "orth", "lkl"):
        assert jnp.isfinite(initial_dict[key])
    initial = initial_dict["total"]

    optim = optax.adam(1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, y):
        loss, grads = eqx.filter_value_and_grad(lambda m, batch: m.loss(batch)["total"])(
            model, y
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(40):
        model, opt_state, _ = step(model, opt_state, y)

    final = model.loss(y)["total"]
    assert jnp.isfinite(final)
    assert float(final) < float(initial)
    assert float(final) < 0.95 * float(initial)


def test_sigma_positive_and_log_prob_depends_on_it():
    model = VAIKAE(4, 2, key=jax.random.PRNGKey(8))
    y0 = jax.random.normal(jax.random.PRNGKey(9), (4,))
    x = jax.random.normal(jax.random.PRNGKey(10), (4,))
    _, sigma = model.embed(y0)
    assert jnp.all(sigma > 0)

    lp = model.log_prob(x, y0, 1)
    d = model.n + model.p
    bias = model.chi.layers[-1].bias
    shifted = bias.at[-d:].add(1.0)
    model2 = eqx.tree_at(lambda m: m.chi.layers[-1].bias, model, shifted)
    lp2 = model2.log_prob(x, y0, 1)
    assert jnp.isfinite(lp) and jnp.isfinite(lp2)
    assert abs(float(lp2) - float(lp)) > 1e-4


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        VAIKAE(3, 2, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        AffineCoupling(5, key=jax.random.PRNGKey(0))

    model = VAIKAE(4, 2, key=jax.random.PRNGKey(11))
    y0 = jnp.zeros((4,))
    mu0, sigma = model.embed(y0)
    with pytest.raises(ValueError):
        model.latent_moments(mu0, sigma, -1)
    with pytest.raises(ValueError):
        model.log_prob(y0, y0, -1)
    with pytest.raises(ValueError):
        model.decode_mean(y0, -1)
    with pytest.raises(ValueError):
        model.sample(y0, -1, jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        model.loss(jnp.zeros((1, 4)))
