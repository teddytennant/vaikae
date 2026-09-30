"""Train a tiny VAIKAE on a 2D rotation embedded in R^4.

Length-8 trajectory: a constant-speed rotation in the first two coordinates,
last two coordinates held constant. 60 Adam steps at learning rate 1e-2.
Exits 0 if the total loss decreased, else 1.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from vaikae import VAIKAE


def rotation_embedded(length, angle=0.25):
    t = jnp.arange(length)
    y = jnp.zeros((length, 4))
    y = y.at[:, 0].set(jnp.cos(angle * t))
    y = y.at[:, 1].set(jnp.sin(angle * t))
    y = y.at[:, 2].set(0.3)
    y = y.at[:, 3].set(-0.4)
    return y


def main():
    model = VAIKAE(4, 2, key=jax.random.PRNGKey(0))
    y = rotation_embedded(8)
    initial = model.loss(y)["total"]

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

    for _ in range(60):
        model, opt_state, _ = step(model, opt_state, y)

    final = model.loss(y)["total"]
    print(f"initial total {float(initial)}")
    print(f"final total {float(final)}")
    return 0 if float(final) < float(initial) else 1


if __name__ == "__main__":
    raise SystemExit(main())
