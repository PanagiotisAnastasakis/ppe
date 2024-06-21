import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from ppe.dirichlet import dirichlet_log_likelihood
from ppe.stochastic_optimization import set_derivative_fn


def get_gaussian_probs(partitions, lam):
    #  Vectorized Gaussian CDF of simple Gaussian model
    mu_1 = lam[0]
    sigma = lam[1]
    sigma_1 = lam[2]

    p1 = jss.norm.cdf(
        (partitions[:, 1] - mu_1) / jnp.sqrt(sigma**2 + sigma_1**2)
    ) - jss.norm.cdf((partitions[:, 0] - mu_1) / jnp.sqrt(sigma**2 + sigma_1**2))

    return p1


# TODO: Add test here
if __name__ == "__main__":
    partitions = jnp.array([[-1000, -2], [-2, 3], [3, 1000]])
    expert_probs = jnp.array([0.2, 0.7, 0.1])

    # Try simple Gaussian example
    alpha = 1.0
    lambd_0 = jnp.ones(3)
    rng_key = jr.key(0)
    num_samples = 1_000_000
    sampler_fn = jr.normal
    cdf_fn = lambda theta, a, lambd: jss.norm.cdf(a, loc=theta, scale=lambd[-1])
    pivot_fn = lambda lambd, z: lambd[0] + lambd[1] * z
    # In this case we can obtain probs in closed form, but in general we would need stochastic estimates
    probs = get_gaussian_probs(partitions, lambd_0)
    total_model_probs = [probs]
    total_expert_probs = [expert_probs]
    nonstochastic_derivative, stochastic_derivative = set_derivative_fn(
        rng_key,
        num_samples,
        sampler_fn,
        cdf_fn,
        pivot_fn,
        total_model_probs,
        total_expert_probs,
    )
    derivative_1 = nonstochastic_derivative(alpha, probs, expert_probs)
    vmap_stochastic_derivative = jax.vmap(stochastic_derivative, in_axes=(None, 0))
    _, derivative_2 = vmap_stochastic_derivative(lambd_0, partitions)
    derivative = jnp.dot(derivative_2.T, derivative_1)

    def test_fn(lambd):
        probs = get_gaussian_probs(partitions, lambd)
        return dirichlet_log_likelihood(alpha, probs, expert_probs)

    test_value = jax.grad(test_fn)(lambd_0)
    print("stochastic gradient", derivative)
    print("Non stochastic gradient", test_value)
    assert jnp.allclose(test_value, derivative, atol=1e-1)
    print("Test passed")
