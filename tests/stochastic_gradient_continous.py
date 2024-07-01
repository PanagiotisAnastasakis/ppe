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
from ppe.stochastic_optimization import set_derivative_continous_fn


def get_gaussian_probs(partitions, lam):
    #  Vectorized Gaussian CDF of simple Gaussian model
    mu_1 = lam[0]
    sigma = lam[1]
    sigma_1 = lam[2]

    p1 = jss.norm.cdf(
        (partitions[:, 1] - mu_1) / jnp.sqrt(sigma**2 + sigma_1**2)
    ) - jss.norm.cdf((partitions[:, 0] - mu_1) / jnp.sqrt(sigma**2 + sigma_1**2))

    return p1


# TODO: Add test for alpha_mle
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

    derivative_fn = set_derivative_continous_fn(
        num_samples,
        sampler_fn,
        cdf_fn,
        pivot_fn,
        alpha,
        partitions,
        expert_probs,
    )
    (value, estimated_probs), derivative = derivative_fn(lambd_0, rng_key)

    def test_fn(lambd):
        probs = get_gaussian_probs(partitions, lambd)
        return -dirichlet_log_likelihood(alpha, probs, expert_probs), probs

    (test_value, probs_test), test_grad = jax.value_and_grad(test_fn, has_aux=True)(
        lambd_0
    )
    assert jnp.allclose(test_value, value, atol=1e-1)
    assert jnp.allclose(probs_test, estimated_probs, atol=1e-1)
    assert jnp.allclose(test_grad, derivative, atol=1e-1)
    print("Test passed")
