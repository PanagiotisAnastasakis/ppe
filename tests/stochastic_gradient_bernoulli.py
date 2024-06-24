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
from ppe.stochastic_optimization import set_derivative_bernoulli_fn


def get_bernoulli_probs(lam, covariate_set):
    mu, sigma = lam
    nom = jnp.dot(covariate_set, mu)
    if jnp.ndim(sigma) == 1:
        den = jnp.sqrt(1 + jnp.dot(covariate_set, sigma * covariate_set))
    else:
        den = jnp.sqrt(1 + covariate_set.T @ sigma @ covariate_set)
    value = jss.norm.cdf(nom / den)
    return jnp.array([1 - value, value])


# TODO: Add test for alpha_mle
if __name__ == "__main__":

    partitions = jnp.array([0, 1])
    expert_probs = jnp.array([0.35, 0.65])
    covariate_set = jnp.array([1.3, 0.7, 0.5, -0.7, -0.5])

    # Try simple Bernoulli example
    lattent_dim = 5
    alpha = 1.0
    mu_0 = jnp.zeros(lattent_dim)
    sigma_0 = jnp.ones(lattent_dim)
    lambd_0 = (mu_0, sigma_0)
    rng_key = jr.key(0)
    num_samples = 1_000_000
    sampler_fn = lambda key, num_samples: jr.normal(key, (num_samples, lattent_dim))
    pivot_fn = lambda lambd, z: lambd[0] + lambd[1] * z
    pmf_fn = lambda theta, x: jss.norm.cdf(jnp.dot(theta, x))
    # In this case we can obtain probs in closed form, but in general we would need stochastic estimates
    probs = get_bernoulli_probs(lambd_0, covariate_set)
    total_model_probs = [probs]
    total_expert_probs = [expert_probs]
    nonstochastic_derivative, stochastic_derivative = set_derivative_bernoulli_fn(
        rng_key,
        num_samples,
        sampler_fn,
        pmf_fn,
        pivot_fn,
        total_model_probs,
        total_expert_probs,
    )
    derivative_1 = nonstochastic_derivative(alpha, probs, expert_probs, index=0)
    _, derivative_2 = stochastic_derivative(lambd_0, covariate_set)
    derivative_2_mu, derivative_2_sigma = derivative_2
    # note that d/dλ 1-p = -d/dλ p
    derivative_mu = jnp.dot(
        jnp.stack((-derivative_2_mu, derivative_2_mu), axis=-1), derivative_1
    )
    derivative_sigma = jnp.dot(
        jnp.stack((-derivative_2_sigma, derivative_2_sigma), axis=-1), derivative_1
    )
    derivative = (derivative_mu, derivative_sigma)

    def test_fn(lambd):
        probs = get_bernoulli_probs(lambd, covariate_set)
        return dirichlet_log_likelihood(alpha, probs, expert_probs)

    test_value = jax.grad(test_fn)(lambd_0)
    print("stochastic gradient", derivative)
    print("Non stochastic gradient", test_value)
    assert jnp.allclose(test_value[0], derivative[0], atol=1e-2)
    assert jnp.allclose(test_value[1], derivative[1], atol=1e-2)
    print("Test passed")
