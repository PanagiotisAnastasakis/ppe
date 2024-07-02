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
from ppe.optimization_loop import optimization_loop
import matplotlib.pyplot as plt

"""
Objective:
 - Optimize parameters of Gaussian prior and likelihood in a simple Gaussian example.
"""


def get_gaussian_probs(partitions, lambd):
    #  Vectorized Gaussian CDF of simple Gaussian model
    mu_1, sigma = lambd[0]
    sigma_1 = lambd[1]

    p1 = jss.norm.cdf(
        (partitions[:, 1] - mu_1) / jnp.sqrt(sigma**2 + sigma_1**2)
    ) - jss.norm.cdf((partitions[:, 0] - mu_1) / jnp.sqrt(sigma**2 + sigma_1**2))

    return p1


def plot_flow_pdf(lambd, directory="figs/prior.png"):
    x = jnp.linspace(-5, 5, 1000)
    y = jss.norm.logpdf(x, loc=lambd[0][0], scale=lambd[0][1])
    plt.plot(x, y)
    plt.savefig(directory)
    plt.close()


if __name__ == "__main__":
    partitions = jnp.array([[-1000, -2], [-2, 3], [3, 1000]])
    expert_probs = jnp.array([0.2, 0.7, 0.1])

    # Try simple Gaussian example
    alpha = 1.0
    initial_value = [jnp.ones(2), 1.0]
    rng_key = jr.key(0)

    def optimize_fn(lambd):
        probs = get_gaussian_probs(partitions, lambd)
        return -dirichlet_log_likelihood(alpha, probs, expert_probs), probs

    @jax.jit
    def derivative_fn(lambd, rng_key):
        return jax.value_and_grad(optimize_fn, has_aux=True)(lambd)

    learning_rate = 1e-3
    num_iterations = 1000

    final_value = optimization_loop(
        initial_value, learning_rate, num_iterations, derivative_fn, rng_key
    )
