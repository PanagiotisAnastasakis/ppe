import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from ppe.stochastic_optimization import set_derivative_continous_fn
from ppe.optimization_loop import optimization_loop
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
import equinox as eqx
from flowjax import wrappers
import matplotlib.pyplot as plt


"""
Objective:
 - Fit Normalizing Flow as a prior for simple Gaussian Example.
 - Optimization loop for flow and likelihood paramters simmultaneously.
 - Plot prior pdf
"""


def plot_flow_pdf(flow, directory="figs/prior.png"):
    x = jnp.linspace(-5, 5, 1000)
    y = jnp.exp(flow.log_prob(x[:, None]))
    plt.plot(x, y)
    plt.savefig(directory)
    plt.close()


if __name__ == "__main__":
    partitions = jnp.array([[-1000, -2], [-2, 3], [3, 1000]])
    expert_probs = jnp.array([0.2, 0.7, 0.1])

    # Try simple Gaussian example
    dim_prior = 1
    alpha = 1.0

    rng_key = jr.key(0)
    rng_key, subkey = jr.split(rng_key)
    num_samples = 100
    sampler_fn = jr.normal
    flow = masked_autoregressive_flow(
        subkey,
        base_dist=Normal(jnp.zeros(dim_prior)),
        transformer=RationalQuadraticSpline(knots=2, interval=2),
        nn_depth=2,
        nn_width=32,
    )
    params, static = eqx.partition(
        flow,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
    )
    cdf_fn = lambda theta, a, lambd: jss.norm.cdf(a, loc=theta, scale=lambd[-1])

    def pivot_fn(lambd, z):
        params = lambd[0]
        flow = eqx.combine(params, static)
        return jax.vmap(flow.bijection.transform)(z[:, None])

    derivative_fn = set_derivative_continous_fn(
        num_samples,
        sampler_fn,
        cdf_fn,
        pivot_fn,
        alpha,
        partitions,
        expert_probs,
    )

    # Plot initial flow
    initial_value = [params, 1.0]
    flow = eqx.combine(initial_value[0], static)
    if not os.path.exists("figs"):
        os.makedirs("figs")
    plot_flow_pdf(flow, directory=f"figs/prior_{0}.png")

    learning_rate = 1e-3
    num_iterations = 1000

    final_value = optimization_loop(
        initial_value, learning_rate, num_iterations, derivative_fn, rng_key
    )
    flow = eqx.combine(final_value[0], static)
    plot_flow_pdf(flow, directory=f"figs/prior_{num_iterations}.png")
