import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from ppe.stochastic_optimization import set_derivative_fn
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
import equinox as eqx
from flowjax import wrappers
import matplotlib.pyplot as plt


"""
Objective:
 - Fit Normalizing Flow as a prior for simple Gaussian Example.
 - Optimization loop for flow and likelihood paramters
 - Plot prior pdf

"""


def optimizaiton_loop(initial_value, learning_rate, num_iterations):
    lambd = initial_value
    for iteration in range(num_iterations):
        vmap_stochastic_derivative = jax.vmap(stochastic_derivative, in_axes=(None, 0))
        # Estimate probs stochastically
        probs, derivative_2 = vmap_stochastic_derivative(lambd, partitions)
        derivative_1 = nonstochastic_derivative(alpha, probs, expert_probs, index=0)
        derivative_params = jax.tree.map(
            lambda x: jnp.dot(x.T, derivative_1), derivative_2[0]
        )
        derivative_sigma = jnp.dot(derivative_2[1].T, derivative_1)

        # Update parameters
        params = jax.tree.map(
            lambda p, dp: p - learning_rate * dp, lambd[0], derivative_params
        )
        sigma = lambd[1] - learning_rate * derivative_sigma

        # Set updated parameters
        lambd = [params, sigma]

        # Optional: print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}")


def plot_flow_pdf(flow):
    x = jnp.linspace(-5, 5, 1000)
    y = jnp.exp(flow.log_prob(x[:, None]))
    plt.plot(x, y)
    plt.show()


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

    total_expert_probs = [expert_probs]
    nonstochastic_derivative, stochastic_derivative = set_derivative_fn(
        rng_key,
        num_samples,
        sampler_fn,
        cdf_fn,
        pivot_fn,
        total_expert_probs,
    )

    # Plot initial flow
    # plot_flow_pdf(flow)

    initial_value = [params, 1.0]
    learning_rate = 1e-3
    num_iterations = 100
    final_value = optimizaiton_loop(initial_value, learning_rate, num_iterations)
    flow = eqx.combine(final_value[0], static)
    plot_flow_pdf(flow)
