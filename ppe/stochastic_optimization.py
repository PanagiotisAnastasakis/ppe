import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from ppe.dirichlet import dirichlet_log_likelihood, alpha_mle_


def likelihood_priorpredprob_grad(alpha, probs, expert_probs):

    def likelihood_fn(probs):
        if alpha is None:
            alpha = alpha_mle_(probs, expert_probs)
        return dirichlet_log_likelihood(alpha, probs, expert_probs)

    grad_fn = jax.grad(likelihood_fn)
    likelihood_gradient = grad_fn(alpha, probs, expert_probs)
    return likelihood_gradient


def stochastic_derivative():
    pivot_sample = sampler_fn(rng_key, num_samples)

    def function_to_optimize(lambd, a, b):
        theta = pivot_fn(lambd, pivot_sample)
        return (cdf_fn(theta, b) - cdf_fn(theta, a)).mean()

    stochastic_gradient = jax.grad(function_to_optimize)
    return stochastic_gradient


if __name__ == "__main__":
    # Try simple Gaussian example
    sigma = 1.0
    rng_key = jr.key(0)
    num_samples = 10
    sampler_fn = jr.normal
    cdf_fn = lambda theta, a: jss.norm.cdf(a, loc=theta, scale=sigma)
    pivot_fn = lambda lambd, z: lambd[0] + lambd[1] * z
