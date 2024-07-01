import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from .dirichlet import dirichlet_log_likelihood, alpha_mle_


def set_derivative_bernoulli_fn(
    rng_key,
    num_samples,
    sampler_fn,
    pmf_fn,
    pivot_fn,
):
    @jax.jit
    def nonstochastic_derivative(alpha, probs, expert_probs):
        # Compute the gradient of the dirichlet likelihood wrt the probabilities

        if alpha is None:
            # TODO: Implement MLE for alpha
            def likelihood_fn(probs):
                # Make the computation of alpha depend on the current probs
                alpha = alpha_mle_(probs, expert_probs)
                return dirichlet_log_likelihood(alpha, probs, expert_probs)

        else:
            likelihood_fn = lambda probs: dirichlet_log_likelihood(
                alpha, probs, expert_probs
            )

        grad_fn = jax.value_and_grad(likelihood_fn)
        return grad_fn(probs)

    @jax.jit
    def stochastic_derivative(lambd, x):
        # Compute stochastic gradient of each prior predictive probability wrt the hyperparameters

        pivot_sample = sampler_fn(rng_key, num_samples)

        def function_to_optimize(lambd, x):
            theta = pivot_fn(lambd, pivot_sample)
            return pmf_fn(theta, x).mean()

        stochastic_gradient_fn = jax.value_and_grad(function_to_optimize)

        return stochastic_gradient_fn(lambd, x)

    return nonstochastic_derivative, stochastic_derivative


def set_derivative_continous_fn(
    num_samples,
    sampler_fn,
    cdf_fn,
    pivot_fn,
    alpha,
    partitions,
    expert_probs,
):

    @jax.jit
    def derivative_fn(lambd, rng_key):

        def function_to_optimize(lambd):
            def get_probabilities(lambd, partition, rng_key):
                a, b = partition
                pivot_sample = sampler_fn(rng_key, (num_samples,))
                theta = pivot_fn(lambd, pivot_sample)
                return (cdf_fn(theta, b, lambd) - cdf_fn(theta, a, lambd)).mean()

            keys = jr.split(rng_key, len(partitions))
            vmap_stochastic_derivative = jax.vmap(
                get_probabilities, in_axes=(None, 0, 0)
            )
            probs = vmap_stochastic_derivative(lambd, partitions, keys)

            if alpha is None:
                # If alpha is not provided, compute the MLE
                def neg_log_likelihood_fn(probs):
                    # Make the computation of alpha depend on the current probs
                    alpha = alpha_mle_(probs, expert_probs)
                    return -dirichlet_log_likelihood(alpha, probs, expert_probs)

            else:
                neg_log_likelihood_fn = lambda probs: -dirichlet_log_likelihood(
                    alpha, probs, expert_probs
                )
            return neg_log_likelihood_fn(probs), probs

        # (loss, aux_info), grads
        grad_fn = jax.value_and_grad(function_to_optimize, has_aux=True)
        return grad_fn(lambd)

    return derivative_fn
