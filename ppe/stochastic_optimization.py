import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from .dirichlet import dirichlet_log_likelihood, alpha_mle_


def replace_first_occurrence(lst, old_value, new_value):
    # Iterate through the list
    for i in range(len(lst)):
        # Check if the current element is the one to replace
        if lst[i] == old_value:
            # Replace it with the new value
            lst[i] = new_value
            # Exit the loop after the first replacement
            break
    return lst


def set_derivative_fn(
    rng_key,
    num_samples,
    sampler_fn,
    cdf_fn,
    pivot_fn,
    total_model_probs,
    total_expert_probs,
):

    @jax.jit
    def nonstochastic_derivative(alpha, probs, expert_probs):
        # Compute the gradient of the dirichlet likelihood wrt the probabilities

        if alpha is None:
            # If alpha is not provided, compute the MLE
            def likelihood_fn(probs):
                # Make the computation of alpha depend on the current probs
                total_model_probs = replace_first_occurrence(
                    total_model_probs, probs, probs
                )
                alpha = alpha_mle_(total_model_probs, total_expert_probs)
                return dirichlet_log_likelihood(alpha, probs, expert_probs)

        else:
            likelihood_fn = lambda probs: dirichlet_log_likelihood(
                alpha, probs, expert_probs
            )

        grad_fn = jax.grad(likelihood_fn)
        likelihood_gradient = grad_fn(probs)
        return likelihood_gradient

    @jax.jit
    def stochastic_derivative(lambd, partition):
        # Compute stochastic gradient of each prior predictive probability wrt the hyperparameters
        a, b = partition
        pivot_sample = sampler_fn(rng_key, (num_samples,))

        def function_to_optimize(lambd, a, b):
            theta = pivot_fn(lambd, pivot_sample)
            return (cdf_fn(theta, b, lambd) - cdf_fn(theta, a, lambd)).mean()

        stochastic_gradient_fn = jax.value_and_grad(function_to_optimize)
        return stochastic_gradient_fn(lambd, a, b)

    return nonstochastic_derivative, stochastic_derivative
