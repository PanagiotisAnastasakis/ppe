import jax.random as jr
import jax.numpy as jnp
import jax


def optimization_loop(
    initial_value, learning_rate, num_iterations, derivative_fn, rng_key
):
    # TODO: Add support for No likelihood params
    lambd = initial_value
    for iteration in range(num_iterations):
        rng_key, _ = jr.split(rng_key)
        (value, probs), derivative = derivative_fn(lambd, rng_key)
        grad_prior_params, grad_likelihood_params = derivative

        # Update parameters
        prior_params, likelihood_params = lambd
        prior_params = jax.tree.map(
            lambda p, dp: p - learning_rate * dp, prior_params, grad_prior_params
        )
        likelihood_params = likelihood_params - learning_rate * grad_likelihood_params

        # Set updated parameters
        lambd = [prior_params, likelihood_params]

        # Optional: print progress
        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1} - Neg Log Likelihood: {value} - Probs: {probs}"
            )
    return lambd
