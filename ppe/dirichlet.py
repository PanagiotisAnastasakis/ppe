import jax.numpy as jnp
from jax import grad
from jax.scipy.special import gammaln


## Class that contains functions related to the dirichlet distribution that are necessary to optimize the hyperparameters \lambda


class Dirichlet:

    def __init__(self, alpha, J):
        self.alpha = alpha
        self.J = J

    ## In the following functions:

    ## sample_probs and sample_expert_probs are the same quantities but for multiple sets of covariates (J), each of which may have different partitions
    ## They are formatted as lists of arrays, with each array corresponding to one covariate set (thus J arrays in total)

    ## Function to calculate the approximation of the MLE of alpha

    def alpha_mle(self, total_model_probs, total_expert_probs):

        # Assert shapes match
        assert self.J == len(total_model_probs)
        assert self.J == len(total_expert_probs)
        # Assert probabilities sum to 1
        self.probabilities_check(total_model_probs)
        self.probabilities_check(total_expert_probs)

        nom = 0
        den = 0

        for j in range(self.J):

            n_j = len(total_model_probs[j])

            nom += (n_j - 1) / 2

            kl_divergence = -jnp.sum(
                total_model_probs[j]
                * (jnp.log(total_expert_probs[j]) - jnp.log(total_model_probs[j]))
            )

            den += kl_divergence

        return nom / den

    ## Function for log likelihood for J=1. We have as inputs sample_probs and sample_expert_probs and an index (j \in {1,...,J}).
    ## If we have a fixed \alpha as input, we use this as input for the computation, alternatively we compute it according to the
    ## MLE formula, using all the covariate sets (all j=1,...,J).

    def llik(self, total_model_probs, total_expert_probs, index=None):

        probs = total_model_probs[index] if index is not None else total_model_probs
        expert_probs = (
            total_expert_probs[index] if index is not None else total_expert_probs
        )

        reset = 0

        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle(
                total_model_probs, total_expert_probs
            )  ## we include all the probabilities to compute alpha!

        output = log_likelihood(
            alpha=self.alpha, probs=probs, expert_probs=expert_probs
        )

        if reset == 1:
            self.alpha = None

        return output

    ## Sum of log-likelihoods for j=1,...,J. Same as before, \alpha is either fixed or computed using the MLE formula

    def sum_llik(self, total_model_probs: list, total_expert_probs: list):

        # Assert probabilities sum to 1
        self.probabilities_check(total_model_probs)
        self.probabilities_check(total_expert_probs)

        reset = 0

        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle(total_model_probs, total_expert_probs)

        total_llik = 0

        for j in range(self.J):

            total_llik += self.llik(total_model_probs, total_expert_probs, j)

        if reset == 1:
            self.alpha = None

        return total_llik

    ## Function to compute the gradient of the dirichlet log likelihood for one specific index (one j \in {1,...,J})
    ## with respect to this probability vector, using automatic differentiation. In order to compute this, we fix all other probabilitity vectors and
    ## we define the log likelihood with respect to the vector with respect to which we compute the gradient.
    ## This supports either fixed \alpha or using the MLE formula. In the latter case, the formula is dependent on the
    ## vector we take the derivative with, meaning that we eventually take the derivative of the MLE formula.

    def grad_dirichlet_p(self, total_model_probs, total_expert_probs, index=None):

        def llik_index(sample_probs_index):

            # Replace the i-th probability vector in total_model_probs with total_model_probs[index], keeping the rest unchanged
            sample_probs_index_new = (
                total_model_probs[:index]
                + [sample_probs_index]
                + total_model_probs[index + 1 :]
            )

            return self.llik(sample_probs_index_new, total_expert_probs, index)

        # Compute the gradient of llik_index with respect to total_model_probs
        return -grad(llik_index)(total_model_probs[index])

    def probabilities_check(self, list_probs):
        assert jnp.all(
            jnp.isclose(
                jnp.array([jnp.sum(probs) for probs in list_probs]),
                jnp.ones(self.J),
            )
        ), "Probabilities must sum to 1"


def log_likelihood(alpha, probs, expert_probs):
    loggamma_alpha = gammaln(alpha)

    num_1 = loggamma_alpha
    den_1 = jnp.sum(jnp.array([gammaln(alpha * probs)]))
    pt_1 = num_1 - den_1
    pt_2 = jnp.sum((alpha * probs - 1) * jnp.log(expert_probs))

    return pt_1 + pt_2
