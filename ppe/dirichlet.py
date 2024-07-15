import jax.numpy as jnp
from jax import grad
from jax.scipy.special import gammaln


## Class that contains functions related to the dirichlet distribution that are necessary to optimize the hyperparameters \lambda


class Dirichlet:
    
    '''
    Inputs:
    
    - "alpha" -> A value for alpha that is used in the Dirichlet likelihood. Can be either a scalar or None.
                 In the latter case, the approximation is used.
    - "J" -> The number of covariate sets that the expert gives probabilistic judgements for
    
    '''
    
    def __init__(self, alpha, J):
        self.alpha = alpha
        self.J = J

    ## Function to calculate the approximation of the MLE of alpha
    
    '''
    Inputs:
    
    - "total_model_probs" -> The prior predictive probabilities for a given hyperparameter vector \lambda.
                             It is a list with each element being the values of the ppd for the target for each covariate set J
                             If there are no covariate sets (J=1), then it is simply a list of one element.
                             
    - "total_expert_probs" -> List of the expert's probabilistic judgements for each covariate set. It has the same
                              length as "total_model_probs".
    '''

    def alpha_mle(self, total_model_probs, total_expert_probs):

        # Assert shapes match
        assert self.J == len(total_model_probs)
        assert self.J == len(total_expert_probs)
        # Assert probabilities sum to 1
        self.probabilities_check(total_model_probs)
        self.probabilities_check(total_expert_probs)
        return alpha_mle_(total_model_probs, total_expert_probs)

    ## Function for log likelihood for one covariate set. We have as inputs total_model_probs and total_expert_probs and an index (jε{1,...,J}).
    ## If we have a fixed \alpha as input, we use this as input for the computation, alternatively we compute it according to the
    ## MLE formula, using all the covariate sets (all j=1,...,J).
    
    '''
    Inputs:
    
    - "total_model_probs", "total_expert_probs" -> Same as before
    - "index" -> the index jε{1,...,J} of the lists "total_model_probs", "total_expert_probs" that we compute the Dirichlet log likelihood for
    
    '''

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

        output = dirichlet_log_likelihood(
            alpha=self.alpha, probs=probs, expert_probs=expert_probs
        )

        if reset == 1:
            self.alpha = None

        return output

    ## Sum of log-likelihoods for j=1,...,J. Same as before, \alpha is either fixed or computed using the MLE formula

    '''
    Inputs:
    
    - "total_model_probs", "total_expert_probs" -> Same as before
    
    '''

    def sum_llik(self, total_model_probs, total_expert_probs):

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

    ## Function to compute the gradient of the Dirichlet log-likelihood for one specific index (one jε{1,...,J})
    ## with respect to this probability vector, using automatic differentiation. In order to compute this, we fix all other probabilitity vectors and
    ## we define the log likelihood as a function of the vector with respect to which we compute the gradient.
    ## This supports either fixed \alpha or using the MLE formula. In the latter case, the formula is dependent on the
    ## vector we take the derivative with, meaning that we eventually take the derivative of the MLE formula.
    
    '''
    Inputs:
    
    - "total_model_probs", "total_expert_probs" -> Same as before
    - "index" -> the index jε{1,...,J} of the lists "total_model_probs", "total_expert_probs" that we compute the Dirichlet log likelihood for
    
    '''

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



## helper functions for the formulas of the Dirirchlet log-likelihood and the approximation of \alpha

def dirichlet_log_likelihood(alpha, probs, expert_probs):
    loggamma_alpha = gammaln(alpha)

    num_1 = loggamma_alpha
    den_1 = jnp.sum(jnp.array([gammaln(alpha * probs)]))
    pt_1 = num_1 - den_1
    pt_2 = jnp.sum((alpha * probs - 1) * jnp.log(expert_probs))

    return pt_1 + pt_2


def alpha_mle_(total_model_probs, total_expert_probs):
    nom = 0
    den = 0
    J = len(total_model_probs)
    for j in range(J):

        n_j = len(total_model_probs[j])

        nom += (n_j - 1) / 2

        kl_divergence = -jnp.sum(
            total_model_probs[j]
            * (jnp.log(total_expert_probs[j]) - jnp.log(total_model_probs[j]))
        )

        den += kl_divergence

    return nom / den
