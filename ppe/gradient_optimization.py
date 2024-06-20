import jax.numpy as jnp
from jax import grad, jacobian, jit
from .dirichlet import Dirichlet


class optimize_ppe(Dirichlet):  ### closed form is assumed!!!

    def __init__(self, alpha, J, ppd):
        super().__init__(alpha, J)
        self.ppd = ppd

    ### We assume that we have as input the prior probability distribution in closed form, for one partition.
    ### For instance, if Y~N(0,1) and we have a partition A=(a,b], then ppd = P(YεA) = Φ(b) - Φ(a)
    ### This input ("ppd") is assumed to have three parameters:
    # 1) the partition (in the form of interval in the continuous case, or a single value in the discrete case).
    # 2) the hyperparameters "lam".
    # 3( the covariates of the model, if any. If there aren't any covariates, "ppd" is only defined by the first two inputs.

    ## To keep track of the dimensions, suppose that we have m hyperparameters and n partitions for a given covariate set j (j \in {1,...,J})

    ## We also account for the presence of covariate sets. Specifically, we add in some functions a parameter called "covariates" that is an array
    ## and represents each individual covariate set. The parameter "total_covariates" represents the quantity that contains all covariate sets and
    ## is either a list of covariate sets, or an array where each row corresponds to one covariate set. We initialize both parameters with None.
    ## If there are no covariates in the data, we simply have None in their place and they are not included in any computation.

    ## Here, we want to define the prior probability distribution for the partition of one covariate set j (j \in {1,...,J})).
    ## We will create an array that has as many components as there are partitions and contains in the i-th position the ppd for the i-th partition
    ## It takes as an input the partition, the hyperparameters \lambda and the covariates (if any).

    def ppd_function(self, partitions, lam, covariates=None):

        if covariates is not None:
            prior_pd = [
                self.ppd(partition, lam, covariates) for partition in partitions
            ]

        else:
            prior_pd = [self.ppd(partition, lam) for partition in partitions]

        return jnp.array(prior_pd)  # shape (n, 1)

    ## Now, we compute the gradient (jacobian) of the prior probability distribution with respect to \lambda for one covariate set j (j \in {1,...,J})).

    def grad_ppd_lambda(self, partitions, lam, covariates=None):

        if covariates is not None:
            return jacobian(
                lambda lam: self.ppd_function(partitions, lam, covariates), argnums=0
            )(
                lam
            )  # shape (m, n)

        else:
            return jacobian(lambda lam: self.ppd_function(partitions, lam), argnums=0)(
                lam
            )  # shape (m, n)

    ## Finally, we compute the dirichlet likelihood gradient with respect to lambda. This will be used to perform gradient descent.

    def grad_dirichlet_lambda(
        self, lam, total_partitions, total_covariates, total_expert_probs, index
    ):

        total_model_probs = [
            jnp.array(self.ppd_function(total_partitions[j], lam, total_covariates[j]))
            for j in range(self.J)
        ]  ## The model probabilities given the hyperparameters \lambda for all j=1,...,J

        grad_dir_p = self.grad_dirichlet_p(
            total_model_probs, total_expert_probs, index
        )  ## The gradient of the Dirichlet llik with respect to the model probabilities for the j'th covariate set

        jac_p_lambda = self.grad_ppd_lambda(
            total_partitions[index], lam, total_covariates[index]
        )

        dir_grad_lambda = jac_p_lambda.T @ (grad_dir_p.T)

        dir_grad_lambda = dir_grad_lambda.T

        return dir_grad_lambda  ## shape (m,1)

    ## Alternative computation of the dirichlet likelihood gradient with respect to \lambda. Here, we define the likelihood with
    ## respect to \lambda and we take the gradient right away, without the need of further computations and the use of chain rule.

    def grad_dirichlet_lambda_2(
        self, lam, total_partitions, total_covariates, total_expert_probs, index
    ):

        total_model_probs = lambda lam: [
            jnp.array(self.ppd_function(total_partitions[j], lam, total_covariates[j]))
            for j in range(self.J)
        ]  ## The model probabilities given the hyperparameters \lambda for all j=1,...,J

        log_lik = lambda lam: self.llik(
            total_model_probs(lam), total_expert_probs, index
        )

        return -grad(log_lik, argnums=0)(lam)  ## shape (m,1)

    ## If we have multiple covariate sets (J), we need to sum the gradients (implemented with "grad_dirichlet_lambda", although "grad_dirichlet_lambda_2") would lead to the exact same results.

    def sum_grad_dirichlet_lambda(
        self, total_partitions, lam, total_expert_probs, total_covariates=None
    ):

        total_dir_grad_lambda = jnp.zeros(len(lam))

        for j in range(self.J):

            total_dir_grad_lambda += self.grad_dirichlet_lambda_2(
                lam, total_partitions, total_covariates, total_expert_probs, j
            )

        return total_dir_grad_lambda  ## shape (m,1)

    def gradient_descent(
        self,
        total_partitions,
        total_expert_probs,
        lam_0,
        iters,
        step_size,
        tol,
        total_covariates=None,
        get_lik_and_grad_progression=True,
    ):

        lam_old = lam_0  ## initial value for the hyperparameters

        total_covariates = (
            total_covariates if total_covariates is not None else [None] * self.J
        )  ## If we have covariates we leave them as is, otherwise we replace them with a list of None

        lik_progression = []
        grad_progression = []

        for i in range(iters):

            prev_model_probs = [
                jnp.array(
                    self.ppd_function(total_partitions[j], lam_old, total_covariates[j])
                )
                for j in range(self.J)
            ]

            prev_lik = self.sum_llik(prev_model_probs, total_expert_probs)

            lik_progression.append(prev_lik)

            grad_dir_lam = self.sum_grad_dirichlet_lambda(
                total_partitions, lam_old, total_expert_probs, total_covariates
            )

            grad_progression.append(jnp.linalg.norm(grad_dir_lam))

            lam_new = lam_old - step_size * grad_dir_lam

            curr_model_probs = [
                jnp.array(
                    self.ppd_function(total_partitions[j], lam_new, total_covariates[j])
                )
                for j in range(self.J)
            ]

            curr_lik = self.sum_llik(curr_model_probs, total_expert_probs)

            if (abs(curr_lik - prev_lik) < tol):  ## Stopping criterion: the dirichlet log likelihood changes less than "tol" between two iterations
                break

            lam_old = lam_new

        if get_lik_and_grad_progression:
            return lam_new, -jnp.array(lik_progression), jnp.array(grad_progression)

        return lam_new

    ## Function to get the \alpha estimate based on the MLE formula, using as inputs the expert probabilities and
    ## the partitions, hyperparameters lambda and covariate sets.

    def get_alpha(
        self, total_partitions, best_lam, total_expert_probs, total_covariates=None
    ):

        total_covariates = (
            total_covariates if total_covariates is not None else [None] * self.J
        )

        best_model_probs = [
            jnp.array(
                self.ppd_function(total_partitions[j], best_lam, total_covariates[j])
            )
            for j in range(self.J)
        ]

        alpha = self.alpha_mle(best_model_probs, total_expert_probs)

        return alpha
