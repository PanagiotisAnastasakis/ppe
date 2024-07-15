import numpy as np
from ax import optimize
import scipy.stats as sps

from ppe.dirichlet import Dirichlet
from ppe.computing_probabilities import PPEProbabilities

## Class to perform Bayesian Optimization to find the optimal hyperparameters for the priors


class Bayesian_Optimization(Dirichlet, PPEProbabilities):
    
    """
    Inputs:

    - "pymc_sampling_func" -> The PyMC function that defines our probabilistic model for the target (i.e. the prior predictive distribution)
    - "J" -> The number of covariate sets that the expert gives probabilistic judgements for
    - "alpha" -> A value for the hyperparameter alpha of the dirichlet likelihood. If None, it is optimized along with all other hyperparameters
    - "target_type" -> parameter for whether the target is continuous or discrete. Options: ["continuous", "discrete"]
    - "target_samples" -> the number of samples that we draw from the prior predictive distribution

    """

    def __init__(self, pymc_sampling_func, J, alpha, target_type, target_samples=500):
        Dirichlet.__init__(self, alpha, J)
        PPEProbabilities.__init__(self, target_type, path=False)
        self.pymc_sampling_func = pymc_sampling_func
        self.target_samples = target_samples

    ## Function to get the model probabilities based on the acquired samples from the prior predictive distribution.

    """
    Inputs:

    - "lam" -> the hyperparameters
    - "partitions" -> the partitioning for the target space for each covariate set. Can be either a list (J=1) or a list of lists for each covariate set (J>1)

    """

    def get_model_probs(self, lam, partitions, num_samples=None):

        idata = (
            self.pymc_sampling_func(lam, self.target_samples)
            if num_samples is None
            else self.pymc_sampling_func(lam, num_samples)
        )

        prior_predictive_samples = idata.prior_predictive["Y_obs"][0]

        samples_list = (
            [prior_predictive_samples[:, j] for j in range(self.J)]
            if self.J > 1
            else prior_predictive_samples
        )

        samples = np.vstack(samples_list).T if self.J > 1 else samples_list

        model_probs = self.ppd_probs(samples=samples, partitions=partitions)

        return model_probs

    ## Function to get the negative of dirichlet log likelihood.

    """
    Inputs:

    - "lam" -> the hyperparameters
    - "partitions" -> the partitioning for the target space for each covariate set. Can be either a list/array (J=1) or a list/array of lists/arrays for each covariate set (J>1)
    - "expert_probs" -> the expert's probabilistic judgements for each covariate set. Can be either a list (J=1) or a list of lists for each covariate set (J>1)

    """

    def dirichlet_neg_llik(self, lam, partitions, expert_probs):

        model_probs = self.get_model_probs(lam, partitions)

        reset = 0
        if self.alpha is None:
            self.alpha = lam["alpha"]
            reset = 1

        dir_llik = self.sum_llik(
            total_model_probs=model_probs, total_expert_probs=expert_probs
        )

        if reset == 1:
            self.alpha = None

        return float(-dir_llik)

    ## Function to include hyperpriors for the hyperparameters.

    ## The idea is that if the user can give an input regarding some of the prior hyperparameters, then we can take that into consideration by adding
    ## hyperpriors in the loss function which encourage values close to that input. Additionally, the user could give a confidence value between 0 and 1
    ## regarding this input, which will act as weight for the hyperprior.

    ## If hyperpriors are employed, the loss will be: -dir_log_likelihood - log_hyperpriors * weights * J

    ## The hyperprior is a truncated gaussian with mean = user input, bounds that are centered around the mean and are of the same width as the ones used for the optimization
    ## and standard deviation is half the length of the bounds.

    """
    Inputs:

    - "lam" -> the hyperparameters
    - "param_bounds" -> the bounds that define the search space for each hyperparameter. List of lists, each corresponding to one hyperparameter and consisting of a lower and upper bound
    - "param_expected_vals" -> the expert's estimates for the hyperparameters. a list of numbers that accepts also None if no such input can be given.
    - "param_weights" -> the level of confidence for these expected value (a list of values)

    """

    def hyperprior_llik(self, lam, param_bounds, param_expected_vals, param_weights):

        llik = 0

        param_values = list(lam.values())

        for m in range(len(param_values)):

            if param_expected_vals[m] is not None:

                bound = param_bounds[m]
                range_length = bound[1] - bound[0]

                mu = param_expected_vals[m]

                lower_trunc_normal = mu - range_length / 2
                upper_trunc_normal = mu + range_length / 2

                sigma = (
                    upper_trunc_normal - lower_trunc_normal
                ) / 2  ## same as range_length/2

                param_llik = sps.truncnorm.logpdf(
                    param_values[m],
                    loc=mu,
                    scale=sigma,
                    a=lower_trunc_normal,
                    b=upper_trunc_normal,
                )

                if np.isinf(param_llik):
                    continue

                llik += param_llik * param_weights[m] * self.J

        return float(-llik)

    ## Function to optimize the hyperparameters using the function "optimize" from "Ax".

    """
    Inputs:

    - "param_names" -> the names for the hyperparameters in a list
    - "param_bounds" -> the bounds that define the search space for each hyperparameter. List of lists, each corresponding to one hyperparameter and consisting of a lower and upper bound
    - "param_expected_vals" -> the expert's estimates for the hyperparameters. a list of numbers that accepts also None if no such input can be given.
    - "param_weights" -> the level of confidence for these expected value (a list of values)
    - "partitions" -> the partitioning for the target space for each covariate set. Can be either a list/array (J=1) or a list/array of lists/arrays for each covariate set (J>1)
    - "expert_probs" -> the expert's probabilistic judgements for each covariate set. Can be either a list (J=1) or a list of lists for each covariate set (J>1)
    - "n_trials" -> the number of trials for the Bayesian Optimization algorithm
    - "return_value" -> Boolean on whether to return the value of the objective function

    """

    def optimize_hyperparams(
        self,
        param_names: list,
        param_types: list,
        param_bounds: list,
        param_expected_vals: list,
        param_weights: list,
        partitions: np.ndarray,
        expert_probs: list,
        n_trials=100,
        return_value=False,
    ):

        dir_neg_llik = lambda lam: self.dirichlet_neg_llik(
            lam, partitions, expert_probs
        ) + self.hyperprior_llik(lam, param_bounds, param_expected_vals, param_weights)

        def create_param_dict(name, type, bound):
            if type == "range":
                return {"name": name, "type": type, "bounds": bound}
            return {"name": name, "type": type, "values": bound}

        parameters = [
            create_param_dict(name, type, bound)
            for name, type, bound in zip(param_names, param_types, param_bounds)
        ]

        best_lam, values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=dir_neg_llik,
            objective_name="Dirichlet_negative_log_likelihood",
            minimize=True,
            total_trials=n_trials,
        )

        if return_value:
            return best_lam, dir_neg_llik(best_lam)

        return best_lam

    ## Function to evaluate a set of hyperparameters by computing alpha

    """
    Inputs:

    - "lam" -> the hyperparameters
    - "partitions" -> the partitioning for the target space for each covariate set. Can be either a list/array (J=1) or a list/array of lists/arrays for each covariate set (J>1)
    - "expert_probs" -> the expert's probabilistic judgements for each covariate set. Can be either a list (J=1) or a list of lists for each covariate set (J>1)

    """

    def eval_function(self, lam, partitions, expert_probs):

        model_probs = self.get_model_probs(lam, partitions)

        alpha = self.alpha_mle(
            total_model_probs=model_probs, total_expert_probs=expert_probs
        )

        return alpha
