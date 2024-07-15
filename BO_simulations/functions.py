import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


from ppe.bayesian_optimization import Bayesian_Optimization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd


## Function to make partitions for a given target
## It is assumed that the target takes values between -10,000 and 10,000

"""
inputs:

- "num_bins" -> The number of bins that we have in the partitioning
- "lower_inner", "upper_inner" -> To define the partitions, we choose an interval that we use to
   to do the splitting instead of using the assumed lower and upper bounds for the target. The reason
   for this is that if we used the global bounds, we would require a very large number of bins to 
   capture useful information, which is impractical. For that reason, we choose a smaller inner interval,
   defined by "lower_inner", "upper_inner", which we use for the partitioning. For instance, if we choose
   to have "num_bins" = 4, then the partitioning would be (-10,000, lower_inner), (lower_inner, (lower_inner + upper_inner)/2)
   ((lower_inner + upper_inner)/2, upper_inner), (upper_inner, 10,000).
"""


def make_partition(num_bins, lower_inner, upper_inner):

    lower, upper = (-10000, 10000)

    assert num_bins >= 2

    if num_bins == 2:
        return np.array(
            [
                [lower, 0.5 * (lower_inner + upper_inner)],
                [0.5 * (lower_inner + upper_inner), upper],
            ]
        )

    bin_edges = np.linspace(lower_inner, upper_inner, num_bins - 1)
    bins = np.array([[bin_edges[i], bin_edges[i + 1]] for i in range(num_bins - 2)])
    bins_total = np.vstack(
        (np.array([lower, lower_inner]), bins, np.array([upper_inner, upper]))
    )

    return bins_total


## The purpose of the function "ppe_simulation" is to facilitate the simulation of PPE for a probabilistic
## model with known hyperparameters. To achieve that, we simulate the expert probabilities by sampling from
## the prior predictive distribution for the true hyperparameters. Then, we perform PPE, optimizing the dirichlet
## likelihood using Bayesian Optimization.

"""
Inputs

- "model" -> the probabilistic PyMC model. It is assumed that if there are covariates, these are defined within the model
- "J" -> The number of different covariate sets. If there are none, J=1 should be used as an input
- "target_type" -> parameter for whether the target is continuous or discrete. Options: ["continuous", "discrete"]
- "lambd_names" -> a list that contains the names for each of the hyperparameters. Required for BO
- "lambd_true_vals" -> a list containing the values of the hyperparameters that we assume are true to conduct the simulation
- "alpha" -> A value for the hyperparameter alpha of the dirichlet likelihood. If None, it is optimized along with all other hyperparameters
- "num_bins" -> The number of bins to have in the partition. If the target is discrete, it is the number of classes for the target. In that case, we assume that if there are n classes, then Yε{0,1,...,n-1}
- "lower_inner", "upper_inner" -> The lower and upper bounds of the interval that we partition
- "param_bounds" -> the bounds that define the search space for each hyperparameter. List of lists, each corresponding to one hyperparameter and consisting of a lower and upper bound
- "target_samples" -> the number of samples that we draw from the prior predictive distribution to compute the probabilities
- "n_trials" -> the number of trials for the Bayesian Optimization algorithm

"""


def ppe_simulation(
    model,
    J,
    target_type,
    lambd_names,
    lambd_true_vals,
    alpha,
    num_bins,
    lower_inner,
    upper_inner,
    param_bounds,
    target_samples,
    n_trials=75,
):

    lambd_true = {name: value for name, value in zip(lambd_names, lambd_true_vals)}

    if target_type == "continuous":
        partition = make_partition(
            num_bins=num_bins, lower_inner=lower_inner, upper_inner=upper_inner
        )
        partition = [partition] * J

    else:
        partition = np.array(range(num_bins))

    BO = Bayesian_Optimization(
        pymc_sampling_func=model,
        J=J,
        alpha=alpha,
        target_type=target_type,
        target_samples=target_samples,
    )

    simulated_expert_probs = BO.get_model_probs(
        lam=lambd_true, partitions=partition, num_samples=20_000
    )

    if alpha is None:
        lambd_names = lambd_names + ["alpha"]  ## We optimize alpha also
        param_bounds = param_bounds + [[0.0001, 2000.0]] ## The bounds used to be (0.,70.], however we increased the upper bound

    param_types = ["range"] * len(lambd_names)
    param_expected_vals = [None] * len(
        lambd_names
    )  ## we only focus on the Dirichlet log-likelihood

    best_params, best_llik = BO.optimize_hyperparams(
        param_names=lambd_names,
        param_types=param_types,
        param_bounds=param_bounds,
        param_expected_vals=param_expected_vals,
        param_weights=None,
        partitions=partition,
        expert_probs=simulated_expert_probs,
        n_trials=n_trials,
        return_value=True,
    )

    best_alpha = BO.eval_function(
        best_params, partition, simulated_expert_probs
    )  ##alpha

    best_probs = BO.get_model_probs(
        lam=best_params, partitions=partition, num_samples=20_000
    )

    return simulated_expert_probs, best_params, best_probs, best_alpha, best_llik


## Function for plotting the distribution of the probabilistic model in the form of overlapping histograms
## for the true hyperparameters of the model and the ones resulting from PPE, for a given partition and for
## all covariates


"""
Inputs

- "model" -> the probabilistic PyMC model. It is assumed that if there are covariates, these are defined within the model
- "J" -> The number of different covariate sets. If there are none, J=1 should be used as an input
- "lambd_names" -> a list that contains the names for each of the hyperparameters. Required for BO
- "lambd_true_vals" -> a list containing the values of the hyperparameters that we assume are true to conduct the simulation
- "best_params" -> The parameters resulting from PPE for the given model
- "alpha" -> A value for the hyperparameter alpha of the dirichlet likelihood given "best_params"
- "num_bins" -> The number of bins to have in the partition
- "partitions" -> The partitions that were used to elicit the expert's probabilistic judgements
- "lower_inner", "upper_inner" -> The lower and upper bounds of the interval that we partition

"""


def plot_histograms(
    model,
    J,
    lambd_names,
    lambd_true_vals,
    best_params,
    alpha,
    num_bins,
    partitions,
    lower_inner,
    upper_inner,
):

    lambd_true = {name: value for name, value in zip(lambd_names, lambd_true_vals)}

    true_samples = model(lambd=lambd_true, n_samples=10_000).prior_predictive["Y_obs"][
        0
    ]

    prior_predictive_samples = model(
        lambd=best_params, n_samples=10_000
    ).prior_predictive["Y_obs"][0]

    if J == 1:

        plt.hist(
            list(filter(lambda x: x > lower_inner and x < upper_inner, true_samples)),
            bins=30,
            alpha=0.5,
            label="True Parameters",
        )
        plt.hist(
            list(
                filter(
                    lambda x: x > lower_inner and x < upper_inner,
                    prior_predictive_samples,
                )
            ),
            bins=30,
            alpha=0.5,
            label="BO Parameters",
        )

        x_lines = list(map(lambda L: L[0], partitions))[1:]
        for x_line in x_lines:
            plt.vlines(
                x=x_line, ymin=0, ymax=20, color="red", linestyle="-", linewidth=2
            )

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.title(
            f"Sampled distributions of Y for {num_bins} partition bins. Alpha = {alpha:.2f}"
        )
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    else:

        num_cols = 3
        num_rows = (J + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        plt.style.use("seaborn-v0_8-whitegrid")

        for j in range(J):

            ax = axes[j]

            prior_predictive_samples = model(
                lambd=best_params, n_samples=10_000
            ).prior_predictive["Y_obs"][0]

            ax.hist(
                list(
                    filter(
                        lambda x: x > lower_inner and x < upper_inner,
                        true_samples[:, j],
                    )
                ),
                bins=30,
                alpha=0.5,
                label="True Parameters",
            )
            ax.hist(
                list(
                    filter(
                        lambda x: x > lower_inner and x < upper_inner,
                        prior_predictive_samples[:, j],
                    )
                ),
                bins=30,
                alpha=0.5,
                label="BO Parameters",
            )

            x_lines = list(map(lambda L: L[0], partitions))[1:]
            for x_line in x_lines:
                ax.vlines(
                    x=x_line, ymin=0, ymax=20, color="red", linestyle="-", linewidth=2
                )

            ax.set_title(
                f"Sampled distribution of Y, J={j+1}, for {num_bins} partition bins. Alpha = {alpha:.2f}"
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()

        for j in range(J, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        
        
## Function for plotting the density of the probabilistic model for the true hyperparameters of the model
## and the ones resulting from PPE, for all partitions and for all covariate sets


"""
Inputs

- "model" -> the probabilistic PyMC model. It is assumed that if there are covariates, these are defined within the model
- "J" -> The number of different covariate sets. If there are none, J=1 should be used as an input
- "lambd_names" -> a list that contains the names for each of the hyperparameters. Required for BO
- "lambd_true_vals" -> a list containing the values of the hyperparameters that we assume are true to conduct the simulation
- "best_params_total" -> List of the parameters resulting from PPE for the given model for all partitions
- "alpha" -> List of values for the hyperparameter alpha of the dirichlet likelihood given the best parameters for each partition
- "num_bins" -> List of number of bins in each partition
- "lower_inner", "upper_inner" -> The lower and upper bounds of the interval that we partition

"""     



def plot_densities(
    model,
    J,
    lambd_names,
    lambd_true_vals,
    best_params_total,
    alpha_total,
    num_bins,
    lower_inner,
    upper_inner,
):

    lambd_true = {name: value for name, value in zip(lambd_names, lambd_true_vals)}

    true_samples = model(lambd=lambd_true, n_samples=10_000).prior_predictive["Y_obs"][
        0
    ]

    if J == 1:

        sns.kdeplot(
            np.array(
                list(
                    filter(lambda x: x > lower_inner and x < upper_inner, true_samples)
                )
            ),
            fill=True,
            label="True Parameters",
        )

        for i in range(len(num_bins)):

            best_params = best_params_total[i]
            prior_predictive_samples = model(
                lambd=best_params, n_samples=10_000
            ).prior_predictive["Y_obs"][0]
            sns.kdeplot(
                np.array(
                    list(
                        filter(
                            lambda x: x > lower_inner and x < upper_inner,
                            prior_predictive_samples,
                        )
                    )
                ),
                fill=False,
                label=f"bins={num_bins[i]}, a={alpha_total[i]:.1f}",
            )

        plt.title("Sampled distributions of Y")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    else:

        total_prior_predictive_samples = [
            model(lambd=best_params, n_samples=10_000).prior_predictive["Y_obs"][0]
            for best_params in best_params_total
        ]

        for j in range(J):

            sns.kdeplot(
                np.array(
                    list(
                        filter(
                            lambda x: x > lower_inner and x < upper_inner,
                            true_samples[:, j],
                        )
                    )
                ),
                fill=True,
                label="True Parameters",
            )

            for i in range(len(num_bins)):

                best_params = best_params_total[i]
                sns.kdeplot(
                    np.array(
                        list(
                            filter(
                                lambda x: x > lower_inner and x < upper_inner,
                                total_prior_predictive_samples[i][:, j],
                            )
                        )
                    ),
                    fill=False,
                    label=f"bins={num_bins[i]}, a={alpha_total[i]:.1f}",
                )

            plt.title(f"Sampled distributions of Y for J={j+1}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.show()
            
## Function to compute the Wasserstein distance between the expert's simulated probabilistic
## judgements and the resulting prior predictive probabilities from PPE. The computations are
## different if there is only one covariate set (J=1, thus the probabilities are 1D) compared
## to multiple covariate sets, which is accounted for in the function by using the correct function
## from scipy.stats

'''
Inputs:

- "best_probs" -> the prior predictive probabilities for a model given the optimized hyperparameters
- "expert_probs" -> the simulated expert probabilities
- "J" -> The number of different covariate sets

'''



def wasserstein_metric(best_probs, expert_probs, J):

    if J == 1:
        return wasserstein_distance(best_probs[0], expert_probs[0])

    return wasserstein_distance_nd(best_probs, expert_probs)
