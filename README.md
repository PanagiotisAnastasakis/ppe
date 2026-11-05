# Probabilistic Predictive Elicitation (PPE)

This repository contains a Python implementation of **Probabilistic Predictive Elicitation (PPE)**, following [“Flexible Prior Elicitation via the Prior Predictive Distribution”](https://proceedings.mlr.press/v124/hartmann20a.html) by Hartmann et al. The aim is to calibrate prior-distribution hyperparameters using an expert's probabilistic judgements about a model's target variable.

The code provides three complementary approaches:

- **Closed-form prior predictive probabilities:** JAX automatic differentiation and gradient descent.
- **Simulation-based probabilities:** prior predictive samples from a user-supplied PyMC-style model, optimized using Bayesian optimization with Ax.
- **Stochastic-gradient experiments:** utilities for estimating derivatives in Bernoulli and continuous-target examples.

This is research code with implementation modules, tests, and simulation notebooks. The descriptions below reflect the code currently in the repository; experimental routines should not be assumed to be general-purpose optimizers.

## 1. Method overview

Suppose an expert provides probabilities for a collection of events or intervals describing a target variable Y. For covariate setting j, let qⱼ be the expert's probability vector and pⱼ(λ) the corresponding probabilities implied by the model's prior predictive distribution, whose hyperparameters are λ.

PPE compares these vectors using a Dirichlet log-likelihood. The implementation evaluates the likelihood for each covariate setting and sums the results over J settings. The hyperparameters are selected to make the prior predictive probabilities compatible with the expert's judgements.

The Dirichlet concentration parameter α controls the concentration of the likelihood. It can be fixed by the user or, in supported methods, estimated from the model and expert probability vectors using the repository's alpha_mle_ calculation.

For a **discrete** target, events are classes. For a **continuous** target, events are intervals (partitions) of the target range. Multiple covariate settings are represented by separate probability vectors, one for each setting.

## 2. Repository structure

    .
    ├── ppe/
    │   ├── __init__.py
    │   ├── bayesian_optimization.py
    │   ├── computing_probabilities.py
    │   ├── dirichlet.py
    │   ├── gradient_optimization.py
    │   ├── optimization_loop.py
    │   └── stochastic_optimization.py
    ├── tests/
    │   ├── Tests_Bayesian_Optimization.ipynb
    │   ├── Tests_dirichlet_probabilities.ipynb
    │   ├── Tests_optimization.ipynb
    │   ├── optimization_normflow_prior.py
    │   ├── optimization_simple_gaussian.py
    │   ├── stochastic_gradient_bernoulli.py
    │   └── stochastic_gradient_continous.py
    ├── BO_simulations/
    │   ├── functions.py
    │   ├── gaussian_models.py
    │   ├── linear_models.py
    │   ├── logistic_models.py
    │   └── simulations.ipynb
    ├── requirements.txt
    └── PPE_final_report.pdf

### Main modules

| Module | Purpose |
|---|---|
| ppe/dirichlet.py | Dirichlet log-likelihood, probability checks, gradient with respect to model probabilities, and alpha_mle_. |
| ppe/computing_probabilities.py | PPEProbabilities: convert prior predictive samples into discrete-class or continuous-interval probabilities; includes an expert-input parser. |
| ppe/gradient_optimization.py | optimize_ppe: closed-form probability evaluation, JAX derivatives, gradient descent, and an α estimate. |
| ppe/bayesian_optimization.py | Bayesian_Optimization: sample-based probabilities, negative Dirichlet log-likelihood, optional hyperpriors, and Ax optimization. |
| ppe/stochastic_optimization.py | JAX utilities for stochastic derivative estimates in Bernoulli and continuous examples. |
| ppe/optimization_loop.py | Iterative update loop used by the stochastic-gradient experiments. |
| BO_simulations/ | Model definitions and helpers for simulation, plotting, and Wasserstein-distance calculations. |
| tests/ | Notebook-based checks and runnable numerical examples. |

The package initializer exports Dirichlet and PPEProbabilities. Import the optimization class and other functions from their defining modules.

## 3. Installation

Create an isolated Python environment and install the declared requirements:

    git clone https://github.com/PanagiotisAnastasakis/ppe.git
    cd ppe
    python -m venv .venv
    # macOS / Linux:
    source .venv/bin/activate
    # Windows PowerShell:
    # .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

The current requirements.txt specifies:

- jax<=0.4.28
- jaxlib<=0.4.28
- numpy<2.0.0
- pandas
- scipy
- matplotlib

**Additional dependencies:** Bayesian optimization imports Ax, but ax-platform is commented out in requirements.txt. Install it separately if you plan to use Bayesian_Optimization:

    python -m pip install ax-platform

The Bayesian optimization interface also expects a PyMC-style model callable returning an object with a prior_predictive group. Specifically, the implementation reads prior_predictive["Y_obs"][0]. Install and configure a compatible PyMC stack for your model separately. PyMC is not declared in the current requirements file, and the repository does not provide a pinned full environment specification.

## 4. Core API

### 4.1 Dirichlet likelihood

Dirichlet is implemented in ppe/dirichlet.py:

    from ppe.dirichlet import Dirichlet, dirichlet_log_likelihood, alpha_mle_

    likelihood = Dirichlet(alpha=1.0, J=1)

- alpha: fixed concentration parameter, or None to request the MLE approximation in methods that support it.
- J: number of covariate settings represented by the probability lists.

Important methods include:

- alpha_mle(total_model_probs, total_expert_probs): estimate the concentration parameter from all supplied settings.
- llik(total_model_probs, total_expert_probs, index=None): evaluate one setting when an index is supplied, or evaluate the supplied vectors directly.
- sum_llik(total_model_probs, total_expert_probs): sum the log-likelihood over all J settings.
- grad_dirichlet_p(..., index=...): compute the negative gradient with respect to one model probability vector using JAX.
- probabilities_check(...): check that each probability vector sums to one.

Probability inputs are lists of vectors, with one vector per covariate setting. The likelihood methods check normalization.

### 4.2 Probabilities from prior predictive samples

PPEProbabilities is implemented in ppe/computing_probabilities.py:

    from ppe.computing_probabilities import PPEProbabilities

    probability_tools = PPEProbabilities(target_type="discrete", path=False)
    model_probs = probability_tools.ppd_probs(
        samples=prior_predictive_samples,
        partitions=[0, 1, 2],
    )

Set target_type to "discrete" or "continuous".

- For discrete targets, partitions contains the class labels. Probabilities are estimated by empirical class frequencies.
- For continuous targets, partitions is a list of interval arrays, one per covariate setting. Probabilities are estimated from sample frequencies within each interval.

For multiple covariate settings, the sample matrix is expected to have one column per setting. The continuous case expects one interval-partition array per setting.

The class also contains get_expert_data for parsing expert judgements from arrays or CSV paths. Its source comments explicitly mark this parser as **not recommended**, because it assumes a specific input-file format. For a new workflow, prepare and validate partition and expert-probability arrays explicitly.

### 4.3 Closed-form gradient optimization

optimize_ppe lives in ppe/gradient_optimization.py:

    from ppe.gradient_optimization import optimize_ppe

    optimizer = optimize_ppe(
        alpha=1.0,
        J=1,
        ppd=prior_predictive_probability,
    )

The supplied ppd callable returns the prior predictive probability of a single partition. It is called as either:

    ppd(partition, lam)
    # With covariates:
    ppd(partition, lam, covariates)

Here, lam is the hyperparameter vector. The class provides ppd_function, grad_ppd_lambda, gradient calculations for the Dirichlet objective, gradient_descent, and get_alpha.

gradient_descent accepts partitions and expert probabilities for all settings, an initial hyperparameter value, an iteration limit, a step size, and a tolerance. Covariates can be passed as a list with one entry per setting; if omitted, the implementation uses None for each setting. By default, the method returns the optimized value together with objective and gradient-norm progression arrays.

Use this route when prior predictive probabilities are available in a differentiable closed form. The probability function must be compatible with JAX tracing and automatic differentiation.

### 4.4 Bayesian optimization using prior predictive samples

Bayesian_Optimization is defined in ppe/bayesian_optimization.py:

    from ppe.bayesian_optimization import Bayesian_Optimization

    bo = Bayesian_Optimization(
        pymc_sampling_func=model_sampler,
        J=1,
        alpha=1.0,
        target_type="continuous",
        target_samples=500,
    )

pymc_sampling_func is supplied by the user. The class calls it with the hyperparameter dictionary and sample count, then reads samples from idata.prior_predictive["Y_obs"][0]. Your callable must therefore accept the arguments in the form used by the class and return an inference-data-like object containing that group and variable.

A typical workflow is:

1. Define target partitions and provide expert probabilities in the same order.
2. Call get_model_probs(lam, partitions, num_samples=None) to estimate model probabilities from prior predictive samples.
3. Call optimize_hyperparams(...) to minimize the negative Dirichlet log-likelihood, plus any supplied hyperprior penalty.
4. Optionally call eval_function(lam, partitions, expert_probs) to calculate the MLE estimate of α for a parameter setting.

optimize_hyperparams expects parallel lists: param_names, param_types, param_bounds, param_expected_vals, and param_weights. The code supports Ax parameter types such as "range"; range parameters are passed to Ax with lower and upper bounds. Set n_trials to control the optimization budget. Set return_value=True to return the best parameter dictionary and objective value.

If alpha=None, the Bayesian-optimization objective expects an "alpha" entry in the parameter dictionary. The simulation helper adds this parameter and a search interval automatically. When alpha is fixed, leave it out of the optimized parameter list.

### 4.5 Hyperpriors

The Bayesian-optimization class includes hyperprior_llik. If an expected value is supplied for a hyperparameter, the objective includes a truncated-normal hyperprior penalty. param_expected_vals contains either a value or None for each parameter, and param_weights supplies the corresponding confidence weights. Parameters without an expected value do not contribute a hyperprior term.

## 5. Stochastic-gradient utilities

ppe/stochastic_optimization.py contains two experimental utilities:

- set_derivative_bernoulli_fn(...) returns two functions: one for the derivative of the Dirichlet likelihood with respect to probability vectors, and one for a stochastic derivative of a Bernoulli predictive probability with respect to model hyperparameters.
- set_derivative_continous_fn(...) (spelled this way in the current source) builds a JAX function that estimates continuous interval probabilities from samples and differentiates the negative Dirichlet log-likelihood.

The companion optimization_loop in ppe/optimization_loop.py applies gradient updates to the parameter structure used in the experiments and prints progress every ten iterations.

These utilities are **experimental and model-specific**, not general-purpose replacements for optimize_ppe or Bayesian_Optimization. The Bernoulli helper assumes compatible sampler, pivot, and PMF functions. The continuous helper expects a sampler, CDF function, pivot function, partitions, and expert probabilities. Review the example scripts before adapting either utility to a new model. The source also contains TODO comments for unfinished generalization and some alpha=None cases.

## 6. Examples, simulations, and tests

The repository's Python examples and notebooks are the best starting points for expected inputs:

- tests/optimization_simple_gaussian.py: simple Gaussian prior/likelihood example with three target intervals and gradient-based optimization.
- tests/stochastic_gradient_bernoulli.py: compares a stochastic Bernoulli-gradient estimate with an automatic-differentiation reference.
- tests/stochastic_gradient_continous.py: checks stochastic continuous interval probabilities and gradients against a closed-form Gaussian calculation.
- tests/optimization_normflow_prior.py: an additional optimization experiment involving a normalizing-flow prior.
- tests/Tests_dirichlet_probabilities.ipynb, tests/Tests_optimization.ipynb, and tests/Tests_Bayesian_Optimization.ipynb: notebook-based tests and demonstrations.
- BO_simulations/simulations.ipynb: simulation experiments using the model and plotting helpers in BO_simulations/.

Run standalone examples from the repository root:

    python tests/optimization_simple_gaussian.py
    python tests/stochastic_gradient_bernoulli.py
    python tests/stochastic_gradient_continous.py

Some examples require additional packages or a working Jupyter environment beyond the base requirements. These scripts are research examples, not a packaged command-line interface. Plotting in the simple Gaussian example is disabled by default.

The simulation helpers in BO_simulations/functions.py include:

- make_partition: construct continuous partitions with outer intervals and a chosen inner range.
- ppe_simulation: simulate expert probabilities from known hyperparameters and run Bayesian optimization.
- plot_histograms and plot_densities: compare prior predictive samples under reference and elicited parameter values.
- wasserstein_metric: compute a Wasserstein distance for single-setting or multi-setting cases.

ppe_simulation uses 20,000 prior predictive samples to generate simulated expert probabilities and defaults to 75 Bayesian-optimization trials. These are helper defaults, not universal recommendations; adjust sample counts and trial budgets to your model's computational cost and precision needs.

## 7. Practical considerations and limitations

- **Sampling noise:** sample-based probabilities depend on the number of prior predictive draws. Increase target_samples or pass a larger num_samples to get_model_probs when Monte Carlo error matters.
- **Partition design:** continuous intervals should cover the plausible target range and reflect questions an expert can answer. The sample-probability routine expands the first or last interval to include observed extremes if needed; consider this when designing partitions.
- **Normalization and ordering:** expert and model probability vectors must refer to the same classes or intervals in the same order and should sum to one.
- **Multiple settings:** keep the number and ordering of covariate settings consistent across samples, partitions, and expert judgements.
- **Optimization bounds and initialization:** gradient descent depends on its initial value and step size; Bayesian optimization depends on parameter bounds and trial budget. Inspect fitted values and predictive probabilities rather than relying on the optimizer output alone.
- **Research-code status:** there is no packaging metadata or pinned full environment file. Some dependencies are optional or used only by notebooks and simulation helpers. The expert CSV parser is marked as not recommended, and stochastic-gradient utilities include unfinished TODOs.
- **Numerical checks:** the Dirichlet expression uses log probabilities. Zero-valued expert probabilities or invalid partitions can cause numerical problems, so validate inputs before optimization.

## 8. Reference

Hartmann, M., et al. [Flexible Prior Elicitation via the Prior Predictive Distribution](https://proceedings.mlr.press/v124/hartmann20a.html). Proceedings of Machine Learning Research, 2020.

---

This project is an implementation and set of experiments for probabilistic prior elicitation. It is not currently distributed as a fully packaged Python library with a stable, versioned public API. Check the source modules and example notebooks when adapting the code to a new probabilistic model.
