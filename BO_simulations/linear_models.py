import pymc as pm
import numpy as np


def linear_model_1(lambd, covs=None, n_samples=1500):

    regression_model = pm.Model()

    with regression_model:

        b_0 = pm.Normal("b_0", mu=2.0, sigma=1.0)
        b_1 = pm.Normal("b_1", mu=lambd["mu_1"], sigma=lambd["sigma_1"])

        b = pm.math.stack([b_0, b_1])

        prod = pm.math.dot(covs, b)

        Y_obs = pm.Normal(
            "Y_obs", mu=prod, sigma=lambd["sigma"], observed=np.ones(covs.shape[0])
        )

    with regression_model:
        idata = pm.sample_prior_predictive(samples=n_samples)

    return idata


def linear_model_2(lambd, covs=None, n_samples=1500):

    regression_model = pm.Model()

    with regression_model:

        b_0 = pm.Normal("b_0", mu=lambd["mu_0"], sigma=lambd["sigma_0"])
        b_1 = pm.Normal("b_1", mu=lambd["mu_1"], sigma=lambd["sigma_1"])
        b_2 = pm.Normal("b_2", mu=lambd["mu_2"], sigma=lambd["sigma_2"])

        b = pm.math.stack([b_0, b_1, b_2])

        prod = pm.math.dot(covs, b)

        Y_obs = pm.Normal(
            "Y_obs", mu=prod, sigma=lambd["sigma"], observed=np.ones(covs.shape[0])
        )

    with regression_model:
        idata = pm.sample_prior_predictive(samples=n_samples)

    return idata


def linear_model_3(lambd, covs=None, n_samples=1500):

    regression_model = pm.Model()

    with regression_model:

        b_0 = pm.Normal("b_0", mu=lambd["mu_0"], sigma=lambd["sigma_0"])
        b_1 = pm.Normal("b_1", mu=lambd["mu_1"], sigma=lambd["sigma_1"])
        b_2 = pm.Normal("b_2", mu=lambd["mu_2"], sigma=lambd["sigma_2"])
        b_3 = pm.Normal("b_3", mu=lambd["mu_3"], sigma=lambd["sigma_3"])
        b_4 = pm.Normal("b_4", mu=lambd["mu_4"], sigma=lambd["sigma_4"])

        b = pm.math.stack([b_0, b_1, b_2, b_3, b_4])

        prod = pm.math.dot(covs, b)

        Y_obs = pm.Normal(
            "Y_obs", mu=prod, sigma=lambd["sigma"], observed=np.ones(covs.shape[0])
        )

    with regression_model:
        idata = pm.sample_prior_predictive(samples=n_samples)

    return idata
