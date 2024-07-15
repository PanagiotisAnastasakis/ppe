import pymc as pm
import numpy as np


def logistic_model_1(lambd, covs=None, n_samples=1500):

    logistic_regression_model = pm.Model()

    with logistic_regression_model:

        b_0 = pm.Normal("b_0", mu=lambd["mu_0"], sigma=1.0)
        b_1 = pm.Normal("b_1", mu=lambd["mu_1"], sigma=lambd["sigma_1"])

        b = pm.math.stack([b_0, b_1])

        xTb = pm.math.dot(covs, b)

        p = pm.math.exp(xTb) / (1 + pm.math.exp(xTb))

        Y_obs = pm.Bernoulli("Y_obs", p=p, observed=np.ones(covs.shape[0]))

    with logistic_regression_model:
        idata = pm.sample_prior_predictive(samples=n_samples)

    return idata


def logistic_model_2(lambd, covs=None, n_samples=1500):

    logistic_regression_model = pm.Model()

    with logistic_regression_model:

        b_0 = pm.Normal("b_0", mu=lambd["mu_0"], sigma=lambd["sigma_0"])
        b_1 = pm.Normal("b_1", mu=lambd["mu_1"], sigma=lambd["sigma_1"])
        b_2 = pm.Normal("b_2", mu=lambd["mu_2"], sigma=lambd["sigma_2"])

        b = pm.math.stack([b_0, b_1, b_2])

        xTb = pm.math.dot(covs, b)

        p = pm.math.exp(xTb) / (1 + pm.math.exp(xTb))

        Y_obs = pm.Bernoulli("Y_obs", p=p, observed=np.ones(covs.shape[0]))

    with logistic_regression_model:
        idata = pm.sample_prior_predictive(samples=n_samples)

    return idata


def logistic_model_3(lambd, covs=None, n_samples=1500):

    logistic_regression_model = pm.Model()

    with logistic_regression_model:

        b_0 = pm.Normal("b_0", mu=lambd["mu_0"], sigma=lambd["sigma_0"])
        b_1 = pm.Normal("b_1", mu=lambd["mu_1"], sigma=lambd["sigma_1"])
        b_2 = pm.Normal("b_2", mu=lambd["mu_2"], sigma=lambd["sigma_2"])
        b_3 = pm.Normal("b_3", mu=lambd["mu_3"], sigma=lambd["sigma_3"])
        b_4 = pm.Normal("b_4", mu=lambd["mu_4"], sigma=lambd["sigma_4"])

        b = pm.math.stack([b_0, b_1, b_2, b_3, b_4])

        xTb = pm.math.dot(covs, b)

        p = pm.math.exp(xTb) / (1 + pm.math.exp(xTb))

        Y_obs = pm.Bernoulli("Y_obs", p=p, observed=np.ones(covs.shape[0]))

    with logistic_regression_model:
        idata = pm.sample_prior_predictive(samples=n_samples)

    return idata
