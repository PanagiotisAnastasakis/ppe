import pymc as pm


def gaussian_model_1(lambd, covs = None, n_samples = 1500):
        
    regression_model = pm.Model()

    with regression_model:

        mu = pm.Normal("mu", mu=lambd["mu_1"], sigma=lambd["sigma_1"])
        sigma = pm.Gamma("sigma", alpha=lambd["a"], beta=lambd["b"])

        Y_obs = pm.Normal("Y_obs", mu = mu, sigma = sigma, observed=0.) ## the observed value does not matter as we only need the prior predictive distribution

    with regression_model:
        idata = pm.sample_prior_predictive(samples = n_samples)

    return idata



def gaussian_model_2(lambd, covs = None, n_samples = 1500):
        
    regression_model = pm.Model()

    with regression_model:
        
        sigma_hyp = pm.LogNormal("sigma_hyp", mu = lambd["mu_s"], sigma = lambd["sigma_s"])
        mu_hyp = pm.Normal("mu_hyp", mu = lambd["mu_m"], sigma = lambd["sigma_m"])

        mu = pm.Normal("mu", mu=mu_hyp, sigma=sigma_hyp)
        sigma = pm.Gamma("sigma", alpha=lambd["a"], beta=lambd["b"])

        Y_obs = pm.Normal("Y_obs", mu = mu, sigma = sigma, observed=0.) ## the observed value does not matter as we only need the prior predictive distribution

    with regression_model:
        idata = pm.sample_prior_predictive(samples = n_samples)

    return idata


