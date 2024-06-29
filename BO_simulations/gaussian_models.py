import pymc as pm

random_seed = 42

def gaussian_model_1(lambd, covs = None, n_samples = 1500):
        
    regression_model = pm.Model()

    with regression_model:

        mu = pm.Normal("mu", mu=lambd["a_1"], sigma=lambd["b_1"])
        sigma = pm.Gamma("sigma", alpha=lambd["a_2"], beta=lambd["b_2"])

        Y_obs = pm.Normal("Y_obs", mu = mu, sigma = sigma, observed=0.) ## the observed value does not matter as we only need the prior predictive distribution

    with regression_model:
        idata = pm.sample_prior_predictive(random_seed=random_seed, samples = n_samples)

    return idata



def gaussian_model_2(lambd, covs = None, n_samples = 1500):
        
    regression_model = pm.Model()

    with regression_model:
        
        sigma_hyp = pm.LogNormal("sigma_hyp", mu = lambd["a_1"], sigma = lambd["b_1"])
        mu_hyp = pm.Normal("mu_hyp", mu = lambd["a_2"], sigma = lambd["b_2"])

        mu = pm.Normal("mu", mu=mu_hyp, sigma=sigma_hyp)
        sigma = pm.Gamma("sigma", alpha=lambd["a_3"], beta=lambd["b_3"])

        Y_obs = pm.Normal("Y_obs", mu = mu, sigma = sigma, observed=0.) ## the observed value does not matter as we only need the prior predictive distribution

    with regression_model:
        idata = pm.sample_prior_predictive(random_seed=random_seed, samples = n_samples)

    return idata


