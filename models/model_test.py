import hssm
import pytensor
import numpy as np
import pandas as pd

# Setting float precision in pytensor
pytensor.config.floatX = "float32"
pytensor.config.optimizer = 'None'
# JAX Configuration
jax.config.update("jax_enable_x64", False)

# Import the data
data = pd.read_csv('../data/dataset_reg_v_hier_full.csv')

# Specify the model
model_reg_v_ddm_hier1A = hssm.HSSM(
    data=data,
    model="ddm",
    loglik_kind="analytical",  # approx_differentiable = LAN likelihood; analytical = Navarro & Fuss
    prior_settings="safe",
    p_outlier=0, # remove the p_outlier for now due to a current bug in HSSM
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + x + y + (1 + x + y | subject)",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 1, "sigma": 2, "initval": 1},
                "x": {"name": "Normal", "mu": 0, "sigma": 1, "initval": 0},
                "y": {"name": "Normal", "mu": 0, "sigma": 1, "initval": 0},
                "1|subject": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 1}, "initval": 0.5},
                "x|subject": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.5}, "initval": 0.5},
                "y|subject": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.5}, "initval": 0.5}
            },
            "link": "identity",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + (1 | subject)",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.5, "sigma": 0.4, "initval": 0.3},
                "1|subject": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.5}, "initval": 0.1}
            },
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1 | subject)",
            "prior": {
                "Intercept": {"name": "HalfNormal", "sigma": 1, "initval": 0.5},
                "1|subject": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.05}, "initval": 0.01}
            },
        },
        {
            "name": "a",
            "formula": "a ~ 1 + (1 | subject)",
            "prior": {
                "Intercept": {"name": "Gamma", "mu": 0.5, "sigma": 1.75, "initval": 1},
                "1|subject": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 1}, "initval": 0.3}
            },
        },
    ]
)

# Sample
samples_model_reg_v_ddm_hier1A = model_reg_v_ddm_hier1A.sample(
    sampler="nuts_numpyro",  # type of sampler to choose, 'nuts_numpyro', 'nuts_blackjax' of default pymc nuts sampler
    cores=3,  # how many cores to use
    chains=3,  # how many chains to run
    draws=200,  # number of draws from the markov chain
    tune=200,  # number of burn-in samples
    idata_kwargs=dict(log_likelihood=True),  # return log likelihood
)

# Save the model
samples_model_reg_v_ddm_hier1A.to_netcdf('../output/Model_test')