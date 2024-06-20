# Import the packages
import hssm
import pytensor  # Graph-based tensor library
import bambi as bmb
import pandas as pd
import numpy as np
import pymc as pm


# Setting float precision in pytensor
pytensor.config.floatX = "float32"
pytensor.config.optimizer = 'None'
from jax.config import config
config.update("jax_enable_x64", False)

# Import the data
data = pd.read_csv('../data/data_hddm_LFXC703_EffClean.csv')

# Specify the model
model = hssm.HSSM(
    model="ddm",
    loglik_kind="blackbox",
    data=data,
    p_outlier={"name": "Uniform", "lower": 0.001, "upper": 0.05},
    lapse=bmb.Prior("Uniform", lower=0.0, upper=5.0),
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.2},
                "1|subject": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
                
                "mbased_efficacy_prev": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|mbased_efficacy_prev": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
                
                "mbased_reward_prev": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|mbased_reward_prev": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},

                "mbased_efficacy_prev:mbased_reward_prev": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|mbased_efficacy_prev:mbased_reward_prev": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
                
                "Congruency": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|Congruency": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
            },
            "formula": "v ~ 1 + Congruency + mbased_efficacy_prev * mbased_reward_prev + (1 + mbased_efficacy_prev * mbased_reward_prev|subject)",
        },
        {
            "name": "a",
            "prior": {
                "Intercept": {"name": "Gamma", "alpha":2.0, "beta": 20.0},
                "1|subject": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
                
                "mbased_efficacy_prev": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|mbased_efficacy_prev": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
                
                "mbased_reward_prev": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|mbased_reward_prev": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},

                "mbased_efficacy_prev:mbased_reward_prev": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|mbased_efficacy_prev:mbased_reward_prev": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
            },
            "formula": "a ~ 1 + mbased_efficacy_prev * mbased_reward_prev + (1 + mbased_efficacy_prev * mbased_reward_prev|subject)",
        },
        {
            "name": "z",
            "prior": {
                "Intercept": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
                "1|subject": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma", "alpha": 2.0, "beta": 20.0}},
                
                "Congruency": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "1|Congruency": {"name": "Normal", "mu":0.0, "sigma":{"name": "Gamma",  "alpha": 2.0, "beta": 20.0}},
            },
            "formula": "z ~ 1 + Congruency + (1|subject)",
        },
    ],
)

# Specify the dictionary of initial values
n_subjects = len(data.subject.unique())
n_subjects

# To check the shapes of all variables:
# model.pymc_model.eval_rv_shapes()

# To check the initial points (init method might changes these; 
#e.g. jitter might be added when you run the model; 
#we use init="adapt_diag" to avoid this):
# model.pymc_model.initial_point()

my_inits = {'t': 0.1,
                                             
            'v_Intercept': 0.0,
            'v_1|subject_sigma': 0.1,
            'v_1|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)
            
            'v_Congruency': np.array([0.0]).astype(np.float32),
            
            'v_mbased_efficacy_prev': 0.00,
            'v_mbased_efficacy_prev|subject_sigma': 0.1,
            'v_mbased_efficacy_prev|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)
            
            'v_mbased_reward_prev': 0.00,
            'v_mbased_reward_prev|subject_sigma': 0.1,
            'v_mbased_reward_prev|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)

            'v_mbased_efficacy_prev:mbased_reward_prev': 0.00,
            'v_mbased_efficacy_prev:mbased_reward_prev|subject_sigma': 0.1,
            'v_mbased_efficacy_prev:mbased_reward_prev|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)
            
            
            'a_Intercept': 1.0,
            'a_1|subject_sigma': 0.1,
            'a_1|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)
            
            'a_mbased_efficacy_prev': 0.00,
            'a_mbased_efficacy_prev|subject_sigma': 0.1,
            'a_mbased_efficacy_prev|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)
            
            'a_mbased_reward_prev': 0.00,
            'a_mbased_reward_prev|subject_sigma': 0.1,
            'a_mbased_reward_prev|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)

            'a_mbased_efficacy_prev:mbased_reward_prev': 0.00,
            'a_mbased_efficacy_prev:mbased_reward_prev|subject_sigma': 0.1,
            'a_mbased_efficacy_prev:mbased_reward_prev|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)

            'z_Intercept': 0.5,
            'z_1|subject_sigma': 0.1,
            'z_1|subject_offset': np.zeros((n_subjects,)).astype(np.float32), # Watch out: shape=(n_subjects,)

            'z_Congruency': np.array([0.0]).astype(np.float32),
           }

# Sample
modelObject = model.sample(
    #sampler="nuts_numpyro", 
    initvals = my_inits, 
    chains=4, 
    cores=4, 
    draws=700, 
    tune=3000,
    #target_accept=0.95
)

# Save the model
modelObject.to_netcdf('../output/Model5')

