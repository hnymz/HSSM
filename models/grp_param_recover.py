import hssm
import numpy as np
import pandas as pd
import pytensor

# Setting float precision in pytensor
pytensor.config.floatX = "float32"
pytensor.config.optimizer = 'None'

# Import the data
param = pd.read_csv('../data/grp_param.csv')

for batch in range(30):
    data_list = []

    mean_v = param['v_Intercept'][batch]
    print(f"mean_v: {mean_v}")
    mean_vx = param['v_x'][batch]
    mean_vy = param['v_y'][batch]
    mean_a = param['a'][batch]
    mean_z = param['z'][batch]
    mean_t = param['t'][batch]

    # Generate data
    n_subjects = 40  # number of subjects
    n_trials = 100  # number of trials per subject - vary from low to high values to check shrinkage

    sd_v = 0.3  # sd for v-intercept
    sd_a = 0.3  # sd for v-intercept
    sd_t=0.1
    sd_z=0.1

    for i in range(n_subjects):
        # Make parameters for subject i
        intercept = np.random.normal(mean_v, sd_v, size=1)
        x = np.random.uniform(-1, 1, size=n_trials)
        y = np.random.uniform(-1, 1, size=n_trials)
        v_x = np.random.normal(mean_vx, sd_v, size=1)
        v_y = np.random.normal(mean_vy, sd_v, size=1)
        v = intercept + (v_x * x) + (v_y * y)
        a = np.random.normal(mean_a, sd_a, size=1)
        z = np.random.normal(mean_z, sd_z, size=1)
        t = np.random.normal(mean_t, sd_t, size=1)

    # v is a vector which differs over trials by x and y, so we have different v for every trial - other params are same for all trials
        true_values = np.column_stack(
        [v, np.repeat(a, axis=0, repeats=n_trials), np.repeat(z, axis=0, repeats=n_trials), np.repeat(t, axis=0, repeats=n_trials)]
    )
        # Simulate data
        obs_ddm_reg_v = hssm.simulate_data(model="ddm", theta=true_values, size=1)

        # Append simulated data to list
        data_list.append(
            pd.DataFrame(
                {
                    "rt": obs_ddm_reg_v["rt"],
                    "response": obs_ddm_reg_v["response"],
                    "x": x,
                    "y": y,
                    "subject": i,
                }
            )
        )
    # Make single dataframe out of subject-wise datasets
    dataset_reg_v_hier_full = pd.concat(data_list)

    # Specify the model
    model_reg_v_ddm_hier1A = hssm.HSSM(
        data=dataset_reg_v_hier_full,
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
                    "Intercept": {"name": "Gamma", "mu": 0.5, "sigma": 0.4, "initval": 0.3},
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
        cores=1,  # how many cores to use
        chains=3,  # how many chains to run
        draws=200,  # number of draws from the markov chain
        tune=200,  # number of burn-in samples
        idata_kwargs=dict(log_likelihood=True),  # return log likelihood
    )

    # Save the model
    samples_model_reg_v_ddm_hier1A.to_netcdf(f'../outputs/grp_param_recover/Model_{batch}')
