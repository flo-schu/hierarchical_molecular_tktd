import time
import numpy as np
from matplotlib import pyplot as plt
import arviz as az
import xarray as xr

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pymob import Config
from pymob.sim.config import Param

from hierarchical_molecular_tktd.sim import (
    NomixHierarchicalSimulation, 
)

config = Config("scenarios/hierarchical_cext_nested_sigma_hyperprior/settings.cfg")

# debugging config
# config = Config("case_studies/hierarchical_molecular_tktd/scenarios/hierarchical_cext_nested_sigma_hyperprior/settings.cfg")
# config.case_study.package = "case_studies"
# config.case_study.data_path = "case_studies/tktd_rna_pulse/data"

sim = NomixHierarchicalSimulation(config)

sim.config.inference_numpyro.gaussian_base_distribution = True
sim.config.jaxsolver.throw_exception = True
sim.config.jaxsolver.max_steps = 10_000
sim.setup()
sim.dispatch_constructor()

sim.config.model_parameters.free.keys()

sim.set_inferer("numpyro")
sim.inferer.load_results(f"numpyro_svi_posterior.nc")

def dataset_to_dict(dataset: xr.Dataset):
    return {k: v.values for k, v in dataset.data_vars.items()}

f, grad = sim.inferer.create_log_likelihood(
    return_type="joint-log-likelihood",
    check=False,
    scaled=True,
    vectorize=False,
    # gradients=True
)

mode = dataset_to_dict(sim.inferer.idata.unconstrained_posterior.mean(("chain", "draw")))

@jax.jit
def func(theta):
    # select first substance
    index: int = 0
    params = {}
    for key, value in mode.items():
        if key in theta:
            new_value = theta[key]
            if new_value.shape == value.shape:
                params.update({key: jnp.array(new_value)})
            else:
                mixed_value = jnp.array([new_value[0] if i == index else v for i, v in enumerate(value)])
                params.update({key: mixed_value})
        else:
            params.update({key: jnp.array(value)})
        
    return f(params)

# Compute the gradient function
grad = jax.grad(func)

print("Jit-compiling likelihood function")
func({"k_i_substance_normal_base": jnp.array([1.47]), "r_rt_substance_normal_base": jnp.array([-1.31])})

diff = time.time()
for i in range(10):
    func({"k_i_substance_normal_base": jnp.array([1.47]), "r_rt_substance_normal_base": jnp.array([-1.31])})
diff -= time.time()
print(f"10 function evaluations took {round(-diff,2)} seconds")

print("Jit-compiling gradient function")
grad({"k_i_substance_normal_base": jnp.array([1.47]), "r_rt_substance_normal_base": jnp.array([-1.31])})
diff = time.time()
for i in range(10):
    grad({"k_i_substance_normal_base": jnp.array([1.47]), "r_rt_substance_normal_base": jnp.array([-1.31])})
diff -= time.time()

print(f"10 gradient evaluations took {round(-diff,2)} seconds")


k_i_0_mode = mode["k_i_substance_normal_base"][0]
r_rt_0_mode = mode["r_rt_substance_normal_base"][0]

dev = 2  # standard deviations
sim.config.model_parameters.k_i_substance_normal_base = Param(min=k_i_0_mode - dev, max=k_i_0_mode + dev)
sim.config.model_parameters.r_rt_substance_normal_base = Param(min=r_rt_0_mode - dev, max=r_rt_0_mode + dev)

ax = sim.inferer.plot_likelihood_landscape(
    parameters=("k_i_substance_normal_base", "r_rt_substance_normal_base"),
    log_likelihood_func=func,
    gradient_func=grad,
    n_grid_points=30,
    n_vector_points=50,
)

ax.figure.savefig(f"{sim.output_path}/test_loglikelihood_gradients.png")
