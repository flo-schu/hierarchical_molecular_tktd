import os
import time
import numpy as np
from matplotlib import pyplot as plt
import arviz as az
import xarray as xr
import click
from click.testing import CliRunner
from functools import partial
from frozendict import frozendict
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pymob import Config
from pymob.sim.config import Param, Modelparameters

from hierarchical_molecular_tktd.sim import (
    NomixHierarchicalSimulation, 
)

@click.command()
@click.option("--config", type=str)
@click.option("--parx", type=str)
@click.option("--pary", type=str)
@click.option("--n_grid_points", type=int, default=50)
@click.option("--n_vector_points", type=int, default=50)
@click.option("--debug/--no-debug", default=False)
def main(config, parx, pary, n_grid_points, n_vector_points, debug):

    if debug:
        import pdb
        pdb.set_trace()

    sim = NomixHierarchicalSimulation(config)

    sim.config.inference_numpyro.gaussian_base_distribution = True
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.max_steps = 10_000
    sim.setup()

    # set up the bounds for likelihood landscapes
    # this must be done here
    sim.config.model_parameters[parx].min = None
    sim.config.model_parameters[parx].max = None
    sim.config.model_parameters[pary].min = None
    sim.config.model_parameters[pary].max = None

    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.inferer.load_results(f"numpyro_svi_posterior.nc")

    def dataset_to_dict(dataset: xr.Dataset):
        return {k: v.values for k, v in dataset.data_vars.items()}
    
    mean = dataset_to_dict(sim.inferer.idata.unconstrained_posterior.mean(("chain", "draw")))
    mpx = mean[f"{parx}_normal_base"]
    mpy = mean[f"{pary}_normal_base"]
    mean_frozen = frozendict({k: tuple(np.array(v, ndmin=1).tolist()) for k, v in mean.items()})


    fig, axes = plt.subplots(ncols=len(mpx), nrows=len(mpy), figsize=(10,10))

    for i, x in enumerate(mpx):
        for j, y in enumerate(mpy):
            sim.dispatch_constructor()
            sim.set_inferer("numpyro")
    
            f, grad = sim.inferer.create_log_likelihood(
                return_type="joint-log-likelihood",
                check=False,
                scaled=True,
                vectorize=False,
                # gradients=True
            )

            @partial(jax.jit, static_argnames=["_i","_j", "_mean", "_parx", "_pary"])
            def func(theta, _i, _j, _mean, _parx, _pary):
                # select first substance
                params = {}
                for key, _value in _mean.items():
                    value = jnp.array(_value)
                    if key in theta and theta[key].shape == value.shape:
                        new_value = theta[key]
                        params.update({key: jnp.array(new_value)})
                    elif key == f"{_parx}_normal_base":
                        value = value.at[_i].set(theta[key][0])
                        params.update({key: value})
                    elif key == f"{_pary}_normal_base":
                        value = value.at[_j].set(theta[key][0])
                        params.update({key: value})
                    else:
                        params.update({key: jnp.array(value)})
                    
                return f(params)
            
            # Compute the gradient function
            grad = jax.grad(partial(func, _i=i, _j=j, _mean=mean_frozen, _parx=parx, _pary=pary))
            func_ = partial(func, _i=i, _j=j, _mean=mean_frozen, _parx=parx, _pary=pary)

            # sim.logger.info("Jit-compiling likelihood function")
            # func_({f"{parx}_normal_base": jnp.array([x]), f"{pary}_normal_base": jnp.array([y])})

            # diff = time.time()
            # for _ in range(10):
            #     func_({f"{parx}_normal_base": jnp.array([x]), f"{pary}_normal_base": jnp.array([y])})
            # diff -= time.time()
            # sim.logger.info(f"10 function evaluations took {round(-diff,2)} seconds")
                        
            # sim.logger.info("Jit-compiling gradient function")
            # grad({f"{parx}_normal_base": jnp.array([x]), f"{pary}_normal_base": jnp.array([y])})
            # diff = time.time()
            # for _ in range(10):
            #     grad({f"{parx}_normal_base": jnp.array([x]), f"{pary}_normal_base": jnp.array([y])})
            # diff -= time.time()
            # sim.logger.info(f"10 gradient evaluations took {round(-diff,2)} seconds")

            # func_({f"{parx}_normal_base": jnp.array([x]), f"{pary}_normal_base": jnp.array([y])})


            dev = 1  # standard deviations
            ax = sim.inferer.plot_likelihood_landscape(
                parameters=(parx, pary),
                # bounds=([x - dev, x + dev], [y - dev, y + dev]),
                bounds=([- dev, + dev], [- dev, + dev]),
                log_likelihood_func=func_,
                # gradient_func=grad,
                n_grid_points=n_grid_points,
                # n_vector_points=n_vector_points,
                normal_base=True,
                ax=axes[i,j]
            )

    folder = os.path.join(sim.output_path, "likelihood_landscapes")
    os.makedirs(folder, exist_ok=True)
    ax.figure.savefig(os.path.join(folder, f"{parx}__{pary}.png"))


if __name__ == "__main__":
    if bool(os.getenv("debug")):
        runner = CliRunner(echo_stdin=True)
        result = runner.invoke(main, catch_exceptions=False, args=[
            "--config=scenarios/hierarchical_cext_nested_sigma_hyperprior/settings.cfg",
            # using --no-debug is important here, because otherwise the pdb interferes
            # with the vscode call I suspect.
            "--parx=k_i_substance",
            "--pary=r_rt_substance",
            "--n_grid_points=5",
            "--n_vector_points=5",
            "--no-debug"
        ])
        if isinstance(result.exception, SystemExit):
            raise KeyError(
                "Invokation of the click command did not execute correctly. " +
                f"Recorded output: {' '.join(result.output.splitlines())}"
            )
        
        else:
            print(result.output)
            
        
    else:
        main()
