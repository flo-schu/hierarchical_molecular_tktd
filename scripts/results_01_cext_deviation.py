import os
import numpy as np
from matplotlib import pyplot as plt
import click
from click.testing import CliRunner
import numpy as np
from matplotlib import pyplot as plt
import arviz as az
from pymob import Config

from hierarchical_molecular_tktd.sim import (
    NomixHierarchicalSimulation, 
)

def compare_external_cocentrations(sim, idata_file):

    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.inferer.load_results(idata_file)


    cext_sim = sim.inferer.idata.posterior.cext_y0.mean(("chain", "draw"))
    # cext_obs = sim.observations.cext.max("time")
    cext_obs = sim.observations.cext.isel(time=0)
    cext_nom = sim.observations.cext_nom

    colors = ["tab:green", "tab:blue", "tab:purple"]


    fig, axes = plt.subplots(3,1, figsize=(4,8))
    for i, (ax, sub) in enumerate(zip(axes, sim.dimension_coords["substance"])):
        obs = cext_obs.where(cext_obs.substance==sub, drop=True)
        x = cext_nom.where(cext_obs.substance==sub, drop=True)
        y = cext_sim.where(cext_obs.substance==sub, drop=True)
        y = y.where(~ cext_obs.isnull(), drop=True)
        x = x.where(~ cext_obs.isnull(), drop=True)
        cmin = float(min(x.min(),y.min())) * 0.9
        cmax = float(max(x.max(),y.max())) * 1.1
        ax.scatter(x, y, c=colors[i], ls="", marker="o", alpha=.5, label="")
        ax.plot(np.linspace(cmin, cmax),np.linspace(cmin,cmax), color="black")
        ax.set_xlabel(r"$C_e$ (nominal)")
        ax.set_ylabel(r"$C_e$ (estimated)")
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(cmin,cmax)
        ax.set_ylim(cmin,cmax)
        ax.set_title(sub.capitalize())
    fig.tight_layout()
    fig.savefig(f"{sim.output_path}/scatter_cext_nom_cext_sim.png")


    fig, axes = plt.subplots(3,1, figsize=(4,8))
    for i, (ax, sub) in enumerate(zip(axes, sim.dimension_coords["substance"])):
        x = cext_obs.where(cext_obs.substance==sub, drop=True)
        y = cext_sim.where(cext_obs.substance==sub, drop=True)
        y = y.where(~x.isnull(), drop=True)
        x = x.where(~x.isnull(), drop=True)
        cmin = float(min(x.min(),y.min())) * 0.9
        cmax = float(max(x.max(),y.max())) * 1.1
        ax.scatter(x, y, c=colors[i], ls="", marker="o", alpha=.5, label="")
        ax.plot(np.linspace(cmin, cmax),np.linspace(cmin,cmax), color="black")
        ax.set_xlabel(r"$C_e$ (observed)")
        ax.set_ylabel(r"$C_e$ (estimated)")
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(cmin,cmax)
        ax.set_ylim(cmin,cmax)
        ax.set_title(sub.capitalize())
    fig.tight_layout()
    fig.savefig(f"{sim.output_path}/scatter_cext_obs_cext_sim.png")

    fig, axes = plt.subplots(3,1, figsize=(4,8))
    for i, (ax, sub) in enumerate(zip(axes, sim.dimension_coords["substance"])):
        x = cext_obs.where(cext_obs.substance==sub, drop=True)
        y = cext_nom.where(cext_obs.substance==sub, drop=True)
        y = y.where(~x.isnull(), drop=True)
        x = x.where(~x.isnull(), drop=True)
        cmin = float(min(x.min(),y.min())) * 0.9
        cmax = float(max(x.max(),y.max())) * 1.1
        ax.scatter(x, y, c=colors[i], ls="", marker="o", alpha=.5, label="")
        ax.plot(np.linspace(cmin, cmax),np.linspace(cmin,cmax), color="black")
        ax.set_xlabel(r"$C_e$ (observed)")
        ax.set_ylabel(r"$C_e$ (nominal)")
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(cmin,cmax)
        ax.set_ylim(cmin,cmax)
        ax.set_title(sub.capitalize())

    fig.tight_layout()
    fig.savefig(f"{sim.output_path}/scatter_cext_obs_cext_nom.png")


@click.command()
@click.option("--config", type=str)
@click.option("--debug/--no-debug", default=False)
@click.option("--idata_file", default="numpyro_posterior.nc")
def main(config, debug, idata_file):

    if debug:
        import pdb
        pdb.set_trace()

    cfg = Config(config)
    cfg.import_casestudy_modules()
    Simulation = cfg.import_simulation_from_case_study()
    sim = Simulation(config)
    sim.setup()

    compare_external_cocentrations(
        sim=sim,
        idata_file=idata_file,
    )



if __name__ == "__main__":
    if bool(os.getenv("debug")):
        runner = CliRunner(echo_stdin=True)
        result = runner.invoke(main, catch_exceptions=False, args=[
            "--config=scenarios/hierarchical_cext_nested_sigma_hyperprior_informed_reduced_dataset_rna_pulse_5/settings.cfg",
            # using --no-debug is important here, because otherwise the pdb interferes
            # with the vscode call I suspect.
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
