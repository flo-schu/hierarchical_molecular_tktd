import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import click
from click.testing import CliRunner
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
import arviz as az
from pymob import Config

from hierarchical_molecular_tktd.sim import (
    NomixHierarchicalSimulation, 
)

def compare_external_cocentrations(sim, idata_file):

    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.inferer.load_results(idata_file)

    colors = ["tab:green", "tab:blue", "tab:purple"]

    Ci = sim.observations.cint
    Ci: xr.DataArray = Ci.where(~Ci.isnull().all("time"), drop=True)


    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for i, (ax, sub) in enumerate(zip(axes, sim.dimension_coords["substance"])):
        Ci_sub = Ci.where(Ci.substance == sub, drop=True)
        Ci_sub = Ci_sub.interpolate_na(dim="time", method="linear")
        cext_color = Normalize(vmin=Ci_sub.cext_nom.min(), vmax=Ci_sub.cext_nom.max())
        cmap = colormaps["cool"]
        for j in Ci_sub.id:
            cii = Ci_sub.sel(id=j)
            ax.plot(Ci_sub.time, cii.values, color=cmap(cext_color(cii.cext_nom)), ls="-", marker="o", alpha=.5, label="")
        ax.set_title(sub.capitalize())
        ax.set_xscale("linear")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(r"$C_i$ µmol/L")
        cbar_ax = ax.inset_axes((0.95,0.0, 0.05, 0.1), transform=ax.transAxes)
        fig.colorbar(ScalarMappable(cext_color, cmap), cbar_ax, label=r"$C_e$")
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

    fig.tight_layout()
    fig.savefig(f"{sim.output_path}/trajectories_observations.png")


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
