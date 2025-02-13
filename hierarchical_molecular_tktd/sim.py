from functools import partial
from typing import Literal
import xarray as xr
import numpy as np
import arviz as az
from matplotlib import pyplot as plt

from pymob import SimulationBase, Config
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

# TODO: Remove
EPS = 9.9e-5

# load the basic TKTD RNA Pulse case study and use as a parent class for the
# hierarchical model
# config = Config()
# config.case_study.name = "tktd_rna_pulse"
# config.case_study.package = "case_studies"
# config.import_casestudy_modules(reset_path=True)
from tktd_rna_pulse.sim import SingleSubstanceSim3

class NomixHierarchicalSimulation(SingleSubstanceSim3):
    def initialize(self, input):
        super().initialize(input)
        self.set_fixed_parameters(None)
        
        # create an index and add it to the simulation
        experiment_index = self.create_index("experiment_id")
        self.indices.update(experiment_index)

        # configure JaxSolver
        self.solver = JaxSolver

        # use_numpyro_backend can be written into the intiialize method
        # self.use_numpyro_backend()
        
    def use_numpyro_backend(self, error_model=None, only_prior=False):
        # configure the Numpyro backend
        self.config.inference_numpyro.user_defined_preprocessing = None
        self.config.inference_numpyro.user_defined_probability_model = None
        
        self.set_inferer("numpyro")
        self.inferer.inference_model = partial( 
            self.inferer.inference_model, 
            user_error_model=error_model,
            only_prior=only_prior
        )

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict

    def setup_data_structure_from_observations(self):
        self.setup()

        # select observations
        obs = [0]
        sim.observations = sim.observations.isel(id=obs)
        sim.observations.attrs["substance"] = list(
            np.unique(sim.observations.substance)
        )
        sim.set_y0()
        sim.indices["substance"] = sim.indices["substance"].isel(id=obs)


    def setup_data_structure_manually(
        self, 
        scenario: Literal[
            "data_structure_01_single_observation",
            "data_structure_02_replicated_observation",
            "data_structure_03_gradient_observation",
            "data_structure_04_unreplicated_multi_substance",
            "data_structure_05_replicated_multi_experiment",
        ] = "data_structure_01_single_observation"
    ):
        self.config.case_study.scenario = scenario
        self.config.create_directory("results")
        self.config.create_directory("scenario")

        # mark existing data variables as unobserved
        for _, datavar in self.config.data_structure.all.items():
            datavar.observed = False
            datavar.min = np.nan
            datavar.max = np.nan

        # copy data structure from survival to lethality
        self.config.data_structure.lethality =\
            self.config.data_structure.survival # type:ignore

        if scenario == "data_structure_01_single_observation":
            self.define_observations_unreplicated()

        elif scenario == "data_structure_02_replicated_observation":
            self.define_observations_replicated()

        elif scenario == "data_structure_03_gradient_observation":
            self.define_observations_replicated_gradient()

        elif scenario == "data_structure_04_unreplicated_multi_substance":
            self.define_observations_unreplicated_multiple_substances()

        elif scenario == "data_structure_05_replicated_multi_experiment":
            self.define_observations_replicated_multi_experiment()

        # set up coordinates
        self.coordinates["time"] = np.arange(0, 120)
        # self.coordinates["substance"] = "diuron"

        # define starting values
        self.config.simulation.y0 = [
            "cext=cext_nom", 
            "cint=Array([0])", 
            "nrf2=Array([1])", 
            "P=Array([0])", 
        ]

        y0 = self.parse_input("y0", reference_data=self.observations, drop_dims=["time"])
        self.model_parameters["y0"] = y0

        # define parameters

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict

        # set up the solver
        self.config.simulation.solver = "JaxSolver"
        self.config.simulation.batch_dimension = "id"

        self.validate()
        self.config.save(force=True)

    def decorate_results(self, results):
        """Convenience function to add attributes and coordinates to simulation
        results needed for other post-processing tasks (e.g. plotting)
        """
        results.attrs["substance"] = np.unique(results.substance)
        results = results.assign_coords({
            "cext_nom": self.model_parameters["y0"]["cext"]
        })
        return results

    def plot(self, results: xr.Dataset):
        if "substance" not in results.coords:
            results = results.assign_coords({"substance": self.observations.substance})
        if "cext_nom" not in results.coords:
            results = results.assign_coords({"cext_nom": self.observations.cext_nom})
        fig = self._plot.plot_simulation_results(results)


    def define_observations_unreplicated(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ], dims=("id"), coords={"id": [0]}),
            "cext_nom":  xr.DataArray([1000    ], dims=("id"), coords={"id": [0]}),
            "substance": xr.DataArray(["diuron"], dims=("id"), coords={"id": [0]})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0],
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * 5, dims=("id"), coords={"id": np.arange(5)}),
            "cext_nom":  xr.DataArray([1000    ] * 5, dims=("id"), coords={"id": np.arange(5)}),
            "substance": xr.DataArray(["diuron"] * 5, dims=("id"), coords={"id": np.arange(5)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * 5,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated_multi_experiment(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        n = 10
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "cext_nom":  xr.DataArray([1000    ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "substance": xr.DataArray(["diuron"] * n, dims=("id"), coords={"id": np.arange(n)}),
            "experiment": xr.DataArray(np.repeat([0,1], int(n/2)), dims=("id"), coords={"id": np.arange(n)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * n,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated_gradient(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        n = 5
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "cext_nom":  xr.DataArray(np.logspace(2,4, n), dims=("id"), coords={"id": np.arange(n)}),
            "substance": xr.DataArray(["diuron"] * n, dims=("id"), coords={"id": np.arange(n)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * n,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_unreplicated_multiple_substances(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      , 10,10], dims=("id"), coords={"id": np.arange(3)}),
            "cext_nom":  xr.DataArray([1000    , 100,10], dims=("id"), coords={"id": np.arange(3)}),
            "substance": xr.DataArray(["diuron", "naproxen","diclofenac"], dims=("id"), coords={"id": np.arange(3)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0,1,2],
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def prior_predictive_checks(self):
        super().prior_predictive_checks()
        plot_y0(
            sim=self, 
            idata=self.inferer.idata, 
            parameter="cext", 
            idata_group="prior", 
            levels=["experiment_id", "substance",], 
            colors={"substance": ["tab:green", "tab:blue", "tab:purple"]}
        )
        

    def posterior_predictive_checks(self):
        super().posterior_predictive_checks()
        plot_y0(
            sim=self, 
            idata=self.inferer.idata, 
            parameter="cext", 
            idata_group="posterior", 
            levels=["experiment_id", "substance",], 
            colors={"substance": ["tab:green", "tab:blue", "tab:purple"]}
        )



def plot_y0(sim, idata, idata_group, parameter, levels, colors={}):
    fig, ax = plt.subplots(1,1, figsize=(16,3.5))
    y0 = sim.model_parameters["y0"]
    samples = idata[idata_group]

    batch_dim = sim.config.simulation.batch_dimension
    sorted_ids = sim.observations[batch_dim].sortby(levels)[batch_dim]

    # plot nominal y0 values
    y0_y = y0[parameter].sel({batch_dim: sorted_ids})
    ax.plot(sorted_ids, y0_y, ls="", marker="o", color="black", ms=1)
    ax.plot([],[], ls="", marker="o", color="black", ms=1, label=r"$y_0$ nominal")
    
    # plot samples from prior/posoterior. THis works, because it maps samples.id to sorted ids
    samples_y = az.hdi(samples[f"{parameter}_y0"])[f"{parameter}_y0"].T
    ax.vlines(samples[batch_dim], *samples_y, color="grey", lw=1)
    ax.vlines([], [], [], color="grey", lw=1, label=f"$y_0$ {idata_group}")

    
    # using unique is fine, here, because I also ordered the IDs by substance
    y_min = np.min([y0_y.min(), samples_y.min()])
    y_max = np.max([y0_y.max(), samples_y.max()])

    ordered_unique_level_0 = np.unique(sorted_ids[levels[0]])
    colors_level_0 = colors.get(levels[0], ["grey"] * len(ordered_unique_level_0)) 
    ordered_unique_level_1 = np.unique(sorted_ids[levels[1]])
    colors_level_1 = colors.get(levels[1], ["grey"] * len(ordered_unique_level_1)) 

    for l0, l0col in zip(ordered_unique_level_0, colors_level_0):
        x_level_0 = np.where(sorted_ids[levels[0]] == l0)[0]
        y_level_0 = (y_max * 12, y_max*120)
        ax.fill_between(x_level_0, *y_level_0, color=l0col, alpha=.25)

        ax.text(x_level_0.mean(), np.exp(np.mean(np.log(y_level_0))), s=str(l0)[:3], 
                ha="center", va="center", color=l0col, fontsize=8)
        # using unique is fine, here, because I also ordered the IDs by substance
        # ordered_unique_experiment_ids = np.unique(sorted_ids.experiment_id)
        # y_max = ax.get_ylim()[1]
        for l1, l1col in zip(ordered_unique_level_1, colors_level_1):
            x_level_1 = np.where(np.logical_and(
                sorted_ids[levels[0]] == l0,
                sorted_ids[levels[1]] == l1
            ))[0]
            if len(x_level_1) == 0:
                continue
            y_level_1 = (y_max, y_max*10)
            ax.fill_between(x_level_1, *y_level_1, color=l1col, alpha=.25)
            ax.text(x_level_1.mean(), np.exp(np.mean(np.log(y_level_1))), s=str(l1)[:3], 
                    ha="center", va="center", color=l1col, fontsize=8)
    


    ax.set_xlim(-5,samples.dims["id"]+5)
    ax.set_title(f"$C_{{e,0}}$ nominal + {idata_group}")
    ax.set_ylabel(r"$C_{e,0}$")
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.set_yticks(10**np.arange(*np.round(np.log10([y_min, y_max*10])), 1))
    ax.set_yticks(10**np.arange(*np.round(np.log10([y_min, y_max*10])), 1))
    ax.set_xlabel("ID")
    ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    fig.savefig(f"{sim.output_path}/y0_nominal_{idata_group}.png")



if __name__ == "__main__":
    cfg = "case_studies/hierarchical_molecular_tktd/scenarios/testing/settings.cfg"
    # cfg = "case_studies/tktd_rna_pulse/scenarios/rna_pulse_3_6c_substance_specific/settings.cfg"
    sim = NomixHierarchicalSimulation(cfg)
    
    # TODO: this will become a problem once I try to load different extra
    # modules. The way to deal with this is to load modules as a list and try
    # to get them in hierarchical order
    sim.config.import_casestudy_modules()

    # sim.setup_data_structure_from_observations()
    sim.setup_data_structure_manually(
        scenario="model_inspection"
    )

    # run a simulation
    sim.dispatch_constructor()
    e = sim.dispatch(theta=sim.model_parameter_dict)
    e()
    sim.plot(e.results)

    # generate artificial data
    sim.dispatch_constructor()
    res = sim.generate_artificial_data(nan_frac=0.0)
    res_ = sim.lethality_to_conditional_survival(res)

    # mark data as observed
    sim.observations = res_
    # mark existing data variables as unobserved
    for key, datavar in sim.config.data_structure.all.items():
        if key in ["cint", "nrf2", "survival"]:
            datavar.observed = True

    sim.plot(res)

    # perform inference
    sim.config.jaxsolver.throw_exception = False
    sim.dispatch_constructor()
    sim.use_numpyro_backend(
        error_model=independent_survival_error_model,
        only_prior=False
    )
    sim.config.inference_numpyro.kernel = "nuts"
    sim.config.inference_numpyro.draws = 2000
    sim.config.inference_numpyro.svi_iterations = 1000

    pp = sim.inferer.prior_predictions()
    plot_kwargs = dict(mode="mean+hdi")
    sim.inferer.plot_prior_predictions("cint", x_dim="time", **plot_kwargs) # type:ignore
    sim.inferer.run()

    # next steps:
    # 1. add substance index to obsertvation like in dict -> dataset
    # 2. convert lethality to conditional probability survival notation
    # 3. Try numpyro plate notation and when batch dimension is returned
    #    Or extend dims if no batch is present



    # define a hierarchical error structure
    # check out murefi for this

    # the long form should always be used for the actual model calculations
    # unless wide form is actually required (i.e. vectors or matrices need)
    # to enter the ODE
    
    # currently I use the substance as an index for broadcasting the parameters
    # from a substance index to the long form.
    # multilevel index or something along these lines would be needed to 
    # bring a multilevel index into the long form.


    # currently parameters are at least broadcasted in the JaxSolver, but this
    # is not happening with the other solvers. 
    # Approach:
    # + Define a module that can handle parameter broadcasting automatically 
    #   during dispatch. This can be adapted from the JaxSolver.
    # + Solvers themselves should only handle the casting of the data to types
    #   they require.
    # + This would mean that it is ensured that parameter, y_0 and x_in shapes
    #   can be handled by the solver, because they have been broadcasted, and
    #   can be vectorized or iterated over.
    #