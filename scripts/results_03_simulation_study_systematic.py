import os
from matplotlib import pyplot as plt
from scipy.stats import lognorm, norm, multinomial
import numpy as np
import itertools as it
import pandas as pd
from hierarchical_molecular_tktd.sim import NomixHierarchicalSimulation
from pymob import Config

def setup_simulation_study(
    true_parameters,
    experimental_variability=0.5, 
    fraction_unmasked_nans=0.0,
    seed=0,
):

    config = Config("scenarios/simulation_study_v2_informed_masked/settings.cfg")
    config.case_study.scenario = "simulation_study_v4_variability_{vari}_overlap_{over}_seed_{seed}".format(
        vari=experimental_variability,
        over=fraction_unmasked_nans,
        seed=seed,
    )
    sim = NomixHierarchicalSimulation(config)
    sim.setup()

    # create random between experiments variation
    ce_nom = sim.observations.cext_nom
    sigma_k = lognorm(scale=1, s=experimental_variability)
    sigma_k_vals = sigma_k.rvs(size=42, random_state=seed)
    ce_true = sim.observations.cext_nom * sigma_k_vals[sim.indices["experiment_id"]]

    _theta = true_parameters.copy()
    _theta["k_i"] = _theta["k_i_substance"][sim.indices["substance"]]
    _theta["k_m"] = _theta["k_m_substance"][sim.indices["substance"]]
    _theta["z_ci"] = _theta["z_ci_substance"][sim.indices["substance"]]
    _theta["ci_max"] = _theta["ci_max_substance"][sim.indices["substance"]]

    # simulate deterministic data based on true external coencentrations
    e = sim.dispatch(
        y0=sim.validate_model_input(ce_true.to_dataset(name="cext")),
        theta=_theta
    )

    e()

    # store original mask of experiments
    mask_zero_overlap = ~sim.observations.isnull()
    mask_zero_overlap["nzfe"] = mask_zero_overlap.nzfe.isel(time=0)

    # add noise to data
    sim.observations["nzfe"] = sim.observations.nzfe.isel(time=0)
    sim.observations["cext"].values = e.results.cext
    sim.observations["cint"].values = e.results.cint * np.exp(norm(loc=0, scale=0.1).rvs((314,23), random_state=seed))
    sim.observations["nrf2"].values = e.results.nrf2 * np.exp(norm(loc=0, scale=0.1).rvs((314,23), random_state=seed+1000))
    multinom_surv = (1-e.results.survival).diff(dim="time")
    multinom_surv = np.column_stack([multinom_surv.values, 1-multinom_surv.sum("time")])
    multinom_surv = np.clip(multinom_surv,0.000001,0.99999) 
    incidence = np.array(list(map(lambda x, n: multinomial(p=x/x.sum(), n=n).rvs(), multinom_surv, sim.observations.nzfe.values) )).squeeze()
    sim.observations["lethality"].values = np.column_stack([np.zeros_like(incidence[:,0]), incidence.cumsum(axis=1)[:,:-1]])
    sim.observations = sim.lethality_to_conditional_survival(sim.observations)
    sim.observations["nzfe"] = sim.observations.nzfe.isel(time=0)


    # mask data
    # shuffling can always be the same, because we can assume this is a deterministic process
    rng = np.random.default_rng(1)
    for endpoint in ["cint", "nrf2", "lethality"]:
        mask_endpoint = mask_zero_overlap[endpoint].copy()
        n_masked_endpoint = int((~mask_zero_overlap[endpoint]).sum())
        n_frac_unmasked_endpoint = int(n_masked_endpoint * fraction_unmasked_nans)
        unmask_indices_endpoint = np.full(n_masked_endpoint, False)
        unmask_indices_endpoint[:n_frac_unmasked_endpoint] = True
        rng.shuffle(unmask_indices_endpoint)
        idx_0, idx_1 =  np.where(~mask_endpoint)
        idx_0[unmask_indices_endpoint]
        idx_1[unmask_indices_endpoint]

        mask_endpoint.values[idx_0[unmask_indices_endpoint], idx_1[unmask_indices_endpoint]] = True

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,2))
        ax1.imshow(mask_zero_overlap[endpoint].T, cmap='hot', interpolation='nearest', )
        ax2.imshow(mask_endpoint.T, cmap='hot', interpolation='nearest', )
        
        sim.observations[endpoint] = sim.observations[endpoint].where(mask_endpoint, np.nan)

    within_experiment_endpoint_overlap = ((
        sim.observations.count("time")[["cint", "nrf2", "lethality"]].to_array() != 0
    ).sum("variable") > 1).sum() / 314

    sim.observations = sim.lethality_to_conditional_survival(sim.observations)
    sim.config.save(force=True)
    return sim, within_experiment_endpoint_overlap


def run_inference(sim, force=False):
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.config.inference_numpyro.svi_iterations = 2000
    sim.config.inference_numpyro.svi_learning_rate = 0.005
    sim.config.inference_numpyro.kernel = "map"
    sim.config.simulation.seed=2

    try: 
        sim.inferer.load_results()
        if force:
            raise FileNotFoundError
    except FileNotFoundError:
        sim.inferer.run()
        sim.inferer.store_results()
        sim.report()

    return sim

def compute_bias(sim, true_parameters):
    compare_params = ["k_i_substance", "k_m_substance", "z_ci_substance", "k_p", "kk", "r_rd", "r_rt", "v_rt", "z"]
    estimated = sim.inferer.posterior[true_parameters.keys()].mean(("chain", "draw"))
    n_params = estimated.count().to_array().sum()

    relative_bias = np.abs((estimated - true_parameters) / true_parameters)[compare_params] / n_params
    absolute_bias = np.abs((estimated - true_parameters))[compare_params] / n_params
    return float(relative_bias.sum().to_array().sum()), float(absolute_bias.sum().to_array().sum())


if __name__ == "__main__":

    if os.path.basename(os.getcwd()) != "hierarchical_molecular_tktd":
        os.chdir("..")

    true_parameters = dict(
        k_i_substance=np.array([4.9, 0.38, 0.36]),
        k_m_substance=np.array([3.8, 0.19, 0.084]),
        z_ci_substance=np.array([1.2, 1.0, 1.2]),
        ci_max_substance=np.array([1757.0,168.1,6364.8]),
        h_b=0.0,
        k_p=0.008,
        kk=0.079,
        r_rd=0.23,
        r_rt=2.3,
        v_rt=3.7,
        z=1.5,
    )

    exploration_params = list(it.product(
        [0.001, 0.1, 0.5, 1.0],
        [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.2],
        np.arange(10),
    ))

    exploration = pd.DataFrame(exploration_params, columns=["variability", "unmasked nans", "seed"])
    
    relbias, absbias, overlap = [], [], []
    for variability, unmasked_nans, seed in exploration_params:
        sim, _overlap = setup_simulation_study(
            true_parameters=true_parameters,
            experimental_variability=variability, 
            fraction_unmasked_nans=unmasked_nans,
            seed=seed
        )

        
        sim = run_inference(sim, force=False)
        rel_bias, abs_bias = compute_bias(sim, true_parameters)

        overlap.append(float(_overlap))
        relbias.append(rel_bias)
        absbias.append(abs_bias)
        plt.close('all')

    exploration["relative bias"] = relbias
    exploration["absolute bias"] = absbias
    exploration["overlap"] = overlap
    

    fig, ax = plt.subplots(1,1)
    for (k, group), ls in zip(exploration.groupby("variability"), ["-", "--", "dotted", "-."]):
        group_mean_reps = group.groupby("unmasked nans").mean().reset_index()
        std = round(lognorm(scale=1, s=k).std(), 1)
        ax.plot(group_mean_reps["overlap"], group_mean_reps["relative bias"] * 100, marker="o", color="black", ls=ls, label=f"Experimental standard deviation {std}")
    ax.plot(group_mean_reps["overlap"], group_mean_reps["unmasked nans"] * 50, color="tab:orange", label="Filled missing values")
    ax.legend()
    ax.set_xlabel("Experiment-wise overlap between endpoints")
    ax.set_ylabel("Parameter bias (%)")


    secax_y = ax.secondary_yaxis(
        'right', functions=(lambda x: x*2, lambda x: x/2))
    secax_y.set_ylabel('Missing values filled in the experimental matrix (%)')

    ax.grid()
    
    fig.savefig("../../results/figures/simulation_study_bias_v4.png")

    exploration.to_csv("../../results/tables/simulation_study_bias.csv")

    print(rel_bias)