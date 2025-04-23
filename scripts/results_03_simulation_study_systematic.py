import os
from matplotlib import pyplot as plt
from scipy.stats import lognorm, norm, multinomial
import numpy as np
from hierarchical_molecular_tktd.sim import NomixHierarchicalSimulation
from pymob import Config

def setup_simulation_study(
    true_parameters,
    experimental_variability=0.5, 
    within_experiment_endpoint_overlap=0.0,
):

    config = Config("scenarios/simulation_study_v2_informed_masked/settings.cfg")
    config.case_study.scenario = "simulation_study_v3_variability_{vari}_overlap_{over}".format(
        vari=experimental_variability,
        over=within_experiment_endpoint_overlap,
    )
    sim = NomixHierarchicalSimulation(config)
    sim.setup()

    # create random between experiments variation
    ce_nom = sim.observations.cext_nom
    sigma_k = lognorm(scale=1, s=experimental_variability)
    sigma_k_vals = sigma_k.rvs(size=42, random_state=6)
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
    sim.observations["cint"].values = e.results.cint * np.exp(norm(loc=0, scale=0.1).rvs((314,23), random_state=2))
    sim.observations["nrf2"].values = e.results.nrf2 * np.exp(norm(loc=0, scale=0.1).rvs((314,23), random_state=3))
    multinom_surv = (1-e.results.survival).diff(dim="time")
    multinom_surv = np.column_stack([multinom_surv.values, 1-multinom_surv.sum("time")])
    multinom_surv = np.clip(multinom_surv,0.000001,0.99999) 
    incidence = np.array(list(map(lambda x, n: multinomial(p=x/x.sum(), n=n).rvs(), multinom_surv, sim.observations.nzfe.values) )).squeeze()
    sim.observations["lethality"].values = np.column_stack([np.zeros_like(incidence[:,0]), incidence.cumsum(axis=1)[:,:-1]])
    sim.observations = sim.lethality_to_conditional_survival(sim.observations)
    sim.observations["nzfe"] = sim.observations.nzfe.isel(time=0)


    # mask data    
    rng = np.random.default_rng(1)
    for endpoint in ["cint", "nrf2", "survival"]:
        mask_endpoint = mask_zero_overlap[endpoint].copy()
        n_masked_endpoint = int((~mask_zero_overlap[endpoint]).sum())
        n_frac_unmasked_endpoint = int(n_masked_endpoint * within_experiment_endpoint_overlap)
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


    sim.observations = sim.lethality_to_conditional_survival(sim.observations)
    sim.config.save(force=True)
    return sim


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
    relative_bias = np.abs((estimated - true_parameters) / true_parameters)[compare_params]
    absolute_bias = np.abs((estimated - true_parameters))[compare_params]
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

    import itertools as it
    import pandas as pd
    exploration_params = list(it.product(
        [0.001, 0.1, 0.5],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ))

    exploration = pd.DataFrame(exploration_params, columns=["variability", "overlap"])
    
    relbias, absbias = [], []
    for variability, overlap in exploration_params:
        sim = setup_simulation_study(
            true_parameters=true_parameters,
            experimental_variability=variability, 
            within_experiment_endpoint_overlap=overlap
        )

        sim = run_inference(sim, force=True)
        rel_bias, abs_bias = compute_bias(sim, true_parameters)

        relbias.append(rel_bias)
        absbias.append(abs_bias)
        plt.close('all')

    exploration["relative bias"] = relbias
    exploration["absolute bias"] = absbias

    fig, ax = plt.subplots(1,1)
    for (k, group), ls in zip(exploration.groupby("variability"), ["-", "--", "dotted"]):
        ax.plot(group["overlap"], group["relative bias"], marker="o", color="black", ls=ls, label=f"Experimental deviation {k}")

    exploration.to_csv("../../results/tables/simulation_study_bias.csv")

    print(rel_bias)