import pytest
from hierarchical_molecular_tktd.sim import NomixHierarchicalSimulation


def construct_sim(scenario, simulation_class):
    """Helper function to construct simulations for debugging"""
    sim = simulation_class(f"scenarios/{scenario}/settings.cfg")
    sim.config.case_study.scenario = "testing"
    sim.setup()
    return sim


# List test scenarios and simulations
@pytest.fixture(scope="module", params=[
    # early tests
    "hierarchical_cext_nested_sigma_hyperprior",
    # informed, reduced_dataset
    "hierarchical_cext_nested_sigma_hyperprior_informed_reduced_dataset_rna_pulse_5",
    "hierarchical_cext_nested_sigma_hyperprior_informed_reduced_dataset_rna_pulse_5_substance_independent",
    # informed, full dataset
    "hierarchical_cext_nested_sigma_hyperprior_informed_rna_pulse_5",
    "hierarchical_cext_nested_sigma_hyperprior_informed_rna_pulse_5_substance_independent",
    # uninformed, reduced dataset
    "hierarchical_cext_nested_sigma_hyperprior_reduced_dataset_rna_pulse_5",
    "hierarchical_cext_nested_sigma_hyperprior_reduced_dataset_rna_pulse_5_substance_independent",
    # uninformed, full dataset
    "hierarchical_cext_nested_sigma_hyperprior_rna_pulse_5",
    "hierarchical_cext_nested_sigma_hyperprior_rna_pulse_5_substance_independent",
])
def scenario(request):
    return request.param

@pytest.fixture(scope="module", params=[
    NomixHierarchicalSimulation
])
def simulation_class(request):
    return request.param


# Derive simulations for testing from fixtures
@pytest.fixture(scope="module")
def sim(scenario, simulation_class):
    yield construct_sim(scenario, simulation_class)


# run tests with the Simulation fixtures
def test_setup(sim):
    """Tests the construction method"""
    assert True


def test_simulation(sim):
    """Tests if a forward simulation pass can be computed"""
    sim.dispatch_constructor()
    evaluator = sim.dispatch({})
    evaluator()
    evaluator.results

    assert True
            
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["numpyro"])
def test_inference(sim, backend):
    """Tests if prior predictions can be computed for arbitrary backends"""
    sim.dispatch_constructor()
    sim.set_inferer(backend)

    # The below does not work with the new backend, because the priors
    # are sampled in the substance dimension, but need to be mapped to the
    # id dimension. This can be done manually, and in complex cases this is
    # required, but for many cases something like:
    # map_to=("id",) should be fine. By default the index for mapping substance
    # to ID is called substance_index. As the dimension of the variable
    # is substance. This can be filled in automatically. The big advantage of 
    # this is that it can be done in the evaluator, if the keyword is set.
    sim.config.inference.n_predictions = 2
    sim.prior_predictive_checks()

    sim.config.inference_numpyro.svi_iterations = 1_000
    sim.config.inference_numpyro.svi_learning_rate = 0.05
    sim.config.inference_numpyro.draws = 100
    sim.config.inference.n_predictions = 10

    sim.inferer.run()

    sim.inferer.idata

    sim.report()
    sim.posterior_predictive_checks()

if __name__ == "__main__":
    import os
    if os.path.basename(os.getcwd()) != "hierarchical_molecular_tktd":
        # change directory to case_studies/beeguts
        # this may work in case the root is a project with case_studies
        os.chdir("case_studies/tktd_rna_pulse")

    test_inference(sim=construct_sim("hierarchical_cext_nested_sigma_hyperprior_informed_rna_pulse_5_substance_independent", NomixHierarchicalSimulation), backend="numpyro")