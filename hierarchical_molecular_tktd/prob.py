from tktd_rna_pulse.prob import *
from guts_base.prob import conditional_survival_from_hazard

EPS = 9.9e-5

def conditional_survival_error_model_old(theta, simulation_results, observations, masks, indices, only_prior=False, make_predictions=False):
    # indexing
    substance_idx = indices["substance_index"]
    sigma_cint_indexed = theta["sigma_cint"][substance_idx]
    sigma_nrf2_indexed = theta["sigma_nrf2"][substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), observations["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), observations["nrf2"].shape)

    # error model
    S = jnp.clip(simulation_results["H"], EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = observations["survivors_before_t"]
    S_mask = masks["survival"]

    if make_predictions:
        obs_cint = None
        obs_nrf2 = None
        obs_survival = None
    else:
        obs_cint = observations["cint"]
        obs_nrf2 = observations["nrf2"]
        obs_survival = observations["survival"]
    
    numpyro.sample("cint_obs", dist.LogNormal(loc=jnp.log(simulation_results["cint"] + EPS), scale=sigma_cint_ix_bc).mask(masks["cint"]), obs=obs_cint)
    numpyro.sample("nrf2_obs", dist.LogNormal(loc=jnp.log(simulation_results["nrf2"]), scale=sigma_nrf2_ix_bc).mask(masks["nrf2"]), obs=obs_nrf2)    
    numpyro.sample(
        "survival_obs", 
        dist.Binomial(probs=S_cond_, total_count=n_surv).mask(S_mask), 
        obs=obs_survival
    )


def conditional_survival_error_model(theta, simulation_results, observations, masks, indices, only_prior=False, make_predictions=False):
    # indexing
    substance_idx = indices["substance_index"]
    # sigma_cint_indexed = theta["sigma_cint"][substance_idx]
    # sigma_nrf2_indexed = theta["sigma_nrf2"][substance_idx]

    # sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), observations["cint"].shape)
    # sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), observations["nrf2"].shape)

    # error model
    S = jnp.clip(simulation_results["survival"], EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = observations["survivors_before_t"]
    S_mask = masks["survival"]
    
    if make_predictions:
        obs_cint = None
        obs_nrf2 = None
        obs_surv = None
    else:
        obs_surv = observations["survival"]
        obs_cint = numpyro.deterministic("cint_res", jnp.log(observations["cint"]+EPS) - jnp.log(simulation_results["cint"]+EPS))
        obs_nrf2 = numpyro.deterministic("nrf2_res", jnp.log(observations["nrf2"]+EPS) - jnp.log(simulation_results["nrf2"]+EPS))

    obs_vars = ["cint", "nrf2", "survival"]
    n = {k: masks[k].sum() for k in obs_vars}
    N = sum(n.values())
    weights = {k: 1/len(obs_vars) for k in obs_vars}
    scaling_factors = {k: 1/n[k]*N*weights[k] for k in obs_vars}

    # calculate likelihoods
    with numpyro.handlers.scale(scale=scaling_factors["cint"]):
        lik_cint = numpyro.sample("cint_obs", dist.Normal(
                loc=0.0,  # type: ignore
                scale=theta["sigma_cint"]  # type: ignore
            ).mask(masks["cint"]),
            obs=obs_cint
        )
    
    with numpyro.handlers.scale(scale=scaling_factors["nrf2"]):
        lik_nrf2 = numpyro.sample("nrf2_obs", dist.Normal(
                loc=0.0,  # type: ignore
                scale=theta["sigma_nrf2"]  # type: ignore
            ).mask(masks["nrf2"]), 
            obs=obs_nrf2
        )    
    
    with numpyro.handlers.scale(scale=scaling_factors["survival"]):    
        lik_surv = numpyro.sample(
            "survival_obs", dist.Binomial(
                probs=S_cond_, 
                total_count=n_surv
            ).mask(S_mask), 
            obs=obs_surv
        )


def independent_survival_error_model(theta, simulation_results, observations, masks):
    # indexing
    substance_idx = observations["substance_index"]
    sigma_cint_indexed = theta["sigma_cint"][substance_idx]
    sigma_nrf2_indexed = theta["sigma_nrf2"][substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), masks["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), masks["nrf2"].shape)

    
    # calculate likelihoods
    numpyro.sample("cint_obs", dist.LogNormal(
            loc=jnp.log(simulation_results["cint"] + EPS),  # type: ignore
            scale=sigma_cint_ix_bc  # type: ignore
        ).mask(masks["cint"]),
        obs=observations["cint"]
    )
    
    numpyro.sample("nrf2_obs", dist.LogNormal(
            loc=jnp.log(simulation_results["nrf2"]),  # type: ignore
            scale=sigma_nrf2_ix_bc  # type: ignore
        ).mask(masks["nrf2"]), 
        obs=observations["nrf2"]
    )    
    
    numpyro.sample(
        "survival_obs", dist.Binomial(
            probs=simulation_results["survival"], 
            total_count=observations["nzfe"]
        ).mask(masks["survival"]), 
        obs=observations["survival"]
    )