from tktd_rna_pulse.plot import *

import numpy as np
import arviz as az
from matplotlib import pyplot as plt
from toopy.plot import letterer, draw_axis_letter

def plot_y0(
    sim, idata, idata_group, parameter, levels, colors={}, 
    show_observed=False,
    show_measured_endpoints=False,
):
    fig, ax = plt.subplots(1,1, figsize=(16,3.5))
    y0 = sim.model_parameters["y0"]
    samples = idata[idata_group]

    batch_dim = sim.config.simulation.batch_dimension
    sorted_ids = sim.observations[batch_dim].sortby(levels)[batch_dim]

    # plot nominal y0 values
    y0_y = y0[parameter].sel({batch_dim: sorted_ids})
    ax.plot(sorted_ids, y0_y, ls="", marker="o", color="black", ms=1, zorder=1.01)
    ax.plot([],[], ls="", marker="o", color="black", ms=1, label=r"$y_0$ nominal")
    
    # plot samples from prior/posoterior. THis works, because it maps samples.id to sorted ids
    samples_y = az.hdi(samples[f"{parameter}_y0"])[f"{parameter}_y0"].T
    ax.vlines(samples[batch_dim], *samples_y, color="grey", lw=1, zorder=1.0)
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
        y_level_0 = (y_max * 6, y_max*25)
        ax.fill_between(x_level_0, *y_level_0, color=l0col, alpha=.25)

        y_mean_level_0 = np.exp(np.mean(np.log(y_level_0)))
        ax.text(x_level_0.mean(), y_mean_level_0, s=str(l0)[:3], 
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
            y_level_1 = (y_max * 1.2, y_max*5)
            y_mean_level_1 = np.exp(np.mean(np.log(y_level_1)))
            ax.fill_between(x_level_1, *y_level_1, color=l1col, alpha=.25)
            ax.text(x_level_1.mean(), y_mean_level_1, s=str(l1)[:3], 
                    ha="center", va="center", color=l1col, fontsize=8)


    if show_observed:
        y0_obs = sim.observations[parameter].sel({batch_dim: sorted_ids}).isel(time=0)
        ax.plot(sorted_ids, y0_obs, ls="", marker="o", color="tab:red", ms=1, alpha=.5, zorder=1.02)
        ax.plot([],[], ls="", marker="o", color="tab:red", ms=1, label=r"$y_0$ observed")
    

    if show_measured_endpoints:
        # Endpoint measurement
        for dv_i, dv in enumerate(sim.config.data_structure.observed_data_variables):
            for l0 in ordered_unique_level_0:
                x_level_0 = np.where(sorted_ids[levels[0]] == l0)[0]
                obs_level_0 = sim.observations.where(sorted_ids[levels[0]] == l0, drop=True)
                is_dv_observed_in_experiment = bool((~obs_level_0[dv].isnull()).sum() > 0)
                str_observed = "✔" if is_dv_observed_in_experiment else "✖️"
                y_level_0 = y_max * 30 * np.exp(dv_i*0.75)

                ax.text(
                    x_level_0.mean(), y_level_0, str_observed,
                    ha="center", va="center", color="black", fontsize=8
                )
            ax.text(-6, y_level_0, s=dv, ha="right", va="center", fontsize=8)

    ax.set_xlim(-5,samples.dims["id"]+5)
    ax.set_ylim(y_min*0.5, y_level_0*1.5)
    ax.set_title(f"$C_{{e,0}}$ nominal + {idata_group}")
    ax.set_ylabel(r"$C_{e,0}$")
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.set_yticks(10**np.arange(*np.round(np.log10([y_min, y_max*10])), 1))
    ax.set_xlabel("ID")
    ax.legend(loc="lower right", ncols=3)
    fig.tight_layout()
    out = f"{sim.output_path}/y0_nominal_{idata_group}.png"
    fig.savefig(out)
    return out



def compare_external_cocentrations(sim):
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


    labels = letterer()
    fig, axes = plt.subplots(2,3, figsize=(5,3), sharex="col")
    for i, (ax, sub) in enumerate(zip(axes[0, :], sim.dimension_coords["substance"])):
        x = cext_obs.where(cext_obs.substance==sub, drop=True)
        y = cext_sim.where(cext_obs.substance==sub, drop=True)

        y = y.where(~x.isnull(), drop=True)
        x = x.where(~x.isnull(), drop=True)

        scaled_residuals = np.round(np.mean(np.abs((x - y) / x).values),2)

        cmin = float(min(x.min(),y.min())) * 0.9
        cmax = float(max(x.max(),y.max())) * 1.1
        ax.scatter(x, y, c=colors[i], ls="", marker="o", alpha=.5, label="")
        ax.plot(np.linspace(cmin, cmax),np.linspace(cmin,cmax), color="black")
        ax.text(0.95, 0.05, s= scaled_residuals,ha="right", va="bottom", transform=ax.transAxes)
        draw_axis_letter(ax, next(labels))

        if i == 0:
            ax.set_ylabel(r"$C_e$ (estimated)")
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(cmin,cmax)
        ax.set_ylim(cmin,cmax)
        ax.set_title(sub.capitalize())

    for i, (ax, sub) in enumerate(zip(axes[1, :], sim.dimension_coords["substance"])):
        x = cext_obs.where(cext_obs.substance==sub, drop=True)
        y = cext_nom.where(cext_obs.substance==sub, drop=True)
        y = y.where(~x.isnull(), drop=True)
        x = x.where(~x.isnull(), drop=True)
        scaled_residuals = np.round(np.mean(np.abs((x - y) / x).values),2)
        
        
        cmin = float(min(x.min(),y.min())) * 0.9
        cmax = float(max(x.max(),y.max())) * 1.1
        ax.scatter(x, y, c=colors[i], ls="", marker="o", alpha=.5, label="")
        ax.plot(np.linspace(cmin, cmax),np.linspace(cmin,cmax), color="black")
        ax.text(0.95, 0.05, s= scaled_residuals,ha="right", va="bottom", transform=ax.transAxes)
        draw_axis_letter(ax, next(labels))

        ax.set_xlabel(r"$C_e$ (observed)")
        if i == 0:
            ax.set_ylabel(r"$C_e$ (nominal)")
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(cmin,cmax)
        ax.set_ylim(cmin,cmax)
        # ax.set_title(sub.capitalize())

    fig.tight_layout()
    out = f"{sim.output_path}/scatter_cext_combined.png"
    fig.savefig(out)
    return out
