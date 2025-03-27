from tktd_rna_pulse.plot import *

import numpy as np
import arviz as az
from matplotlib import pyplot as plt

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
    out = f"{sim.output_path}/y0_nominal_{idata_group}.png"
    fig.savefig(out)
    return out

