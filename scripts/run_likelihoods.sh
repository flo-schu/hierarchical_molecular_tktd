#!/usr/bin/env bash
#SBATCH --job-name=likelihood-landscape                 # name of job
#SBATCH --time=0-12:00:00                               # maximum time until job is cancelled
#SBATCH --ntasks=1                                      # number of tasks
#SBATCH --cpus-per-task=8                               # number of cpus requested
#SBATCH --mem-per-cpu=8G                                # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@uos.de              # email of user
#SBATCH --output=/home/staff/f/fschunck/logs/job-%x-%A_%a.out    # output file of stdout messages
#SBATCH --error=/home/staff/f/fschunck/logs/job-%x-%A_%a.err     # output file of stderr messages

strings=(
    "k_i_substance"
    "r_rt_substance"
    "r_rd_substance"
    "v_rt_substance"
    "z_ci_substance"
    "k_p_substance"
    "k_m_substance"
    "h_b_substance"
    "z_substance"
    "kk_substance"
    "sigma_nrf2"
    "sigma_cint"
)

parameters=()

# Iterate over the array to create pairs
for ((i=0; i<${#strings[@]}-1; i++)); do
    for ((j=i+1; j<${#strings[@]}; j++)); do
        # Store the pair in the pairs array
        parameters+=("${strings[i]} ${strings[j]}")
    done
done

spack load miniconda3
source activate hmt
spack unload miniconda3

export JAX_ENABLE_X64=True

index=$SLURM_ARRAY_TASK_ID

# Extract the two elements from the pair into separate variables
pair=${parameters[index]}
read -r parx pary <<< "$pair"

python scripts/likelihood_landscape.py \
    --config=scenarios/hierarchical_cext_nested_sigma_hyperprior/settings.cfg \
    --parx="${parx}" \
    --pary="${pary}" \
    --n_grid_points=50 \
    --n_vector_points=0