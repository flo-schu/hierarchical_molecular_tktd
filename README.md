# Hierarchical RNA pulse TKTD model

Short summary of the case study

## Instructions

### Prerequisites

Install git (https://git-scm.com/downloads), conda (https://docs.anaconda.com/free/miniconda/) and datalad (https://handbook.datalad.org/en/latest/intro/installation.html)

### Installation

**Prerequisites**

**If the case study was not already installed as a submodule** with a meta package, you can install the package as follows.

Open a command line utility (cmd, bash, ...) and execute

```bash
git clone git@github.com:flo-schu/CASE_STUDY
cd CASE_STUDY
```

Create environment, activate it and install model package. 
```bash
conda create -n CASE_STUDY
conda activate guts
conda install python=3.11
pip install -r requirements.txt
```

In order to install additional inference backends, use:

```bash
pip install pymob[numpyro]
```

For the available backends see https://pymob.readthedocs.io/en/latest/

### Download or start tracking datasets

Download potentially existing results datasets

```bash
datalad clone git@gin.g-node.org:/flo-schu/CASE_STUDY__results.git results
```

if this is not possible create a new dataset (see section below) 
```bash
datalad create -c text2git results
datalad create -c text2git data
``` 

### Case study layout

This is the layout of your folder. Files like README, requirements, LICENSE and .gitignore are not shown, because they are not strictly necessary for the case study but contain important metadata
```
└─ lotka_volterra_case_study
    ├─ data
    │   └─ ...
    ├─ results
    │   └─ ...
    ├─ scenarios
    │   ├─ test_scenario
    │   │   └─ settings.cfg
    │   └─ test_scenario_2
    ├─ scripts
    │   └─ ...
    ├─ __init__.py
    ├─ data.py
    ├─ mod.py
    ├─ plot.py
    ├─ prob.py
    ├─ sim.py
    ├─ ...
```

## Usage

The case studies should now be ready to use.
To get started, see: https://pymob.readthedocs.io/en/latest/

### Command line

there is a command line script provided by the `pymob` package which will directly
run inference accroding to the scenario settings.cfg file provided in the scenario
folder of the respective case study. For details see https://pymob.readthedocs.io/en/latest/case_studies.html

`pymob-infer --case_study CASE_STUDY --scenario SCENARIO --inference_backend numpyro`

The options in the `settings.cfg` are the same as used when preparing the publication

The results will be stored in the results directory of the respective case study 
unless otherwise specified in the settings.cfg


## Tracking results and data with datalad and gin

1. create a new `results` dataset in the root of the repository

`datalad create -c text2git results`

if the directory already exists, use the `-f` flag:

`datalad create -f -c text2git results`

2. save the results as they come in in order to keep a version history.

`datalad save -m "Computed posterior with nuts"`


### Upload the dataset to gin.


1. Create a new repository on gin https://gin.g-node.org/

2. add the repository as a remote for the dataset

`datalad siblings add -d . --name gin --url git@gin.g-node.org:/flo-schu/CASE_STUDY__results.git`

The remote is now connected to the sibling `gin`. 

3. Push new results to gin `datalad push --to gin` 


## Using a case study as a foundation for building new models

You may have noticed an `__init__.py` file. This file makes the case study a package. This way, the package can be installed as a submodule in any other package. 

This means, case-studies can be stacked on top of each other. This way, only the changes need to be captured in the 
new case study and can be transferred into the next project.

The downside is that pymob versions need to be compatible for this use case.