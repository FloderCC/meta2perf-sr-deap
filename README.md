# meta2perf-symbolic-regression-deap

Code and experiment artifacts for the paper **“Balancing Accuracy and Interpretability in Symbolic Modeling of Machine Learning Performance.”** This repository implements a two-stage symbolic regression workflow for predicting the performance of machine learning classifiers from dataset meta-features and model descriptors. The target variable is the **Matthews Correlation Coefficient (MCC)** of the best tuned configuration for each learning algorithm. The paper studies how a strong first-stage symbolic predictor can be refined into a more interpretable second-stage equation while preserving useful predictive accuracy.

The repository reuses a previously constructed meta-dataset with **25,836 meta-instances**, derived from **65 classification datasets**, **5 sampling ratios** (20%, 40%, 60%, 80%, and 100%), **16 machine learning algorithms**, and **5 random seeds**. Each meta-instance represents one `(seed, dataset variant, algorithm)` experiment and combines dataset meta-features with model descriptors. The current study focuses on **MCC > 0** instances, matching the evaluation protocol described in the paper.

## Repository overview

```text
.
├── README.md
└── src/
    ├── __init__.py
    ├── datasets/                    # per-dataset notes/READMEs
    ├── docs/                        # paper-aligned appendices and reference material
    │   ├── datasets.md
    │   ├── ml-hyperparams.md
    │   └── model-descriptors.md
    ├── logs/
    │   └── exp_stage_perf.txt       # execution log for the stage-1 search
    ├── plots/
    │   ├── pred_vs_true_stage1.pdf
    │   └── pred_vs_true_stage2.pdf
    ├── results/
    │   ├── exp_stage_sim_hall_of_fame.csv
    │   └── meta_dataset.csv
    ├── results_analysis/
    │   ├── __init__.py
    │   ├── reg_model_perf.py
    │   ├── reg_model_simp.py
    │   └── reg_model_simp_fact.py
    ├── utils/
    │   ├── __init__.py
    │   └── ga_init_ext.py
    ├── exp_stage_perf.py            # stage 1: accuracy-oriented DEAP + OBLESA search
    └── exp_stage_simp.py            # stage 2: penalized refinement for interpretability
```

## What this repository contains

### 1. Stage 1: performance-oriented symbolic regression

`src/exp_stage_perf.py` runs the first symbolic regression stage in DEAP

Outputs produced by this stage include:

- `src/logs/exp_stage_perf.txt`
- `src/plots/pred_vs_true_stage1.pdf`
- the best evolved symbolic expression printed in the log

### 2. Stage 2: penalized refinement for a more interpretable model

`src/exp_stage_simp.py` implements the second stage described in the paper. It reuses the same meta-dataset and core GP machinery, but adds a penalized objective intended to improve the accuracy–interpretability balance of the final expression.


Main output:

- `src/results/exp_stage_sim_hall_of_fame.csv`
- `src/plots/pred_vs_true_stage2.pdf`

### 3. Analysis scripts for the selected equations

The `src/results_analysis/` folder contains analysis and plotting scripts for the selected symbolic predictors:

- `reg_model_perf.py` analyzes and plots the stage-1 predictor.
- `reg_model_simp.py` analyzes and plots the stage-2 balanced predictor.
- `reg_model_simp_fact.py` contains a factored/block-style version of the simplified predictor.

### 4. OBLESA-based initialization utilities

`src/utils/ga_init_ext.py` contains the external population-initialization logic used by the DEAP runs. In the methodology of the paper, this component is responsible for generating a stronger initial population through the OBLESA procedure before the GP evolution begins.

## Running the code

The experiment scripts use paths such as `./results/meta_dataset.csv` and `logs/exp_stage_perf.txt`, so they are intended to be executed **from inside the `src/` directory**.

Typical workflow:

```bash
cd src
python exp_stage_perf.py
python exp_stage_simp.py
python results_analysis/reg_model_perf.py
python results_analysis/reg_model_simp.py
```

## Python dependencies

A `requirements.txt` file is not included in the attached source archive, but the checked-in scripts clearly depend on at least the following Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `deap`
- `tqdm`
