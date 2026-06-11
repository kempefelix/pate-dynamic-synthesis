# PATE Dynamic Synthesis

**Improving the Privacy of Federated Learning via Optimized Synthetic Data Generation**

Master Thesis – Felix Kempe, TU Wien
Supervised by Ao.Univ.Prof. Dr. Andreas Rauber

## Overview

This repository implements two dynamic synthesis strategies that close the *Unlabeled Data Gap* in the PATE framework under Non-IID conditions:

- **Variant A (Distribution-Aware Synthesis):** Inverse confidence weighting based on the student's class confidence map
- **Variant B (Error-Aware Synthesis):** Error-proportional weighting with confusion-based contrast pairs

The PATE aggregation logic and teacher CNN architecture are adapted from the NEWSROOM/Saferlearn Framework (Thales Research & Technology).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (smoke test)

```bash
# Single run to verify the pipeline works
# (configs/default.yaml = sigma=10, MNIST only — NOT the reported conditions)
python -m src.experiments.run_experiment --config configs/default.yaml \
    --dataset MNIST --beta 0.1 --strategy variant_a --seed 0
```

## Reproducing the evaluation

### From the committed results (tables + figures)

```bash
# 1. Re-derive all epsilon values under the corrected sensitivity (Delta_2 = sqrt(2))
python3 recompute_epsilon.py

# 2. Regenerate all tables, t-tests, savings and the 8 thesis figures
python3 make_results.py --fig-dir figures
```

`make_results.py` is the canonical evaluation pipeline; the older
`src/experiments/evaluate.py` aggregated the three configurations of each
variant as if they were seeds and is superseded.

### Re-running the experiments (raw results)

The reported results use **sigma=40, epsilon_max=100**, set by the SLURM
submission scripts (NOT `configs/default.yaml`). On a GPU cluster:

```bash
bash run_main_experiments.sh   # 224 main runs (MNIST + CIFAR-10)
bash run_grid_search.sh        # 57 sensitivity-analysis runs
```

GPU/cuDNN nondeterminism makes a re-run statistically equivalent but not
bit-identical to the committed `results/` (up to ~6 pp on identical configs).
The committed `results/` plus the analysis scripts above reproduce all tables
and figures exactly.

## Provenance of `results/` (grid search)

* The sensitivity analysis (grid search) was **submitted via
  `run_grid_search.sh`** (57 jobs: 27 Variant-A + 27 Variant-B + 3 Static
  configurations over σ ∈ {10, 20, 40}, MNIST, β = 0.1, one seed).
* The grid-search JSONs in `results/grid_search/` were **reconstructed from
  the SLURM log files via `recover_grid.py`** (the original JSON outputs were
  lost); they therefore contain only the per-round series
  `round/accuracy/macro_f1/epsilon/teacher_queries`, without
  `per_class_accuracy`/`synthesis_distribution`. The SLURM log files
  themselves are not part of the supplement.
* `generate_grid_search.py` is an **unused early draft** of a different grid
  (it varies σ ∈ {10, 20, 40, 80}, `samples_per_round` and the Dirichlet β,
  but **not** the strategy hyperparameters m/λ) and was never used for the
  reported experiments. It is kept for documentation only.
* Seven σ = 40 grid configurations (`MNIST_beta0.1_seed0_{static,
  varA_a1.0_m0.5, varA_a2.0_m0.3, varA_a4.0_m0.1, varB_b1.0_l0.1,
  varB_b1.0_l0.3, varB_b2.0_l0.5}`) coincide with main-experiment
  configurations; both result files are kept (separate runs of identical
  configurations, accuracy deviations up to ~6 pp, see thesis
  Sec. "Reproduzierbarkeit").
* One additional file (`results/grid_search/MNIST_beta0.1_seed0.json`) did
  not correspond to any documented configuration (its filename encodes
  neither strategy nor hyperparameters; the run is a σ = 40 Variant-B run
  whose b/λ values cannot be identified). It was **removed** so that
  `results/` contains exactly the 281 documented runs
  (112 MNIST + 112 CIFAR-10 + 57 grid search).

## Acknowledgments

The PATE aggregation logic and teacher CNN architecture are adapted from the NEWSROOM/Saferlearn framework (Romain Ferrari et al., Thales Research & Technology).