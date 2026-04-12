"""
Generate Grid Search Configurations and SLURM Submission Script.

Grid search over:
  - sigma:            {10, 20, 40, 80}
  - alpha (Var A):    {1.0, 2.0, 4.0}
  - beta_focus (Var B): {1.0, 2.0, 4.0}
  - samples_per_round: {2000, 5000, 10000}
  - dirichlet_beta:   {0.1, 0.5, 1.0}

Strategy: Run each config with 1 seed first, then best configs with 5 seeds.
"""

import yaml
import os
from itertools import product

OUTPUT_DIR = "configs/grid_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base config
base = {
    "seed": 42,
    "device": "auto",
    "num_seeds": 1,
    "dataset": {"name": "MNIST", "num_classes": 10, "data_dir": "./data"},
    "federation": {"num_teachers": 250},
    "teacher": {
        "architecture": "TeacherCNN",
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "adam",
    },
    "student": {
        "architecture": "TeacherCNN",
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "adam",
    },
    "cgan": {
        "latent_dim": 100,
        "num_epochs": 200,
        "batch_size": 64,
        "lr_g": 0.0002,
        "lr_d": 0.0002,
        "beta1": 0.5,
        "feature_maps_g": 64,
        "feature_maps_d": 64,
        "min_samples_to_train": 100,
    },
    "evaluation": {
        "target_accuracy": 0.95,
        "convergence_threshold": 0.005,
        "metrics": ["accuracy", "macro_f1", "epsilon", "teacher_queries"],
    },
    "baselines": {"run_upper_bound": False, "run_classical_pate": False},
    "logging": {"save_dir": "./results/grid_search", "save_checkpoints": False, "log_interval": 1},
}

# Grid search parameters
sigmas = [10, 20, 40, 80]
alphas = [1.0, 2.0, 4.0]           # Variant A focusing exponent
beta_focuses = [1.0, 2.0, 4.0]     # Variant B focusing exponent
samples_per_round = [2000, 5000, 10000]
dirichlet_betas = [0.1, 0.5, 1.0]

configs = []
job_lines = []

# Variant A grid
for sigma, alpha, spr, dbeta in product(sigmas, alphas, samples_per_round, dirichlet_betas):
    cfg = yaml.safe_load(yaml.dump(base))  # deep copy
    cfg["pate"] = {
        "mechanism": "gaussian",
        "sigma": float(sigma),
        "laplace_scale": 20.0,
        "delta": 1.0e-5,
        "epsilon_max": 100.0,
    }
    cfg["synthesis"] = {
        "num_rounds": 10,
        "samples_per_round": spr,
        "variant_a": {"alpha": alpha},
        "variant_b": {"beta": 2.0, "lambda_contrast": 0.3, "top_m_confusions": 3, "n_base_ratio": 0.7, "n_contrast_ratio": 0.3},
    }
    cfg["strategies"] = ["variant_a"]
    cfg["federation"]["dirichlet_beta"] = [dbeta]

    name = f"va_s{sigma}_a{alpha}_spr{spr}_b{dbeta}"
    fname = f"{OUTPUT_DIR}/{name}.yaml"
    with open(fname, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    configs.append((name, fname, dbeta))

# Variant B grid
for sigma, bf, spr, dbeta in product(sigmas, beta_focuses, samples_per_round, dirichlet_betas):
    cfg = yaml.safe_load(yaml.dump(base))
    cfg["pate"] = {
        "mechanism": "gaussian",
        "sigma": float(sigma),
        "laplace_scale": 20.0,
        "delta": 1.0e-5,
        "epsilon_max": 100.0,
    }
    cfg["synthesis"] = {
        "num_rounds": 10,
        "samples_per_round": spr,
        "variant_a": {"alpha": 2.0},
        "variant_b": {"beta": bf, "lambda_contrast": 0.3, "top_m_confusions": 3, "n_base_ratio": 0.7, "n_contrast_ratio": 0.3},
    }
    cfg["strategies"] = ["variant_b"]
    cfg["federation"]["dirichlet_beta"] = [dbeta]

    name = f"vb_s{sigma}_bf{bf}_spr{spr}_b{dbeta}"
    fname = f"{OUTPUT_DIR}/{name}.yaml"
    with open(fname, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    configs.append((name, fname, dbeta))

# Static baseline grid (only over sigma and samples_per_round)
for sigma, spr, dbeta in product(sigmas, samples_per_round, dirichlet_betas):
    cfg = yaml.safe_load(yaml.dump(base))
    cfg["pate"] = {
        "mechanism": "gaussian",
        "sigma": float(sigma),
        "laplace_scale": 20.0,
        "delta": 1.0e-5,
        "epsilon_max": 100.0,
    }
    cfg["synthesis"] = {
        "num_rounds": 10,
        "samples_per_round": spr,
        "variant_a": {"alpha": 2.0},
        "variant_b": {"beta": 2.0, "lambda_contrast": 0.3, "top_m_confusions": 3, "n_base_ratio": 0.7, "n_contrast_ratio": 0.3},
    }
    cfg["strategies"] = ["static"]
    cfg["federation"]["dirichlet_beta"] = [dbeta]

    name = f"static_s{sigma}_spr{spr}_b{dbeta}"
    fname = f"{OUTPUT_DIR}/{name}.yaml"
    with open(fname, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    configs.append((name, fname, dbeta))

# Generate SLURM submission script
with open("run_grid_search.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"# Grid Search: {len(configs)} total jobs\n")
    f.write(f"# Variant A: {len(sigmas)}σ × {len(alphas)}α × {len(samples_per_round)}spr × {len(dirichlet_betas)}β = {len(sigmas)*len(alphas)*len(samples_per_round)*len(dirichlet_betas)} jobs\n")
    f.write(f"# Variant B: {len(sigmas)}σ × {len(beta_focuses)}bf × {len(samples_per_round)}spr × {len(dirichlet_betas)}β = {len(sigmas)*len(beta_focuses)*len(samples_per_round)*len(dirichlet_betas)} jobs\n")
    f.write(f"# Static:    {len(sigmas)}σ × {len(samples_per_round)}spr × {len(dirichlet_betas)}β = {len(sigmas)*len(samples_per_round)*len(dirichlet_betas)} jobs\n\n")
    f.write("SUBMITTED=0\n\n")

    for name, fname, dbeta in configs:
        f.write(f'sbatch --job-name="gs_{name}" \\\n')
        f.write(f'       --account=p73142 \\\n')
        f.write(f'       --partition=zen3_0512_a100x2 \\\n')
        f.write(f'       --qos=zen3_0512_a100x2 \\\n')
        f.write(f'       --gres=gpu:1 \\\n')
        f.write(f'       --ntasks=1 \\\n')
        f.write(f'       --time=04:00:00 \\\n')
        f.write(f'       --output="results/grid_search/{name}_%j.log" \\\n')
        f.write(f'       --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \\$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\\$DATA/pate-dynamic-synthesis:\\$PYTHONPATH && python -m src.experiments.run_experiment --config {fname} --dataset MNIST --beta {dbeta} --seed 0"\n')
        f.write(f'SUBMITTED=$((SUBMITTED+1))\n')
        f.write(f'echo "[$SUBMITTED/{len(configs)}] {name}"\n\n')

    f.write(f'\necho "\\n=== TOTAL: $SUBMITTED jobs submitted ==="\n')

print(f"\nGenerated {len(configs)} configs in {OUTPUT_DIR}/")
print(f"Generated run_grid_search.sh")

va_count = len(sigmas)*len(alphas)*len(samples_per_round)*len(dirichlet_betas)
vb_count = len(sigmas)*len(beta_focuses)*len(samples_per_round)*len(dirichlet_betas)
static_count = len(sigmas)*len(samples_per_round)*len(dirichlet_betas)
print(f"  Variant A: {va_count} jobs")
print(f"  Variant B: {vb_count} jobs")
print(f"  Static:    {static_count} jobs")
print(f"  TOTAL:     {len(configs)} jobs")
