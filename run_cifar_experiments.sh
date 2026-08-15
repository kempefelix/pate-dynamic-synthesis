#!/bin/bash
# =============================================================================
# CIFAR-10 main experiments — 120 SLURM jobs (3 β × 5 seeds × 8 job kinds)
#
#   *** RECONSTRUCTION — NOT THE ORIGINAL SUBMISSION SCRIPT ***
#
# The script that originally submitted the CIFAR-10 runs was excluded from the
# repository by .gitignore (see the `run_all_mnist.sh` / `run_gpu.sh` entries
# and the `configs/cifar_*.yaml` pattern) and is not recoverable. This file was
# reconstructed from `run_main_experiments.sh` so that the CIFAR-10 half of the
# experiment matrix can be re-executed; it is NOT evidence of how the committed
# results in `results/cifar/` were produced.
#
# What is verifiable, and what is not:
#   - VERIFIABLE: the configuration matrix (7 configurations × 3 β × 5 seeds),
#     σ = 40, δ = 1e-5, ε_max = 100, R = 10, N_syn = 5000, 250 teachers, and
#     the A1–A3 / B1–B3 hyperparameters. These are documented in Table 7.2 and
#     Section 7.6 of the thesis, and the resulting file names in
#     `results/cifar/` encode them.
#   - NOT VERIFIABLE: the exact per-run YAML. `run_experiment.py` persists only
#     `{dataset, beta, seed}` into the `config` block of each result JSON, so
#     the remaining fields below are taken from `run_main_experiments.sh` on the
#     strength of the thesis's statement that "the configuration matrix is
#     identical to that on MNIST" (Section 7.6). They are not read back from the
#     committed artifacts.
#   - NOT VERIFIABLE: the SLURM account, partition and wall-clock limits are
#     copied from the MNIST submitter.
#
# Note on the dataset switch: the teacher and cGAN architectures are selected
# from the `--dataset` COMMAND-LINE flag (run_experiment.py:492-495), not from
# the `dataset.name` YAML key. The YAML key is carried along for documentation
# only and has no effect on the channel count or the image resolution.
# =============================================================================
WORKDIR=$DATA/pate-dynamic-synthesis
cd $WORKDIR
mkdir -p results/cifar configs
TOTAL=0

for beta in 0.1 0.5 1.0; do
for seed in 0 1 2 3 4; do

# --- Baselines ---
JN="cifar_baseline_b${beta}_s${seed}"
cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "CIFAR10", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [${beta}]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: 40.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis: {num_rounds: 10, samples_per_round: 5000, variant_a: {alpha: 2.0, min_class_ratio: 0.3}, variant_b: {beta: 1.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: true, run_classical_pate: true}
strategies: []
logging: {save_dir: "./results/cifar", save_checkpoints: false, log_interval: 1}
YEOF
sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=02:00:00 --output="results/cifar/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset CIFAR10 --beta ${beta} --seed ${seed}"
echo "Submitted: ${JN}"
TOTAL=$((TOTAL + 1))

# --- Static ---
JN="cifar_static_b${beta}_s${seed}"
cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "CIFAR10", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [${beta}]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: 40.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis: {num_rounds: 10, samples_per_round: 5000, variant_a: {alpha: 2.0, min_class_ratio: 0.3}, variant_b: {beta: 1.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["static"]
logging: {save_dir: "./results/cifar", save_checkpoints: false, log_interval: 1}
YEOF
sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/cifar/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset CIFAR10 --beta ${beta} --seed ${seed}"
echo "Submitted: ${JN}"
TOTAL=$((TOTAL + 1))

# --- Variant A: 3 configs ---
for cfg in "A1 1.0 0.5" "A2 2.0 0.3" "A3 4.0 0.1"; do
    set -- $cfg; cid=$1; alpha=$2; mcr=$3
    JN="cifar_${cid}_b${beta}_s${seed}"
    cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "CIFAR10", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [${beta}]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: 40.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis:
  num_rounds: 10
  samples_per_round: 5000
  variant_a: {alpha: ${alpha}, min_class_ratio: ${mcr}}
  variant_b: {beta: 1.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["variant_a"]
logging: {save_dir: "./results/cifar", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/cifar/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset CIFAR10 --beta ${beta} --seed ${seed}"
    echo "Submitted: ${JN} (α=${alpha}, mcr=${mcr})"
    TOTAL=$((TOTAL + 1))
done

# --- Variant B: 3 configs ---
for cfg in "B1 1.0 0.1" "B2 1.0 0.3" "B3 2.0 0.5"; do
    set -- $cfg; cid=$1; bf=$2; lc=$3
    JN="cifar_${cid}_b${beta}_s${seed}"
    cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "CIFAR10", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [${beta}]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: 40.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis:
  num_rounds: 10
  samples_per_round: 5000
  variant_a: {alpha: 2.0, min_class_ratio: 0.3}
  variant_b: {beta: ${bf}, lambda_contrast: ${lc}, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["variant_b"]
logging: {save_dir: "./results/cifar", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/cifar/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset CIFAR10 --beta ${beta} --seed ${seed}"
    echo "Submitted: ${JN} (β=${bf}, λ=${lc})"
    TOTAL=$((TOTAL + 1))
done

done
done

echo "=========================================="
echo "TOTAL: ${TOTAL} jobs submitted"
echo "=========================================="
