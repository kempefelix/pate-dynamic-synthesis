#!/bin/bash
# =============================================================================
# Grid Search: PATE Dynamic Synthesis – Felix Kempe, TU Wien
# 57 jobs total: 27 Variant A + 27 Variant B + 3 Static
# All at beta_noniid=0.1 (strongest Non-IID), seed=0, MNIST
# =============================================================================

WORKDIR=$DATA/pate-dynamic-synthesis
cd $WORKDIR
mkdir -p results/grid_search configs
TOTAL=0

echo "=========================================="
echo "GRID SEARCH: Variant A (27 jobs)"
echo "sigma x alpha x min_class_ratio"
echo "=========================================="

for sigma in 10 20 40; do
for alpha in 1.0 2.0 4.0; do
for mcr in 0.1 0.3 0.5; do
    JN="gs_A_s${sigma}_a${alpha}_m${mcr}"
    cat > configs/grid_${JN}.yaml << YEOF
seed: 42
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [0.1]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: ${sigma}.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis:
  num_rounds: 10
  samples_per_round: 5000
  variant_a: {alpha: ${alpha}, min_class_ratio: ${mcr}}
  variant_b: {beta: 2.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["variant_a"]
logging: {save_dir: "./results/grid_search", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/grid_search/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/grid_${JN}.yaml --dataset MNIST --beta 0.1 --seed 0"
    echo "Submitted: ${JN}"
    TOTAL=$((TOTAL + 1))
done
done
done

echo ""
echo "=========================================="
echo "GRID SEARCH: Variant B (27 jobs)"
echo "sigma x beta_focus x lambda_contrast"
echo "=========================================="

for sigma in 10 20 40; do
for bf in 1.0 2.0 4.0; do
for lc in 0.1 0.3 0.5; do
    JN="gs_B_s${sigma}_b${bf}_l${lc}"
    cat > configs/grid_${JN}.yaml << YEOF
seed: 42
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [0.1]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: ${sigma}.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis:
  num_rounds: 10
  samples_per_round: 5000
  variant_a: {alpha: 2.0, min_class_ratio: 0.3}
  variant_b: {beta: ${bf}, lambda_contrast: ${lc}, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["variant_b"]
logging: {save_dir: "./results/grid_search", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/grid_search/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/grid_${JN}.yaml --dataset MNIST --beta 0.1 --seed 0"
    echo "Submitted: ${JN}"
    TOTAL=$((TOTAL + 1))
done
done
done

echo ""
echo "=========================================="
echo "GRID SEARCH: Static Baseline (3 jobs)"
echo "=========================================="

for sigma in 10 20 40; do
    JN="gs_static_s${sigma}"
    cat > configs/grid_${JN}.yaml << YEOF
seed: 42
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [0.1]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: ${sigma}.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis:
  num_rounds: 10
  samples_per_round: 5000
  variant_a: {alpha: 2.0, min_class_ratio: 0.3}
  variant_b: {beta: 2.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["static"]
logging: {save_dir: "./results/grid_search", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/grid_search/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/grid_${JN}.yaml --dataset MNIST --beta 0.1 --seed 0"
    echo "Submitted: ${JN}"
    TOTAL=$((TOTAL + 1))
done

echo ""
echo "=========================================="
echo "TOTAL: ${TOTAL} jobs submitted"
echo "=========================================="
