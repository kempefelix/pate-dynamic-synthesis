#!/bin/bash
WORKDIR=$DATA/pate-dynamic-synthesis
cd $WORKDIR
mkdir -p results/main configs
TOTAL=0

for beta in 0.1 0.5 1.0; do
for seed in 0 1 2 3 4; do

# --- Baselines ---
JN="main_baseline_b${beta}_s${seed}"
cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [${beta}]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: 40.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis: {num_rounds: 10, samples_per_round: 5000, variant_a: {alpha: 2.0, min_class_ratio: 0.3}, variant_b: {beta: 1.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: true, run_classical_pate: true}
strategies: []
logging: {save_dir: "./results/main", save_checkpoints: false, log_interval: 1}
YEOF
sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=02:00:00 --output="results/main/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset MNIST --beta ${beta} --seed ${seed}"
echo "Submitted: ${JN}"
TOTAL=$((TOTAL + 1))

# --- Static ---
JN="main_static_b${beta}_s${seed}"
cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
federation: {num_teachers: 250, dirichlet_beta: [${beta}]}
teacher: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
student: {architecture: "TeacherCNN", epochs: 10, batch_size: 128, learning_rate: 0.001, optimizer: "adam"}
pate: {mechanism: "gaussian", sigma: 40.0, delta: 1.0e-5, epsilon_max: 100.0}
cgan: {latent_dim: 100, num_epochs: 200, batch_size: 64, lr_g: 0.0002, lr_d: 0.0002, beta1: 0.5, feature_maps_g: 64, feature_maps_d: 64, min_samples_to_train: 100}
synthesis: {num_rounds: 10, samples_per_round: 5000, variant_a: {alpha: 2.0, min_class_ratio: 0.3}, variant_b: {beta: 1.0, lambda_contrast: 0.3, top_m_confusions: 3, n_base_ratio: 0.7, n_contrast_ratio: 0.3}}
evaluation: {target_accuracy: 0.95, convergence_threshold: 0.005}
baselines: {run_upper_bound: false, run_classical_pate: false}
strategies: ["static"]
logging: {save_dir: "./results/main", save_checkpoints: false, log_interval: 1}
YEOF
sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/main/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset MNIST --beta ${beta} --seed ${seed}"
echo "Submitted: ${JN}"
TOTAL=$((TOTAL + 1))

# --- Variant A: 3 configs ---
for cfg in "A1 1.0 0.5" "A2 2.0 0.3" "A3 4.0 0.1"; do
    set -- $cfg; cid=$1; alpha=$2; mcr=$3
    JN="main_${cid}_b${beta}_s${seed}"
    cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
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
logging: {save_dir: "./results/main", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/main/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset MNIST --beta ${beta} --seed ${seed}"
    echo "Submitted: ${JN} (α=${alpha}, mcr=${mcr})"
    TOTAL=$((TOTAL + 1))
done

# --- Variant B: 3 configs ---
for cfg in "B1 1.0 0.1" "B2 1.0 0.3" "B3 2.0 0.5"; do
    set -- $cfg; cid=$1; bf=$2; lc=$3
    JN="main_${cid}_b${beta}_s${seed}"
    cat > configs/${JN}.yaml << YEOF
seed: ${seed}
device: "auto"
num_seeds: 1
dataset: {name: "MNIST", num_classes: 10, data_dir: "./data"}
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
logging: {save_dir: "./results/main", save_checkpoints: false, log_interval: 1}
YEOF
    sbatch --job-name="${JN}" --account=p73142 --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 --gres=gpu:1 --ntasks=1 --time=04:00:00 --output="results/main/${JN}_%j.log" --wrap="module load python/3.11.0-gcc-12.2.0-4cmli4d && module load cuda/11.8.0-gcc-12.2.0-bplw5nu && cd \$DATA/pate-dynamic-synthesis && source .venv/bin/activate && export PYTHONPATH=\$DATA/pate-dynamic-synthesis:\$PYTHONPATH && python -m src.experiments.run_experiment --config configs/${JN}.yaml --dataset MNIST --beta ${beta} --seed ${seed}"
    echo "Submitted: ${JN} (β=${bf}, λ=${lc})"
    TOTAL=$((TOTAL + 1))
done

done
done

echo "=========================================="
echo "TOTAL: ${TOTAL} jobs submitted"
echo "=========================================="
