"""
Main Experiment Runner.

Implements the full FL-PATE training loop with dynamic synthesis
(Algorithm 1 from Chapter 5). Supports all five experimental
configurations from Table 4.4:
  - Upper Bound (no privacy, all data)
  - Classical PATE Reference (real public dataset)
  - Static Synthesis Baseline
  - Variant A (Distribution-Aware)
  - Variant B (Error-Aware)
"""

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import yaml

from src.models.teacher_cnn import TeacherCNN
from src.models.cgan import Generator, Discriminator
from src.data.partition import dirichlet_partition, get_client_classes, load_dataset
from src.pate.aggregation import pate_label_dataset
from src.privacy.rdp import RDPAccountant
from src.synthesis.generators.local_cgan import LocalCGAN
from src.synthesis.coordinator import SynthesisCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiment")


# =========================================================================
# Helper Functions
# =========================================================================

def get_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    optimizer_name: str = "adam",
    momentum: float = 0.9,
) -> List[float]:
    """Train a model with Adam or SGD + NLLLoss."""
    model.train()
    model.to(device)
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.NLLLoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if num_batches > 0:
            losses.append(epoch_loss / num_batches)

    return losses


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, float]:
    """Evaluate accuracy, per-class accuracy, and macro F1."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    # For confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                confusion[t, p] += 1

    accuracy = correct / total if total > 0 else 0.0

    # Per-class precision, recall, F1
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)

    for k in range(num_classes):
        tp = confusion[k, k]
        fp = confusion[:, k].sum() - tp
        fn = confusion[k, :].sum() - tp

        precision_per_class[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per_class[k] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision_per_class[k] + recall_per_class[k] > 0:
            f1_per_class[k] = (
                2 * precision_per_class[k] * recall_per_class[k]
                / (precision_per_class[k] + recall_per_class[k])
            )

    macro_f1 = f1_per_class.mean()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_accuracy": (np.diag(confusion) / confusion.sum(axis=1)).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


# =========================================================================
# Phase 1: Train Teachers
# =========================================================================

def train_teachers(
    client_datasets: List[Subset],
    num_channels: int,
    num_classes: int,
    config: dict,
    device: torch.device,
) -> List[nn.Module]:
    """Train one teacher model per client on its private data partition."""
    teachers = []
    for i, dataset in enumerate(client_datasets):
        logger.info(f"Training Teacher {i+1}/{len(client_datasets)} "
                     f"({len(dataset)} samples)")
        model = TeacherCNN(num_channels=num_channels, num_classes=num_classes)
        dataloader = DataLoader(dataset, batch_size=config["teacher"]["batch_size"],
                                shuffle=True, num_workers=0)
        train_model(
            model, dataloader,
            epochs=config["teacher"]["epochs"],
            lr=float(config["teacher"]["learning_rate"]),
            device=device,
            optimizer_name=config["teacher"].get("optimizer", "adam"),
        )
        model.eval()
        model.cpu()
        teachers.append(model)
    return teachers


# =========================================================================
# Phase 2: Train Local cGANs
# =========================================================================

def train_local_cgans(
    client_datasets: List[Subset],
    client_class_map: Dict[int, List[int]],
    num_classes: int,
    num_channels: int,
    image_size: int,
    config: dict,
    device: torch.device,
) -> Dict[int, LocalCGAN]:
    """Train one local cGAN per client."""
    cgans = {}
    cgan_cfg = config["cgan"]

    for client_id, dataset in enumerate(client_datasets):
        available_classes = client_class_map[client_id]
        min_samples = cgan_cfg.get("min_samples_to_train", 200)
        if len(available_classes) == 0 or len(dataset) < min_samples:
            logger.warning(f"Client {client_id}: skipping cGAN training "
                           f"({len(dataset)} samples < {min_samples} minimum)")
            continue

        logger.info(f"Training cGAN for Client {client_id} "
                     f"(classes: {available_classes}, {len(dataset)} samples)")

        local_cgan = LocalCGAN(
            client_id=client_id,
            available_classes=available_classes,
            num_classes=num_classes,
            latent_dim=cgan_cfg["latent_dim"],
            num_channels=num_channels,
            image_size=image_size,
            feature_maps=cgan_cfg["feature_maps_g"],
            device=device,
        )
        local_cgan.train(
            dataset=dataset,
            num_epochs=cgan_cfg["num_epochs"],
            batch_size=cgan_cfg["batch_size"],
            lr_g=cgan_cfg["lr_g"],
            lr_d=cgan_cfg["lr_d"],
            beta1=cgan_cfg["beta1"],
        )
        # Move generator to CPU to save GPU memory
        local_cgan.generator.cpu()
        local_cgan.device = torch.device("cpu")
        cgans[client_id] = local_cgan

    return cgans


# =========================================================================
# Phase 3: Dynamic Synthesis Loop (Algorithm 1)
# =========================================================================

def run_synthesis_loop(
    strategy: str,
    teachers: List[nn.Module],
    coordinator: SynthesisCoordinator,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    num_channels: int,
    num_classes: int,
) -> Dict[str, list]:
    """Run the main FL-PATE training loop with dynamic synthesis.

    This is the implementation of Algorithm 1 from Chapter 5.

    Returns:
        Dictionary with per-round metrics.
    """
    synth_cfg = config["synthesis"]
    pate_cfg = config["pate"]

    # Initialize student
    student = TeacherCNN(num_channels=num_channels, num_classes=num_classes)
    student.to(device)

    # Initialize RDP accountant
    accountant = RDPAccountant(sigma=float(pate_cfg["sigma"]), delta=float(pate_cfg["delta"]))

    # Metrics tracking
    metrics = {
        "round": [],
        "accuracy": [],
        "macro_f1": [],
        "epsilon": [],
        "teacher_queries": [],
        "per_class_accuracy": [],
        "synthesis_distribution": [],
    }

    prev_synthetic_data = None
    prev_synthetic_labels = None
    prev_accuracy = 0.0

    for round_num in range(synth_cfg["num_rounds"]):
        round_start = time.time()

        # Check budget
        current_eps = accountant.get_epsilon()
        if current_eps >= pate_cfg["epsilon_max"]:
            logger.warning(
                f"Round {round_num}: Privacy budget exhausted "
                f"(ε={current_eps:.2f} ≥ {pate_cfg['epsilon_max']}). Stopping."
            )
            break

        # ----- Step 1-4: Generate synthetic dataset -----
        synthetic_data, requested_labels = coordinator.generate_synthetic_dataset(
            strategy=strategy,
            student=student if round_num > 0 else None,
            prev_synthetic_data=prev_synthetic_data,
            prev_synthetic_labels=prev_synthetic_labels,
            num_samples=synth_cfg["samples_per_round"],
            round_num=round_num,
            alpha=synth_cfg.get("variant_a", {}).get("alpha", 2.0),
            min_class_ratio=synth_cfg.get("variant_a", {}).get("min_class_ratio", 0.30),
            beta=synth_cfg.get("variant_b", {}).get("beta", 2.0),
            lambda_contrast=synth_cfg.get("variant_b", {}).get("lambda_contrast", 0.3),
            top_m=synth_cfg.get("variant_b", {}).get("top_m_confusions", 3),
            n_base_ratio=synth_cfg.get("variant_b", {}).get("n_base_ratio", 0.7),
            n_contrast_ratio=synth_cfg.get("variant_b", {}).get("n_contrast_ratio", 0.3),
        )

        # ----- Step 5: PATE Labeling -----
        pate_labels, vote_matrix, consensus = pate_label_dataset(
            teacher_models=teachers,
            data=synthetic_data,
            num_classes=num_classes,
            sigma=float(pate_cfg["sigma"]),
            device=device,
        )

        # RDP accounting for this batch
        eps_after = accountant.step_batch(vote_matrix)
        logger.info(
            f"Round {round_num}: ε = {eps_after:.4f} | "
            f"Queries so far: {accountant.get_num_queries()} | "
            f"Mean consensus: {consensus.mean():.3f}"
        )

        # ----- Step 6: Train Student -----
        labeled_dataset = TensorDataset(
            synthetic_data,
            torch.tensor(pate_labels, dtype=torch.long),
        )
        student_loader = DataLoader(
            labeled_dataset,
            batch_size=config["student"]["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        train_model(
            student, student_loader,
            epochs=config["student"]["epochs"],
            lr=float(config["student"]["learning_rate"]),
            device=device,
            optimizer_name=config["student"].get("optimizer", "adam"),
        )

        # Evaluate
        eval_results = evaluate_model(student, test_loader, device, num_classes)
        round_time = time.time() - round_start

        # Log metrics
        metrics["round"].append(round_num)
        metrics["accuracy"].append(eval_results["accuracy"])
        metrics["macro_f1"].append(eval_results["macro_f1"])
        metrics["epsilon"].append(eps_after)
        metrics["teacher_queries"].append(accountant.get_num_queries())
        metrics["per_class_accuracy"].append(eval_results["per_class_accuracy"])
        metrics["synthesis_distribution"].append(
            np.bincount(requested_labels, minlength=num_classes).tolist()
        )

        logger.info(
            f"Round {round_num}: Acc={eval_results['accuracy']:.4f} | "
            f"F1={eval_results['macro_f1']:.4f} | "
            f"ε={eps_after:.4f} | Time={round_time:.1f}s"
        )

        # Store for next round's feedback
        prev_synthetic_data = synthetic_data.clone()
        prev_synthetic_labels = pate_labels.copy()

        # Check convergence
        if round_num > 0:
            delta_acc = abs(eval_results["accuracy"] - prev_accuracy)
            if delta_acc < config["evaluation"]["convergence_threshold"]:
                logger.info(
                    f"Round {round_num}: Converged "
                    f"(Δacc={delta_acc:.5f} < {config['evaluation']['convergence_threshold']})"
                )
        prev_accuracy = eval_results["accuracy"]

    return metrics


# =========================================================================
# Upper Bound Baseline
# =========================================================================

def run_upper_bound(
    train_dataset,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    num_channels: int,
    num_classes: int,
) -> Dict[str, float]:
    """Train on all data without privacy constraints (theoretical upper bound)."""
    logger.info("=" * 60)
    logger.info("Running UPPER BOUND (no privacy, full data access)")
    logger.info("=" * 60)

    model = TeacherCNN(num_channels=num_channels, num_classes=num_classes)
    loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    train_model(model, loader, epochs=15, lr=0.001, device=device, optimizer_name="adam")
    results = evaluate_model(model, test_loader, device, num_classes)
    logger.info(f"Upper Bound: Acc={results['accuracy']:.4f} | F1={results['macro_f1']:.4f}")
    return results


# =========================================================================
# Classical PATE Reference
# =========================================================================

def run_classical_pate(
    teachers: List[nn.Module],
    test_dataset,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    num_channels: int,
    num_classes: int,
) -> Dict[str, float]:
    """Classical PATE with real test data as public dataset (reference)."""
    logger.info("=" * 60)
    logger.info("Running CLASSICAL PATE REFERENCE (real public data)")
    logger.info("=" * 60)

    pate_cfg = config["pate"]

    # Use a subset of test data as "public" dataset for querying
    num_public = min(5000, len(test_dataset))
    indices = np.random.choice(len(test_dataset), num_public, replace=False)
    public_data = torch.stack([test_dataset[i][0] for i in indices])

    # PATE labeling
    pate_labels, vote_matrix, consensus = pate_label_dataset(
        teacher_models=teachers,
        data=public_data,
        num_classes=num_classes,
        sigma=float(pate_cfg["sigma"]),
        device=device,
    )

    # RDP accounting
    accountant = RDPAccountant(sigma=float(pate_cfg["sigma"]), delta=float(pate_cfg["delta"]))
    eps = accountant.step_batch(vote_matrix)

    # Train student
    student = TeacherCNN(num_channels=num_channels, num_classes=num_classes)
    labeled_dataset = TensorDataset(
        public_data, torch.tensor(pate_labels, dtype=torch.long)
    )
    student_loader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)
    train_model(student, student_loader, epochs=10, lr=0.001, device=device, optimizer_name="adam")

    results = evaluate_model(student, test_loader, device, num_classes)
    results["epsilon"] = eps
    results["teacher_queries"] = num_public
    logger.info(
        f"Classical PATE: Acc={results['accuracy']:.4f} | "
        f"F1={results['macro_f1']:.4f} | ε={eps:.4f}"
    )
    return results


# =========================================================================
# Main Entry Point
# =========================================================================

def run_single_experiment(
    config: dict,
    dataset_name: str,
    beta: float,
    seed: int,
    strategies: List[str],
) -> Dict[str, any]:
    """Run a complete experiment for one (dataset, beta, seed) configuration."""
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device(config.get("device", "auto"))
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {dataset_name} | β={beta} | seed={seed} | device={device}")
    logger.info(f"{'='*60}")

    # Dataset parameters
    if dataset_name.upper() == "MNIST":
        num_channels, image_size = 1, 28
    else:
        num_channels, image_size = 3, 32
    num_classes = config["dataset"]["num_classes"]
    num_teachers = config["federation"]["num_teachers"]

    # Load data
    train_dataset = load_dataset(dataset_name, config["dataset"]["data_dir"], train=True)
    test_dataset = load_dataset(dataset_name, config["dataset"]["data_dir"], train=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Non-IID partition
    logger.info(f"Partitioning into {num_teachers} clients with Dir(β={beta})")
    client_datasets, client_class_counts = dirichlet_partition(
        train_dataset, num_teachers, beta, num_classes, seed
    )
    client_class_map = get_client_classes(client_class_counts)

    # Log partition statistics
    for cid in range(min(5, num_teachers)):
        logger.info(f"  Client {cid}: classes={client_class_map[cid]}, "
                     f"counts={client_class_counts[cid]}")

    all_results = {"config": {"dataset": dataset_name, "beta": beta, "seed": seed}}

    # Upper Bound
    if config["baselines"]["run_upper_bound"]:
        all_results["upper_bound"] = run_upper_bound(
            train_dataset, test_loader, config, device, num_channels, num_classes
        )

    # Train teachers (shared across all strategies)
    logger.info("=" * 60)
    logger.info("PHASE 1: Training Teachers")
    logger.info("=" * 60)
    teachers = train_teachers(
        client_datasets, num_channels, num_classes, config, device
    )

    # Classical PATE Reference
    if config["baselines"]["run_classical_pate"]:
        all_results["classical_pate"] = run_classical_pate(
            teachers, test_dataset, test_loader, config, device,
            num_channels, num_classes,
        )

    # Train local cGANs (shared across synthesis strategies)
    logger.info("=" * 60)
    logger.info("PHASE 2: Training Local cGANs")
    logger.info("=" * 60)
    local_cgans = train_local_cgans(
        client_datasets, client_class_map, num_classes,
        num_channels, image_size, config, device,
    )

    # Create synthesis coordinator
    coordinator = SynthesisCoordinator(
        local_cgans=local_cgans,
        client_classes=client_class_map,
        num_classes=num_classes,
        device=device,
    )

    # Run each strategy
    for strategy in strategies:
        logger.info("=" * 60)
        logger.info(f"PHASE 3: Running strategy '{strategy}'")
        logger.info("=" * 60)

        metrics = run_synthesis_loop(
            strategy=strategy,
            teachers=teachers,
            coordinator=coordinator,
            test_loader=test_loader,
            config=config,
            device=device,
            num_channels=num_channels,
            num_classes=num_classes,
        )
        all_results[strategy] = metrics

    return all_results


def main():
    parser = argparse.ArgumentParser(description="PATE Dynamic Synthesis Experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration YAML")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset (MNIST or CIFAR10)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override Dirichlet beta")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy (static/variant_a/variant_b)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Determine experiment parameters
    datasets = [args.dataset] if args.dataset else [config["dataset"]["name"]]
    betas = [args.beta] if args.beta else config["federation"]["dirichlet_beta"]
    seeds = [args.seed] if args.seed else list(range(config["num_seeds"]))
    strategies = [args.strategy] if args.strategy else config.get("strategies", ["static", "variant_a", "variant_b"])

    # Output directory
    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run all combinations
    all_experiments = []
    for dataset_name in datasets:
        for beta in betas:
            for seed in seeds:
                results = run_single_experiment(
                    config, dataset_name, beta, seed, strategies
                )
                all_experiments.append(results)

                # Save intermediate results
                # Build unique filename per run
                sigma = config.get("pate", {}).get("sigma", "")
                strat_str = "_".join(strategies)
                va = config.get("synthesis", {}).get("variant_a", {})
                vb = config.get("synthesis", {}).get("variant_b", {})
                if "variant_a" in strategies and len(strategies) == 1:
                    param_str = f"_varA_s{sigma}_a{va.get('alpha','')}_m{va.get('min_class_ratio','')}"
                elif "variant_b" in strategies and len(strategies) == 1:
                    param_str = f"_varB_s{sigma}_b{vb.get('beta','')}_l{vb.get('lambda_contrast','')}"
                elif "static" in strategies and len(strategies) == 1:
                    param_str = f"_static_s{sigma}"
                else:
                    param_str = f"_s{sigma}_{strat_str}"
                result_file = save_dir / f"{dataset_name}_beta{beta}_seed{seed}{param_str}.json"
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {result_file}")

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
