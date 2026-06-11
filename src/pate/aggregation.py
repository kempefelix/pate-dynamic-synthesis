"""
PATE Aggregation Mechanism.

Core logic adapted from NEWSROOM/Saferlearn framework (orchestrator.py,
method pate_aggregate). Extended with Gaussian noise support for
RDP-compatible privacy accounting.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def pate_aggregate_votes(
    teacher_predictions: np.ndarray,
    num_classes: int,
    sigma: float = 0.0,
) -> Tuple[int, np.ndarray]:
    """Aggregate teacher votes for a single sample with optional DP noise.

    Adapted from Saferlearn's orchestrator.pate_aggregate().

    Args:
        teacher_predictions: Array of shape (num_teachers,) with class predictions.
        num_classes: Total number of classes.
        sigma: Gaussian noise standard deviation. 0 = no noise.

    Returns:
        Tuple of (winning_class, vote_counts). winning_class is the argmax
        of the (noisy, if sigma > 0) vote vector. vote_counts is the vote
        vector on which the argmax was taken, i.e. it INCLUDES the Gaussian
        noise when sigma > 0. Callers that need the raw (pre-noise) counts
        must recompute them from teacher_predictions (as pate_label_dataset
        does for its vote_matrix/consensus outputs).
    """
    # Count votes per class
    vote_counts = np.zeros(num_classes, dtype=np.float64)
    for vote in teacher_predictions:
        vote_counts[int(vote)] += 1

    # Add Gaussian noise for DP (if sigma > 0)
    if sigma > 0:
        noise = np.random.normal(0, sigma, num_classes)
        vote_counts = vote_counts + noise

    winning_class = int(np.argmax(vote_counts))
    return winning_class, vote_counts


def pate_label_dataset(
    teacher_models: List[torch.nn.Module],
    data: torch.Tensor,
    num_classes: int,
    sigma: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label a dataset using the PATE mechanism.

    Each teacher independently classifies all samples, then votes are
    aggregated with noisy argmax.

    Args:
        teacher_models: List of T trained teacher models.
        data: Unlabeled data tensor of shape (N, C, H, W).
        num_classes: Number of classes.
        sigma: Gaussian noise std for DP.
        device: Torch device.

    Returns:
        Tuple of:
            - labels: Aggregated noisy labels, shape (N,)
            - vote_counts: Raw vote counts, shape (N, num_classes)
            - consensus: Consensus score per sample (max_votes / num_teachers)
    """
    num_samples = data.shape[0]
    num_teachers = len(teacher_models)

    # Collect all teacher predictions: shape (num_teachers, num_samples)
    all_predictions = np.zeros((num_teachers, num_samples), dtype=np.int64)

    for t_idx, teacher in enumerate(teacher_models):
        teacher.eval()
        teacher.to(device)
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = 256
            preds = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch = data[start:end].to(device)
                outputs = teacher(batch)
                _, predicted = torch.max(outputs, 1)
                preds.append(predicted.cpu().numpy())
            all_predictions[t_idx] = np.concatenate(preds)

    # Aggregate votes per sample
    labels = np.zeros(num_samples, dtype=np.int64)
    vote_matrix = np.zeros((num_samples, num_classes), dtype=np.float64)
    consensus = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        teacher_votes = all_predictions[:, i]
        winning_class, noisy_votes = pate_aggregate_votes(
            teacher_votes, num_classes, sigma
        )
        labels[i] = winning_class

        # Store raw (pre-noise) vote counts for consensus measurement
        raw_votes = np.zeros(num_classes)
        for v in teacher_votes:
            raw_votes[int(v)] += 1
        vote_matrix[i] = raw_votes
        consensus[i] = np.max(raw_votes) / num_teachers

    return labels, vote_matrix, consensus
