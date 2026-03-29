"""
Variant B: Error-Aware Synthesis.

Targets the fine-grained error structure of the student model.
Uses the normalized confusion matrix as feedback signal and generates
contrast pairs for systematically confused class pairs.

Implements Algorithm 3 from Chapter 5 of the thesis.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.special import softmax


def extract_feedback_variant_b(
    student: torch.nn.Module,
    synthetic_data: torch.Tensor,
    synthetic_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the normalized confusion matrix (Φ_B) from the student model.

    Corresponds to Equations 5.5 and 5.6 in the thesis:
        C_{k,k'} = #{true=k, pred=k'} / #{true=k}
        e_k = 1 - C_{k,k}

    Args:
        student: Current student model S^(t-1).
        synthetic_data: Synthetic images from previous round.
        synthetic_labels: PATE-assigned labels (treated as ground truth).
        num_classes: Total number of classes K.
        device: Torch device.

    Returns:
        Tuple of:
            - confusion_matrix: Normalized C of shape (K, K)
            - error_scores: e vector of shape (K,)
    """
    student.eval()

    # Get student predictions
    all_preds = []
    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(synthetic_data), batch_size):
            end = min(start + batch_size, len(synthetic_data))
            batch = synthetic_data[start:end].to(device)
            outputs = student(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
    all_preds = np.concatenate(all_preds)

    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for true_label, pred_label in zip(synthetic_labels, all_preds):
        confusion[int(true_label), int(pred_label)] += 1

    # Normalize row-wise
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    confusion_normalized = confusion / row_sums

    # Error scores: e_k = 1 - C_{k,k}
    error_scores = 1.0 - np.diag(confusion_normalized)

    return confusion_normalized, error_scores


def build_query_variant_b(
    confusion_matrix: np.ndarray,
    error_scores: np.ndarray,
    num_classes: int,
    n_base: int,
    n_contrast: int,
    client_classes: Dict[int, List[int]],
    beta: float = 2.0,
    lambda_contrast: float = 0.3,
    top_m: int = 3,
) -> Dict[int, Dict[int, int]]:
    """Build synthesis queries with error-proportional weighting + contrast pairs.

    Corresponds to Equations 5.7, 5.8, 5.9 in the thesis.

    Step 1: Error-proportional class weighting
        w_k = softmax(e * β)

    Step 2: Confusion-directed contrast pairs
        C_k = top-m confusion partners for each class k

    Args:
        confusion_matrix: Normalized C of shape (K, K).
        error_scores: Error vector e of shape (K,).
        num_classes: Total number of classes K.
        n_base: Base synthesis budget N_base.
        n_contrast: Contrast pair budget N_contrast.
        client_classes: Dict mapping client_id -> available classes.
        beta: Focusing exponent (β ≥ 1).
        lambda_contrast: Contrast pair scaling factor λ.
        top_m: Number of top confusion partners m.

    Returns:
        Query dict: {client_id: {class_label: num_samples_to_generate}}
    """
    # Step 1: Error-proportional weights via softmax
    weights = softmax(error_scores * beta)

    # Step 2: Identify top-m confusion partners per class
    confusion_partners = {}
    for k in range(num_classes):
        # Get off-diagonal confusion values for class k
        off_diag = confusion_matrix[k].copy()
        off_diag[k] = 0  # exclude diagonal
        # Top-m indices
        if off_diag.sum() > 0:
            top_indices = np.argsort(off_diag)[::-1][:top_m]
            confusion_partners[k] = [
                (int(idx), off_diag[idx])
                for idx in top_indices
                if off_diag[idx] > 0
            ]
        else:
            confusion_partners[k] = []

    # Build per-class sample counts
    # Base: error-proportional
    base_per_class = np.floor(weights * n_base).astype(int)

    # Contrast: additional samples for confused class pairs
    contrast_per_class = np.zeros(num_classes, dtype=int)
    for k in range(num_classes):
        for k_prime, c_val in confusion_partners[k]:
            additional = int(np.floor(lambda_contrast * c_val * n_contrast))
            contrast_per_class[k] += additional
            contrast_per_class[k_prime] += additional  # both sides of the pair

    total_per_class = base_per_class + contrast_per_class

    # Find which clients can generate each class
    class_to_clients = {}
    for k in range(num_classes):
        class_to_clients[k] = [
            cid for cid, classes in client_classes.items() if k in classes
        ]

    # Build per-client queries
    query = {cid: {} for cid in client_classes.keys()}
    for k in range(num_classes):
        clients_for_k = class_to_clients[k]
        if len(clients_for_k) == 0 or total_per_class[k] == 0:
            continue
        per_client = total_per_class[k] // len(clients_for_k)
        if per_client > 0:
            for cid in clients_for_k:
                query[cid][k] = per_client

    return query
