"""
Variant A: Distribution-Aware Synthesis (Fixed).

Addresses the global class distribution gap caused by Non-IID data.
Uses the student model's class confidence map as feedback signal
and applies inverse confidence weighting to oversample underrepresented classes.

FIX: Added smoothing via min_class_ratio to prevent catastrophic forgetting.
The synthesis distribution is now a blend of uniform and adaptive:
    w_k = (1 - min_class_ratio) * adaptive_w_k + min_class_ratio * (1/K)
This ensures every class receives at least some samples each round.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


def extract_feedback_variant_a(
    student: torch.nn.Module,
    synthetic_data: torch.Tensor,
    synthetic_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Extract the class confidence map (Phi_A) from the student model.

    For each class k, computes the mean maximum prediction probability
    of the student on synthetic samples of that class.

    Args:
        student: Current student model S^(t-1).
        synthetic_data: Synthetic images from previous round.
        synthetic_labels: PATE-assigned labels.
        num_classes: Total number of classes K.
        device: Torch device.

    Returns:
        Confidence map Phi_A of shape (K,), values in [0, 1].
    """
    student.eval()
    confidence_map = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        batch_size = 256
        all_max_probs = []

        for start in range(0, len(synthetic_data), batch_size):
            end = min(start + batch_size, len(synthetic_data))
            batch = synthetic_data[start:end].to(device)
            logits = student(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            all_max_probs.append(max_probs.cpu().numpy())

        all_max_probs = np.concatenate(all_max_probs)

    for k in range(num_classes):
        mask = synthetic_labels == k
        if mask.sum() > 0:
            confidence_map[k] = all_max_probs[mask].mean()
        else:
            confidence_map[k] = 0.0

    return confidence_map


def build_query_variant_a(
    confidence_map: np.ndarray,
    num_classes: int,
    num_samples: int,
    client_classes: Dict[int, List[int]],
    alpha: float = 2.0,
    min_class_ratio: float = 0.30,
) -> Dict[int, Dict[int, int]]:
    """Build synthesis queries using smoothed inverse confidence weighting.

    The synthesis distribution blends adaptive and uniform components:
        w_k = (1 - min_class_ratio) * adaptive_w_k + min_class_ratio * (1/K)

    This prevents catastrophic forgetting: even well-learned classes
    receive at least 30% of their uniform share each round.

    Args:
        confidence_map: Phi_A of shape (K,).
        num_classes: Total number of classes K.
        num_samples: Total synthesis budget N_syn.
        client_classes: Dict mapping client_id -> available classes.
        alpha: Focusing exponent.
        min_class_ratio: Blend ratio for uniform floor (0.0 = fully adaptive, 1.0 = fully uniform).

    Returns:
        Query dict: {client_id: {class_label: num_samples_to_generate}}
    """
    # Compute inverse confidence weights (adaptive component)
    inv_confidence = (1.0 - confidence_map) ** alpha
    weight_sum = inv_confidence.sum()
    if weight_sum > 0:
        adaptive_weights = inv_confidence / weight_sum
    else:
        adaptive_weights = np.ones(num_classes) / num_classes

    # Uniform component
    uniform_weights = np.ones(num_classes) / num_classes

    # Blend: smooth transition between adaptive and uniform
    weights = (1.0 - min_class_ratio) * adaptive_weights + min_class_ratio * uniform_weights
    weights = weights / weights.sum()  # Re-normalize

    # Compute per-class sample counts
    samples_per_class = np.floor(weights * num_samples).astype(int)

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
        if len(clients_for_k) == 0 or samples_per_class[k] == 0:
            continue
        per_client = samples_per_class[k] // len(clients_for_k)
        if per_client > 0:
            for cid in clients_for_k:
                query[cid][k] = per_client

    return query
