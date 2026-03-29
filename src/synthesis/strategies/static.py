"""
Static Synthesis Baseline.

Generates a fixed synthetic dataset once (uniform class distribution)
without any feedback from the student model. Serves as the baseline
against which Variants A and B are compared.
"""

import numpy as np
import torch
from typing import Dict, List

from src.synthesis.generators.local_cgan import LocalCGAN


def compute_static_query(
    num_classes: int,
    num_samples: int,
    client_classes: Dict[int, List[int]],
) -> Dict[int, Dict[int, int]]:
    """Compute a uniform synthesis query (no feedback).

    Distributes samples equally across all classes and all clients
    that can generate each class.

    Args:
        num_classes: Total number of classes.
        num_samples: Total samples to generate.
        client_classes: Dict mapping client_id -> list of available classes.

    Returns:
        Query dict: {client_id: {class_label: num_samples_to_generate}}
    """
    samples_per_class = num_samples // num_classes

    # Find which clients can generate each class
    class_to_clients = {}
    for k in range(num_classes):
        class_to_clients[k] = [
            cid for cid, classes in client_classes.items() if k in classes
        ]

    query = {cid: {} for cid in client_classes.keys()}
    for k in range(num_classes):
        clients_for_k = class_to_clients[k]
        if len(clients_for_k) == 0:
            continue
        per_client = samples_per_class // len(clients_for_k)
        for cid in clients_for_k:
            if per_client > 0:
                query[cid][k] = per_client

    return query
