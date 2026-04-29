"""
Synthesis Coordinator.

Central component that manages the dynamic synthesis loop:
1. Extract feedback from student model
2. Build synthesis queries based on strategy (A or B)
3. Dispatch queries to local cGANs
4. Aggregate synthetic samples into D_syn^(t)

Implements Algorithm 1 from Chapter 5 of the thesis.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from src.synthesis.generators.local_cgan import LocalCGAN
from src.synthesis.strategies.static import compute_static_query
from src.synthesis.strategies.distribution_aware import (
    extract_feedback_variant_a,
    build_query_variant_a,
)
from src.synthesis.strategies.error_aware import (
    extract_feedback_variant_b,
    build_query_variant_b,
)

logger = logging.getLogger(__name__)


class SynthesisCoordinator:
    """Coordinates the synthesis loop between student, generators, and PATE.

    This is the novel component introduced by this thesis (see Section 5.1).
    """

    def __init__(
        self,
        local_cgans: Dict[int, LocalCGAN],
        client_classes: Dict[int, List[int]],
        num_classes: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        self.local_cgans = local_cgans
        self.client_classes = client_classes
        self.num_classes = num_classes
        self.device = device

    def generate_synthetic_dataset(
        self,
        strategy: str,
        student: Optional[torch.nn.Module],
        prev_synthetic_data: Optional[torch.Tensor],
        prev_synthetic_labels: Optional[np.ndarray],
        num_samples: int,
        round_num: int,
        # Variant A params
        alpha: float = 2.0,
        min_class_ratio: float = 0.30,
        # Variant B params
        beta: float = 2.0,
        lambda_contrast: float = 0.3,
        top_m: int = 3,
        n_base_ratio: float = 0.7,
        n_contrast_ratio: float = 0.3,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Generate a synthetic dataset for one round.

        Implements the core synthesis loop (Steps 1-4 of Algorithm 1).

        Args:
            strategy: "static", "variant_a", or "variant_b".
            student: Current student model (None for round 0 or static).
            prev_synthetic_data: Synthetic data from previous round.
            prev_synthetic_labels: Labels from previous round.
            num_samples: Total synthesis budget N_syn.
            round_num: Current round number t.
            alpha: Focusing exponent for Variant A.
            beta: Focusing exponent for Variant B.
            lambda_contrast: Contrast scaling for Variant B.
            top_m: Number of confusion partners for Variant B.
            n_base_ratio: Fraction of budget for base synthesis (Variant B).
            n_contrast_ratio: Fraction for contrast pairs (Variant B).

        Returns:
            Tuple of:
                - synthetic_data: Tensor of shape (N, C, H, W)
                - requested_labels: Array of shape (N,) with requested class labels
        """
        # Step 1 & 2: Extract feedback and build query
        if strategy == "static" or round_num == 0 or student is None:
            query = compute_static_query(
                self.num_classes, num_samples, self.client_classes
            )
            logger.info(f"Round {round_num}: Static synthesis query")

        elif strategy == "variant_a":
            confidence_map = extract_feedback_variant_a(
                student, prev_synthetic_data, prev_synthetic_labels,
                self.num_classes, self.device,
            )
            query = build_query_variant_a(
                confidence_map, self.num_classes, num_samples,
                self.client_classes, alpha, min_class_ratio,
            )
            logger.info(
                f"Round {round_num}: Variant A | "
                f"Confidence map: {np.round(confidence_map, 3)}"
            )

        elif strategy == "variant_b":
            n_base = int(num_samples * n_base_ratio)
            n_contrast = int(num_samples * n_contrast_ratio)
            confusion_matrix, error_scores = extract_feedback_variant_b(
                student, prev_synthetic_data, prev_synthetic_labels,
                self.num_classes, self.device,
            )
            query = build_query_variant_b(
                confusion_matrix, error_scores, self.num_classes,
                n_base, n_contrast, self.client_classes,
                beta, lambda_contrast, top_m,
            )
            logger.info(
                f"Round {round_num}: Variant B | "
                f"Error scores: {np.round(error_scores, 3)}"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Step 3: Dispatch queries to local cGANs
        all_samples = []
        all_labels = []

        for client_id, class_counts in query.items():
            if client_id not in self.local_cgans:
                continue
            cgan = self.local_cgans[client_id]
            for class_label, count in class_counts.items():
                if count <= 0:
                    continue
                try:
                    samples = cgan.generate(class_label, count)
                    all_samples.append(samples)
                    all_labels.extend([class_label] * count)
                except ValueError as e:
                    logger.warning(f"Client {client_id}: {e}")

        # Step 4: Aggregate into D_syn^(t)
        if len(all_samples) == 0:
            raise RuntimeError("No synthetic samples generated!")

        synthetic_data = torch.cat(all_samples, dim=0)
        requested_labels = np.array(all_labels, dtype=np.int64)

        logger.info(
            f"Round {round_num}: Generated {len(synthetic_data)} samples | "
            f"Class distribution: {np.bincount(requested_labels, minlength=self.num_classes)}"
        )

        return synthetic_data, requested_labels
