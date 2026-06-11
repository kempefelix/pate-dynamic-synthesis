"""
Rényi Differential Privacy (RDP) Accounting.

Implements RDP budget tracking for the PATE mechanism following
Mironov (2017) and Papernot et al. (2018).
"""

import math
import numpy as np
from typing import List, Tuple


def compute_rdp_gnmax(
    vote_counts: np.ndarray,
    sigma: float,
    alpha_orders: np.ndarray,
) -> np.ndarray:
    """Compute RDP guarantee for a single GNMax query.

    This implementation uses the data-independent Gaussian-mechanism
    bound: the per-query RDP cost is constant and does NOT depend on
    the vote margin (consensus) among teachers. The data-dependent
    GNMax analysis of Papernot et al. (2018) is deliberately not used.

    Args:
        vote_counts: Raw vote counts of shape (num_classes,).
        sigma: Gaussian noise standard deviation.
        alpha_orders: Array of RDP orders to evaluate.

    Returns:
        RDP epsilon values for each alpha order.
    """
    # L2-sensitivity of the vote histogram is sqrt(2): changing one
    # record moves one teacher's vote from one class to another, i.e.
    # the histogram changes by -1 in one coordinate and +1 in another
    # (||(-1, +1)||_2 = sqrt(2); cf. Papernot et al. 2018, Prop. 8).
    sensitivity = math.sqrt(2)

    # For the Gaussian mechanism applied to the vote histogram:
    # epsilon(alpha) = alpha * sensitivity^2 / (2 * sigma^2)
    #               = alpha / sigma^2          (with sensitivity = sqrt(2))
    rdp_eps = alpha_orders * (sensitivity ** 2) / (2.0 * sigma ** 2)

    return rdp_eps


def rdp_to_dp(rdp_epsilon: float, alpha: float, delta: float) -> float:
    """Convert RDP guarantee to (epsilon, delta)-DP.

    Uses the conversion formula:
        epsilon = rdp_epsilon + log(1/delta) / (alpha - 1)

    Args:
        rdp_epsilon: RDP epsilon at order alpha.
        alpha: RDP order (> 1).
        delta: Target delta for (eps, delta)-DP.

    Returns:
        The (epsilon, delta)-DP epsilon value.
    """
    if alpha <= 1:
        raise ValueError(f"Alpha must be > 1, got {alpha}")
    return rdp_epsilon + math.log(1.0 / delta) / (alpha - 1)


class RDPAccountant:
    """Tracks cumulative RDP budget across multiple PATE queries.

    Usage:
        accountant = RDPAccountant(sigma=150.0, delta=1e-5)
        for each query:
            accountant.step(vote_counts)
        eps = accountant.get_epsilon()
    """

    def __init__(
        self,
        sigma: float,
        delta: float = 1e-5,
        alpha_orders: np.ndarray = None,
    ):
        self.sigma = float(sigma)
        self.delta = float(delta)

        if alpha_orders is None:
            # Standard set of alpha orders for RDP optimization
            self.alpha_orders = np.array(
                [1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]
            )
        else:
            self.alpha_orders = alpha_orders

        # Cumulative RDP epsilon per alpha order
        self._cumulative_rdp = np.zeros_like(self.alpha_orders, dtype=np.float64)
        self._num_queries = 0

    def step(self, vote_counts: np.ndarray) -> float:
        """Account for one PATE query.

        Args:
            vote_counts: Raw vote counts of shape (num_classes,).

        Returns:
            Current cumulative (epsilon, delta)-DP epsilon after this query.
        """
        rdp_eps = compute_rdp_gnmax(vote_counts, self.sigma, self.alpha_orders)
        self._cumulative_rdp += rdp_eps
        self._num_queries += 1
        return self.get_epsilon()

    def step_batch(self, vote_matrix: np.ndarray) -> float:
        """Account for a batch of PATE queries.

        Args:
            vote_matrix: Vote counts of shape (num_samples, num_classes).

        Returns:
            Current cumulative epsilon after all queries in this batch.
        """
        for i in range(vote_matrix.shape[0]):
            rdp_eps = compute_rdp_gnmax(
                vote_matrix[i], self.sigma, self.alpha_orders
            )
            self._cumulative_rdp += rdp_eps
        self._num_queries += vote_matrix.shape[0]
        return self.get_epsilon()

    def get_epsilon(self) -> float:
        """Get the tightest (epsilon, delta)-DP guarantee.

        Optimizes over all alpha orders to find the smallest epsilon.

        Returns:
            The best (smallest) epsilon value.
        """
        eps_values = np.array([
            rdp_to_dp(self._cumulative_rdp[i], self.alpha_orders[i], self.delta)
            for i in range(len(self.alpha_orders))
        ])
        return float(np.min(eps_values))

    def get_num_queries(self) -> int:
        """Return total number of PATE queries made."""
        return self._num_queries

    @property
    def spent(self) -> Tuple[float, float]:
        """Return (epsilon, delta) spent so far."""
        return (self.get_epsilon(), self.delta)
