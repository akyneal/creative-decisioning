"""Contextual bandit policies: LinTS, LinUCB, and baselines.

LinTS uses Bayesian linear regression with hierarchical priors that pool
by coach tier, so cold-start coaches shrink toward the tier mean.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .arms import enumerate_arms, phi, phi_all_arms, PHI_DIM
from .data import p_click, COACH_TIERS


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Minimal policy interface: select an arm given context."""

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        ...


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

ARMS_DF = enumerate_arms()
# The A/B champion: first_class_free + lifestyle + social_proof + above_fold
AB_CHAMPION_ID = int(ARMS_DF.loc[
    (ARMS_DF["headline"] == "social_proof") &
    (ARMS_DF["image"] == "lifestyle") &
    (ARMS_DF["offer"] == "first_class_free") &
    (ARMS_DF["cta"] == "above_fold"),
    "arm_id",
].iloc[0])


class ABChampionPolicy:
    """Always serve the A/B-test champion (click-winner)."""

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        return AB_CHAMPION_ID


class RandomPolicy:
    """Uniform random arm selection."""

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        return rng.integers(0, len(ARMS_DF))


class SegmentRulesPolicy:
    """Hand-crafted heuristic: pick a plausible creative per segment.

    These are decent but not optimal — the bandit should beat them.
    """

    _RULES: dict[str, int] = {}

    def __init__(self) -> None:
        arms = ARMS_DF
        for seg, hl, img, ofr in [
            ("beginner", "identity", "coach_portrait", "first_class_free"),
            ("fitness", "outcome", "class_in_action", "first_class_free"),  # wrong offer
            ("prenatal", "therapeutic_identity", "coach_portrait", "3_class_pack"),  # wrong offer
            ("lapsed", "social_proof", "lifestyle", "first_class_free"),  # wrong offer
        ]:
            row = arms.loc[
                (arms["headline"] == hl) & (arms["image"] == img) &
                (arms["offer"] == ofr) & (arms["cta"] == "above_fold")
            ]
            self._RULES[seg] = int(row["arm_id"].iloc[0])

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        return self._RULES.get(segment, AB_CHAMPION_ID)


class ClickOptimisedPolicy:
    """Argmax p_click — demonstrates that optimising clicks ≠ conversion."""

    def __init__(self) -> None:
        self._arms = enumerate_arms()
        self._best: dict[tuple[str, str], int] = {}

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        key = (segment, coach_tier)
        if key not in self._best:
            arms = self._arms
            n = len(arms)
            segs = np.full(n, segment)
            tiers = np.full(n, coach_tier)
            pc = p_click(
                segs, tiers,
                arms["headline"].values, arms["image"].values,
                arms["offer"].values, arms["cta"].values,
            )
            self._best[key] = int(np.argmax(pc))
        return self._best[key]


# ---------------------------------------------------------------------------
# LinTS — Linear Thompson Sampling with hierarchical priors
# ---------------------------------------------------------------------------

class LinTS:
    """Linear Thompson Sampling with Bayesian linear regression.

    Hierarchical priors: each coach tier maintains a prior mean, and
    individual-level posteriors shrink toward the tier mean for cold-start
    contexts.

    Parameters
    ----------
    dim : int
        Dimensionality of phi(context, arm).
    v_sq : float
        Noise variance (assumed known).
    lambda_prior : float
        Prior precision (diagonal).
    """

    def __init__(self, dim: int = PHI_DIM, v_sq: float = 0.25, lambda_prior: float = 1.0) -> None:
        self.dim = dim
        self.v_sq = v_sq
        self.lambda_prior = lambda_prior

        # Per-tier posterior sufficient statistics (hierarchical pooling)
        self._B: dict[str, np.ndarray] = {}   # precision matrix
        self._f: dict[str, np.ndarray] = {}   # B @ mu
        self._n_obs: dict[str, int] = {}

        # Global prior (pooled across all tiers)
        self._B_global = lambda_prior * np.eye(dim)
        self._f_global = np.zeros(dim)
        self._n_global = 0

        for tier in COACH_TIERS:
            self._B[tier] = lambda_prior * np.eye(dim)
            self._f[tier] = np.zeros(dim)
            self._n_obs[tier] = 0

    def _get_posterior(self, coach_tier: str) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and covariance for a coach tier.

        For tiers with few observations, the posterior is shrunk toward the
        global (all-tier) mean — this is the hierarchical pooling.
        """
        tier_n = self._n_obs.get(coach_tier, 0)
        # Shrinkage weight: more tier data -> less pooling
        alpha = min(tier_n / max(tier_n + 50, 1), 0.95)

        B_tier = self._B.get(coach_tier, self.lambda_prior * np.eye(self.dim))
        f_tier = self._f.get(coach_tier, np.zeros(self.dim))

        # Blend tier and global posteriors
        B_blend = alpha * B_tier + (1 - alpha) * self._B_global
        f_blend = alpha * f_tier + (1 - alpha) * self._f_global

        cov = np.linalg.inv(B_blend) * self.v_sq
        mu = np.linalg.solve(B_blend, f_blend)
        return mu, cov

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        """Thompson-sample a reward for each arm, return the argmax."""
        mu, cov = self._get_posterior(coach_tier)

        # Sample theta from posterior
        theta = rng.multivariate_normal(mu, cov)

        # Vectorised: compute x @ theta for all 72 arms at once
        X = phi_all_arms(segment, experience_level, recency_days, coach_tier)
        scores = X @ theta
        return int(np.argmax(scores))

    def update(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, headline: str, image: str, offer: str, cta: str,
        reward: float,
    ) -> None:
        """Update posterior with an observed (context, arm, reward) tuple."""
        x = phi(segment, experience_level, recency_days, coach_tier,
                headline, image, offer, cta)
        outer = np.outer(x, x)

        # Update tier-specific posterior
        if coach_tier not in self._B:
            self._B[coach_tier] = self.lambda_prior * np.eye(self.dim)
            self._f[coach_tier] = np.zeros(self.dim)
            self._n_obs[coach_tier] = 0

        self._B[coach_tier] += outer / self.v_sq
        self._f[coach_tier] += x * reward / self.v_sq
        self._n_obs[coach_tier] += 1

        # Update global posterior
        self._B_global += outer / self.v_sq
        self._f_global += x * reward / self.v_sq
        self._n_global += 1


# ---------------------------------------------------------------------------
# LinUCB — comparison policy
# ---------------------------------------------------------------------------

class LinUCB:
    """LinUCB (disjoint model, shared features).

    Parameters
    ----------
    dim : int
        Feature dimension.
    alpha : float
        Exploration parameter (UCB width).
    """

    def __init__(self, dim: int = PHI_DIM, alpha: float = 1.0) -> None:
        self.dim = dim
        self.alpha = alpha
        self._A = np.eye(dim)
        self._b = np.zeros(dim)

    def select_arm(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, rng: np.random.Generator,
    ) -> int:
        A_inv = np.linalg.inv(self._A)
        theta_hat = A_inv @ self._b

        X = phi_all_arms(segment, experience_level, recency_days, coach_tier)
        exploit = X @ theta_hat
        explore = self.alpha * np.sqrt(np.einsum("ij,jk,ik->i", X, A_inv, X))
        ucb = exploit + explore
        return int(np.argmax(ucb))

    def update(
        self, segment: str, experience_level: int, recency_days: int,
        coach_tier: str, headline: str, image: str, offer: str, cta: str,
        reward: float,
    ) -> None:
        x = phi(segment, experience_level, recency_days, coach_tier,
                headline, image, offer, cta)
        self._A += np.outer(x, x)
        self._b += x * reward
