"""Off-policy evaluation: IPS, SNIPS, doubly-robust estimators.

Demonstrates why logged propensities are essential for counterfactual
evaluation and provides bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .arms import enumerate_arms, phi, PHI_DIM


def generate_logged_data(
    students: pd.DataFrame,
    coaches: pd.DataFrame,
    n_impressions: int,
    logging_policy_fn,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a logged dataset from a known logging policy.

    Parameters
    ----------
    students : pd.DataFrame
    coaches : pd.DataFrame
    n_impressions : int
        Number of impressions to log.
    logging_policy_fn : callable
        (segment, experience_level, recency_days, coach_tier, rng) -> (arm_id, propensity)
    rng : numpy.random.Generator

    Returns
    -------
    pd.DataFrame
        Logged data with propensities.
    """
    from .data import p_convert

    arms_df = enumerate_arms()
    n_students = len(students)
    n_coaches = len(coaches)

    records = []
    for _ in range(n_impressions):
        si = rng.integers(0, n_students)
        ci = rng.integers(0, n_coaches)
        s = students.iloc[si]
        c = coaches.iloc[ci]

        arm_id, propensity = logging_policy_fn(
            s["segment"], int(s["experience_level"]),
            int(s["recency_days"]), c["tier"], rng,
        )
        arm = arms_df.iloc[arm_id]

        pv = p_convert(
            np.array([s["segment"]]), np.array([c["tier"]]),
            np.array([arm["headline"]]), np.array([arm["image"]]),
            np.array([arm["offer"]]), np.array([arm["cta"]]),
        )[0]
        booking = rng.binomial(1, pv)

        records.append({
            "student_id": s["student_id"],
            "coach_id": c["coach_id"],
            "segment": s["segment"],
            "experience_level": s["experience_level"],
            "recency_days": s["recency_days"],
            "tier": c["tier"],
            "arm_id": arm_id,
            "headline": arm["headline"],
            "image": arm["image"],
            "offer": arm["offer"],
            "cta": arm["cta"],
            "propensity": propensity,
            "booking": booking,
        })

    return pd.DataFrame(records)


def epsilon_greedy_logging_policy(champion_id: int, n_arms: int = 72, epsilon: float = 0.10):
    """Return a logging policy function that serves champion (1-eps) and explores eps.

    Returns
    -------
    callable
        (segment, exp, rec, tier, rng) -> (arm_id, propensity)
    """
    def policy(segment, experience_level, recency_days, coach_tier, rng):
        if rng.random() < epsilon:
            arm = rng.integers(0, n_arms)
            prop = epsilon / n_arms
            if arm == champion_id:
                prop += (1 - epsilon)
            return int(arm), prop
        else:
            return champion_id, (1 - epsilon) + epsilon / n_arms
    return policy


# ---------------------------------------------------------------------------
# Off-policy estimators (vectorised for performance)
# ---------------------------------------------------------------------------

def _compute_target_arms(
    logged: pd.DataFrame,
    target_policy_fn,
) -> np.ndarray:
    """Precompute target policy arm assignments for all rows.

    Parameters
    ----------
    logged : pd.DataFrame
    target_policy_fn : callable
        (row) -> (arm_id, propensity)

    Returns
    -------
    np.ndarray
        Array of target arm IDs.
    """
    target_arms = np.zeros(len(logged), dtype=int)
    for i, (_, row) in enumerate(logged.iterrows()):
        target_arms[i], _ = target_policy_fn(row)
    return target_arms


def ips_estimate(
    logged: pd.DataFrame,
    target_arms: np.ndarray,
    reward_col: str = "booking",
) -> float:
    """Inverse propensity scoring (IPS) estimator.

    Parameters
    ----------
    logged : pd.DataFrame
        Must contain propensity, arm_id, and reward_col.
    target_arms : np.ndarray
        Precomputed target arm for each row.

    Returns
    -------
    float
        IPS estimate of target policy's expected reward.
    """
    match = logged["arm_id"].values == target_arms
    rewards = logged[reward_col].values
    propensities = logged["propensity"].values
    return np.sum(rewards[match] / propensities[match]) / len(logged)


def snips_estimate(
    logged: pd.DataFrame,
    target_arms: np.ndarray,
    reward_col: str = "booking",
) -> float:
    """Self-normalised IPS (SNIPS) estimator -- lower variance than IPS."""
    match = logged["arm_id"].values == target_arms
    rewards = logged[reward_col].values
    propensities = logged["propensity"].values
    weights = 1.0 / propensities[match]
    num = np.sum(rewards[match] * weights)
    denom = np.sum(weights)
    return num / denom if denom > 0 else 0.0


def fit_reward_model(logged: pd.DataFrame) -> LogisticRegression:
    """Fit a logistic regression reward model q(x, a) for DR estimation.

    Parameters
    ----------
    logged : pd.DataFrame
        Logged data with features and booking outcome.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    X = np.array([
        phi(
            row["segment"], int(row["experience_level"]), int(row["recency_days"]),
            row["tier"], row["headline"], row["image"], row["offer"], row["cta"],
        )
        for _, row in logged.iterrows()
    ])
    y = logged["booking"].values
    model = LogisticRegression(max_iter=1000, random_state=917)
    model.fit(X, y)
    return model


def _precompute_dr_components(
    logged: pd.DataFrame,
    target_arms: np.ndarray,
    reward_model: LogisticRegression,
    reward_col: str = "booking",
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-row DR components for fast bootstrap.

    Returns
    -------
    tuple of (q_targets, corrections)
        q_targets: reward model predictions for target arm (n,)
        corrections: IPS correction terms (n,)
    """
    arms_df = enumerate_arms()
    n = len(logged)
    q_targets = np.zeros(n)
    corrections = np.zeros(n)

    # Batch-compute phi for target arms
    X_target = np.zeros((n, PHI_DIM))
    for i, (_, row) in enumerate(logged.iterrows()):
        ta = target_arms[i]
        arm_info = arms_df.iloc[ta]
        X_target[i] = phi(
            row["segment"], int(row["experience_level"]), int(row["recency_days"]),
            row["tier"],
            arm_info["headline"], arm_info["image"], arm_info["offer"], arm_info["cta"],
        )

    q_targets = reward_model.predict_proba(X_target)[:, 1]

    match = logged["arm_id"].values == target_arms
    rewards = logged[reward_col].values
    propensities = logged["propensity"].values

    corrections[match] = (rewards[match] - q_targets[match]) / propensities[match]

    return q_targets, corrections


def dr_estimate_from_components(
    q_targets: np.ndarray,
    corrections: np.ndarray,
    indices: np.ndarray | None = None,
) -> float:
    """Compute DR estimate from precomputed components.

    Parameters
    ----------
    q_targets, corrections : precomputed arrays
    indices : optional subset indices (for bootstrap)

    Returns
    -------
    float
    """
    if indices is not None:
        return np.mean(q_targets[indices] + corrections[indices])
    return np.mean(q_targets + corrections)


def dr_estimate(
    logged: pd.DataFrame,
    target_arms: np.ndarray,
    reward_model: LogisticRegression,
    reward_col: str = "booking",
) -> float:
    """Doubly-robust (DR) estimator.

    Combines the IPS correction with a fitted reward model to reduce
    variance while remaining unbiased if either component is correct.
    """
    q_targets, corrections = _precompute_dr_components(
        logged, target_arms, reward_model, reward_col
    )
    return dr_estimate_from_components(q_targets, corrections)


def bootstrap_dr_ci(
    logged: pd.DataFrame,
    target_arms: np.ndarray,
    reward_model: LogisticRegression,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for the DR estimator.

    Uses precomputed DR components for fast bootstrap resampling.

    Returns
    -------
    tuple
        (point_estimate, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng(917)

    q_targets, corrections = _precompute_dr_components(
        logged, target_arms, reward_model
    )
    point = dr_estimate_from_components(q_targets, corrections)

    n = len(logged)
    boot_estimates = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_estimates[b] = dr_estimate_from_components(q_targets, corrections, idx)

    ci_lo = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_hi = np.percentile(boot_estimates, 100 * (1 - alpha / 2))
    return point, ci_lo, ci_hi


def demonstrate_no_propensities() -> str:
    """Show that off-policy evaluation fails without logged propensities.

    Returns
    -------
    str
        Explanation of the failure mode.
    """
    msg = (
        "OFF-POLICY EVALUATION REQUIRES LOGGED PROPENSITIES\n"
        "====================================================\n\n"
        "Without knowing the probability that the logging policy assigned\n"
        "to each served arm, importance weights cannot be computed.\n\n"
        "If you set propensity = 1.0 for all logged actions (a common\n"
        "mistake when propensities weren't recorded), the IPS estimator\n"
        "degenerates to the naive mean of rewards only for actions that\n"
        "match the target policy -- which is biased because the logging\n"
        "policy's selection mechanism is not corrected for.\n\n"
        "Example: if the logging policy serves arm A 90% of the time,\n"
        "the naive estimator will be dominated by arm A's outcomes,\n"
        "even when evaluating a target policy that prefers arm B.\n\n"
        "PRACTICAL IMPLICATION: instrument your serving system to record\n"
        "the propensity (probability) of the served creative at serve time.\n"
        "This is a prerequisite for any counterfactual evaluation."
    )
    return msg
