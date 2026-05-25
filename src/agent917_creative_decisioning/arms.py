"""Creative arm enumeration and feature encoding phi(context, arm).

The 72 creatives = 4 headlines x 3 images x 3 offers x 2 CTA placements.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .data import HEADLINES, IMAGES, OFFERS, CTA_PLACEMENTS, STUDENT_SEGMENTS


def enumerate_arms() -> pd.DataFrame:
    """Return a DataFrame of all 72 creative arms.

    Returns
    -------
    pd.DataFrame
        Columns: arm_id, headline, image, offer, cta.
    """
    combos = list(product(HEADLINES, IMAGES, OFFERS, CTA_PLACEMENTS))
    return pd.DataFrame(combos, columns=["headline", "image", "offer", "cta"]).assign(
        arm_id=lambda df: np.arange(len(df))
    )[["arm_id", "headline", "image", "offer", "cta"]]


# ---------------------------------------------------------------------------
# Feature maps for one-hot encoding
# ---------------------------------------------------------------------------
_HEADLINE_MAP = {h: i for i, h in enumerate(HEADLINES)}
_IMAGE_MAP = {im: i for i, im in enumerate(IMAGES)}
_OFFER_MAP = {o: i for i, o in enumerate(OFFERS)}
_CTA_MAP = {c: i for i, c in enumerate(CTA_PLACEMENTS)}
_SEGMENT_MAP = {s: i for i, s in enumerate(STUDENT_SEGMENTS)}


def _one_hot(value: str, mapping: dict[str, int]) -> np.ndarray:
    """Return a one-hot vector for a categorical value."""
    vec = np.zeros(len(mapping))
    vec[mapping[value]] = 1.0
    return vec


def phi(
    segment: str,
    experience_level: int,
    recency_days: int,
    coach_tier: str,
    headline: str,
    image: str,
    offer: str,
    cta: str,
) -> np.ndarray:
    """Build feature vector phi(context, arm).

    Layout (30 dims):
        [0:4]   segment one-hot (4)
        [4]     experience_level (normalised)
        [5]     recency_days (normalised)
        [6:10]  coach_tier one-hot (4) — boutique_solo, studio_brand, celebrity, cold_start
        [10:14] headline one-hot (4)
        [14:17] image one-hot (3)
        [17:20] offer one-hot (3)
        [20:22] cta one-hot (2)
        [22:26] segment x headline interaction (top 4)
        [26:30] segment x offer interaction (top 4)

    Parameters
    ----------
    segment, experience_level, recency_days, coach_tier : context features
    headline, image, offer, cta : arm (creative) features

    Returns
    -------
    np.ndarray
        Feature vector of length 30.
    """
    tier_map = {"boutique_solo": 0, "studio_brand": 1, "celebrity": 2, "cold_start": 3}

    parts = [
        _one_hot(segment, _SEGMENT_MAP),            # 4
        np.array([experience_level / 7.0]),          # 1
        np.array([recency_days / 365.0]),            # 1
        _one_hot(coach_tier, tier_map),              # 4
        _one_hot(headline, _HEADLINE_MAP),           # 4
        _one_hot(image, _IMAGE_MAP),                 # 3
        _one_hot(offer, _OFFER_MAP),                 # 3
        _one_hot(cta, _CTA_MAP),                     # 2
        # Key interactions: segment × headline
        np.array([
            float(segment == "beginner" and headline == "identity"),
            float(segment == "fitness" and headline == "outcome"),
            float(segment == "prenatal" and headline == "therapeutic_identity"),
            float(segment == "lapsed" and headline == "social_proof"),
        ]),                                          # 4
        # Key interactions: segment × offer
        np.array([
            float(segment == "beginner" and offer == "first_class_free"),
            float(segment == "fitness" and offer == "3_class_pack"),
            float(segment == "prenatal" and offer == "first_class_free"),
            float(segment == "lapsed" and offer == "20pct_off_first_month"),
        ]),                                          # 4
    ]
    return np.concatenate(parts)


PHI_DIM = 30  # total dimensionality of phi

# Precomputed arm-only features (constant across contexts)
_ARMS_DF_CACHED = None
_ARM_ONLY_FEATURES = None  # shape (72, 12): headline(4) + image(3) + offer(3) + cta(2)


def _ensure_arm_cache() -> tuple[pd.DataFrame, np.ndarray]:
    """Lazily precompute arm-only one-hot features."""
    global _ARMS_DF_CACHED, _ARM_ONLY_FEATURES
    if _ARM_ONLY_FEATURES is None:
        arms = enumerate_arms()
        _ARMS_DF_CACHED = arms
        n = len(arms)
        feats = np.zeros((n, 12))
        for i, (_, row) in enumerate(arms.iterrows()):
            feats[i, 0:4] = _one_hot(row["headline"], _HEADLINE_MAP)
            feats[i, 4:7] = _one_hot(row["image"], _IMAGE_MAP)
            feats[i, 7:10] = _one_hot(row["offer"], _OFFER_MAP)
            feats[i, 10:12] = _one_hot(row["cta"], _CTA_MAP)
        _ARM_ONLY_FEATURES = feats
    return _ARMS_DF_CACHED, _ARM_ONLY_FEATURES


def phi_all_arms(
    segment: str, experience_level: int, recency_days: int, coach_tier: str,
) -> np.ndarray:
    """Build feature matrix for ALL 72 arms given a single context.

    Much faster than calling phi() 72 times — precomputes arm-only features
    and broadcasts context features.

    Returns
    -------
    np.ndarray
        Shape (72, PHI_DIM).
    """
    tier_map = {"boutique_solo": 0, "studio_brand": 1, "celebrity": 2, "cold_start": 3}
    arms_df, arm_feats = _ensure_arm_cache()
    n = len(arms_df)

    X = np.zeros((n, PHI_DIM))

    # Context features (same for all arms) — broadcast
    seg_oh = _one_hot(segment, _SEGMENT_MAP)
    tier_oh = _one_hot(coach_tier, tier_map)
    X[:, 0:4] = seg_oh
    X[:, 4] = experience_level / 7.0
    X[:, 5] = recency_days / 365.0
    X[:, 6:10] = tier_oh

    # Arm features (precomputed)
    X[:, 10:22] = arm_feats

    # Interaction features (segment × headline, segment × offer)
    hl_vals = arms_df["headline"].values
    ofr_vals = arms_df["offer"].values

    X[:, 22] = (segment == "beginner") * (hl_vals == "identity")
    X[:, 23] = (segment == "fitness") * (hl_vals == "outcome")
    X[:, 24] = (segment == "prenatal") * (hl_vals == "therapeutic_identity")
    X[:, 25] = (segment == "lapsed") * (hl_vals == "social_proof")

    X[:, 26] = (segment == "beginner") * (ofr_vals == "first_class_free")
    X[:, 27] = (segment == "fitness") * (ofr_vals == "3_class_pack")
    X[:, 28] = (segment == "prenatal") * (ofr_vals == "first_class_free")
    X[:, 29] = (segment == "lapsed") * (ofr_vals == "20pct_off_first_month")

    return X


def phi_batch(contexts: pd.DataFrame, arms: pd.DataFrame) -> np.ndarray:
    """Vectorised feature construction for a batch.

    Parameters
    ----------
    contexts : pd.DataFrame
        Must have: segment, experience_level, recency_days, tier.
    arms : pd.DataFrame
        Must have: headline, image, offer, cta (same length as contexts).

    Returns
    -------
    np.ndarray
        Shape (n, PHI_DIM).
    """
    n = len(contexts)
    X = np.zeros((n, PHI_DIM))

    seg_vals = contexts["segment"].values
    exp_vals = contexts["experience_level"].values
    rec_vals = contexts["recency_days"].values
    tier_vals = contexts["tier"].values

    hl_vals = arms["headline"].values
    img_vals = arms["image"].values
    ofr_vals = arms["offer"].values
    cta_vals = arms["cta"].values

    tier_map = {"boutique_solo": 0, "studio_brand": 1, "celebrity": 2, "cold_start": 3}

    for i in range(n):
        X[i] = phi(
            seg_vals[i], int(exp_vals[i]), int(rec_vals[i]), tier_vals[i],
            hl_vals[i], img_vals[i], ofr_vals[i], cta_vals[i],
        )
    return X
