"""Synthetic data-generating process (DGP) for the two-sided yoga marketplace.

All numbers are ILLUSTRATIVE — designed so that click-maximising and
conversion-maximising creatives diverge.  That divergence is the teaching
point, not a found empirical result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADLINES = ["identity", "outcome", "social_proof", "therapeutic_identity"]
IMAGES = ["coach_portrait", "class_in_action", "lifestyle"]
OFFERS = ["first_class_free", "3_class_pack", "20pct_off_first_month"]
CTA_PLACEMENTS = ["above_fold", "below_fold"]

COACH_TIERS = ["boutique_solo", "studio_brand", "celebrity", "cold_start"]
COACH_TIER_WEIGHTS = [0.50, 0.30, 0.05, 0.15]

STUDENT_SEGMENTS = ["beginner", "fitness", "prenatal", "lapsed"]
STUDENT_SEGMENT_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

COACH_STYLES = ["vinyasa", "hatha", "yin", "power", "restorative"]
PRICE_TIERS = ["budget", "mid", "premium"]
ACQ_SOURCES = ["organic", "paid_social", "referral", "search"]


def generate_coaches(n: int = 2400, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate synthetic coach supply table.

    Parameters
    ----------
    n : int
        Number of coaches.
    rng : numpy.random.Generator, optional
        Seeded RNG for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: coach_id, tier, solo_vs_studio, style, price_tier, tenure_days.
    """
    if rng is None:
        rng = np.random.default_rng(917)

    tiers = rng.choice(COACH_TIERS, size=n, p=COACH_TIER_WEIGHTS)
    styles = rng.choice(COACH_STYLES, size=n)
    price_tiers = rng.choice(PRICE_TIERS, size=n)

    solo_vs_studio = np.where(
        np.isin(tiers, ["boutique_solo", "cold_start"]), "solo", "studio"
    )
    # Celebrity coaches are a mix
    celebrity_mask = tiers == "celebrity"
    solo_vs_studio[celebrity_mask] = rng.choice(
        ["solo", "studio"], size=celebrity_mask.sum()
    )

    tenure_days = np.zeros(n, dtype=int)
    for tier in COACH_TIERS:
        mask = tiers == tier
        if tier == "cold_start":
            tenure_days[mask] = rng.integers(0, 90, size=mask.sum())
        elif tier == "celebrity":
            tenure_days[mask] = rng.integers(365, 2000, size=mask.sum())
        elif tier == "studio_brand":
            tenure_days[mask] = rng.integers(180, 1500, size=mask.sum())
        else:
            tenure_days[mask] = rng.integers(30, 1200, size=mask.sum())

    return pd.DataFrame({
        "coach_id": np.arange(n),
        "tier": tiers,
        "solo_vs_studio": solo_vs_studio,
        "style": styles,
        "price_tier": price_tiers,
        "tenure_days": tenure_days,
    })


def generate_students(n: int = 95000, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate synthetic student demand table.

    Parameters
    ----------
    n : int
        Number of students.
    rng : numpy.random.Generator, optional
        Seeded RNG for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: student_id, segment, experience_level, recency_days, acquisition_source.
    """
    if rng is None:
        rng = np.random.default_rng(917)

    segments = rng.choice(STUDENT_SEGMENTS, size=n, p=STUDENT_SEGMENT_WEIGHTS)

    experience = np.zeros(n, dtype=int)
    for seg in STUDENT_SEGMENTS:
        mask = segments == seg
        if seg == "beginner":
            experience[mask] = rng.integers(0, 2, size=mask.sum())
        elif seg == "fitness":
            experience[mask] = rng.integers(1, 5, size=mask.sum())
        elif seg == "prenatal":
            experience[mask] = rng.integers(0, 4, size=mask.sum())
        else:  # lapsed
            experience[mask] = rng.integers(2, 8, size=mask.sum())

    recency = np.zeros(n, dtype=int)
    for seg in STUDENT_SEGMENTS:
        mask = segments == seg
        if seg == "lapsed":
            recency[mask] = rng.integers(90, 365, size=mask.sum())
        else:
            recency[mask] = rng.integers(0, 60, size=mask.sum())

    acq = rng.choice(ACQ_SOURCES, size=n)

    return pd.DataFrame({
        "student_id": np.arange(n),
        "segment": segments,
        "experience_level": experience,
        "recency_days": recency,
        "acquisition_source": acq,
    })


# ---------------------------------------------------------------------------
# Outcome models  (click ≠ conversion — this is the whole point)
# ---------------------------------------------------------------------------

def _logistic(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def p_click(
    segment: np.ndarray,
    coach_tier: np.ndarray,
    headline: np.ndarray,
    image: np.ndarray,
    offer: np.ndarray,
    cta: np.ndarray,
) -> np.ndarray:
    """Compute click probability.  Price-led creative wins broadly.

    The "first_class_free" + "lifestyle" combination drives clicks across
    all segments — this is the A/B champion that the team shipped.
    """
    z = np.full(len(segment), -2.8)  # base intercept

    # Strong click drivers (the champion components)
    z += 0.90 * (offer == "first_class_free")
    z += 0.65 * (image == "lifestyle")
    z += 0.30 * (headline == "social_proof")
    z += 0.20 * (cta == "above_fold")

    # Mild segment effects on click (clicks are fairly uniform)
    z += 0.10 * (segment == "lapsed")
    z += 0.05 * (segment == "beginner")

    # Coach tier: celebrity coaches get slightly more clicks
    z += 0.25 * (coach_tier == "celebrity")
    z += 0.05 * (coach_tier == "studio_brand")

    # Other creative components — mild click effects
    z += 0.15 * (offer == "20pct_off_first_month")
    z += 0.10 * (headline == "outcome")

    return _logistic(z)


def p_convert(
    segment: np.ndarray,
    coach_tier: np.ndarray,
    headline: np.ndarray,
    image: np.ndarray,
    offer: np.ndarray,
    cta: np.ndarray,
) -> np.ndarray:
    """Compute conversion (booking) probability — SEGMENT-DEPENDENT.

    The optimal creative differs by segment.  The click champion
    (first_class_free + lifestyle) converts poorly for most segments.

    Target per-segment optima (with weighted coach-tier mix):
        beginner  ~ 3.4%   identity + coach_portrait + first_class_free
        fitness   ~ 2.9%   outcome + class_in_action + 3_class_pack
        prenatal  ~ 4.1%   therapeutic_identity + coach_portrait + first_class_free
        lapsed    ~ 2.3%   social_proof + lifestyle + 20pct_off_first_month

    Target blended conversion under A/B champion ~ 2.1%.
    """
    z = np.full(len(segment), -3.87)  # base intercept (low base rate)

    # --- Segment main effects (sets each segment's base level) ---
    z += 0.05 * (segment == "beginner")
    z += 0.00 * (segment == "fitness")
    z += 0.06 * (segment == "prenatal")
    z -= 0.24 * (segment == "lapsed")

    # --- Coach tier effects ---
    z += 0.20 * (coach_tier == "celebrity")
    z += 0.07 * (coach_tier == "studio_brand")
    z += 0.02 * (coach_tier == "boutique_solo")
    z -= 0.06 * (coach_tier == "cold_start")

    # --- CTA placement (mild) ---
    z += 0.03 * (cta == "above_fold")

    # --- Segment × creative interactions (the core signal) ---
    # These create the divergence between click-optimal and conversion-optimal.

    # BEGINNER: identity headline + coach portrait + first class free
    z += 0.20 * (segment == "beginner") * (headline == "identity")
    z += 0.11 * (segment == "beginner") * (image == "coach_portrait")
    z += 0.08 * (segment == "beginner") * (offer == "first_class_free")
    z -= 0.10 * (segment == "beginner") * (image == "lifestyle")
    z -= 0.05 * (segment == "beginner") * (headline == "social_proof")

    # FITNESS: outcome headline + class in action + 3-class pack
    z += 0.13 * (segment == "fitness") * (headline == "outcome")
    z += 0.08 * (segment == "fitness") * (image == "class_in_action")
    z += 0.12 * (segment == "fitness") * (offer == "3_class_pack")
    z -= 0.10 * (segment == "fitness") * (offer == "first_class_free")
    z -= 0.06 * (segment == "fitness") * (image == "lifestyle")

    # PRENATAL: therapeutic identity headline + coach portrait + first class free
    z += 0.26 * (segment == "prenatal") * (headline == "therapeutic_identity")
    z += 0.14 * (segment == "prenatal") * (image == "coach_portrait")
    z += 0.20 * (segment == "prenatal") * (offer == "first_class_free")
    z -= 0.12 * (segment == "prenatal") * (image == "lifestyle")
    z -= 0.08 * (segment == "prenatal") * (headline == "social_proof")

    # LAPSED: social proof + lifestyle + 20% off (the only segment where price nudge fits)
    z += 0.12 * (segment == "lapsed") * (headline == "social_proof")
    z += 0.08 * (segment == "lapsed") * (image == "lifestyle")
    z += 0.13 * (segment == "lapsed") * (offer == "20pct_off_first_month")
    z -= 0.08 * (segment == "lapsed") * (offer == "first_class_free")

    # --- Mild main effects from creative components ---
    z += 0.03 * (headline == "identity")
    z += 0.02 * (image == "coach_portrait")

    return _logistic(z)


def generate_impressions(
    students: pd.DataFrame,
    coaches: pd.DataFrame,
    creative_assignments: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate impression-level outcomes (click + booking).

    Parameters
    ----------
    students : pd.DataFrame
        Student table.
    coaches : pd.DataFrame
        Coach table.
    creative_assignments : pd.DataFrame
        Must have columns: student_id, coach_id, headline, image, offer, cta.
    rng : numpy.random.Generator, optional
        Seeded RNG.

    Returns
    -------
    pd.DataFrame
        One row per impression with click and booking outcomes.
    """
    if rng is None:
        rng = np.random.default_rng(917)

    df = creative_assignments.merge(
        students[["student_id", "segment", "experience_level", "recency_days", "acquisition_source"]],
        on="student_id",
    ).merge(
        coaches[["coach_id", "tier", "solo_vs_studio", "style", "price_tier", "tenure_days"]],
        on="coach_id",
    )

    seg = df["segment"].values
    tier = df["tier"].values
    hl = df["headline"].values
    img = df["image"].values
    ofr = df["offer"].values
    cta = df["cta"].values

    pc = p_click(seg, tier, hl, img, ofr, cta)
    pv = p_convert(seg, tier, hl, img, ofr, cta)

    df["p_click"] = pc
    df["p_convert"] = pv
    df["click"] = rng.binomial(1, pc)
    df["booking"] = rng.binomial(1, pv)

    return df
