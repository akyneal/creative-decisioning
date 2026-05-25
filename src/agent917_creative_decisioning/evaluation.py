"""Online simulation, hypothesis testing, and business-impact translation.

Simulates a 6-week experiment with 90% bandit / 10% holdout (A/B champion),
computes conversion lift, and translates to business metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .arms import enumerate_arms, phi
from .data import (
    p_convert, generate_students, generate_coaches,
    STUDENT_SEGMENTS, COACH_TIERS,
)
from .bandits import LinTS, ABChampionPolicy, ARMS_DF


def run_online_simulation(
    students: pd.DataFrame,
    coaches: pd.DataFrame,
    bandit: LinTS,
    n_weeks: int = 6,
    impressions_per_week: int = 15000,
    bandit_fraction: float = 0.90,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate an online experiment: bandit (90%) vs A/B champion holdout (10%).

    Parameters
    ----------
    students : pd.DataFrame
    coaches : pd.DataFrame
    bandit : LinTS
        Pre-trained or fresh bandit.
    n_weeks : int
    impressions_per_week : int
    bandit_fraction : float
    rng : numpy.random.Generator

    Returns
    -------
    pd.DataFrame
        Impression-level data with 'group' (bandit/holdout) and 'booking'.
    """
    if rng is None:
        rng = np.random.default_rng(917)

    champion = ABChampionPolicy()
    arms_df = enumerate_arms()
    n_students = len(students)
    n_coaches = len(coaches)

    records = []
    for week in range(n_weeks):
        for _ in range(impressions_per_week):
            si = rng.integers(0, n_students)
            ci = rng.integers(0, n_coaches)
            s = students.iloc[si]
            c = coaches.iloc[ci]

            seg = s["segment"]
            exp = int(s["experience_level"])
            rec = int(s["recency_days"])
            tier = c["tier"]

            is_bandit = rng.random() < bandit_fraction

            if is_bandit:
                arm_id = bandit.select_arm(seg, exp, rec, tier, rng)
                group = "bandit"
            else:
                arm_id = champion.select_arm(seg, exp, rec, tier, rng)
                group = "holdout"

            arm = arms_df.iloc[arm_id]
            pv = p_convert(
                np.array([seg]), np.array([tier]),
                np.array([arm["headline"]]), np.array([arm["image"]]),
                np.array([arm["offer"]]), np.array([arm["cta"]]),
            )[0]
            booking = rng.binomial(1, pv)

            # Update bandit if in treatment group
            if is_bandit:
                bandit.update(
                    seg, exp, rec, tier,
                    arm["headline"], arm["image"], arm["offer"], arm["cta"],
                    booking,
                )

            records.append({
                "week": week,
                "student_id": s["student_id"],
                "coach_id": c["coach_id"],
                "segment": seg,
                "tier": tier,
                "arm_id": arm_id,
                "headline": arm["headline"],
                "image": arm["image"],
                "offer": arm["offer"],
                "cta": arm["cta"],
                "group": group,
                "booking": booking,
            })

    return pd.DataFrame(records)


def two_proportion_z_test(
    n_treatment: int, conv_treatment: int,
    n_control: int, conv_control: int,
) -> tuple[float, float, float, float]:
    """Two-proportion z-test for conversion lift.

    Parameters
    ----------
    n_treatment, conv_treatment : treatment group counts
    n_control, conv_control : control group counts

    Returns
    -------
    tuple
        (p_treatment, p_control, z_stat, p_value)
    """
    p_t = conv_treatment / n_treatment
    p_c = conv_control / n_control
    p_pool = (conv_treatment + conv_control) / (n_treatment + n_control)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_treatment + 1 / n_control))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return p_t, p_c, z, p_value


def compute_mde(
    n_treatment: int, n_control: int,
    baseline_rate: float, alpha: float = 0.05, power: float = 0.80,
) -> float:
    """Compute minimum detectable effect (MDE) for a two-proportion test.

    Pre-registered MDE: with ~81,000 treatment and ~9,000 control at a
    2.1% baseline, the MDE is approximately 0.35 percentage points
    (relative lift ~17%).  Our expected effect (+33%) is well above MDE.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    se = np.sqrt(
        baseline_rate * (1 - baseline_rate) * (1 / n_treatment + 1 / n_control)
    )
    mde = (z_alpha + z_beta) * se
    return mde


def slice_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute conversion rate by segment x group.

    Returns
    -------
    pd.DataFrame
        Pivot: rows=segment, columns=group, values=conversion_rate.
    """
    grouped = (
        results
        .groupby(["segment", "group"])["booking"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grouped["conversion_rate"] = grouped["sum"] / grouped["count"]
    pivot = grouped.pivot(index="segment", columns="group", values="conversion_rate")
    return pivot


def per_segment_winning_creative(results: pd.DataFrame) -> pd.DataFrame:
    """Find the best-performing creative per segment in bandit traffic.

    Returns
    -------
    pd.DataFrame
        Rows=segment, columns=headline, image, offer, conversion_rate.
    """
    bandit_data = results[results["group"] == "bandit"]
    records = []
    for seg in STUDENT_SEGMENTS:
        seg_data = bandit_data[bandit_data["segment"] == seg]
        if len(seg_data) == 0:
            continue
        grouped = (
            seg_data
            .groupby(["headline", "image", "offer"])["booking"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grouped["conversion_rate"] = grouped["sum"] / grouped["count"]
        # Filter to arms with enough data
        grouped = grouped[grouped["count"] >= 20]
        if len(grouped) == 0:
            continue
        best = grouped.loc[grouped["conversion_rate"].idxmax()]
        records.append({
            "segment": seg,
            "headline": best["headline"],
            "image": best["image"],
            "offer": best["offer"],
            "conversion_rate": best["conversion_rate"],
        })
    return pd.DataFrame(records)


def business_impact(
    conv_champion: float,
    conv_bandit: float,
    weekly_sends: int = 95000,
    avg_class_price: float = 25.0,
    classes_per_activated: float = 8.0,
    platform_take_rate: float = 0.20,
    weeks_per_quarter: int = 13,
) -> dict[str, float]:
    """Translate conversion lift to business impact.

    ASSUMPTIONS (explicit and documented):
    - weekly_sends: ~95,000 emails per week (total student base)
    - avg_class_price: $25 average class price
    - classes_per_activated: each activated student books ~8 classes/quarter
    - platform_take_rate: marketplace takes 20% of GMV
    - weeks_per_quarter: 13 weeks

    Returns
    -------
    dict
        incremental_students_per_quarter, incremental_gmv_quarterly,
        annualised_gmv, annualised_revenue.
    """
    lift_abs = conv_bandit - conv_champion
    incremental_per_week = weekly_sends * lift_abs
    incremental_per_quarter = incremental_per_week * weeks_per_quarter

    gmv_per_student = avg_class_price * classes_per_activated
    incremental_gmv_quarterly = incremental_per_quarter * gmv_per_student
    annualised_gmv = incremental_gmv_quarterly * 4
    annualised_revenue = annualised_gmv * platform_take_rate

    return {
        "conv_champion": conv_champion,
        "conv_bandit": conv_bandit,
        "relative_lift": (conv_bandit - conv_champion) / conv_champion,
        "incremental_students_per_quarter": incremental_per_quarter,
        "incremental_gmv_quarterly": incremental_gmv_quarterly,
        "annualised_gmv": annualised_gmv,
        "annualised_revenue": annualised_revenue,
    }
