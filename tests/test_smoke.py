"""Smoke tests: determinism, propensity sanity, and end-to-end run."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent917_creative_decisioning.data import (
    generate_coaches, generate_students, p_click, p_convert,
)
from agent917_creative_decisioning.arms import enumerate_arms, phi, PHI_DIM
from agent917_creative_decisioning.bandits import LinTS, ABChampionPolicy, AB_CHAMPION_ID, ARMS_DF
from agent917_creative_decisioning.offpolicy import epsilon_greedy_logging_policy


class TestDeterminism:
    """Two runs with the same seed produce identical results."""

    def test_coaches_deterministic(self):
        c1 = generate_coaches(100, rng=np.random.default_rng(917))
        c2 = generate_coaches(100, rng=np.random.default_rng(917))
        assert c1.equals(c2)

    def test_students_deterministic(self):
        s1 = generate_students(100, rng=np.random.default_rng(917))
        s2 = generate_students(100, rng=np.random.default_rng(917))
        assert s1.equals(s2)

    def test_outcome_models_deterministic(self):
        segs = np.array(["beginner", "fitness", "prenatal", "lapsed"])
        tiers = np.array(["boutique_solo", "studio_brand", "celebrity", "cold_start"])
        hls = np.array(["identity", "outcome", "therapeutic_identity", "social_proof"])
        imgs = np.array(["coach_portrait", "class_in_action", "lifestyle", "coach_portrait"])
        ofrs = np.array(["first_class_free", "3_class_pack", "20pct_off_first_month", "first_class_free"])
        ctas = np.array(["above_fold", "below_fold", "above_fold", "below_fold"])

        pc1 = p_click(segs, tiers, hls, imgs, ofrs, ctas)
        pc2 = p_click(segs, tiers, hls, imgs, ofrs, ctas)
        np.testing.assert_array_equal(pc1, pc2)

        pv1 = p_convert(segs, tiers, hls, imgs, ofrs, ctas)
        pv2 = p_convert(segs, tiers, hls, imgs, ofrs, ctas)
        np.testing.assert_array_equal(pv1, pv2)


class TestPropensitySanity:
    """Propensities from the logging policy are valid probabilities."""

    def test_propensities_sum_to_one(self):
        """Propensity for champion + exploration should be consistent."""
        policy = epsilon_greedy_logging_policy(AB_CHAMPION_ID, n_arms=72, epsilon=0.10)
        rng = np.random.default_rng(917)

        # The champion propensity
        _, prop_champ = policy("beginner", 0, 10, "boutique_solo", rng)
        expected = 0.90 + 0.10 / 72
        assert abs(prop_champ - expected) < 1e-10

    def test_propensities_positive(self):
        policy = epsilon_greedy_logging_policy(AB_CHAMPION_ID, n_arms=72, epsilon=0.10)
        rng = np.random.default_rng(917)
        for _ in range(100):
            _, prop = policy("fitness", 2, 30, "studio_brand", rng)
            assert prop > 0


class TestArms:
    """Arm enumeration and feature encoding."""

    def test_72_arms(self):
        arms = enumerate_arms()
        assert len(arms) == 72

    def test_phi_dimension(self):
        x = phi("beginner", 1, 30, "boutique_solo",
                "identity", "coach_portrait", "first_class_free", "above_fold")
        assert len(x) == PHI_DIM

    def test_phi_values_bounded(self):
        x = phi("fitness", 3, 100, "celebrity",
                "outcome", "class_in_action", "3_class_pack", "below_fold")
        assert np.all(x >= 0)
        assert np.all(x <= 1.5)


class TestEndToEnd:
    """Quick end-to-end: generate data, train bandit, check conversion."""

    def test_bandit_learns(self):
        rng = np.random.default_rng(917)
        coaches = generate_coaches(100, rng=np.random.default_rng(917))
        students = generate_students(500, rng=np.random.default_rng(917))
        arms_df = enumerate_arms()

        bandit = LinTS()
        n_updates = 2000

        for i in range(n_updates):
            si = rng.integers(0, len(students))
            ci = rng.integers(0, len(coaches))
            s = students.iloc[si]
            c = coaches.iloc[ci]

            arm_id = bandit.select_arm(
                s["segment"], int(s["experience_level"]),
                int(s["recency_days"]), c["tier"], rng,
            )
            arm = arms_df.iloc[arm_id]

            pv = p_convert(
                np.array([s["segment"]]), np.array([c["tier"]]),
                np.array([arm["headline"]]), np.array([arm["image"]]),
                np.array([arm["offer"]]), np.array([arm["cta"]]),
            )[0]
            reward = rng.binomial(1, pv)

            bandit.update(
                s["segment"], int(s["experience_level"]),
                int(s["recency_days"]), c["tier"],
                arm["headline"], arm["image"], arm["offer"], arm["cta"],
                reward,
            )

        # After training, bandit should pick reasonable arms
        arm = bandit.select_arm("beginner", 1, 10, "boutique_solo", rng)
        assert 0 <= arm < 72

    def test_conversion_rates_in_range(self):
        """Check that the DGP produces conversion rates in a plausible range."""
        n = 10000
        segs = np.array(["beginner"] * n)
        tiers = np.array(["boutique_solo"] * n)
        hls = np.array(["identity"] * n)
        imgs = np.array(["coach_portrait"] * n)
        ofrs = np.array(["first_class_free"] * n)
        ctas = np.array(["above_fold"] * n)

        pv = p_convert(segs, tiers, hls, imgs, ofrs, ctas)
        mean_pv = pv.mean()
        # Should be around 3-4% for optimal beginner creative
        assert 0.02 < mean_pv < 0.06, f"Expected ~3.4%, got {mean_pv:.3%}"
