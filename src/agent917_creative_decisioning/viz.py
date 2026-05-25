"""Agent917-branded matplotlib visualisation helpers.

Palette: navy #0A1628, gold #F9CB28.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Agent917 palette
NAVY = "#0A1628"
GOLD = "#F9CB28"
LIGHT_NAVY = "#1B3A5C"
LIGHT_GOLD = "#FBE17A"
GRAY = "#8C8C8C"
WHITE = "#FFFFFF"
PALETTE = [NAVY, GOLD, LIGHT_NAVY, LIGHT_GOLD, GRAY]

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"


def set_agent917_style() -> None:
    """Apply Agent917 styling to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def plot_clicks_vs_conversion(
    weeks: np.ndarray,
    click_rates: np.ndarray,
    conv_rates: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """The 'clicks up, conversion flat' framing chart.

    Parameters
    ----------
    weeks : array of week numbers
    click_rates : click-through rates over time
    conv_rates : conversion rates over time
    save : bool
        Save to figures/clicks_vs_conversion.png.
    """
    set_agent917_style()
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(weeks, click_rates * 100, color=GOLD, linewidth=2.5, label="Click-through rate")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Click-through rate (%)", color=GOLD)
    ax1.tick_params(axis="y", labelcolor=GOLD)

    ax2 = ax1.twinx()
    ax2.plot(weeks, conv_rates * 100, color=NAVY, linewidth=2.5, linestyle="--", label="Conversion rate")
    ax2.set_ylabel("Conversion rate (%)", color=NAVY)
    ax2.tick_params(axis="y", labelcolor=NAVY)

    fig.suptitle("Clicks went up and revenue didn't", fontsize=16, fontweight="bold", color=NAVY)
    ax1.set_title("A/B champion drives clicks but conversion stays flat", fontsize=11, color=GRAY)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "clicks_vs_conversion.png")
    return fig


def plot_cumulative_regret(
    regret_dict: dict[str, np.ndarray],
    save: bool = True,
) -> plt.Figure:
    """Cumulative regret / learning curves for bandit vs baselines.

    Parameters
    ----------
    regret_dict : dict
        Policy name -> array of cumulative regret at each time step.
    """
    set_agent917_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [NAVY, GOLD, LIGHT_NAVY, GRAY, LIGHT_GOLD]
    for (name, regret), color in zip(regret_dict.items(), colors):
        ax.plot(regret, label=name, color=color, linewidth=2)

    ax.set_xlabel("Impressions")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Learning curve: contextual bandit vs baselines", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "cumulative_regret.png")
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save: bool = True,
) -> plt.Figure:
    """Reliability / calibration plot for the reward model.

    Parameters
    ----------
    y_true : binary outcomes
    y_prob : predicted probabilities
    """
    set_agent917_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_means.append(y_true[mask].mean())

    ax.plot([0, 1], [0, 1], "--", color=GRAY, label="Perfect calibration")
    ax.scatter(bin_centers, bin_means, color=NAVY, s=80, zorder=5)
    ax.plot(bin_centers, bin_means, color=GOLD, linewidth=2, label="Reward model")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reward model calibration", fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 0.12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "calibration.png")
    return fig


def plot_impression_concentration(
    shares_before: pd.Series,
    shares_after: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Side-by-side impression concentration before/after fairness.

    Parameters
    ----------
    shares_before : pd.Series
        tier -> fraction (unconstrained).
    shares_after : pd.Series
        tier -> fraction (constrained).
    """
    set_agent917_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tiers = shares_before.index.tolist()
    x = np.arange(len(tiers))
    width = 0.6

    ax1.bar(x, shares_before.values * 100, width, color=NAVY)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, rotation=30, ha="right")
    ax1.set_ylabel("Impression share (%)")
    ax1.set_title("Unconstrained (greedy)", fontweight="bold")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax2.bar(x, shares_after.values * 100, width, color=GOLD)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers, rotation=30, ha="right")
    ax2.set_ylabel("Impression share (%)")
    ax2.set_title("With fairness constraints", fontweight="bold")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("Impression concentration by coach tier", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "impression_concentration.png")
    return fig


def plot_conversion_lift(
    conv_champion: float,
    conv_bandit: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
    save: bool = True,
) -> plt.Figure:
    """Conversion lift bar chart: champion vs bandit with CI.

    Parameters
    ----------
    conv_champion, conv_bandit : conversion rates
    ci_low, ci_high : optional confidence interval bounds for bandit
    """
    set_agent917_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["A/B Champion", "Contextual Bandit"],
        [conv_champion * 100, conv_bandit * 100],
        color=[GRAY, GOLD],
        edgecolor=NAVY,
        linewidth=1.5,
        width=0.5,
    )

    if ci_low is not None and ci_high is not None:
        ax.errorbar(
            1, conv_bandit * 100,
            yerr=[[conv_bandit * 100 - ci_low * 100], [ci_high * 100 - conv_bandit * 100]],
            fmt="none", color=NAVY, capsize=8, linewidth=2,
        )

    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Conversion lift: champion vs contextual bandit", fontweight="bold")

    lift = (conv_bandit - conv_champion) / conv_champion * 100
    ax.annotate(
        f"+{lift:.0f}%",
        xy=(1, conv_bandit * 100),
        xytext=(1.3, conv_bandit * 100),
        fontsize=16, fontweight="bold", color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD),
    )

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "conversion_lift.png")
    return fig


def plot_segment_heatmap(
    segment_table: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Heatmap/table of per-segment winning creatives and conversion rates.

    Parameters
    ----------
    segment_table : pd.DataFrame
        Must have: segment, headline, image, offer, conversion_rate.
    """
    set_agent917_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    cols = ["Segment", "Headline", "Image", "Offer", "Conv. Rate"]
    cell_text = []
    for _, row in segment_table.iterrows():
        cell_text.append([
            row["segment"].title(),
            row["headline"].replace("_", " ").title(),
            row["image"].replace("_", " ").title(),
            row["offer"].replace("_", " ").title(),
            f"{row['conversion_rate']:.1%}",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(cols)):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color=WHITE, fontweight="bold")

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F0F0F0")

    ax.set_title("Per-segment winning creatives (bandit policy)", fontweight="bold", pad=20)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "segment_heatmap.png")
    return fig
