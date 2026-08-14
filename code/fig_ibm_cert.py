# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Render the frozen IBM-CERT scalar receipt for archival inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from plotting_helpers import apply_classy_defaults  # noqa: E402


SUMMARY = REPO / "results" / "ibm_cert" / "summary.json"
OUTPUT = REPO / "results" / "manuscript_recut" / "visual_archive" / "fig_ibm_cert.pdf"


def main() -> None:
    """Render exact accuracy factors and the measured setup intercept."""
    apply_classy_defaults()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text())
    strength = summary["mechanism"]["bmc9_accuracy_strength"]
    cost = summary["mechanism"]["cost_mechanism"]

    error_factors = [
        strength["wall_trace_worst_shift_exact_factor"],
        strength["exterior_relative_l2_worst_shift_exact_factor"],
    ]
    total_ratios = [
        cost["wall_trace_blocker_total_time_ratio"],
        cost["exterior_blocker_total_time_ratio"],
    ]
    setup_ratios = [
        cost["wall_trace_blocker_setup_ratio"],
        cost["exterior_blocker_setup_ratio"],
    ]
    schur_ratios = [
        cost["wall_trace_blocker_schur_assembly_ratio"],
        cost["exterior_blocker_schur_assembly_ratio"],
    ]

    blue = "#22517a"
    orange = "#b06a2c"
    gray = "#686868"
    labels = ["wall trace", r"exterior $L^2$"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))

    ax = axes[0]
    x = [0.0, 1.0]
    bars = ax.bar(x, error_factors, width=0.58, color=[blue, orange],
                  edgecolor="black", linewidth=0.55)
    thresholds = [4.0, 2.0]
    for xpos, bar, value, threshold in zip(x, bars, error_factors, thresholds):
        ax.hlines(threshold, xpos - 0.35, xpos + 0.35, color=gray,
                  linestyle="--", linewidth=1.0)
        ax.text(xpos, threshold + 1.2, f"gate {threshold:.0f}x", ha="center",
                va="bottom", fontsize=7.5, color=gray)
        ax.text(bar.get_x() + bar.get_width() / 2, value - 2.2,
                f"{value:.2f}x", ha="center", va="bottom",
                fontsize=9.0, color="white")
    ax.set_xticks(x, labels)
    ax.set_ylabel("strongest-compact error / BMC9 error")
    ax.set_ylim(0.0, 72.0)
    ax.set_title("(a) accuracy-first separation", loc="left", fontsize=10)
    ax.text(0.5, 0.97, "BMC9 lies on both all-route fronts in all four shifts",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
            color="#2f4858")

    ax = axes[1]
    categories = ["total solve", "setup", "Schur assembly"]
    centers = [0.0, 1.0, 2.0]
    width = 0.34
    wall = [total_ratios[0], setup_ratios[0], schur_ratios[0]]
    exterior = [total_ratios[1], setup_ratios[1], schur_ratios[1]]
    bars_wall = ax.bar([v - width / 2 for v in centers], wall, width=width,
                       color=blue, edgecolor="black", linewidth=0.45,
                       label="wall-trace cell")
    bars_ext = ax.bar([v + width / 2 for v in centers], exterior, width=width,
                      color=orange, edgecolor="black", linewidth=0.45,
                      label=r"exterior-$L^2$ cell")
    ax.axhline(1.0, color="black", linewidth=0.75)
    ax.axhline(1.25, color=gray, linestyle="--", linewidth=1.0,
               label="1.25 total-time gate")
    for bar, value in zip([*bars_wall, *bars_ext], [*wall, *exterior]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.055,
                f"{value:.3f}", ha="center", va="bottom", fontsize=7.2,
                rotation=90 if value > 2.0 else 0)
    ax.set_xticks(centers, categories)
    ax.set_ylabel("BMC9 / C10 plus common L12 time ratio")
    ax.set_ylim(0.0, 3.05)
    ax.set_title("(b) measured setup intercept", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.88", linewidth=0.55)
        ax.set_axisbelow(True)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
