# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Fixed-grid practice comparison of the eight Taylor-Green routes.

The Gate 4c campaign runs the Taylor-Green vortex at Re = 1600 on 64^3 and
128^3 and compares the shell summed kinetic energy spectrum against the
restricted 384^3 reference over the resolved band 1 <= m < N/3. Each route is
run twice, once without dealiasing (base) and once with the two-thirds rule
(dealiased), so the two markers of a row differ only in that treatment.

Panel (a) carries 64^3 and panel (b) 128^3, both on a linear axis. Rows carry
the same route order in both panels, ranked by the base 128^3 value. Filled
markers are the base runs and open markers the dealiased runs.

Every value is read from the tracked Gate 4c run receipt
results/exponential_family/gate4c/runs.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402

RUNS = ROOT / "results" / "exponential_family" / "gate4c" / "runs.csv"
OUTDIR = ROOT / "figures_paper"

ROUTE_LABEL = {
    "exp": r"band member $E_{6,2}$",
    "sl": "SLC q4",
    "opt": "tuned q8",
    "cw12b": "CW12B",
    "bmc_p9": "BMC9",
    "bmc_p11": "BMC11",
    "c10": "C10",
    "c12": "C12",
}
ROUTE_COLOR = {
    "exp": "#0072B2",
    "sl": "#D55E00",
    "opt": "#CC79A7",
    "cw12b": "#8C510A",
    "bmc_p9": "#3A3A3A",
    "bmc_p11": "#000000",
    "c10": "#56B4E9",
    "c12": "#E69F00",
}
C_DEALIASED = "0.50"


def _cells() -> dict:
    """Return the unit-step cells of the Gate 4c run receipt."""
    with RUNS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["label"], int(row["N"]), int(row["dealias"])): float(row["e_band"])
            for row in rows if float(row["dt_scale"]) == 1.0}


def main() -> int:
    cells = _cells()
    order = sorted(ROUTE_LABEL, key=lambda name: cells[(name, 128, 0)])

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 11.4,
        "axes.labelsize": 11.4,
        "xtick.labelsize": 11.4,
        "ytick.labelsize": 11.4,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.9))

    receipt = []
    limits = {64: (0.150, 0.235), 128: (0.010, 0.115)}
    for ax, tag, grid in zip(axes, ("(a)", "(b)"), (64, 128)):
        for position, name in enumerate(order):
            y = len(order) - 1 - position
            base = cells[(name, grid, 0)]
            dealiased = cells[(name, grid, 1)]
            ax.plot([base, dealiased], [y, y], color="0.80", lw=0.7, zorder=2)
            ax.plot([base], [y], marker="o", ms=5.4, color=ROUTE_COLOR[name],
                    ls="none", zorder=4)
            ax.plot([dealiased], [y], marker="o", ms=5.0, mfc="none",
                    mec=C_DEALIASED, mew=1.0, ls="none", zorder=3)
            if grid == 128:
                receipt.append((name, base, dealiased))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([ROUTE_LABEL[name] for name in reversed(order)])
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(*limits[grid])
        ax.set_xlabel(f"resolved-band error at ${grid}^3$")
        ax.grid(axis="x", color="0.92", lw=0.5)
        ph.quiet_axis(ax)
        ax.text(0.015, 0.985, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=11.4, color=ph.CLASSY_ACCENT)
    axes[1].set_yticklabels([])

    handles = [
        Line2D([], [], color="0.3", marker="o", ls="none", ms=5.4,
               label="base run"),
        Line2D([], [], color=C_DEALIASED, marker="o", ls="none", ms=5.0,
               mfc="none", mew=1.0, label="two-thirds dealiased"),
    ]
    ph.make_overhead_legend(fig, handles, [h.get_label() for h in handles],
                            ncol=2, bbox=(0.5, 1.005), rect=(0, 0, 1, 0.92),
                            fontsize=11.4, handslcngth=1.2, columnspacing=1.5,
                            handletextpad=0.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_practice.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("resolved-band errors at 128^3 (gate4c/runs.csv):")
    for name, base, dealiased in receipt:
        print(f"  {name:8s} base={base:.6f} dealiased={dealiased:.6f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
