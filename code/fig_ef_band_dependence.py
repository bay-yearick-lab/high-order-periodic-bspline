# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Band objective against band edge at the SLC conditioning budget.

The Gate 1 objective compares a member against the SLC 1992 pair on the same
band,

    J = max( E^{(1)}_max / E_ref1 , E^{(2)}_max / E_ref2 ),

with both suprema taken on (0, T] and both reference values recomputed on the
same band. Sweeping the edge T and reoptimizing the frequencies at the fixed
left-side condition number C_SLC1 gives the best J of each polynomial core m,
so the horizontal line J = 1 is parity with the comparator on that band.

The three bands B_0 = 2 pi/3, B_1 = 0.75 pi and B_2 = 0.8 pi used elsewhere in
the evidence are marked. Every plotted value is read from the tracked receipt
results/exponential_family/gate1/figures/band_dependence.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402

DATA = (ROOT / "results" / "exponential_family" / "gate1" / "figures"
        / "band_dependence.json")
OUTDIR = ROOT / "figures_paper"

CORE_COLOR = {6: "#0072B2", 8: "#009E73"}
CORE_MARKER = {6: "o", 8: "s"}
C_PARITY = "#D55E00"
BAND_MARKS = ((2.0 / 3.0, r"$I_0$", "right"), (0.75, r"$I_1$", "right"),
              (0.80, r"$I_2$", "left"))


def _parity_crossing(edges: np.ndarray, values: np.ndarray) -> float:
    """Return the edge at which the objective first reaches parity."""
    above = np.flatnonzero(values >= 1.0)
    if above.size == 0 or above[0] == 0:
        return float("nan")
    j = int(above[0])
    x0, x1 = edges[j - 1], edges[j]
    y0, y1 = np.log(values[j - 1]), np.log(values[j])
    return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))


def main() -> int:
    data = json.loads(DATA.read_text())
    edges = np.asarray(data["edges"], dtype=float) / np.pi

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
    })
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    crossings = {}
    for core in (6, 8):
        raw = data["cores"][str(core)]["J"]
        good = np.array([value is not None for value in raw])
        values = np.array([value if value is not None else np.nan
                           for value in raw], dtype=float)
        ax.semilogy(edges[good], values[good], color=CORE_COLOR[core], lw=1.6,
                    marker=CORE_MARKER[core], ms=4.4, label=f"$m = {core}$")
        crossings[core] = _parity_crossing(edges[good], values[good])

    ax.axhline(1.0, color=C_PARITY, lw=0.9, ls=(0, (4, 2.5)))
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylim(2.0e-3, 3.0e1)
    ax.text(float(edges[0]) + 0.004, 1.25, "parity with SLC q4", fontsize=8.4,
            color=C_PARITY, ha="left", va="bottom")
    for position, text, side in BAND_MARKS:
        ax.axvline(position, color="0.70", lw=0.6, ls=(0, (1, 2.2)))
        nudge = -0.003 if side == "right" else 0.003
        ax.text(position + nudge, 2.4e-3, text, fontsize=8.4, color="0.45",
                ha=side, va="bottom")
    ax.set_xlabel(r"band edge $T/\pi$")
    ax.set_ylabel(r"$J = \max\left(E^{(1)}_{\max}/E_{\mathrm{ref}1},"
                  r"\ E^{(2)}_{\max}/E_{\mathrm{ref}2}\right)$")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    ph.make_overhead_legend(fig, handles, labels, ncol=2, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.92), fontsize=9.0,
                            handslcngth=1.8, columnspacing=1.4,
                            handletextpad=0.45)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_band_dependence.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"band sweep at {data['budget_name']} = {data['budget']:.6f}")
    for core in (6, 8):
        print(f"  m={core}: parity crossing at T/pi = {crossings[core]:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
