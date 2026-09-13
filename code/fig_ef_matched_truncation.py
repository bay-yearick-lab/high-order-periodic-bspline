# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Matched-truncation operator fidelity inside the Taylor-Green vortex.

All ten routes run the same dealiased Taylor-Green vortex at Re = 1600 from
the same initial condition with the same time integrator and step, and each is
compared against the dealiased Fourier solution on its own grid at the common
landing time. The remaining field difference is therefore attributable to the
derivative operator alone.

Panel (a) ranks the routes by the relative field error e_full at 128^3 and
places the two realized errors beside the two symbol-level predictions P of
the same cells, so realized and predicted read on one logarithmic axis. The
joint member and the C^2 record ran the same campaign afterwards on the same
reference fields, and their two rows sit in the same ranking. Panel (b)
resolves the 128^3 error over the shell summed kinetic energy spectrum for the
champion, for the SLC 1992 pair and for the polynomial rung BMC11; shells
whose reference energy sits at the round-off floor carry no relative error and
are omitted.

The eight stored routes are read from results/exponential_family/gate4e,
namely runs.csv and shell_errors.json. The two member rows are read from
results/exponential_family/gate9, namely e2_runs.csv with e2_prediction.json
for the C^2 record and e2_runs_core6.csv with e2_prediction_core6.json for the
joint member. The producer checks that the stored routes re-reduce identically
in the later receipts before the figure is written.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402

GATE = ROOT / "results" / "exponential_family" / "gate4e"
RUNS = GATE / "runs.csv"
SHELLS = GATE / "shell_errors.json"
GATE9 = ROOT / "results" / "exponential_family" / "gate9"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"
RECORD_LABEL = r"$C^2$ record"

# The later campaign of each member: its run table, its prediction file and the
# label its rows carry inside them.
MEMBER_RUNS = {
    "core6": (GATE9 / "e2_runs_core6.csv", GATE9 / "e2_prediction_core6.json"),
    "pert5": (GATE9 / "e2_runs.csv", GATE9 / "e2_prediction.json"),
}

ROUTE_LABEL = {
    "exp": r"band member $E_{6,2}$",
    "sl": "SLC q4",
    "opt": "tuned q8",
    "cw12b": "CW12B",
    "bmc_p9": "BMC9",
    "bmc_p11": "BMC11",
    "c10": "C10",
    "c12": "C12",
    "core6": JOINT_LABEL,
    "pert5": RECORD_LABEL,
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
    "core6": "#7B3294",
    "pert5": "#009E73",
}
STORED_ROUTES = ("exp", "sl", "opt", "cw12b", "bmc_p9", "bmc_p11", "c10", "c12")
GRIDS = (64, 128)
SHELL_ROUTES = ("exp", "sl", "bmc_p11")
C_PREDICTED = "0.55"
ENERGY_FLOOR = 1.0e-12


def _table(path: Path) -> dict:
    """Return the dealiased unit-step cells of one run receipt."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["label"], int(row["N"])): row
            for row in rows if float(row["dt_scale"]) == 1.0}


def _rows() -> tuple[dict, list]:
    """Return every plotted cell and the re-reduction log of the stored routes."""
    cells = _table(RUNS)
    log = []
    for name, (runs, prediction) in MEMBER_RUNS.items():
        later = _table(runs)
        stored_P = json.loads(prediction.read_text())["grids"]
        for grid in GRIDS:
            for route in STORED_ROUTES:
                if (route, grid) not in later:
                    continue
                for field in ("e_full", "predicted_P"):
                    a = cells[(route, grid)][field]
                    b = later[(route, grid)][field]
                    if a != b:
                        raise RuntimeError(
                            f"{route} at {grid} {field} drift: {a} against {b}")
                    log.append((runs.name, route, grid, field, a))
            cell = later[(name, grid)]
            quoted = float(stored_P[str(grid)]["P"][name])
            if float(cell["predicted_P"]) != quoted:
                raise RuntimeError(
                    f"{name} at {grid} predicted P drift: "
                    f"{cell['predicted_P']} against {quoted}")
            cells[(name, grid)] = cell
    return cells, log


def main() -> int:
    cells, log = _rows()
    shells = json.loads(SHELLS.read_text())["cells"]
    order = sorted(ROUTE_LABEL, key=lambda name: float(cells[(name, 128)]["e_full"]))

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 11.4,
        "axes.labelsize": 11.4,
        "xtick.labelsize": 11.4,
        "ytick.labelsize": 11.4,
    })
    fig, (ax_rank, ax_shell) = plt.subplots(
        1, 2, figsize=(7.9, 3.9), gridspec_kw={"width_ratios": [1.0, 1.02]})

    receipt = []
    for position, name in enumerate(order):
        y = len(order) - 1 - position
        color = ROUTE_COLOR[name]
        realized = [float(cells[(name, n)]["e_full"]) for n in GRIDS]
        predicted = [float(cells[(name, n)]["predicted_P"]) for n in GRIDS]
        ax_rank.plot(predicted, [y, y], color=C_PREDICTED, lw=0.7, zorder=2)
        ax_rank.plot(realized, [y, y], color=color, lw=0.9, alpha=0.75, zorder=3)
        ax_rank.plot([predicted[0]], [y], marker="o", ms=4.4, mfc="none",
                     mec=C_PREDICTED, mew=0.9, ls="none", zorder=4)
        ax_rank.plot([predicted[1]], [y], marker="s", ms=4.0, mfc="none",
                     mec=C_PREDICTED, mew=0.9, ls="none", zorder=4)
        ax_rank.plot([realized[0]], [y], marker="o", ms=5.2, color=color,
                     ls="none", zorder=5)
        ax_rank.plot([realized[1]], [y], marker="s", ms=4.8, color=color,
                     ls="none", zorder=5)
        receipt.append((name, realized[0], realized[1], predicted[0], predicted[1]))

    ax_rank.set_xscale("log")
    ax_rank.set_yticks(range(len(order)))
    ax_rank.set_yticklabels([ROUTE_LABEL[name] for name in reversed(order)])
    ax_rank.set_ylim(-0.7, len(order) + 0.15)
    ax_rank.set_xlim(2.0e-6, 4.0e-1)
    ax_rank.set_xlabel("relative field error and symbol prediction")
    ax_rank.grid(axis="x", color="0.92", lw=0.5)
    ph.quiet_axis(ax_rank)
    ax_rank.text(0.015, 0.985, "(a)", transform=ax_rank.transAxes, ha="left",
                 va="top", fontsize=11.4, color=ph.CLASSY_ACCENT)

    for name in SHELL_ROUTES:
        cell = shells[f"{name}_N128_da1_dt100"]
        index = np.asarray(cell["shells"], dtype=float)
        reference = np.asarray(cell["shell_reference_spectrum"], dtype=float)
        error = np.abs(np.asarray(cell["shell_relative_error"], dtype=float))
        keep = reference > ENERGY_FLOOR * float(np.max(reference))
        ax_shell.semilogy(index[keep], np.maximum(error[keep], 2.0e-7),
                          color=ROUTE_COLOR[name], lw=1.4,
                          label=ROUTE_LABEL[name])
    ax_shell.set_xlabel(r"spectral shell $m$")
    ax_shell.set_ylabel("relative shell energy error")
    ax_shell.set_xlim(0.0, 42.0)
    ax_shell.set_ylim(2.0e-7, 1.0e-2)
    ax_shell.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax_shell)
    ax_shell.legend(loc="lower right", frameon=False, fontsize=11.4,
                    handslcngth=1.8, borderaxespad=0.4)
    ax_shell.text(0.015, 0.985, "(b)", transform=ax_shell.transAxes, ha="left",
                  va="top", fontsize=11.4, color=ph.CLASSY_ACCENT)

    handles = [
        Line2D([], [], color="0.3", marker="o", ls="none", ms=5.2,
               label=r"realized, $64^3$"),
        Line2D([], [], color="0.3", marker="s", ls="none", ms=4.8,
               label=r"realized, $128^3$"),
        Line2D([], [], color=C_PREDICTED, marker="o", ls="none", ms=4.4,
               mfc="none", mew=0.9, label=r"predicted $P$, $64^3$"),
        Line2D([], [], color=C_PREDICTED, marker="s", ls="none", ms=4.0,
               mfc="none", mew=0.9, label=r"predicted $P$, $128^3$"),
    ]
    ph.make_overhead_legend(fig, handles, [h.get_label() for h in handles],
                            ncol=4, bbox=(0.5, 1.005), rect=(0, 0, 1, 0.90),
                            fontsize=11.4, handslcngth=1.2, columnspacing=1.3,
                            handletextpad=0.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_matched_truncation.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"the eight stored routes re-reduce identically in the later "
          f"receipts on {len(log)} compared fields")
    print("matched-truncation field errors:")
    for name, e64, e128, p64, p128 in receipt:
        print(f"  {name:8s} e_full 64={e64:.6e} 128={e128:.6e} "
              f"P 64={p64:.6e} 128={p128:.6e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
