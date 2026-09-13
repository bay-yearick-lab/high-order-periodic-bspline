# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Immersed smooth wall read from one periodic field.

Three smooth closed obstacles are embedded in a periodic box and the Dirichlet
trace is imposed at marker points through a Schur complement. The wall trace is
then read back at off-marker positions and the solution is measured in the
exterior, so both diagnostics probe the reconstruction between the nodes
rather than at the nodes.

Panel (a) carries the off-marker wall-trace error and panel (b) the exterior
relative l2 error, each taken as the worst of the four registered cell-unit
shifts, against grid size. The joint member, the C^2 record, the band champion
E_{6,2} and the polynomial rung BMC9 read all channels from one field; the two
compact routes carry a common twelve-point Lagrange evaluator.

Every value is read from results/exponential_family/gate9/e5_wall_runs.csv. The
four stored routes of that table are checked here against the earlier
results/exponential_family/gate5/5c/runs.csv character for character, and their
worst-shift blockers against 5c/decision.json, before the figure is written.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402

EF = ROOT / "results" / "exponential_family"
GATE = EF / "gate5" / "5c"
STORED_RUNS = GATE / "runs.csv"
DECISION = GATE / "decision.json"
RUNS = EF / "gate9" / "e5_wall_runs.csv"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"
RECORD_LABEL = r"$C^2$ record"

MEMBER_ROUTE = "EXP-X9_m6_C_SLC1_SLC1992_6IIq2"
RECORD_ROUTE = "EXP-X8_n10_C_SLC1_pert5"
STORED_ROUTES = ("EXP-E62", "BMC9", "C10|common_L12", "SLC-SL|common_L12")
ROUTE_ORDER = (MEMBER_ROUTE, RECORD_ROUTE) + STORED_ROUTES
ROUTE_LABEL = {
    MEMBER_ROUTE: JOINT_LABEL,
    RECORD_ROUTE: RECORD_LABEL,
    "EXP-E62": r"band member $E_{6,2}$",
    "BMC9": "BMC9",
    "C10|common_L12": "C10 + L12",
    "SLC-SL|common_L12": "SLC q4 + L12",
}
ROUTE_COLOR = {
    MEMBER_ROUTE: "#7B3294",
    RECORD_ROUTE: "#009E73",
    "EXP-E62": "#0072B2",
    "BMC9": "#3A3A3A",
    "C10|common_L12": "#56B4E9",
    "SLC-SL|common_L12": "#D55E00",
}
ROUTE_MARKER = {
    MEMBER_ROUTE: "D",
    RECORD_ROUTE: "P",
    "EXP-E62": "o",
    "BMC9": "s",
    "C10|common_L12": "^",
    "SLC-SL|common_L12": "v",
}
METRICS = (("trace_error", "(a)", "worst-shift wall-trace error"),
           ("exterior_relative_l2", "(b)", r"worst-shift exterior relative $\ell^2$"))
COMPARED_FIELDS = ("trace_error", "trace_error_disk", "trace_error_trefoil",
                   "trace_error_kite", "exterior_relative_l2",
                   "marker_constraint_residual", "schur_condition")
RELATIVE_TOLERANCE = 1.0e-9


def _worst() -> tuple[dict, int]:
    """Return the worst-shift value of each route, grid and metric."""
    with RUNS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with STORED_RUNS.open(newline="") as stream:
        earlier = {(row["route"], int(row["N"]), row["shift_id"]): row
                   for row in csv.DictReader(stream)}
    out: dict = defaultdict(dict)
    compared = 0
    for row in rows:
        for metric, _, _ in METRICS:
            key = (row["route"], metric)
            grid = int(row["N"])
            value = float(row[metric])
            out[key][grid] = max(out[key].get(grid, 0.0), value)
        if row["route"] not in STORED_ROUTES:
            continue
        held = earlier[(row["route"], int(row["N"]), row["shift_id"])]
        for field in COMPARED_FIELDS:
            if held[field] != row[field]:
                raise RuntimeError(
                    f"{row['route']} N={row['N']} {row['shift_id']} {field} "
                    f"drift: {row[field]} against {held[field]}")
            compared += 1
    return out, compared


def main() -> int:
    worst, compared = _worst()
    decision = json.loads(DECISION.read_text())
    blockers = decision["champion_versus_slc_q4_plus_L12"]

    for metric, _, _ in METRICS:
        key = "wall_trace" if metric == "trace_error" else metric
        blocker = blockers[key]["worst_shift_blocker"]
        grid = int(blocker["N"])
        stored = float(blocker["competitor_error"])
        fresh = worst[("SLC-SL|common_L12", metric)][grid]
        if fresh < stored * (1.0 - RELATIVE_TOLERANCE):
            raise RuntimeError(
                f"{metric} worst-shift value {fresh} below the blocker {stored}")

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 10.9,
        "axes.labelsize": 10.9,
        "xtick.labelsize": 10.9,
        "ytick.labelsize": 10.9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))

    grids = sorted(worst[("EXP-E62", "trace_error")])
    receipt = []
    for ax, (metric, tag, ylabel) in zip(axes, METRICS):
        for route in ROUTE_ORDER:
            values = [worst[(route, metric)][grid] for grid in grids]
            ax.loglog(grids, values, color=ROUTE_COLOR[route], lw=1.4,
                      marker=ROUTE_MARKER[route], ms=4.4,
                      label=ROUTE_LABEL[route])
            if metric == "trace_error":
                receipt.append((route, values))
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(grids)
        ax.set_xticklabels([str(grid) for grid in grids])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(grids[0] / 1.09, grids[-1] * 1.09)
        # Pinned so the larger tick font cannot thin the decade ladder.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.grid(axis="y", color="0.93", lw=0.5)
        ph.quiet_axis(ax)
        ax.text(0.015, 0.985, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.9, color=ph.CLASSY_ACCENT)

    handles, labels = axes[0].get_legend_handles_labels()
    ph.make_overhead_legend(fig, handles, labels, ncol=3, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.84), fontsize=10.9,
                            handslcngth=1.7, columnspacing=1.2,
                            handletextpad=0.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_wall.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"the four stored routes re-reduce identically on {compared} "
          "compared fields of gate5/5c/runs.csv")
    print("worst-shift wall-trace error (gate9/e5_wall_runs.csv):")
    for route, values in receipt:
        rendered = ", ".join(f"N={grid}: {value:.3e}"
                             for grid, value in zip(grids, values))
        print(f"  {ROUTE_LABEL[route]:22s} {rendered}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
