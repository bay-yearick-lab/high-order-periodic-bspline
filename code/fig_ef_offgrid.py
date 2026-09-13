# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Coupled off-grid reading of value, first and second derivative.

Six routes answer the same question at an arbitrary point between the nodes.
Four of them read all three channels from one field, namely the joint member,
the band champion E_{6,2} and the polynomial rungs BMC9 and BMC11. The
remaining two bolt a Lagrange layer of twelve or sixteen nodes onto the SLC
1992 pair in its operator-faithful form, which is the construction the wall
experiment of the paper fields and the stronger of the two augmentation
variants in the two derivative channels.

Panel (a) carries the coupled symbol error

    e_c(theta) = max over xi in [-1/2, 1/2] of max_q |T_q(theta, xi) - 1|,

with the band edge B_0 = 2 pi/3 marked. Panels (b) and (c) carry the realized
relative supremum error of the second-derivative channel alone, the channel on
which the routes separate most, in the two realized regimes: (b) the
resolution-scaled packet, whose top mode sits at the dealiased band edge on
every grid, and (c) the well-resolved packet, whose top mode is fixed as the
grid refines so that formal order takes over. Showing one channel instead of
three keeps every curve legible; the other two channels are tabulated in the
receipts and quoted in the prose.

The curves of panel (a) are recomputed with the Gate 5a symbol routine, the
joint member entering through the member route of gate9_e4; the stored routes
are checked against the amendment block of
results/exponential_family/gate5/5a/decision.json and the joint member against
the symbol block of results/exponential_family/gate9/e4_offgrid.json. Panels
(b) and (c) are read from results/exponential_family/gate9/e4_runs.csv, whose
rows for the five stored routes are checked here against the earlier
results/exponential_family/gate5/5a/runs.csv character for character.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate5a_offgrid as g5a  # noqa: E402
import gate9_e4 as g9e4  # noqa: E402
import plotting_helpers as ph  # noqa: E402

EF = ROOT / "results" / "exponential_family"
GATE = EF / "gate5" / "5a"
DECISION = GATE / "decision.json"
STORED_RUNS = GATE / "runs.csv"
OFFGRID = EF / "gate9" / "e4_offgrid.json"
RUNS = EF / "gate9" / "e4_runs.csv"
JOINT_MEMBER = EF / "gate8" / "members" / "X9_m6_C_SLC1_SLC1992_6IIq2.json"
MEMBER_KEY = "core6"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"

HEADLINE = (MEMBER_KEY, "champion", "BMC9", "BMC11", "slc_operator_L12",
            "slc_operator_L16")
STORED = ("champion", "BMC9", "BMC11", "slc_operator_L12", "slc_operator_L16")
ROUTE_LABEL = {
    MEMBER_KEY: JOINT_LABEL,
    "champion": r"band member $E_{6,2}$",
    "BMC9": "BMC9",
    "BMC11": "BMC11",
    "slc_operator_L12": "SLC q4 + L12",
    "slc_operator_L16": "SLC q4 + L16",
}
ROUTE_COLOR = {
    MEMBER_KEY: "#7B3294",
    "champion": "#0072B2",
    "BMC9": "#3A3A3A",
    "BMC11": "#000000",
    "slc_operator_L12": "#D55E00",
    "slc_operator_L16": "#CC79A7",
}
# The name each route carries in the earlier stored table, per regime.
STORED_NAME = {
    "resolution_scaled": {name: f"{name}|resolution_scaled" for name in STORED},
    "anchor": {"champion": "champion", "BMC9": "BMC9", "BMC11": "BMC11",
               "slc_operator_L12": "slc_operator_L12|well_resolved",
               "slc_operator_L16": "slc_operator_L16|well_resolved"},
}
SCALED_GRIDS = (48, 64, 96, 128, 192, 256)
WELL_GRIDS = (64, 128, 256)
RELATIVE_TOLERANCE = 1.0e-9


def _realized() -> tuple[dict, int]:
    """Return the realized errors and the count of re-reduced stored cells."""
    with RUNS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with STORED_RUNS.open(newline="") as stream:
        earlier = {(row["route"], row["channel"], int(row["N"])): row
                   for row in csv.DictReader(stream)}
    out: dict = {}
    compared = 0
    for row in rows:
        key = (row["route"], row["regime"], row["channel"], int(row["N"]))
        out[key] = float(row["rel_sup_error"])
        name = STORED_NAME[row["regime"]].get(row["route"])
        if name is None:
            continue
        held = earlier[(name, row["channel"], int(row["N"]))]
        for field in ("max_abs_error", "rel_sup_error", "rel_l2_error"):
            if held[field] != row[field]:
                raise RuntimeError(
                    f"{row['route']} {row['regime']} N={row['N']} "
                    f"{row['channel']} {field} drift: "
                    f"{row[field]} against {held[field]}")
            compared += 1
    return out, compared


def _series(realized: dict, name: str, regime: str) -> list[float]:
    """Return the second-derivative errors of one route in one regime."""
    grids = SCALED_GRIDS if regime == "resolution_scaled" else WELL_GRIDS
    return [realized[(name, regime, "d2", npts)] for npts in grids]


def _routes() -> dict:
    """Return the six plotted routes, the joint member built from its record."""
    routes = {route.name: route for route in g5a.build_amendment_routes()}
    routes[MEMBER_KEY] = g9e4.MemberRoute(MEMBER_KEY, JOINT_MEMBER)
    return routes


def main() -> int:
    amendment = json.loads(DECISION.read_text())["amendment_1"]
    stored = {entry["route"]: entry for entry in amendment["symbol_level"]}
    stored.update({entry["route"]: entry
                   for entry in json.loads(OFFGRID.read_text())["symbol_level"]
                   if entry["route"] == MEMBER_KEY})
    realized, compared = _realized()

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 12.2,
        "axes.labelsize": 12.2,
        "xtick.labelsize": 12.2,
        "ytick.labelsize": 12.2,
    })
    fig = plt.figure(figsize=(7.9, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.96, 0.72], wspace=0.52,
                            top=0.80, bottom=0.14, left=0.075, right=0.99)
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_scaled = fig.add_subplot(grid[0, 1])
    ax_well = fig.add_subplot(grid[0, 2])

    edge = g5a.BANDS["B0"]
    receipt = []
    routes = _routes()
    for name in HEADLINE:
        row, theta, ec = g5a.symbol_level(routes[name])
        reference = float(stored[name]["ec_max_B0"])
        if abs(row["ec_max_B0"] - reference) > RELATIVE_TOLERANCE * reference:
            raise RuntimeError(
                f"{name} ec_max_B0 drift: {row['ec_max_B0']} against {reference}")
        ax_curve.semilogy(theta / np.pi, np.maximum(ec, 1.0e-13),
                          color=ROUTE_COLOR[name], lw=1.4,
                          label=ROUTE_LABEL[name])
        receipt.append((name, row["ec_max_B0"], reference,
                        row["reach_ec_1e-2_over_pi"]))

    ax_curve.axvline(edge / np.pi, color="0.55", lw=0.7, ls=(0, (1, 2.2)))
    ax_curve.text(edge / np.pi - 0.012, 1.6e-13, r"$I_0$", fontsize=12.2,
                  color="0.45", ha="right", va="bottom")
    ax_curve.axhline(1.0e-2, color="0.62", lw=0.6, ls=(0, (4, 2.5)))
    ax_curve.text(0.055, 1.45e-2, r"$10^{-2}$", fontsize=12.2, color="0.62",
                  ha="left", va="bottom")
    ax_curve.set_xlabel(r"$\theta/\pi$")
    ax_curve.set_ylabel(r"$e_c(\theta)$")
    ax_curve.set_xlim(0.0, 1.0)
    # Pinned so the larger tick font cannot re-space the axis.
    ax_curve.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_curve.set_ylim(1.0e-13, 5.0)
    ax_curve.grid(axis="y", color="0.93", lw=0.5)
    ph.quiet_axis(ax_curve)
    ax_curve.set_title("(a) coupled symbol error", loc="left", fontsize=12.2,
                       color=ph.CLASSY_ACCENT, fontstyle="italic", pad=4.0)

    for ax, regime, grids, tag in (
            (ax_scaled, "resolution_scaled", SCALED_GRIDS,
             "(b) resolution-scaled packet"),
            (ax_well, "anchor", WELL_GRIDS,
             "(c) well-resolved packet")):
        for name in HEADLINE:
            ax.loglog(grids, np.maximum(_series(realized, name, regime),
                                        1.0e-16),
                      color=ROUTE_COLOR[name], lw=1.3, marker="o", ms=3.0)
        ax.set_xlabel(r"$N$")
        ax.set_xticks(list(grids))
        ax.set_xticklabels([str(n) for n in grids])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(grids[0] / 1.12, grids[-1] * 1.12)
        ax.grid(axis="y", color="0.93", lw=0.5)
        ph.quiet_axis(ax)
        ax.set_title(tag, loc="left", fontsize=12.2, color=ph.CLASSY_ACCENT,
                     fontstyle="italic", pad=4.0)
    ax_scaled.set_ylabel("second-derivative relative supremum error")
    ax_scaled.set_ylim(1.0e-5, 1.0e-1)
    ax_well.set_ylim(1.0e-13, 1.0e-2)
    for index in (1, 3):
        ax_scaled.get_xticklabels()[index].set_visible(False)

    handles, labels = ax_curve.get_legend_handles_labels()
    legend = ph.make_overhead_legend(fig, handles, labels, ncol=3,
                                     bbox=(0.5, 1.005), rect=(0, 0, 1, 0.78),
                                     fontsize=12.2, handslcngth=1.5,
                                     columnspacing=0.9, handletextpad=0.35)
    for handle in legend.legend_handles:
        handle.set_linewidth(2.2)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_offgrid.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"the five stored routes re-reduce identically on {compared} "
          "compared fields of gate5/5a/runs.csv")
    print("coupled off-grid symbol receipt:")
    for name, fresh, reference, reach in receipt:
        print(f"  {name:20s} ec_max_B0={fresh:.6e} against {reference:.6e}, "
              f"reach={reach:.6f} pi")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
