# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Semi-Lagrangian matched-accuracy mission on a divergence-free packet.

The mission transports a divergence-free packet of K modes by exact periodic
translation, reading the value, the three gradient components and the Laplacian
off grid at every step, and it admits a grid only when all five relative errors
stay below one percent through the requested number of steps while the mass
drift stays below its own tolerance.

Panels (a) through (c) carry the three step histories at K = 24 and shift A,
each route drawn on its own minimum admissible grid of the refined ladder, with
the dashed line at the one percent admissibility tolerance. The three gradient
components are collapsed into one panel by taking their maximum, because the
three components carry the same history to plotting accuracy; the mass channel
is a number rather than a panel, since it sits below 2e-17 on every history
drawn here and below 4e-14 over the whole sweep. Panel (d) carries the worst of
the five channels after M = 32 steps against grid size, the sweep that fixes
each route's minimum admissible grid. Panel (e) carries those grids beside the
wall time of the whole mission on them, all six timings measured in one session
under the same one-thread protocol.

The stored routes are read from results/exponential_family/gate5/5b, namely the
histories directory, the amendment block of decision.json and runs.csv. The
joint member is read from results/exponential_family/gate9, namely
e5_histories, e5_semilagrangian.json and e5_runs.csv; the producer checks that
the later receipt repeats the stored minimum grids of the four stored routes
before the figure is written.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402

EF = ROOT / "results" / "exponential_family"
GATE = EF / "gate5" / "5b"
HISTORIES = GATE / "histories"
DECISION = GATE / "decision.json"
RUNS = GATE / "runs.csv"
GATE9 = EF / "gate9"
MEMBER_HISTORIES = GATE9 / "e5_histories"
MEMBER_DECISION = GATE9 / "e5_semilagrangian.json"
MEMBER_RUNS = GATE9 / "e5_runs.csv"
MEMBER_KEY = "core6"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"

PACKET_K = 24
SHIFT = "A"
STEPS = 32
TOL = 1.0e-2
CHANNELS = ("value", "grad_x", "grad_y", "grad_z", "laplacian")
STORED_ROUTES = ("champion", "BMC9", "BMC11", "slc_operator_L12")
ROUTES = (MEMBER_KEY,) + STORED_ROUTES
ROUTE_LABEL = {
    MEMBER_KEY: JOINT_LABEL,
    "champion": r"band member $E_{6,2}$",
    "BMC9": "BMC9",
    "BMC11": "BMC11",
    "slc_operator_L12": "SLC q4 + L12",
}
ROUTE_COLOR = {
    MEMBER_KEY: "#7B3294",
    "champion": "#0072B2",
    "BMC9": "#3A3A3A",
    "BMC11": "#000000",
    "slc_operator_L12": "#D55E00",
}
PANELS = (("(a) value", "value"),
          ("(b) gradient, worst component", "gradient"),
          ("(c) Laplacian", "laplacian"))


def _sweep() -> dict:
    """Return the worst channel after M = 32 steps against grid size."""
    out: dict = {}
    for path, keep in ((RUNS, STORED_ROUTES), (MEMBER_RUNS, (MEMBER_KEY,))):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if (row["kind"] != "accuracy" or int(row["K"]) != PACKET_K
                    or row["shift"] != SHIFT or row["route"] not in keep):
                continue
            worst = max(float(row[f"{channel}_M{STEPS}"]) for channel in CHANNELS)
            out.setdefault(row["route"], {})[int(row["N"])] = worst
    return out


def _history(name: str, npts: int) -> dict:
    """Return the step axis and the three plotted channels of one history."""
    folder = MEMBER_HISTORIES if name == MEMBER_KEY else HISTORIES
    path = folder / f"{name}_K{PACKET_K}_{SHIFT}_N{npts}_M{STEPS}.npz"
    with np.load(path) as data:
        return {
            "step": data["step"],
            "value": data["value"],
            "gradient": np.maximum.reduce(
                [data["grad_x"], data["grad_y"], data["grad_z"]]),
            "laplacian": data["laplacian"],
            "mass": np.abs(data["mass"]),
        }


def _grids_and_times() -> tuple[dict, dict]:
    """Return the minimum admissible grid and the mission time of every route."""
    amendment = json.loads(DECISION.read_text())["amendment_1"]
    stored = amendment["minimum_admissible_N_per_shift"]
    later = json.loads(MEMBER_DECISION.read_text())
    repeated = later["minimum_admissible_N_fine"]
    grids = {}
    for name in STORED_ROUTES:
        grids[name] = int(stored[f"{name}|{PACKET_K}|{SHIFT}|{STEPS}"])
        held = int(repeated[f"{name}|{PACKET_K}|{STEPS}"])
        if held != grids[name]:
            raise RuntimeError(
                f"{name} minimum admissible grid drift: {held} against {grids[name]}")
    grids[MEMBER_KEY] = int(
        later["member_minimum_N_per_shift_fine"][
            f"{MEMBER_KEY}|{PACKET_K}|{SHIFT}|{STEPS}"])

    times = {}
    for row in later["timing_rows"]:
        if (row["route"] not in ROUTES or int(row["K"]) != PACKET_K
                or row["shift"] != SHIFT or int(row["steps"]) != STEPS):
            continue
        if int(row["N"]) != grids[row["route"]]:
            raise RuntimeError(
                f"{row['route']} timing grid {row['N']} is not its minimum "
                f"admissible grid {grids[row['route']]}")
        times[row["route"]] = float(row["wall_seconds"])
    missing = [name for name in ROUTES if name not in times]
    if missing:
        raise RuntimeError(f"no mission time for {missing}")
    return grids, times


def main() -> int:
    grids, times = _grids_and_times()
    histories = {name: _history(name, grids[name]) for name in ROUTES}
    sweep = _sweep()

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 11.6,
        "axes.labelsize": 11.6,
        "xtick.labelsize": 11.6,
        "ytick.labelsize": 11.6,
    })
    fig = plt.figure(figsize=(7.9, 5.0))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.04], hspace=0.72,
                            wspace=1.45, top=0.815, bottom=0.085, left=0.075,
                            right=0.985)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6])]
    ax_sweep = fig.add_subplot(grid[1, 0:3])
    ax_cost = fig.add_subplot(grid[1, 3:6])

    for ax, (tag, key) in zip(axes, PANELS):
        for name in ROUTES:
            record = histories[name]
            ax.semilogy(record["step"], np.maximum(record[key], 1.0e-18),
                        color=ROUTE_COLOR[name], lw=1.3)
        ax.axhline(TOL, color="0.55", lw=0.7, ls=(0, (4, 2.5)))
        ax.set_xlabel("step")
        ax.set_xlim(0, STEPS)
        ax.set_xticks([0, 16, 32])
        ax.set_ylim(2.0e-6, 4.0e-2)
        ax.grid(axis="y", color="0.93", lw=0.5)
        ph.quiet_axis(ax)
        ax.set_title(tag, loc="left", fontsize=11.6, color=ph.CLASSY_ACCENT,
                     fontstyle="italic", pad=3.0)
    axes[0].set_ylabel(r"relative $L^2$ error")

    ladder = sorted({n for values in sweep.values() for n in values})
    for name in ROUTES:
        points = sorted(sweep[name])
        ax_sweep.loglog(points, [sweep[name][n] for n in points],
                        color=ROUTE_COLOR[name], lw=1.3, marker="o", ms=3.0,
                        label=ROUTE_LABEL[name])
    ax_sweep.axhline(TOL, color="0.55", lw=0.7, ls=(0, (4, 2.5)))
    ax_sweep.set_xlabel(r"$N$")
    ax_sweep.set_ylabel("worst channel error")
    ax_sweep.set_xticks(ladder)
    ax_sweep.set_xticklabels([str(n) for n in ladder])
    ax_sweep.xaxis.set_minor_locator(NullLocator())
    ax_sweep.xaxis.set_minor_formatter(NullFormatter())
    ax_sweep.set_xlim(ladder[0] / 1.1, ladder[-1] * 1.1)
    # Pinned so the larger tick font cannot thin the decade ladder.
    ax_sweep.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax_sweep.grid(axis="y", color="0.93", lw=0.5)
    ph.quiet_axis(ax_sweep)
    for index, label in enumerate(ax_sweep.get_xticklabels()):
        label.set_visible(index in (0, 3, 5, len(ladder) - 1))
    ax_sweep.set_title(r"(d) worst channel after $M = 32$ steps against $N$",
                       loc="left", fontsize=11.6, color=ph.CLASSY_ACCENT,
                       fontstyle="italic", pad=3.0)

    for slot, name in enumerate(ROUTES):
        y = len(ROUTES) - 1 - slot
        ax_cost.barh(y, grids[name], height=0.58, color=ROUTE_COLOR[name],
                     linewidth=0.0, zorder=3)
        ax_cost.text(grids[name] + 3.0, y, f"{times[name]:.1f} s",
                     fontsize=11.6, color=ph.CLASSY_ACCENT, ha="left",
                     va="center", zorder=4)
    ax_cost.set_yticks(range(len(ROUTES)))
    ax_cost.set_yticklabels([ROUTE_LABEL[name] for name in reversed(ROUTES)])
    ax_cost.set_ylim(-0.62, len(ROUTES) - 0.38)
    ax_cost.set_xlim(0.0, 178.0)
    ax_cost.set_xticks([0, 32, 64, 96, 128])
    ax_cost.set_xlabel(r"minimum admissible $N$")
    ax_cost.grid(axis="x", color="0.93", lw=0.5, zorder=0)
    ph.quiet_axis(ax_cost)
    ax_cost.set_title("(e) minimum admissible grid,\nwith the mission time",
                      loc="left", fontsize=11.6, color=ph.CLASSY_ACCENT,
                      fontstyle="italic", pad=3.0)

    handles = [Line2D([], [], color=ROUTE_COLOR[name], lw=1.6,
                      label=ROUTE_LABEL[name]) for name in ROUTES]
    handles.append(Line2D([], [], color="0.55", lw=0.7, ls=(0, (4, 2.5)),
                          label="one percent tolerance"))
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
               fontsize=11.6, handslcngth=1.7, columnspacing=1.1,
               handletextpad=0.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_semilagrangian.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"semi-Lagrangian mission at K = {PACKET_K}, shift {SHIFT}, "
          f"M = {STEPS}:")
    for name in ROUTES:
        record = histories[name]
        print(f"  {name:18s} minimum admissible N = {grids[name]:4d}, "
              f"mission {times[name]:7.3f} s, "
              f"worst mass drift {record['mass'].max():.2e}")
    for name in ROUTES:
        rendered = ", ".join(f"N={n}: {sweep[name][n]:.3e}"
                             for n in sorted(sweep[name]))
        print(f"  {name:18s} {rendered}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
