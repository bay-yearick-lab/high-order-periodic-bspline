# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Band error against formal order at a fixed conditioning budget.

Each member E_{m,r}(omega) of the family carries two dials: the number m of
zero exponents, which fixes the formal first-derivative order, and the r
conjugate pairs, which place exact symbol contacts inside the band without
raising the order. Holding the left-side condition number at a comparator's
value and minimizing the band objective over omega produces one optimum per
cell (m, r), and the figure places those optima against m.

The two dials are made readable by connecting the optima twice: a solid thin
line joins the cells that share a pair count, which is the polynomial dial,
and a dashed thin line joins the cells that share a polynomial core, which is
the frequency dial. Every cell is labeled. Filled markers satisfy every Gate 1
constraint, open markers are the threshold class that violates the declared
formal-order floor. Published and optimized comparators are placed at their own
formal order.

At the SLC q4 budget every cell except (m, r) = (4, 3) is an interior optimum,
so raising the budget to the degree-eleven rung's condition number moves only
that one cell; the moved cell is drawn and labeled rather than the whole second
series, which is identical.

Two files are written: fig_ef_two_dial.pdf carries the first-derivative channel
for the main article, and fig_ef_two_dial_d2.pdf carries the second-derivative
channel for the supplement. Every plotted value is read from the tracked Gate 1
ladder receipt results/exponential_family/gate1/ladder.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402

LADDER = ROOT / "results" / "exponential_family" / "gate1" / "ladder.json"
OUTDIR = ROOT / "figures_paper"

BASE_BUDGET = "C_SLC1"
WIDE_BUDGET = "C_BMC11"
FAMILY_COLOR = "#0072B2"
WIDE_COLOR = "#009E73"
PAIR_MARKER = {1: "o", 2: "s", 3: "D"}

# Offsets in points for the per-cell labels, keyed by (m, r).
CELL_LABEL_OFFSET = {
    (4, 1): (-7.5, 6.0),
    (4, 2): (-7.5, 6.0),
    (4, 3): (-7.5, 6.0),
    (6, 1): (-7.5, 6.0),
    (6, 2): (6.5, 0.0),
    (8, 1): (6.5, 0.0),
}
COMPARATOR_STYLE = {
    "SLC1992": ("SLC q4", "#D55E00", "*", 11.0, (7.5, 0.0)),
    "TunedQ8": ("tuned q8", "#CC79A7", "P", 7.0, (7.0, -8.0)),
    "BMC7": ("BMC7", "#777777", "x", 7.0, (7.0, 4.0)),
    "BMC9": ("BMC9", "#3A3A3A", "+", 8.0, (7.0, 3.0)),
    "CW12A": ("CW12A", "#56B4E9", "^", 6.5, (7.0, 3.0)),
    "CW12B": ("CW12B", "#8C510A", "v", 6.5, (7.0, 5.0)),
    "BMC11": ("BMC11", "#000000", "1", 9.5, (7.0, -8.0)),
}
CHANNEL = {
    "e1_max_B0": ("fig_ef_two_dial.pdf", r"$\max_{I_0}|e_1|$"),
    "e2_max_B0": ("fig_ef_two_dial_d2.pdf", r"$\max_{I_0}|e_2|$"),
}


def cells(ladder: list[dict], budget: str) -> dict:
    """Return the B0 optima of one budget, keyed by the cell (m, r)."""
    out = {}
    for entry in ladder:
        if (entry["band"] != "B0" or entry["status"] != "ok"
                or entry["budget_name"] != budget):
            continue
        out[(entry["m"], entry["r"])] = entry
    return out


def draw(key: str, base: dict, wide: dict, named: dict) -> Path:
    """Render one channel and return the file written."""
    filename, ylabel = CHANNEL[key]
    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 10.6,
        "axes.labelsize": 10.6,
        "xtick.labelsize": 10.6,
        "ytick.labelsize": 10.6,
    })
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.2))

    def x_of(m: int) -> float:
        return m - 0.13

    # thin connectors: solid along a fixed pair count, dashed along a fixed core
    for pair_count in sorted({r for _, r in base}):
        cores = sorted(m for m, r in base if r == pair_count)
        if len(cores) > 1:
            ax.semilogy([x_of(m) for m in cores],
                        [base[(m, pair_count)]["metrics"][key] for m in cores],
                        color=FAMILY_COLOR, lw=0.8, ls="-", alpha=0.55,
                        zorder=1)
    for core in sorted({m for m, _ in base}):
        counts = sorted(r for m, r in base if m == core)
        if len(counts) > 1:
            ax.semilogy([x_of(core)] * len(counts),
                        [base[(core, r)]["metrics"][key] for r in counts],
                        color=FAMILY_COLOR, lw=0.8, ls=(0, (3.0, 2.0)),
                        alpha=0.55, zorder=1)

    for (core, pair_count), entry in sorted(base.items()):
        value = entry["metrics"][key]
        clear = not entry["violations"]
        ax.semilogy([x_of(core)], [value], marker=PAIR_MARKER[pair_count],
                    ms=6.0, mfc=FAMILY_COLOR if clear else "none",
                    mec=FAMILY_COLOR, mew=1.1, ls="none", zorder=3)
        dx, dy = CELL_LABEL_OFFSET[(core, pair_count)]
        ax.annotate(rf"$\mathcal{{E}}_{{{core},{pair_count}}}$",
                    (x_of(core), value), textcoords="offset points",
                    xytext=(dx, dy), fontsize=10.6, color=FAMILY_COLOR,
                    ha="right" if dx < 0 else "left", va="center", zorder=4)

    moved = [cell for cell in base
             if abs(wide[cell]["metrics"][key] - base[cell]["metrics"][key])
             > 1.0e-4 * base[cell]["metrics"][key]]
    for cell in moved:
        core, pair_count = cell
        value = wide[cell]["metrics"][key]
        ax.semilogy([x_of(core)], [value], marker=PAIR_MARKER[pair_count],
                    ms=6.0, mfc="none", mec=WIDE_COLOR, mew=1.1, ls="none",
                    zorder=3)
        ax.semilogy([x_of(core)] * 2,
                    [value, base[cell]["metrics"][key]], color=WIDE_COLOR,
                    lw=0.8, ls=(0, (1.0, 1.6)), zorder=1)
        ax.annotate(rf"$\mathcal{{E}}_{{{core},{pair_count}}}$ at the wider budget",
                    (x_of(core), value), textcoords="offset points",
                    xytext=(7.5, 0.0), fontsize=10.6, color=WIDE_COLOR,
                    ha="left", va="center", zorder=4)

    for name, (label, color, marker, size, offset) in COMPARATOR_STYLE.items():
        row = named[name]
        ax.semilogy([row["order_d1"] + 0.13], [row[key]], marker=marker,
                    ms=size, color=color,
                    mfc="none" if marker in "ov^s" else color, mew=1.2,
                    ls="none", zorder=3)
        ax.annotate(label, (row["order_d1"] + 0.13, row[key]),
                    textcoords="offset points", xytext=offset, fontsize=10.6,
                    color=color, va="center", ha="left", zorder=4)

    ax.set_xlabel("formal first-derivative order of the member")
    ax.set_ylabel(ylabel)
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.set_xlim(3.3, 13.9)
    ax.set_ylim(1.2e-5, 3.0e-2)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax)

    handles = [
        Line2D([], [], color=FAMILY_COLOR, marker="o", ls="none", ms=6.0,
               label="optimum at the SLC q4 budget"),
        Line2D([], [], color=WIDE_COLOR, marker="D", ls="none", ms=6.0,
               mfc="none", label="the one cell the wider budget moves"),
        Line2D([], [], color="0.3", marker="o", ls="none", ms=6.2, mfc="none",
               label="threshold class"),
        Line2D([], [], color=FAMILY_COLOR, lw=0.9, ls="-", alpha=0.6,
               label="fixed pair count"),
        Line2D([], [], color=FAMILY_COLOR, lw=0.9, ls=(0, (3.0, 2.0)),
               alpha=0.6, label="fixed polynomial core"),
    ]
    ph.make_overhead_legend(fig, handles, [h.get_label() for h in handles],
                            ncol=3, bbox=(0.5, 1.005), rect=(0, 0, 1, 0.85),
                            fontsize=10.6, handslcngth=1.6, columnspacing=1.2,
                            handletextpad=0.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def main() -> int:
    record = json.loads(LADDER.read_text())
    base = cells(record["ladder"], BASE_BUDGET)
    wide = cells(record["ladder"], WIDE_BUDGET)
    named = record["comparators"]

    print("ladder optima on B0 (from gate1/ladder.json):")
    for (core, pair_count), entry in sorted(base.items()):
        flag = "clear" if not entry["violations"] else "threshold"
        other = wide[(core, pair_count)]["metrics"]["e1_max_B0"]
        print(f"  m={core} r={pair_count}: "
              f"e1_max_B0={entry['metrics']['e1_max_B0']:.6e} ({flag}), "
              f"wider budget {other:.6e}")
    for key in CHANNEL:
        print(f"wrote {draw(key, base, wide, named)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
