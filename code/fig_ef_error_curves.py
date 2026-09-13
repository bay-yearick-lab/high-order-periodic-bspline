# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Nodal symbol error of the two members against three comparators.

For a periodic operator on the uniform grid the sampled first and second
derivative symbols are kappa_1(theta) and kappa_2(theta), and the relative
symbol errors are

    e_1(theta) = kappa_1(theta) / theta      - 1,
    e_2(theta) = kappa_2(theta) / theta^2    - 1.

Both panels draw |e_1| and |e_2| on (0, pi] for the joint member, for the band
champion E_{6,2}(omega_1, omega_2), for the SLC 1992 fourth-order compact
pair in its published form, that is his first derivative beside his own tuned
second derivative, for the tuned eighth-order pair and for the polynomial rung
BMC11 = E_{12,0}, together with the three band edges and the three tolerance
levels of the design. Five curves per panel keep every crossing readable; the
C^2 record carries the same shape as the joint member and is left out.

The curves are recomputed from the symbol functions, the joint member through
gate8_family.from_record on its member record and the champion through
family.ExponentialGenerator. The band suprema of the three comparators and of
the champion are then checked against the stored receipt
results/exponential_family/gate1/champions.json, and those of the joint member
against results/exponential_family/gate9/battery.json, before the figure is
written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import family as fam  # noqa: E402
import gate8_family as g8  # noqa: E402
import metrics as met  # noqa: E402
import plotting_helpers as ph  # noqa: E402

EF = ROOT / "results" / "exponential_family"
CHAMPIONS = EF / "gate1" / "champions.json"
BATTERY = EF / "gate9" / "battery.json"
JOINT_MEMBER = EF / "gate8" / "members" / "X9_m6_C_SLC1_SLC1992_6IIq2.json"
BATTERY_ROW = "X9_m6_C_SLC1_SLC1992_6IIq2"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"

C_CHAMPION = "#0072B2"
C_SLC = "#D55E00"
C_Q8 = "#CC79A7"
C_JOINT = "#7B3294"
C_BMC11 = "#000000"
C_GUIDE = "0.62"

FINE_TOL = 1.0e-14
POINTS = 20001
RELATIVE_TOLERANCE = 1.0e-6

TOLERANCE_LABELS = ((1.0e-2, r"$10^{-2}$"), (1.0e-3, r"$10^{-3}$"),
                    (1.0e-4, r"$10^{-4}$"))


def _curves(record: dict):
    """Return the five error-curve entries of the comparison."""
    champion = record["champions"]["B0"]["C_SLC1"]["champion"]
    generator = fam.ExponentialGenerator(int(champion["m"]),
                                         tuple(champion["omegas"]))
    joint = g8.from_record(json.loads(JOINT_MEMBER.read_text()))
    bmc11 = fam.ExponentialGenerator(12, ())
    entries = [
        (JOINT_LABEL, C_JOINT, 1.8, "joint",
         lambda t: joint.symbols(t)[1],
         lambda t: joint.symbols(t)[2]),
        ("band member $E_{6,2}$", C_CHAMPION, 1.5, "champion",
         lambda t: generator.symbols(t, FINE_TOL)[1],
         lambda t: generator.symbols(t, FINE_TOL)[2]),
        ("SLC q4", C_SLC, 1.1, "SLC1992",
         comp.compact_kappa1(comp.SLC_D1["alpha"], comp.SLC_D1["a"]),
         comp.compact_kappa2(comp.SLC_D2["alpha"], comp.SLC_D2["a0"],
                             comp.SLC_D2["a"])),
        ("tuned q8", C_Q8, 1.1, "TunedQ8",
         comp.compact_kappa1(comp.Q8_D1["alpha"], comp.Q8_D1["a"]),
         comp.compact_kappa2(comp.Q8_D2["alpha"], comp.Q8_D2["a0"],
                             comp.Q8_D2["a"])),
        ("BMC11", C_BMC11, 1.1, "BMC11",
         lambda t: bmc11.symbols(t, FINE_TOL)[1],
         lambda t: bmc11.symbols(t, FINE_TOL)[2]),
    ]
    return champion, entries


def _stored(record: dict, key: str, field: str) -> float:
    """Return one stored band supremum of the receipts."""
    if key == "champion":
        return float(record["champions"]["B0"]["C_SLC1"]["champion"]["metrics"][field])
    if key == "joint":
        battery = {row["label"]: row
                   for row in json.loads(BATTERY.read_text())["results"]}
        return float(battery[BATTERY_ROW]["metrics"][field])
    return float(record["comparators"][key][field])


def main() -> int:
    record = json.loads(CHAMPIONS.read_text())
    bands = record["conventions"]["bands"]
    champion, entries = _curves(record)

    theta = met.theta_grid(POINTS)
    receipt = []
    errors = {}
    for label, color, width, key, kappa1, kappa2 in entries:
        e1 = np.asarray(kappa1(theta), dtype=float) / theta - 1.0
        e2 = np.asarray(kappa2(theta), dtype=float) / theta ** 2 - 1.0
        errors[label] = (color, width, e1, e2)
        for order, err in ((1, e1), (2, e2)):
            for band, edge in bands.items():
                fresh = met.band_max(theta, err, float(edge))
                stored = _stored(record, key, f"e{order}_max_{band}")
                if abs(fresh - stored) > RELATIVE_TOLERANCE * abs(stored):
                    raise RuntimeError(
                        f"{key} e{order}_max_{band} drift: {fresh} against {stored}")
                if band == "B0":
                    receipt.append((key, order, fresh, stored))

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 10.9,
        "axes.labelsize": 10.9,
        "xtick.labelsize": 10.9,
        "ytick.labelsize": 10.9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    edge_marks = ((bands["B0"], r"$I_0$"), (bands["B1"], r"$I_1$"),
                  (bands["B2"], r"$I_2$"))

    for ax, tag, slot in zip(axes, ("(a)", "(b)"), (2, 3)):
        for label, (color, width, e1, e2) in errors.items():
            err = e1 if slot == 2 else e2
            ax.semilogy(theta / np.pi, np.maximum(np.abs(err), 1.0e-13),
                        color=color, lw=width, label=label)
        for tol in (1.0e-2, 1.0e-3, 1.0e-4):
            ax.axhline(tol, color=C_GUIDE, lw=0.6, ls=(0, (4, 2.5)))
        for (edge, text), side in zip(edge_marks, ("right", "right", "left")):
            ax.axvline(edge / np.pi, color="0.70", lw=0.6, ls=(0, (1, 2.2)))
            nudge = -0.006 if side == "right" else 0.006
            ax.text(edge / np.pi + nudge, 1.6e-13, text, fontsize=10.9,
                    color="0.45", ha=side, va="bottom")
        for tol, text in TOLERANCE_LABELS:
            ax.text(0.012, tol * 1.35, text, fontsize=9.8, color=C_GUIDE,
                    ha="left", va="bottom")
        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0e-13, 2.0)
        ax.grid(axis="y", color="0.93", lw=0.5)
        ph.quiet_axis(ax)
        ax.text(0.015, 0.985, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.9, color=ph.CLASSY_ACCENT)
    axes[0].set_ylabel(r"$|e_1(\theta)|$")
    axes[1].set_ylabel(r"$|e_2(\theta)|$")
    axes[1].tick_params(labelleft=True)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = ph.make_overhead_legend(fig, handles, labels, ncol=5,
                                     bbox=(0.5, 1.005), rect=(0, 0, 1, 0.90),
                                     fontsize=10.9, handslcngth=1.9,
                                     columnspacing=1.2, handletextpad=0.45)
    for handle in legend.legend_handles:
        handle.set_linewidth(2.2)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_error_curves.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("band-B0 symbol suprema, recomputed against the stored receipts:")
    for key, order, fresh, stored in receipt:
        print(f"  {key:10s} e{order}: {fresh:.6e} against {stored:.6e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
