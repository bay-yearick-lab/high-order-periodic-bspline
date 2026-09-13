# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""The generator of the dominance member, cell by cell.

Panel (a) draws the three convolution factors on [-1, 1]: the unit box beta_0,
the hat beta_1 = beta_0 * beta_0, and the sine arch

    P_omega(x) = omega sin(omega (1 - |x|)) / (4 sin^2(omega / 2)),

the factor a conjugate pair at frequency omega contributes. The two arches are
drawn at the frequencies 0.5870 pi and 0.7757 pi of the dominance member
results/exponential_family/gate8/members/X9_m6_C_SLC1_SLC1992_6IIq2.json.
Each factor has unit mass and support of width one or two, so a member with
core m and r pairs has support of width m + 2 r.

Panel (b) draws psi of that member on [-5, 5]. The ten unit cells are shaded
alternately, the knots sit at the integers, and the stems carry the integer
samples. The inset magnifies the two negative samples at |q| = 3 and |q| = 4
and the vanishing samples at |q| = 5, which is why a node sees nine
coefficients while a generic point sees ten.

Panel (c) draws psi' and psi'' on the same cells, with their own integer
samples. All three readings come from one function, so the value, the slope and
the curvature between the nodes cannot drift apart.

The pointwise values come from general_evaluator.GeneralEvaluator, which folds
the Taylor jet of every root term into one polynomial per cell in 50-digit
arithmetic; the integer samples it returns are checked against the samples
stored in the member record before the figure is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import family as fam  # noqa: E402
import general_evaluator as ge  # noqa: E402
import plotting_helpers as ph  # noqa: E402
import spline_analysis as sa  # noqa: E402

DOMINANCE = (ROOT / "results" / "exponential_family" / "gate8" / "members"
             / "X9_m6_C_SLC1_SLC1992_6IIq2.json")
OUTDIR = ROOT / "figures_paper"

C_BOX = "#6B6B6B"
C_HAT = "#009E73"
C_ARCH1 = "#D55E00"
C_ARCH2 = "#0072B2"
C_PSI = "#7B3294"
C_D1 = "#0072B2"
C_D2 = "#D55E00"
C_CELL = "#F1F3F6"

BASE_FONT = 11.1
SAMPLE_TOLERANCE = 1.0e-12
MASS_TOLERANCE = 1.0e-12


def _cells(ax, radius: int, lo: float, hi: float) -> None:
    """Shade the unit cells alternately and draw the knots at the integers."""
    for left in range(-radius, radius):
        if (left + radius) % 2 == 0:
            ax.axvspan(left, left + 1, color=C_CELL, lw=0.0, zorder=0)
    for knot in range(-radius, radius + 1):
        ax.plot([knot, knot], [lo, hi], color="0.80", lw=0.5, zorder=1)
    ax.axhline(0.0, color="0.62", lw=0.6, zorder=2)


def _stems(ax, index, values, color, marker="o", size=4.6) -> None:
    """Draw the integer samples as stems on the zero line."""
    for position, value in zip(index, values):
        ax.plot([position, position], [0.0, value], color=color, lw=0.9, zorder=5)
    ax.plot(index, values, ls="none", marker=marker, ms=size, mfc="white",
            mec=color, mew=1.2, zorder=6)


def main() -> int:
    record = json.loads(DOMINANCE.read_text())
    omegas = [float(w) for w in record["omegas"]]
    radius = int(record["radius"])

    evaluator = ge.evaluator_from_record(record)
    index = np.arange(-radius, radius + 1, dtype=float)
    beta = evaluator.value(index)
    dbeta = evaluator.d1(index)
    ddbeta = evaluator.d2(index)
    half = slice(radius, 2 * radius + 1)
    drift = max(
        float(np.max(np.abs(beta[half] - np.asarray(record["beta"], dtype=float)))),
        float(np.max(np.abs(dbeta[half] - np.asarray(record["dbeta"], dtype=float)))),
        float(np.max(np.abs(ddbeta[half] - np.asarray(record["ddbeta"], dtype=float)))),
    )
    if drift > SAMPLE_TOLERANCE:
        raise RuntimeError(f"integer sample drift {drift:.3e}")

    masses = [fam.pair_kernel_mass(w) for w in omegas]
    if max(abs(value - 1.0) for value in masses) > MASS_TOLERANCE:
        raise RuntimeError("the sine arch does not carry unit mass")

    x_unit = np.linspace(-1.0, 1.0, 3001)
    box = sa.centered_cardinal_bspline(0, x_unit)
    hat = sa.centered_cardinal_bspline(1, x_unit)
    arches = [fam.pair_kernel(w, x_unit) for w in omegas]

    x = np.linspace(-radius, radius, 8001)
    psi = evaluator.value(x)
    psi1 = evaluator.d1(x)
    psi2 = evaluator.d2(x)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    })
    fig = plt.figure(figsize=(7.6, 5.75))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.95],
                            height_ratios=[1.0, 1.0], hspace=0.52, wspace=0.26)
    ax_factor = fig.add_subplot(grid[0, 0])
    ax_psi = fig.add_subplot(grid[0, 1])
    ax_tail = fig.add_subplot(grid[1, 0])
    ax_der = fig.add_subplot(grid[1, 1])

    ax_factor.axhline(0.0, color="0.62", lw=0.6)
    ax_factor.plot(x_unit, box, color=C_BOX, lw=1.6, ls=(0, (1, 1.6)),
                   label=r"box $\beta_0$")
    ax_factor.plot(x_unit, hat, color=C_HAT, lw=1.6, ls=(0, (5, 2)),
                   label=r"hat $\beta_1$")
    ax_factor.plot(x_unit, arches[0], color=C_ARCH1, lw=1.7,
                   label=r"arch at $\omega_1$")
    ax_factor.plot(x_unit, arches[1], color=C_ARCH2, lw=1.7,
                   label=r"arch at $\omega_2$")
    ax_factor.set_xlim(-1.08, 1.08)
    ax_factor.set_ylim(-0.06, 1.16)
    ax_factor.set_xticks([-1, 0, 1])
    ax_factor.set_yticks([0.0, 0.5, 1.0])
    ax_factor.set_xlabel(r"$x$")
    ax_factor.set_ylabel("factor")
    ax_factor.grid(axis="y", color="0.92", lw=0.5)
    ax_factor.set_title("(a) the three factors", fontsize=BASE_FONT,
                        color=ph.CLASSY_ACCENT, pad=3.0)
    ph.quiet_axis(ax_factor)

    lo_psi, hi_psi = -0.09, 0.56
    _cells(ax_psi, radius, lo_psi, hi_psi)
    ax_psi.plot(x, psi, color=C_PSI, lw=1.8)
    _stems(ax_psi, index, beta, C_PSI)
    ax_psi.set_xlim(-radius - 0.22, radius + 0.22)
    ax_psi.set_ylim(lo_psi, hi_psi)
    ax_psi.set_xticks(range(-radius, radius + 1))
    ax_psi.set_yticks([0.0, 0.2, 0.4])
    ax_psi.set_xlabel(r"$x$")
    ax_psi.set_ylabel(r"$\psi(x)$")
    ax_psi.set_title("(b) the field on its ten cells", fontsize=BASE_FONT,
                     color=C_PSI, pad=3.0)
    ph.quiet_axis(ax_psi)

    tail_mask = x >= 2.0
    tail_span = float(np.max(np.abs(psi[tail_mask])))
    tail_lo, tail_hi = -1.28 * tail_span, 1.28 * tail_span
    ax_psi.add_patch(Rectangle((1.9, -0.05), radius - 1.9 + 0.3, 0.10,
                               fill=False, edgecolor="0.55", lw=0.7, zorder=7))
    _cells(ax_tail, radius, tail_lo, tail_hi)
    ax_tail.plot(x[tail_mask], psi[tail_mask], color=C_PSI, lw=1.7)
    tail_index = index >= 2.0
    _stems(ax_tail, index[tail_index], beta[tail_index], C_PSI)
    ax_tail.set_xlim(1.86, radius + 0.30)
    ax_tail.set_ylim(tail_lo, tail_hi)
    ax_tail.set_xticks([2, 3, 4, 5])
    ax_tail.set_yticks([-0.01, 0.0, 0.01])
    ax_tail.set_xlabel(r"$x$")
    ax_tail.set_ylabel(r"$\psi(x)$")
    ax_tail.set_title("(c) the tail magnified", fontsize=BASE_FONT, color=C_PSI,
                      pad=3.0)
    ph.quiet_axis(ax_tail)

    lo_der, hi_der = -0.66, 0.44
    _cells(ax_der, radius, lo_der, hi_der)
    ax_der.plot(x, psi1, color=C_D1, lw=1.7, label=r"$\psi'(x)$")
    ax_der.plot(x, psi2, color=C_D2, lw=1.5, ls=(0, (5, 2)), label=r"$\psi''(x)$")
    _stems(ax_der, index, dbeta, C_D1, size=4.2)
    _stems(ax_der, index, ddbeta, C_D2, marker="s", size=4.0)
    ax_der.set_xlim(-radius - 0.22, radius + 0.22)
    ax_der.set_ylim(lo_der, hi_der)
    ax_der.set_xticks(range(-radius, radius + 1))
    ax_der.set_yticks([-0.6, -0.3, 0.0, 0.3])
    ax_der.set_xlabel(r"$x$")
    ax_der.set_ylabel(r"$\psi'$, $\psi''$")
    ax_der.set_title("(d) slope and curvature on the same cells",
                     fontsize=BASE_FONT, color=ph.CLASSY_ACCENT, pad=3.0)
    ph.quiet_axis(ax_der)

    handles, labels = ax_factor.get_legend_handles_labels()
    handles += [Line2D([], [], color=C_PSI, lw=1.8),
                Line2D([], [], color=C_D1, lw=1.7),
                Line2D([], [], color=C_D2, lw=1.5, ls=(0, (5, 2)))]
    labels += [r"$\psi$", r"$\psi'$", r"$\psi''$"]
    ph.make_overhead_legend(fig, handles, labels, ncol=4, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.87), fontsize=BASE_FONT,
                            handslcngth=2.0, columnspacing=1.3,
                            handletextpad=0.45)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_generator_cells.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    moment_0 = float(np.sum(beta))
    moment_1 = float(2.0 * np.sum(index[index > 0] * dbeta[index > 0]))
    moment_2 = float(np.sum(index[index > 0] ** 2 * ddbeta[index > 0]))
    print("generator summary:")
    print(f"  frequencies {omegas[0] / np.pi:.4f} pi, {omegas[1] / np.pi:.4f} pi; "
          f"arch masses {masses[0]:.12f}, {masses[1]:.12f}")
    print(f"  integer sample drift against the member record {drift:.2e}")
    print("  samples psi(q), q = 0..5: "
          + ", ".join(f"{value:.6f}" for value in beta[half]))
    print(f"  moment sums: sum psi = {moment_0:.12f}, "
          f"2 sum n psi' = {moment_1:.12f}, sum n^2 psi'' = {moment_2:.12f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
