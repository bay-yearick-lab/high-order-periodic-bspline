# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""The degree shift: a Gram matrix of degree two is a sample matrix of degree five.

Panel (a) climbs the convolution ladder: the unit box beta_0, the hat
beta_1 = beta_0 * beta_0, and beta_2 = beta_1 * beta_0, three boxes in all, on
the support [-3/2, 3/2].

Panels (b) and (c) slide one copy of beta_2 against another at the shifts n = 0
and n = 2 and shade the integrand. The area under the shaded region is the
Gram entry, and it equals beta_5(n): the six boxes of the two degree-two
factors reassemble into the degree-five cardinal B-spline.

Panel (d) draws the overlap as a function of the shift together with beta_5 and
marks the integer samples. The Gram matrix of the degree-two basis is therefore
the sample matrix of the degree-five basis entry by entry, which is the
identity BMC5 = BMG2 between the two ladders.

The computation is the one behind the supplement figure fig_overlap_identity,
plot_overlap_identity in code/experiment_d1.py: the same
spline_analysis.centered_cardinal_bspline values and the same quadrature of the
sliding product.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting_helpers as ph  # noqa: E402
import spline_analysis as sa  # noqa: E402

OUTDIR = ROOT / "figures_paper"

C_BOX = "#6B6B6B"
C_HAT = "#009E73"
C_BASE = "#22517A"
C_SHIFT = "#B06A2C"
C_PRODUCT = "#6F8FAF"
C_TARGET = "#7B3294"

BASE_FONT = 11.1
QUADRATURE_POINTS = 4001
IDENTITY_TOLERANCE = 1.0e-6


def _overlap(shift: float, grid: np.ndarray, fixed: np.ndarray) -> float:
    """Return the quadrature of beta_2(t) beta_2(t - shift) on the grid."""
    return float(np.trapezoid(fixed * sa.centered_cardinal_bspline(2, grid - shift),
                              grid))


def _slide_panel(ax, grid, fixed, shift, tag) -> None:
    """Draw one sliding overlap with the integrand shaded and the area printed."""
    shifted = sa.centered_cardinal_bspline(2, grid - shift)
    product = fixed * shifted
    ax.fill_between(grid, 0.0, product, color=C_PRODUCT, alpha=0.38, lw=0.0)
    ax.plot(grid, fixed, color=C_BASE, lw=1.7)
    ax.plot(grid, shifted, color=C_SHIFT, lw=1.7, ls=(0, (5, 2)))
    area = _overlap(shift, grid, fixed)
    sample = float(sa.centered_cardinal_bspline(5, np.array([shift]))[0])
    if abs(area - sample) > IDENTITY_TOLERANCE:
        raise RuntimeError(f"overlap {area:.9f} against beta_5 {sample:.9f}")
    ax.text(0.975, 0.94,
            "area " + f"{area:.6f}" + "\n" + rf"$\beta_5({int(shift)})={sample:.6f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=BASE_FONT,
            color="0.25", linespacing=1.3)
    ax.axhline(0.0, color="0.70", lw=0.5)
    ax.set_xlim(-1.9, 3.9)
    ax.set_ylim(-0.03, 1.24)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.set_xlabel(r"$t$")
    ax.grid(axis="y", color="0.93", lw=0.5)
    ax.set_title(tag, fontsize=BASE_FONT, color=ph.CLASSY_ACCENT, pad=3.0)
    ph.quiet_axis(ax)


def main() -> int:
    grid = np.linspace(-2.6, 4.6, QUADRATURE_POINTS)
    fixed = sa.centered_cardinal_bspline(2, grid)

    ladder = np.linspace(-2.0, 2.0, 2001)
    box = sa.centered_cardinal_bspline(0, ladder)
    hat = sa.centered_cardinal_bspline(1, ladder)
    quadratic = sa.centered_cardinal_bspline(2, ladder)

    shifts = np.linspace(-3.0, 3.0, 601)
    overlap = np.array([_overlap(float(value), grid, fixed) for value in shifts])
    target = sa.centered_cardinal_bspline(5, shifts)
    identity = float(np.max(np.abs(overlap - target)))
    if identity > IDENTITY_TOLERANCE:
        raise RuntimeError(f"overlap identity deviation {identity:.3e}")
    nodes = np.arange(-3.0, 4.0)
    node_values = sa.centered_cardinal_bspline(5, nodes)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.15))
    ax_ladder, ax_zero = axes[0]
    ax_two, ax_curve = axes[1]
    fig.subplots_adjust(hspace=0.55, wspace=0.26)

    ax_ladder.axhline(0.0, color="0.70", lw=0.5)
    ax_ladder.plot(ladder, box, color=C_BOX, lw=1.5, ls=(0, (1, 1.6)),
                   label=r"$\beta_0$")
    ax_ladder.plot(ladder, hat, color=C_HAT, lw=1.5, ls=(0, (5, 2)),
                   label=r"$\beta_1=\beta_0*\beta_0$")
    ax_ladder.plot(ladder, quadratic, color=C_BASE, lw=1.9,
                   label=r"$\beta_2=\beta_1*\beta_0$")
    ax_ladder.set_xlim(-2.0, 2.0)
    ax_ladder.set_ylim(-0.04, 2.30)
    ax_ladder.set_xticks([-2, -1, 0, 1, 2])
    ax_ladder.set_yticks([0.0, 0.5, 1.0])
    ax_ladder.set_xlabel(r"$t$")
    ax_ladder.set_title("(a) three boxes give $\\beta_2$", fontsize=BASE_FONT,
                        color=ph.CLASSY_ACCENT, pad=3.0)
    ax_ladder.grid(axis="y", color="0.93", lw=0.5)
    ph.quiet_axis(ax_ladder)
    ph.quiet_legend(ax_ladder, loc="upper left", bbox_to_anchor=(0.0, 1.02))

    _slide_panel(ax_zero, grid, fixed, 0.0, "(b) the copies at shift $n=0$")
    _slide_panel(ax_two, grid, fixed, 2.0, "(c) the copies at shift $n=2$")

    ax_curve.axhline(0.0, color="0.70", lw=0.5)
    ax_curve.plot(shifts, overlap, color=C_BASE, lw=2.1,
                  label=r"$\int\beta_2(t)\beta_2(t-n)\,dt$")
    ax_curve.plot(shifts, target, color=C_TARGET, lw=1.3, ls=(0, (4, 2)),
                  label=r"$\beta_5(n)$, six boxes")
    ax_curve.plot(nodes, node_values, ls="none", marker="o", ms=5.4, mfc="white",
                  mec=C_SHIFT, mew=1.4, label="integer samples", zorder=6)
    ax_curve.set_xlim(-3.0, 3.0)
    ax_curve.set_ylim(-0.03, 1.58)
    ax_curve.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax_curve.set_yticks([0.0, 0.25, 0.5])
    ax_curve.set_xlabel(r"shift $n$")
    ax_curve.set_title("(d) the overlap is the degree-five sample",
                       fontsize=BASE_FONT, color=ph.CLASSY_ACCENT, pad=3.0)
    ax_curve.grid(axis="y", color="0.93", lw=0.5)
    ph.quiet_axis(ax_curve)
    ph.quiet_legend(ax_curve, loc="upper left", bbox_to_anchor=(0.0, 1.02))

    ax_ladder.set_ylabel("value")
    ax_two.set_ylabel("value")
    ax_zero.set_ylabel("value")
    ax_curve.set_ylabel("value")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_degree_shift.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("degree shift summary:")
    print(f"  quadrature points {QUADRATURE_POINTS}; largest deviation of the "
          f"overlap from beta_5 over {shifts.size} shifts {identity:.3e}")
    for shift in (0.0, 1.0, 2.0):
        area = _overlap(shift, grid, fixed)
        sample = float(sa.centered_cardinal_bspline(5, np.array([shift]))[0])
        print(f"  n = {int(shift)}: overlap {area:.9f}, beta_5 {sample:.9f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
