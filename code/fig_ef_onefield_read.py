# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Reading value, slope and curvature between the nodes from one field.

Panel (a) sketches a query at x = (j + xi) h with xi = 0.35, inside the cell
[x_j, x_{j+1}]. Panels (b), (c) and (d) draw the coefficients that query
touches: psi(x/h - k), psi'(x/h - k)/h and psi''(x/h - k)/h^2 over the same ten
indices k = j - 4, ..., j + 5, for the dominance member
results/exponential_family/gate8/members/X9_m6_C_SLC1_SLC1992_6IIq2.json.
The tap at k = j + 5 carries the width-ten support: at a node it vanishes and
nine coefficients are read, at a generic point ten are.

Panel (e) draws the off-grid transfers of the band member
results/exponential_family/members/E62_champion.json at theta = 0.6 pi as
functions of the query offset xi in [-1/2, 1/2]: the deviations |T_0 - 1|,
|T_1 - 1| and |T_2 - 1|, and the two coherence residuals of the off-grid
transfer proposition,

    |T_1 - (i theta)^{-1} d_xi (e^{i theta xi} T_0) e^{-i theta xi}|,
    |T_2 + theta^{-2} d_xi^2 (e^{i theta xi} T_0) e^{-i theta xi}|,

with the xi derivatives taken spectrally from the value channel alone. The
residuals sit at the rounding floor, so the three readings are three readings of
one differentiable function rather than three separate approximations.

The coefficients come from general_evaluator.GeneralEvaluator and the transfers
from family.ExponentialGenerator.offgrid_transfers; the integer samples the
evaluator returns are checked against the samples stored in the member record
before the figure is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import family as fam  # noqa: E402
import general_evaluator as ge  # noqa: E402
import plotting_helpers as ph  # noqa: E402

BAND_MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
DOMINANCE = (ROOT / "results" / "exponential_family" / "gate8" / "members"
             / "X9_m6_C_SLC1_SLC1992_6IIq2.json")
OUTDIR = ROOT / "figures_paper"

C_VALUE = "#7B3294"
C_SLOPE = "#0072B2"
C_CURV = "#D55E00"
C_T0 = "#7B3294"
C_T1 = "#0072B2"
C_T2 = "#D55E00"
C_COH = "#4A4A4A"
C_CELL = "#F1F3F6"

BASE_FONT = 11.1
OFFSET = 0.35
THETA = 0.6 * np.pi
XI_POINTS = 256
SAMPLE_TOLERANCE = 1.0e-12
FLOOR = 1.0e-17


def _stem_row(ax, taps, values, color, label, lo, hi, ticks) -> None:
    """Draw one row of coefficient stems over the tap indices."""
    ax.axhline(0.0, color="0.62", lw=0.6)
    for tap, value in zip(taps, values):
        ax.plot([tap, tap], [0.0, value], color=color, lw=1.2, zorder=5)
    ax.plot(taps, values, ls="none", marker="o", ms=5.0, mfc="white", mec=color,
            mew=1.3, zorder=6)
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.grid(axis="y", color="0.93", lw=0.5)
    ax.text(0.012, 0.90, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=BASE_FONT, color=color)
    ph.quiet_axis(ax)


def _coherence(generator, theta: float, xi: np.ndarray):
    """Return the two coherence residuals from spectral xi derivatives of T_0."""
    uniform = -0.5 + np.arange(XI_POINTS) / XI_POINTS
    T0, T1, T2 = generator.offgrid_transfers(np.array([theta]), uniform)
    T0, T1, T2 = T0[0], T1[0], T2[0]
    head = max(generator.required_K(q) for q in (0, 1, 2))
    order = np.fft.fftfreq(XI_POINTS, d=1.0 / XI_POINTS)
    wavenumbers = 2.0 * np.pi * 1j * order
    spectrum = np.fft.fft(T0)
    spectrum[np.abs(order) > head] = 0.0
    first = np.fft.ifft(wavenumbers * spectrum)
    second = np.fft.ifft(wavenumbers ** 2 * spectrum)
    slope_residual = np.abs(T1 - T0 - first / (1j * theta))
    curve_residual = np.abs(T2 - T0 + 2j * first / theta + second / theta ** 2)
    return uniform, slope_residual, curve_residual


def main() -> int:
    dom_record = json.loads(DOMINANCE.read_text())
    band_record = json.loads(BAND_MEMBER.read_text())
    radius = int(dom_record["radius"])

    evaluator = ge.evaluator_from_record(dom_record)
    index = np.arange(-radius, radius + 1, dtype=float)
    drift = float(np.max(np.abs(evaluator.value(index)[radius:]
                                - np.asarray(dom_record["beta"], dtype=float))))
    if drift > SAMPLE_TOLERANCE:
        raise RuntimeError(f"integer sample drift {drift:.3e}")

    taps = np.arange(-(radius - 1), radius + 1, dtype=float)
    argument = OFFSET - taps
    value = evaluator.value(argument)
    slope = evaluator.d1(argument)
    curvature = evaluator.d2(argument)
    nodal = evaluator.value(-taps)

    band = fam.ExponentialGenerator(int(band_record["m"]), band_record["omegas"])
    xi = np.linspace(-0.5, 0.5, 513)
    T0, T1, T2 = band.offgrid_transfers(np.array([THETA]), xi)
    deviations = [np.abs(T0[0] - 1.0), np.abs(T1[0] - 1.0), np.abs(T2[0] - 1.0)]
    uniform, slope_residual, curve_residual = _coherence(band, THETA, xi)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    })
    fig = plt.figure(figsize=(7.6, 4.95))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.08, 1.0],
                            height_ratios=[0.62, 1.0, 1.0, 1.0], hspace=0.30,
                            wspace=0.26)
    ax_cell = fig.add_subplot(grid[0, 0])
    ax_value = fig.add_subplot(grid[1, 0], sharex=ax_cell)
    ax_slope = fig.add_subplot(grid[2, 0], sharex=ax_cell)
    ax_curve = fig.add_subplot(grid[3, 0], sharex=ax_cell)
    ax_transfer = fig.add_subplot(grid[:, 1])

    ax_cell.axvspan(0.0, 1.0, color=C_CELL, lw=0.0, zorder=0)
    ax_cell.axhline(0.0, color="0.55", lw=0.9)
    ax_cell.plot(taps, np.zeros(taps.shape), ls="none", marker="|", ms=7.0,
                 color="0.45", mew=1.0)
    ax_cell.plot([OFFSET], [0.0], ls="none", marker="o", ms=6.0, color=C_VALUE,
                 zorder=6)
    ax_cell.text(OFFSET, 0.42, r"$x=(j+\xi)h$", ha="center", va="bottom",
                 fontsize=BASE_FONT, color=C_VALUE)
    ax_cell.text(0.5, -2.35, r"cell $[x_j,\,x_{j+1}]$", ha="center", va="bottom",
                 fontsize=BASE_FONT, color="0.35")
    ax_cell.set_ylim(-2.5, 1.6)
    ax_cell.set_yticks([])
    ax_cell.set_title("(a) one cell and its ten taps", fontsize=BASE_FONT,
                      color=ph.CLASSY_ACCENT, pad=3.0)
    for spine in ("left", "bottom"):
        ax_cell.spines[spine].set_visible(False)
    ax_cell.tick_params(axis="both", length=0.0)

    _stem_row(ax_value, taps, value, C_VALUE, r"(b) $\psi(x/h-k)$", -0.10, 0.60,
              [0.0, 0.25, 0.5])
    _stem_row(ax_slope, taps, slope, C_SLOPE, r"(c) $\psi'(x/h-k)/h$", -0.45, 0.55,
              [-0.25, 0.0, 0.25])
    _stem_row(ax_curve, taps, curvature, C_CURV, r"(d) $\psi''(x/h-k)/h^2$",
              -0.72, 0.64, [-0.5, 0.0, 0.5])
    ax_value.plot(taps, nodal, ls="none", marker="+", ms=6.0, color="0.45",
                  mew=1.0, zorder=7)
    ax_value.annotate("tenth tap", xy=(taps[-1], value[-1]),
                      xytext=(2.05, 0.40), fontsize=BASE_FONT, color="0.35",
                      ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7,
                                      shrinkA=1.0, shrinkB=3.0))
    for ax in (ax_cell, ax_value, ax_slope):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_curve.set_xlabel(r"$k-j$")
    ax_cell.set_xlim(taps[0] - 0.7, taps[-1] + 0.7)
    ax_curve.set_xticks([int(value) for value in taps])

    for values, color, style, width, label in (
            (deviations[0], C_T0, "-", 1.8, r"$|T_0-1|$"),
            (deviations[1], C_T1, (0, (5, 2)), 1.5, r"$|T_1-1|$"),
            (deviations[2], C_T2, (0, (1, 1.6)), 1.6, r"$|T_2-1|$")):
        ax_transfer.semilogy(xi, np.maximum(values, FLOOR), color=color, lw=width,
                             ls=style, label=label)
    ax_transfer.semilogy(uniform, np.maximum(slope_residual, FLOOR), color=C_COH,
                         lw=1.1, label="coherence residual")
    ax_transfer.semilogy(uniform, np.maximum(curve_residual, FLOOR), color=C_COH,
                         lw=1.1)
    ax_transfer.set_xlim(-0.5, 0.5)
    ax_transfer.set_ylim(FLOOR, 3.0e-4)
    ax_transfer.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax_transfer.set_yticks([1.0e-16, 1.0e-13, 1.0e-10, 1.0e-7, 1.0e-4])
    ax_transfer.set_xlabel(r"$\xi$")
    ax_transfer.set_ylabel("transfer deviation")
    ax_transfer.grid(axis="y", color="0.92", lw=0.5)
    ax_transfer.set_title(r"(e) transfers at $\theta=0.6\pi$", fontsize=BASE_FONT,
                          color=ph.CLASSY_ACCENT, pad=3.0)
    ph.quiet_axis(ax_transfer)
    ax_transfer.text(-0.47, 1.4e-16, "coherence residual", ha="left", va="bottom",
                     fontsize=BASE_FONT, color=C_COH)

    handles = [
        Line2D([], [], color=C_T0, lw=1.8),
        Line2D([], [], color=C_T1, lw=1.5, ls=(0, (5, 2))),
        Line2D([], [], color=C_T2, lw=1.6, ls=(0, (1, 1.6))),
        Line2D([], [], color="0.45", ls="none", marker="+", ms=6.0, mew=1.0),
    ]
    labels = [r"$|T_0-1|$", r"$|T_1-1|$", r"$|T_2-1|$", "nodal tap"]
    ph.make_overhead_legend(fig, handles, labels, ncol=4, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.91), fontsize=BASE_FONT,
                            handslcngth=2.0, columnspacing=1.4,
                            handletextpad=0.45)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_onefield_read.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("one-field reading summary:")
    print(f"  query offset xi = {OFFSET}; taps k - j = "
          f"{int(taps[0])} to {int(taps[-1])}; sample drift {drift:.2e}")
    print(f"  tenth tap psi({OFFSET - taps[-1]:.2f}) = {value[-1]:.6e}, "
          f"nodal value psi({-taps[-1]:.0f}) = {nodal[-1]:.3e}")
    print(f"  sum of the value taps {float(np.sum(value)):.12f}, "
          f"slope taps {float(np.sum(slope)):.3e}, "
          f"curvature taps {float(np.sum(curvature)):.3e}")
    print(f"  at theta = 0.6 pi: max |T_0 - 1| {deviations[0].max():.3e}, "
          f"max |T_1 - 1| {deviations[1].max():.3e}, "
          f"max |T_2 - 1| {deviations[2].max():.3e}")
    print(f"  coherence residuals: slope {slope_residual.max():.3e}, "
          f"curvature {curve_residual.max():.3e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
