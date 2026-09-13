# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Equioscillation: the levelled optimum, a perturbation, and the iteration.

Panel (a) draws the first-derivative error e_1 of E_{6,2} on the band
(0, 2 pi / 3] at the levelled optimum of
results/exponential_family/gate2/rule.json, frequencies 0.5856 pi and
0.6584 pi. The r + 1 = 3 alternation abscissae carry alternating signs and a
common magnitude E_max = 2.696e-5, drawn as the band +- E_max.

Panel (b) draws the same member with omega_1 moved up by 0.02 pi. The three
extremal magnitudes separate, the largest grows to 3.536e-5, and the equal
ripple is gone.

Panel (c) draws the levelling residual max_i (|e_1(x_i)| - E_max) / E_max of
the iteration in gate2_rule.py. The start is the weighted Chebyshev solution of
the one-alias model; two damped Newton steps on the r level conditions carry
the residual to 2.0e-10. The residual of the perturbed member of panel (b) is
drawn as the reference line.

The frequencies, the alternation abscissae and the iteration history are
recomputed here by gate2_rule.remez_frequencies and checked against the stored
record before the figure is written.
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

import gate2_rule as g2  # noqa: E402
import plotting_helpers as ph  # noqa: E402

RULE = ROOT / "results" / "exponential_family" / "gate2" / "rule.json"
OUTDIR = ROOT / "figures_paper"

C_OPT = "#0072B2"
C_PERT = "#D55E00"
C_ITER = "#7B3294"
C_GUIDE = "0.45"
C_BAND = "#E8EEF4"

BASE_FONT = 11.1
M_CORE = 6
R_PAIRS = 2
SHIFT = 0.02 * np.pi
SCALE = 1.0e5
RECORD_TOLERANCE = 1.0e-9


def _error_panel(ax, theta, values, points, point_values, level, color, title,
                 omegas, band_edge):
    """Draw one error curve with its alternation points and its levelled band."""
    ax.axhspan(-level * SCALE, level * SCALE, color=C_BAND, lw=0.0, zorder=0)
    for value in omegas:
        ax.axvline(value / np.pi, color="0.74", lw=0.6, ls=(0, (1, 2.2)))
    ax.axvline(band_edge / np.pi, color="0.35", lw=0.8)
    for line in (-level * SCALE, level * SCALE):
        ax.axhline(line, color=C_GUIDE, lw=0.8, ls=(0, (4, 2.4)))
    ax.axhline(0.0, color="0.78", lw=0.6)
    ax.plot(theta / np.pi, values * SCALE, color=color, lw=1.7)
    ax.plot(points / np.pi, point_values * SCALE, ls="none", marker="o", ms=5.4,
            mfc="white", mec=color, mew=1.4, zorder=6)
    for point, value in zip(points, point_values):
        sign = "$+$" if value > 0.0 else "$-$"
        offset = 0.62 if value > 0.0 else -0.62
        ax.text(point / np.pi, value * SCALE + offset, sign, ha="center",
                va="center", fontsize=BASE_FONT, color=color, zorder=7)
    ax.set_xlim(0.0, 0.70)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel(r"$\theta/\pi$")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.set_title(title, fontsize=BASE_FONT, color=color, pad=3.0)
    ph.quiet_axis(ax)


def main() -> int:
    stored = json.loads(RULE.read_text())["band_B0"]["m6_r2"]
    band_edge = float(stored["T"])

    result = g2.remez_frequencies(M_CORE, R_PAIRS, band_edge)
    omegas = np.asarray(result.omegas, dtype=float)
    drift = float(np.max(np.abs(omegas - np.asarray(stored["omegas"], dtype=float))))
    if drift > RECORD_TOLERANCE:
        raise RuntimeError(f"frequency drift {drift:.3e} against the stored record")

    perturbed = omegas.copy()
    perturbed[0] = perturbed[0] + SHIFT

    optimum_error = g2.ExactError(M_CORE, omegas)
    perturbed_error = g2.ExactError(M_CORE, perturbed)
    theta = np.linspace(band_edge / 4001.0, band_edge, 4001)
    optimum_values = optimum_error(theta)
    perturbed_values = perturbed_error(theta)

    perturbed_points, perturbed_at = g2.alternation_set(perturbed_error, perturbed,
                                                        band_edge)
    perturbed_residual = float(np.max(np.abs(
        g2.equioscillation_residual(perturbed_at))))
    perturbed_level = float(np.max(np.abs(perturbed_values)))

    start = np.asarray(result.start, dtype=float)
    _, start_values, start_residual = g2._state(
        g2.exact_error_factory(M_CORE), start, band_edge)
    residuals = [float(np.max(np.abs(start_residual)))]
    residuals += [float(step["residual"]) for step in result.history]
    steps = np.arange(len(residuals), dtype=float)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    })
    fig, (ax_opt, ax_pert, ax_iter) = plt.subplots(1, 3, figsize=(7.6, 3.45))

    _error_panel(ax_opt, theta, optimum_values, np.asarray(result.points),
                 np.asarray(result.values), result.level, C_OPT,
                 "(a) levelled optimum", omegas, band_edge)
    _error_panel(ax_pert, theta, perturbed_values, perturbed_points, perturbed_at,
                 perturbed_level, C_PERT, r"(b) $\omega_1$ moved by $0.02\pi$",
                 perturbed, band_edge)
    span = 1.45 * perturbed_level * SCALE
    for ax in (ax_opt, ax_pert):
        ax.set_ylim(-span, span)
    ax_opt.set_ylabel(r"$e_1(\theta)\times 10^{5}$")
    ax_pert.set_yticklabels([])

    ax_iter.semilogy(steps, residuals, color=C_ITER, lw=1.5, marker="o", ms=6.0,
                     mfc="white", mec=C_ITER, mew=1.5)
    ax_iter.axhline(perturbed_residual, color=C_PERT, lw=1.2, ls=(0, (4, 2.4)))
    ax_iter.text(len(residuals) - 1.05, perturbed_residual * 1.5,
                 r"$\omega_1+0.02\pi$", ha="right", va="bottom",
                 fontsize=BASE_FONT, color=C_PERT)
    ax_iter.set_xlim(-0.35, len(residuals) - 0.65)
    ax_iter.set_ylim(1.0e-11, 6.0)
    ax_iter.set_xticks(steps)
    ax_iter.set_xticklabels(["start"] + [str(int(value)) for value in steps[1:]])
    ax_iter.set_yticks([1.0e-10, 1.0e-7, 1.0e-4, 1.0e-1])
    ax_iter.set_xlabel("Newton step")
    ax_iter.set_ylabel("levelling residual")
    ax_iter.grid(axis="y", color="0.92", lw=0.5)
    ax_iter.set_title("(c) the iteration", fontsize=BASE_FONT, color=C_ITER,
                      pad=3.0)
    ph.quiet_axis(ax_iter)

    handles = [
        Line2D([], [], color="0.35", ls="none", marker="o", ms=5.4, mfc="white",
               mec="0.35", mew=1.4),
        Line2D([], [], color=C_GUIDE, lw=0.8, ls=(0, (4, 2.4))),
    ]
    labels = ["alternation point", r"$\pm\max|e_1|$"]
    ph.make_overhead_legend(fig, handles, labels, ncol=2, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.89), fontsize=BASE_FONT,
                            handslcngth=2.0, columnspacing=1.6,
                            handletextpad=0.5)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_equioscillation.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("equioscillation summary:")
    print(f"  optimum omegas {omegas[0] / np.pi:.6f} pi, {omegas[1] / np.pi:.6f} pi; "
          f"drift against the stored record {drift:.2e}")
    print(f"  alternation abscissae "
          + ", ".join(f"{p / np.pi:.4f} pi" for p in result.points)
          + f"; E_max {result.level:.6e}")
    print(f"  perturbed omegas {perturbed[0] / np.pi:.6f} pi, "
          f"{perturbed[1] / np.pi:.6f} pi; band maximum {perturbed_level:.6e} "
          f"({perturbed_level / result.level:.3f} times the levelled value); "
          f"levelling residual {perturbed_residual:.3f}")
    print("  residual by step: "
          + ", ".join(f"{value:.3e}" for value in residuals))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
