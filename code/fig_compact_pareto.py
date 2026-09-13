# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Matched-order short-wave resolution of the periodic BMC degree ladder.

Panel (a) compares each parameter-free BMC rung with the unique
maximum-formal-order tridiagonal Pade formula at the same derivative order.
Panel (b) exposes the order-12 mechanism through the complete relative
modified-wavenumber error curves of BMC11 and C12.

The stored threshold metrics remain the canonical annotation source. Dense
symbol evaluation independently reproduces every plotted cutoff before the
figure is written to ``figures_paper/fig_compact_pareto.pdf``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plotting_helpers as ph
import spline_analysis as sa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = HERE / "results" / "compact_frontier" / "symbol_metrics.csv"
OUTDIR = ROOT / "figures_paper"

C_PADE = ph.SCHEME_COLORS["C"]
C_BMC = ph.SCHEME_COLORS["BMC"]
C_THRESHOLD = "0.42"

PAIRS = (
    (6, 5, "C6 (Taylor-maximal tridiag)"),
    (8, 7, "C8 (Taylor-maximal tridiag)"),
    (10, 9, "C10 (Taylor-maximal tridiag)"),
    (12, 11, "C12 (Taylor-maximal tridiag)"),
)


def _stored_cutoffs() -> dict[str, float]:
    """Return canonical one-percent modified-wavenumber cutoffs."""
    with METRICS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["name"]: float(row["omega_cut_0.01"]) for row in rows}


def _first_contiguous_cut(x: np.ndarray, relative_error: np.ndarray,
                          tolerance: float = 1.0e-2) -> float:
    """Locate the last point in the contiguous tolerance interval from zero."""
    failed = np.flatnonzero(relative_error > tolerance)
    if not len(failed):
        return float(x[-1])
    index = int(failed[0])
    if index == 0:
        return float(x[0])
    x0, x1 = x[index - 1], x[index]
    e0, e1 = relative_error[index - 1], relative_error[index]
    return float(x0 + (tolerance - e0) * (x1 - x0) / (e1 - e0))


def _symbol_receipt(stored: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute matched-order cutoffs and verify the stored values."""
    omega = np.linspace(1.0e-6, np.pi, 400_001)
    bmc_cut = []
    pade_cut = []
    for order, degree, pade_name in PAIRS:
        bmc_symbol = sa.kprime_collocation(omega, degree)
        pade_symbol = sa.kprime_compact(omega, order)
        bmc_error = np.abs(bmc_symbol / omega - 1.0)
        pade_error = np.abs(pade_symbol / omega - 1.0)
        bmc_value = _first_contiguous_cut(omega, bmc_error) / np.pi
        pade_value = _first_contiguous_cut(omega, pade_error) / np.pi
        if not np.isclose(bmc_value, stored[f"BMC{degree}"], atol=2.0e-5, rtol=0.0):
            raise RuntimeError(f"BMC{degree} cutoff drift: {bmc_value}")
        if not np.isclose(pade_value, stored[pade_name], atol=2.0e-5, rtol=0.0):
            raise RuntimeError(f"C{order} cutoff drift: {pade_value}")
        bmc_cut.append(stored[f"BMC{degree}"])
        pade_cut.append(stored[pade_name])
    return omega, np.asarray(bmc_cut), np.asarray(pade_cut)


def main() -> int:
    stored = _stored_cutoffs()
    omega, bmc_cut, pade_cut = _symbol_receipt(stored)
    orders = np.asarray([pair[0] for pair in PAIRS])
    gains = 100.0 * (bmc_cut / pade_cut - 1.0)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 13.0,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12.2,
        "ytick.labelsize": 12.2,
    })
    fig, (ax_ladder, ax_curve) = plt.subplots(
        1, 2, figsize=(8.6, 4.35), gridspec_kw={"width_ratios": [0.94, 1.18]}
    )

    ax_ladder.plot(orders, pade_cut, color=C_PADE, lw=1.8, marker="s",
                   ms=5.5, label="maximum-order tridiagonal Pade")
    ax_ladder.plot(orders, bmc_cut, color=C_BMC, lw=2.2, marker="o",
                   ms=5.8, label="BMC degree ladder")
    for order, lower, upper, gain in zip(orders, pade_cut, bmc_cut, gains):
        ax_ladder.annotate(
            "", xy=(order, upper - 0.004), xytext=(order, lower + 0.004),
            arrowprops={"arrowstyle": "-|>", "color": C_BMC, "lw": 1.0,
                        "mutation_scale": 8.0},
        )
        ax_ladder.text(order + 0.17, 0.5 * (lower + upper), f"+{gain:.1f}%",
                       color=C_BMC, fontsize=12.4, ha="left", va="center")
    ax_ladder.set_xlabel("formal derivative order")
    ax_ladder.set_ylabel(r"1\% reach $\omega_c/\pi$")
    ax_ladder.set_xticks(orders)
    ax_ladder.set_xlim(5.4, 12.9)
    ax_ladder.set_ylim(0.47, 0.81)
    ax_ladder.set_title("(a) matched-order reach", loc="left", fontsize=13.0,
                        color="#2F4858", fontstyle="italic")
    ax_ladder.grid(axis="y", color="0.88", lw=0.6)
    ph.quiet_axis(ax_ladder)

    bmc11 = sa.kprime_collocation(omega, 11)
    c12 = sa.kprime_compact(omega, 12)
    bmc11_error = np.maximum(np.abs(bmc11 / omega - 1.0), 1.0e-12)
    c12_error = np.maximum(np.abs(c12 / omega - 1.0), 1.0e-12)
    x = omega / np.pi
    ax_curve.semilogy(x, c12_error, color=C_PADE, lw=1.8, label="C12")
    ax_curve.semilogy(x, bmc11_error, color=C_BMC, lw=2.2, label="BMC11")
    ax_curve.axhline(1.0e-2, color=C_THRESHOLD, lw=0.9, ls=(0, (4, 2)))
    ax_curve.axvspan(pade_cut[-1], bmc_cut[-1], color=C_BMC, alpha=0.10,
                     lw=0.0)
    ax_curve.vlines([pade_cut[-1], bmc_cut[-1]], 1.0e-12, 1.0e-2,
                    colors=[C_PADE, C_BMC], linestyles=(0, (2, 2)), lw=0.9)
    ax_curve.scatter([pade_cut[-1]], [1.0e-2], marker="s", s=28,
                     color=C_PADE, zorder=5)
    ax_curve.scatter([bmc_cut[-1]], [1.0e-2], marker="o", s=32,
                     color=C_BMC, zorder=5)
    ax_curve.text(0.38, 1.45e-2, "1% error", color=C_THRESHOLD,
                  fontsize=12.2, ha="center", va="bottom")
    ax_curve.text(0.5 * (pade_cut[-1] + bmc_cut[-1]), 2.4e-8,
                  "BMC11 extension", color=C_BMC, fontsize=12.4,
                  ha="center", va="center")
    cut_halo = dict(facecolor="white", edgecolor="none", pad=0.9)
    ax_curve.text(pade_cut[-1] - 0.008, 2.0e-2, f"{pade_cut[-1]:.3f}",
                  color=C_PADE, fontsize=12.0, ha="right", va="bottom",
                  bbox=cut_halo, zorder=6)
    ax_curve.text(bmc_cut[-1] + 0.008, 2.0e-2, f"{bmc_cut[-1]:.3f}",
                  color=C_BMC, fontsize=12.0, ha="left", va="bottom",
                  bbox=cut_halo, zorder=6)
    ax_curve.set_xlabel(r"scaled wavenumber $\omega/\pi$")
    ax_curve.set_ylabel("relative phase error")
    ax_curve.set_xlim(0.05, 0.90)
    ax_curve.set_ylim(1.0e-11, 1.0)
    ax_curve.set_title("(b) order-12 error curve", loc="left", fontsize=13.0,
                       color="#2F4858", fontstyle="italic")
    ax_curve.grid(axis="y", color="0.90", lw=0.55)
    ph.quiet_axis(ax_curve)

    handles = [
        Line2D([], [], color=C_BMC, lw=2.2, marker="o", ms=5.5,
               label="BMC"),
        Line2D([], [], color=C_PADE, lw=1.8, marker="s", ms=5.0,
               label="maximum-order tridiagonal Padé"),
        Line2D([], [], color=C_THRESHOLD, lw=0.9, ls=(0, (4, 2)),
               label="1% error"),
    ]
    ph.make_overhead_legend(fig, handles, [h.get_label() for h in handles],
                            ncol=3, bbox=(0.5, 1.005), rect=(0, 0, 1, 0.87),
                            fontsize=12.6, handslcngth=2.1,
                            columnspacing=1.15, handletextpad=0.45)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_compact_pareto.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("matched-order reach receipt:")
    for order, degree, bmc, pade, gain in zip(
            orders, [p[1] for p in PAIRS], bmc_cut, pade_cut, gains):
        print(f"  q={order}: BMC{degree}={bmc:.6f}, C{order}={pade:.6f}, "
              f"gain={gain:.3f}%")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
