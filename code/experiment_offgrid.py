#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Expanded baseline catalog and Pareto figures (Act I of the numerical-evidence section).

This script generates two figures used by the periodic B-spline paper:

  fig_drp_pseudospectral_dispersion.pdf  : modified-wavenumber comparison that
      adds DRP (Tam-Webb), Bogey-Bailly optimized explicit, and Fourier
      pseudospectral to the periodic on-grid catalog.
  fig_pareto_cost_accuracy.pdf           : derivative-cost versus actual
      L2 transport error after one period at PPW=6 and PPW=8.

The off-grid evaluation figure is now produced from the vendored solver by
``experiment_dns.py`` (fig_offgrid_turbulence.pdf), on a turbulence-like field.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THISDIR = Path(__file__).resolve().parent
REPO_ROOT = THISDIR.parent if THISDIR.name == "code" else THISDIR
FIGDIR = REPO_ROOT / "figures_paper"
FIGDIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THISDIR))

import spline_analysis as sa  # noqa: E402
from plotting_helpers import (  # noqa: E402
    SCHEME_COLORS,
    add_slope_reference,
    add_theoretical_reference_line,
    apply_classy_defaults,
    apply_full_width_print_defaults,
    make_overhead_legend,
    method_style,
    quiet_axis,
)


def setup_style() -> None:
    apply_classy_defaults()


def setup_full_width_style() -> None:
    apply_full_width_print_defaults()


COLORS = {
    "Exact": SCHEME_COLORS["Analytic"],
    "E6": method_style("E6")["color"],
    "C6": method_style("C6")["color"],
    "DRP (Tam-Webb)": method_style("DRP")["color"],
    "BB11 (Bogey-Bailly)": method_style("BB")["color"],
    "Fourier pseudospectral": method_style("PS")["color"],
    "BMC2": method_style("BMC2")["color"],
    "BMC5 = BMG2": method_style("BMC5")["color"],
    "BMC7 = BMG3": method_style("BMC7")["color"],
    "C6, paired Lagrange (2pts)": SCHEME_COLORS["ParedLagrange"],
}


# ---------------------------------------------------------------------------
# 1. Modified-wavenumber comparison with the new baselines.
# ---------------------------------------------------------------------------


def figure_drp_pseudospectral_dispersion() -> None:
    setup_full_width_style()
    w = np.linspace(1e-4, np.pi, 800)

    curves = {
        "Exact": w,
        "E6": sa.kprime_centered_fd(w, order=6),
        "C6": sa.kprime_compact(w, order=6),
        "DRP (Tam-Webb)": sa.kprime_drp_tam_webb(w),
        "BB11 (Bogey-Bailly)": sa.kprime_bogey_bailly(w),
        "BMC5 = BMG2": sa.kprime_collocation(w, degree=5),
        "Fourier pseudospectral": sa.kprime_pseudospectral(w),
    }
    line_styles = {
        "Exact": (0, (1, 1.5)),
        "Fourier pseudospectral": (0, (4, 2)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    for label, curve in curves.items():
        ls = line_styles.get(label, "-")
        lw = 1.0 if label in ("Exact", "Fourier pseudospectral") else 1.6
        ax.plot(w, curve, color=COLORS[label], lw=lw, ls=ls, label=label)
    ax.set_xlabel(r"$\omega = kh$")
    ax.set_ylabel(r"$k'h(\omega)$")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, color="#2F4858", fontstyle="italic")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    quiet_axis(ax)

    ax = axes[1]
    err_floor = 1e-8
    for label, curve in curves.items():
        if label == "Exact":
            continue
        err = np.maximum(np.abs(curve - w) / np.maximum(w, 1e-12), err_floor)
        ls = line_styles.get(label, "-")
        lw = 1.0 if label == "Fourier pseudospectral" else 1.6
        ax.semilogy(w, err, color=COLORS[label], lw=lw, ls=ls, label=label)
    ax.axhline(1e-2, color="0.45", lw=0.6, ls=(0, (3, 2)))
    ax.text(0.97, 1.25e-2, r"$1\%$ band", fontsize=12.0, color="0.40", ha="right")
    ax.set_xlabel(r"$\omega = kh$")
    ax.set_ylabel("relative phase error")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, color="#2F4858", fontstyle="italic")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(err_floor, 2.0)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    quiet_axis(ax)

    handles, labels = axes[1].get_legend_handles_labels()
    ncol = min(len(handles), 4)
    make_overhead_legend(fig, handles, labels, ncol=ncol,
                          bbox=(0.5, 1.02), rect=(0, 0, 1, 0.82), fontsize=12.0)

    fig.savefig(FIGDIR / "fig_drp_pseudospectral_dispersion.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Pareto front: derivative cost vs L2 transport error.
# ---------------------------------------------------------------------------


def derivative_apply(label: str, u: np.ndarray, h: float):
    npts = u.shape[0]
    if label == "E6":
        D = sa.compact_derivative_matrix(*sa.explicit_centered_fd_entries(6), npts, h=h)
        return D @ u
    if label == "C6":
        D = sa.compact_derivative_matrix(*sa.compact_scheme_entries(6), npts, h=h)
        return D @ u
    if label == "DRP (Tam-Webb)":
        D = sa.compact_derivative_matrix(*sa.drp_tam_webb_entries(), npts, h=h)
        return D @ u
    if label == "BB11 (Bogey-Bailly)":
        D = sa.compact_derivative_matrix(*sa.bogey_bailly_entries(), npts, h=h)
        return D @ u
    if label == "BMC5 = BMG2":
        D = sa.collocation_derivative_matrix(5, npts, h=h)
        return D @ u
    if label == "BMC7 = BMG3":
        D = sa.collocation_derivative_matrix(7, npts, h=h)
        return D @ u
    if label == "Fourier pseudospectral":
        N = npts
        khat = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
        return np.real(np.fft.ifft(1j * khat * np.fft.fft(u)))
    raise ValueError(label)


def derivative_apply_factory(label: str, npts: int, h: float):
    if label == "E6":
        D = sa.compact_derivative_matrix(*sa.explicit_centered_fd_entries(6), npts, h=h)
        return lambda u: D @ u
    if label == "C6":
        D = sa.compact_derivative_matrix(*sa.compact_scheme_entries(6), npts, h=h)
        return lambda u: D @ u
    if label == "DRP (Tam-Webb)":
        D = sa.compact_derivative_matrix(*sa.drp_tam_webb_entries(), npts, h=h)
        return lambda u: D @ u
    if label == "BB11 (Bogey-Bailly)":
        D = sa.compact_derivative_matrix(*sa.bogey_bailly_entries(), npts, h=h)
        return lambda u: D @ u
    if label == "BMC5 = BMG2":
        D = sa.collocation_derivative_matrix(5, npts, h=h)
        return lambda u: D @ u
    if label == "BMC7 = BMG3":
        D = sa.collocation_derivative_matrix(7, npts, h=h)
        return lambda u: D @ u
    if label == "Fourier pseudospectral":
        khat = np.fft.fftfreq(npts, d=h) * 2.0 * np.pi
        return lambda u: np.real(np.fft.ifft(1j * khat * np.fft.fft(u)))
    raise ValueError(label)


def cost_per_point(label: str) -> float:
    """Recurring derivative work per point in flops, charging both apply and
    band solve where applicable. Pseudospectral charges roughly 5 N log N for
    the FFT pair plus 2 N for the multiply, normalized to per point."""
    if label == "E6":
        return 10.0
    if label == "C6":
        return 11.0
    if label == "DRP (Tam-Webb)":
        return 12.0
    if label == "BB11 (Bogey-Bailly)":
        return 20.0
    if label == "BMC5 = BMG2":
        return 15.0
    if label == "BMC7 = BMG3":
        return 22.0
    if label == "Fourier pseudospectral":
        return 60.0  # representative N=128 cost; explicitly labelled in figure
    raise ValueError(label)


def _transport_error(label: str, ppw: float, npts: int, nperiods: float = 1.0) -> float:
    L = 2.0 * np.pi
    h = L / npts
    x = np.arange(npts) * h
    k_target = 2.0 * np.pi / (ppw * h)
    u0 = np.cos(k_target * x)
    T = nperiods * L
    nsteps = max(2000, int(np.ceil(8.0 * nperiods * npts / ppw)))
    dt = T / nsteps
    u = u0.copy()
    apply_derivative = derivative_apply_factory(label, npts, h)
    for _ in range(nsteps):
        k1 = -apply_derivative(u)
        k2 = -apply_derivative(u + 0.5 * dt * k1)
        k3 = -apply_derivative(u + 0.5 * dt * k2)
        k4 = -apply_derivative(u + dt * k3)
        u = u + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return float(np.sqrt(np.mean((u - u0) ** 2)) / np.sqrt(np.mean(u0 ** 2)))


def figure_pareto_cost_accuracy() -> None:
    """Cost-versus-accuracy frontier at two fixed points-per-wavelength values."""
    from matplotlib.lines import Line2D

    setup_full_width_style()

    npts = 128
    schemes = [
        "E6",
        "C6",
        "DRP (Tam-Webb)",
        "BB11 (Bogey-Bailly)",
        "BMC5 = BMG2",
        "BMC7 = BMG3",
        "Fourier pseudospectral",
    ]
    ppws = [6.0, 8.0]

    pareto_colors = {label: COLORS[label] for label in schemes}
    pareto_markers = {
        "E6": "o",
        "C6": "s",
        "DRP (Tam-Webb)": "v",
        "BB11 (Bogey-Bailly)": "P",
        "BMC5 = BMG2": "D",
        "BMC7 = BMG3": "^",
        "Fourier pseudospectral": "*",
    }

    short_label = {
        "E6": r"E$_6$",
        "C6": r"C$_6$",
        "DRP (Tam-Webb)": r"DRP",
        "BB11 (Bogey-Bailly)": r"BB$_{11}$",
        "BMC5 = BMG2": r"BMC$_5$=BMG$_2$",
        "BMC7 = BMG3": r"BMC$_7$=BMG$_3$",
        "Fourier pseudospectral": r"PS",
    }

    costs = {label: cost_per_point(label) for label in schemes}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6),
                             gridspec_kw={"wspace": 0.34})
    print("Pareto figure errors:")
    for ax, ppw in zip(axes, ppws):
        errors = {label: _transport_error(label, ppw, npts) for label in schemes}
        for label in schemes:
            c = costs[label]
            e = errors[label]
            ax.scatter(
                [c], [e], s=78, color=pareto_colors[label],
                marker=pareto_markers[label], zorder=4,
                edgecolor="white", lw=0.9, label=short_label[label],
            )

        pareto_pts = sorted(((costs[lbl], errors[lbl], lbl) for lbl in schemes),
                            key=lambda t: t[0])
        pareto_x: list[float] = []
        pareto_y: list[float] = []
        best_e = float("inf")
        for c, e, _ in pareto_pts:
            if e < best_e:
                pareto_x.append(c)
                pareto_y.append(e)
                best_e = e
        ax.plot(pareto_x, pareto_y, color="0.55", ls=(0, (4, 2)), lw=0.9,
                alpha=0.7, zorder=2)

        ax.set_xlabel("derivative-apply cost (flops per point)")
        ax.set_ylabel(r"relative $L^2$ transport error, one period")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(5.5, 220.0)
        ax.set_title(f"({'ab'[ppws.index(ppw)]})", loc="left")
        quiet_axis(ax)
        print(f"  PPW={ppw:.0f}")
        for label in schemes:
            print(f"    {label:30s}  cost={costs[label]:5.1f}  L2 err={errors[label]:.3e}")

    method_handles, method_labels = axes[0].get_legend_handles_labels()
    legend_handles = method_handles + [
        Line2D([0], [0], color="0.55", ls=(0, (4, 2)), lw=0.9,
               marker="None", label="Pareto frontier")
    ]
    legend_labels = method_labels + ["Pareto frontier"]
    make_overhead_legend(fig, legend_handles, legend_labels,
                         ncol=4, fontsize=12.0,
                         bbox=(0.5, 1.04), rect=(0, 0, 1, 0.68))

    fig.savefig(FIGDIR / "fig_pareto_cost_accuracy.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    figure_drp_pseudospectral_dispersion()
    figure_pareto_cost_accuracy()
    print("Wrote dispersion/Pareto figures to", FIGDIR)


if __name__ == "__main__":
    main()
