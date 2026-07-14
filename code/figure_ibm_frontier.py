#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Figures for the matched immersed-boundary head-to-head (Section 8.8.2/8.8.3).

fig_ibm_setup_receipt.pdf
    The actual Eulerian grid, the marker set on the three obstacles, the
    interpolation-support footprint of E for one marker under each route family
    (local spline, local compact+Lagrange, global Fourier), and the
    worst-conditioned marker neighborhood.
fig_ibm_error_walltime.pdf
    Boundary-trace error and exterior field error against measured total wall
    time, all routes, matched pairs paired by marker shape.

Both are parked in results/ibm_frontier/ until the controller integrates them.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

THISDIR = Path(__file__).resolve().parent
REPO_ROOT = THISDIR.parent if THISDIR.name == "code" else THISDIR
OUTDIR = REPO_ROOT / "results" / "ibm_frontier"
sys.path.insert(0, str(THISDIR))

import experiment_ibm as ibm  # noqa: E402
import experiment_ibm_frontier as fr  # noqa: E402
from plotting_helpers import SCHEME_COLORS, apply_classy_defaults  # noqa: E402

TWO_PI = 2.0 * np.pi
RESULTS = json.load(open(OUTDIR / "results.json"))

ROUTE_STYLE = {
    "BMC5":       dict(color=SCHEME_COLORS["BMC"], ls="-",  marker="o", label="BMC5"),
    "C6+L6":      dict(color=SCHEME_COLORS["C"],   ls="--", marker="o", label="C6+L6"),
    "BMC7":       dict(color=SCHEME_COLORS["BMG"], ls="-",  marker="s", label="BMC7"),
    "C8+L8":      dict(color="#c98a3c",            ls="--", marker="s", label="C8+L8"),
    "BMC9":       dict(color="#7a2d2d",            ls="-",  marker="^", label="BMC9"),
    "C10+L10":    dict(color="#e0a95e",            ls="--", marker="^", label="C10+L10"),
    "Fourier-PS": dict(color=SCHEME_COLORS["PS"],  ls=":",  marker="D", label="Fourier PS"),
}
ORDER = ["BMC5", "C6+L6", "BMC7", "C8+L8", "BMC9", "C10+L10", "Fourier-PS"]


# --------------------------------------------------------------------------
def _E_row_grid(route, npts, h, markers, weights, m0):
    """The interpolation weights E[m0, :] laid on the grid: E^T e_{m0} = S e_{m0}/W."""
    ops = fr.coupling_ops(route, npts, h, markers, weights)
    unit = np.zeros(markers.shape[0])
    unit[m0] = 1.0
    return ops["S"](unit) / weights[m0]


def figure_setup_receipt():
    apply_classy_defaults()
    obstacles = ibm.standard_obstacles()
    npts = 64
    h = TWO_PI / npts
    n_markers = 40
    markers, weights, _ = fr.markers_weights_queries(obstacles, n_markers)

    # worst-conditioned marker: tightest local spacing relative to h (min weight)
    m_worst = int(np.argmin(weights))
    # a representative smooth-boundary marker for the footprint comparison
    m_rep = 8  # on the disk

    fig = plt.figure(figsize=(12.6, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.25, 0.9, 0.9],
                          height_ratios=[1, 1], hspace=0.32, wspace=0.34)
    ax_main = fig.add_subplot(gs[:, 0:2])

    # ---- main panel: grid + obstacles + markers -------------------------
    for gx in np.arange(npts + 1) * h:
        ax_main.axvline(gx, color="#e7e2d6", lw=0.35, zorder=0)
        ax_main.axhline(gx, color="#e7e2d6", lw=0.35, zorder=0)
    for obs in obstacles:
        th = np.linspace(0, TWO_PI, 400)
        bx, by = obs.parametrization(th)
        ax_main.fill(bx, by, facecolor="#eceff2", edgecolor="#2F4858",
                     lw=1.1, zorder=2)
    ax_main.scatter(markers[:, 0], markers[:, 1], s=7, color=SCHEME_COLORS["BMC"],
                    zorder=3, linewidths=0)
    # highlight the representative and worst markers
    ax_main.scatter(markers[m_rep, 0], markers[m_rep, 1], s=70, marker="o",
                    facecolor="none", edgecolor="#1a1a1a", lw=1.2, zorder=5)
    win = 4.5 * h
    x0, y0 = markers[m_worst]
    ax_main.add_patch(plt.Rectangle((x0 - win, y0 - win), 2 * win, 2 * win,
                                    fill=False, edgecolor="#a83232", lw=1.3,
                                    zorder=6))
    ax_main.annotate("worst-conditioned\nneighborhood", (x0, y0 - win),
                     xytext=(x0 - 1.7, y0 - 1.9), fontsize=8.5, color="#a83232",
                     ha="center", va="top",
                     arrowprops=dict(arrowstyle="-", color="#a83232", lw=0.8))
    ax_main.annotate("footprint\nmarker", markers[m_rep],
                     xytext=(markers[m_rep, 0] - 1.9, markers[m_rep, 1] - 0.5),
                     fontsize=8.5, color="#1a1a1a", ha="right", va="top",
                     arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=0.8))
    ax_main.set_xlim(0, TWO_PI)
    ax_main.set_ylim(0, TWO_PI)
    ax_main.set_aspect("equal")
    ax_main.set_xticks([0, np.pi, TWO_PI])
    ax_main.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax_main.set_yticks([0, np.pi, TWO_PI])
    ax_main.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")
    ax_main.set_title(r"Eulerian grid ($64^2$), markers, obstacles",
                      fontsize=10.0, color="#2F4858")

    # ---- footprint panels: |E[m_rep, :]| for three routes --------------
    xr = markers[m_rep, 0]
    yr = markers[m_rep, 1]
    span = 6.5 * h
    xs = np.arange(npts) * h
    footprint_routes = [("BMC5", "spline, local"),
                        ("C6+L6", "compact+Lagrange, local"),
                        ("Fourier-PS", "Fourier, global")]
    foot_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
                 fig.add_subplot(gs[1, 2])]
    vmax = None
    fields = {}
    for route, _ in footprint_routes:
        f = np.abs(_E_row_grid(route, npts, h, markers, weights, m_rep))
        fields[route] = f
        vmax = f.max() if vmax is None else max(vmax, f.max())
    norm = LogNorm(vmin=vmax * 1e-6, vmax=vmax)
    pcm = None
    for ax, (route, tag) in zip(foot_axes, footprint_routes):
        f = fields[route].copy()
        f[f < vmax * 1e-6] = np.nan
        pcm = ax.pcolormesh(xs, xs, f.T, norm=norm, cmap="rocket_r"
                            if "rocket_r" in plt.colormaps() else "magma_r",
                            shading="nearest", rasterized=True)
        ax.scatter([xr], [yr], marker="*", s=60, color="#0b3d5c",
                   edgecolor="white", lw=0.5, zorder=5)
        ax.set_xlim(xr - span, xr + span)
        ax.set_ylim(yr - span, yr + span)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ROUTE_STYLE[route]['label']}\n{tag}", fontsize=8.6,
                     color="#2F4858")

    # ---- worst-conditioned neighborhood zoom ---------------------------
    axw = fig.add_subplot(gs[1, 3])
    for gx in np.arange(npts + 1) * h:
        axw.axvline(gx, color="#e7e2d6", lw=0.4, zorder=0)
        axw.axhline(gx, color="#e7e2d6", lw=0.4, zorder=0)
    for obs in obstacles:
        th = np.linspace(0, TWO_PI, 800)
        bx, by = obs.parametrization(th)
        axw.fill(bx, by, facecolor="#eceff2", edgecolor="#2F4858", lw=1.0,
                 zorder=2)
    axw.scatter(markers[:, 0], markers[:, 1], s=16, color="#a83232", zorder=3,
                linewidths=0)
    axw.set_xlim(x0 - win, x0 + win)
    axw.set_ylim(y0 - win, y0 + win)
    axw.set_aspect("equal")
    axw.set_xticks([])
    axw.set_yticks([])
    axw.set_title("worst-conditioned\nmarker neighborhood", fontsize=8.6,
                  color="#a83232")

    cbar = fig.colorbar(pcm, ax=foot_axes, fraction=0.045, pad=0.02,
                        location="right")
    cbar.set_label(r"$|E_{m,\cdot}|$ (interpolation weight)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(r"Immersed-boundary setup receipt: grid, markers, and the "
                 r"interpolation footprint of $E$", fontsize=11.5, y=0.99,
                 color="#2F4858")
    fig.savefig(OUTDIR / "fig_ibm_setup_receipt.pdf", dpi=200,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("Wrote", OUTDIR / "fig_ibm_setup_receipt.pdf")


def figure_error_walltime():
    apply_classy_defaults()
    conv = RESULTS["convergence"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    for ax, key, title, ylab in (
            (ax1, "trace", r"boundary trace $\|u_h-u^\ast\|_\infty$ vs wall time",
             "max-norm trace error"),
            (ax2, "field", r"exterior field $L^2$ error vs wall time",
             "relative field error")):
        for route in ORDER:
            st = ROUTE_STYLE[route]
            t = np.asarray(conv[route]["total_time"])
            e = np.asarray(conv[route][key])
            ax.loglog(t, e, ls=st["ls"], marker=st["marker"], color=st["color"],
                      ms=4.6, lw=1.4, mec="white", mew=0.5, label=st["label"])
        ax.set_xlabel("measured total wall time [s]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.0)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)
        from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5]))
        ax.xaxis.set_minor_locator(NullLocator())
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, 1.02), columnspacing=1.3,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTDIR / "fig_ibm_error_walltime.pdf", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print("Wrote", OUTDIR / "fig_ibm_error_walltime.pdf")


def main():
    figure_setup_receipt()
    figure_error_walltime()


if __name__ == "__main__":
    main()
