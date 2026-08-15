"""Convergence-figure primitives for pre-roundoff order fits and clean axes."""
from __future__ import annotations

import numpy as np

from plotting_helpers import METHOD_STYLES

# Canonical spline-rung styles used by both convergence figures.
DEG_COLOR = {p: METHOD_STYLES[f"BMC{p}"]["color"] for p in range(2, 10)}
DEG_MARKER = {p: METHOD_STYLES[f"BMC{p}"]["marker"] for p in range(2, 10)}

FLOOR = 1e-13          # errors at/below this are machine-precision saturated


def slope_triangle(ax, h_anchor, e_anchor, order, color, dy_dec=0.6, label=None):
    """A small right slope triangle anchored bottom-right at (h_anchor, e_anchor),
    extending up and to the left. The vertical leg spans ``dy_dec`` decades for
    every order, so the triangle stays compact regardless of slope."""
    ha = h_anchor
    hb = ha / 10 ** (dy_dec / order)
    e0, e1 = e_anchor, e_anchor * 10 ** dy_dec
    ax.plot([hb, ha, ha, hb], [e0, e0, e1, e0], color=color, lw=1.0, alpha=0.9, zorder=5)
    ax.text(ha * 1.07, (e0 * e1) ** 0.5, label if label is not None else f"{order}",
            color=color, fontsize=19.0, va="center", ha="left", zorder=6)


def fit_indices(h, e, expected_order=None, floor=FLOOR):
    """Return the resolved monotone prefix before roundoff flattening.

    Samples are ordered from coarse to fine. A point is excluded once the error
    reaches the numerical floor, stops decreasing, or its local rate falls below
    half the expected asymptotic order. The final criterion detects saturation
    above the absolute floor without fitting through a visibly flat tail.
    """
    h = np.asarray(h, float); e = np.asarray(e, float)
    if len(e) < 2:
        return np.asarray([], int)
    keep = [0]
    for i in range(1, len(e)):
        if not np.isfinite(e[i]) or e[i] <= floor or e[i] >= e[i - 1]:
            break
        local_order = np.log(e[i - 1] / e[i]) / np.log(h[i - 1] / h[i])
        if expected_order is not None and local_order < 0.5 * expected_order:
            break
        keep.append(i)
    if len(keep) < 2:
        return np.asarray([], int)
    return np.asarray(keep[-4:], int)


def measured_order(h, e, expected_order=None, floor=FLOOR):
    idx = fit_indices(h, e, expected_order=expected_order, floor=floor)
    if len(idx) < 2:
        return None
    return float(np.polyfit(np.log(np.asarray(h)[idx]), np.log(np.asarray(e)[idx]), 1)[0])


def anchor_point(h, e, floor=FLOOR):
    """A coarse, well-separated point on the asymptotic tail to hang the slope
    triangle from (returns h, e or None)."""
    idx = fit_indices(h, e, floor=floor)
    if len(idx) < 2:
        return None
    a = idx[min(1, len(idx) - 1)]
    return float(np.asarray(h)[a]), float(np.asarray(e)[a])


def style_axis(ax, xlabel, ylabel, title, headroom=3.0):
    """Style a full-width two-panel convergence axis for print."""
    ax.set_xlabel(xlabel, fontsize=17.0)
    ax.set_ylabel(ylabel, fontsize=17.0)
    ax.set_title(title, fontsize=17.0, loc="left")
    ax.tick_params(axis="both", labelsize=15.5)
    ax.grid(True, which="major", axis="y", ls=":", lw=0.5, alpha=0.40)
    lo, hi = ax.get_ylim()
    ax.set_ylim(top=hi * headroom)            # fine grid on the left (no x inversion)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", which="both", labelrotation=0)


def resolution_axis(ax, grid_sizes):
    """Use every executed resolution as a horizontal major tick."""
    grid_sizes = np.asarray(grid_sizes, dtype=float)
    ax.set_xscale("log")
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([str(int(n)) for n in grid_sizes])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlim(grid_sizes[0] / 1.10, grid_sizes[-1] * 1.18)


def order_table(axt, title, col_headers, rows, footnote=None):
    """Render a side table. ``col_headers`` is e.g. ["D1", "D2"]; ``rows`` is a
    list of (label, color, [cell_strings]) with one cell per header.

    Font sizes match the ``style_axis`` print-legibility scaling above: this
    panel carries the smallest text in the figure (the footnote), so its size
    is the binding constraint for clearing ~7.5pt once printed at \\textwidth.
    """
    axt.axis("off")
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.set_title(title, fontsize=20.0)
    ncol = len(col_headers)
    # Wide enough that an 8-character "formal / measured" cell (e.g. "8 / 7.16")
    # at fontsize 19 does not run into its neighbor; verified empirically at
    # 250 dpi against the two figures that use this table (fig_superconvergence,
    # fig_offgrid_turbulence). The gridspec that hosts this axes gives it a
    # generous width_ratio precisely so this column layout has room to breathe.
    xcol = [0.44 + 0.32 * i for i in range(ncol)]
    y = 0.95
    axt.text(0.02, y, "curve", fontsize=19.0, fontweight="bold")
    for x, hdr in zip(xcol, col_headers):
        axt.text(x, y, hdr, fontsize=19.0, fontweight="bold")
    y -= 0.058
    axt.plot([0.01, 0.99], [y, y], color="0.7", lw=0.8)
    y -= 0.075
    n_rows = max(len(rows), 1)
    row_pitch = min(0.105, 0.82 / n_rows)
    for label, color, cells in rows:
        axt.plot([0.02, 0.08], [y + 0.010, y + 0.010], color=color, lw=2.6)
        axt.text(0.10, y, label, fontsize=19.0, color=color, va="center")
        for x, cell in zip(xcol, cells):
            axt.text(x, y, cell, fontsize=19.0, va="center")
        y -= row_pitch
    if footnote:
        axt.text(0.02, y - 0.01, footnote, fontsize=19.0, color="0.4", va="top")


def fmt_order(formal, measured, spectral=False):
    if spectral:
        return f"{formal} / spec."
    if measured is None:
        return f"{formal} / --"
    return f"{formal} / {measured:.2f}"
