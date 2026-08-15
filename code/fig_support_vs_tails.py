#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy"]
# ///
"""Support-vs-tails figure for coefficient-space and nodal-space reads.

One row, two panels, built from the ACTUAL periodic operators at degree
p = 9, N = 64 in 1D:

  (a) the coefficient-space footprint of one wall read u(x*): the row of
      evaluation weights phi_j = B_p(x* - x_j) is nonzero on exactly p+1
      consecutive coefficients and identically zero elsewhere;
  (b) the same read expressed on nodal values, w = B^{-1} phi (B the
      circulant collocation basis matrix): dense in principle, but |w_j|
      decays geometrically; the measured per-point ratio is printed on the
      panel and cross-checked against the exact interior Euler-Frobenius
      root of the basis symbol.

"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THISDIR = Path(__file__).resolve().parent
REPO_ROOT = THISDIR.parent if THISDIR.name == "code" else THISDIR
OUTDIR = REPO_ROOT / "figures_paper"
sys.path.insert(0, str(THISDIR))

import spline_analysis as sa  # noqa: E402
from plotting_helpers import apply_classy_defaults, quiet_axis, SCHEME_COLORS  # noqa: E402

P = 9          # representative upper-ladder degree (BMC9)
N = 64         # 1D periodic grid
FRAC = 0.37    # off-grid wall-read location x* = (j0 + FRAC) h
ACCENT = SCHEME_COLORS["BMC"]
GREY = "#8a8a8a"


def periodic_offset(j: np.ndarray, j0: float, n: int) -> np.ndarray:
    """Signed minimal-image offset j - j0 on the periodic index ring."""
    d = (j - j0 + n / 2.0) % n - n / 2.0
    return d


def build_operators():
    """Assemble B (circulant collocation basis matrix) and one wall-read row."""
    j = np.arange(N)
    # B[i, k] = B_p(i - k), periodic minimal image: nodal values of basis k.
    off = periodic_offset(j[:, None], j[None, :], N)
    B = sa.centered_cardinal_bspline(P, off)
    j0 = N // 2 + FRAC  # wall read between nodes 32 and 33
    phi = sa.centered_cardinal_bspline(P, periodic_offset(j, j0, N))
    # u(x*) = phi . c with B c = u  =>  u(x*) = (B^{-1} phi) . u  (B symmetric)
    w = np.linalg.solve(B, phi)
    return j, j0, phi, w


def exact_decay_ratio() -> float:
    """Interior Euler-Frobenius root of the degree-P basis symbol.

    The nodal-map tails decay like rho^{|j|} where rho is the largest-
    magnitude root of sum_k B_p(k) z^k inside the unit disk.
    """
    k = np.arange(-(P // 2) - 1, P // 2 + 2)
    coeff = sa.centered_cardinal_bspline(P, k.astype(float))
    keep = np.abs(coeff) > 1e-14
    roots = np.roots(coeff[keep][::-1])
    inside = roots[np.abs(roots) < 1.0 - 1e-9]
    return float(np.max(np.abs(inside)))


def main() -> None:
    apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 13.0,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
    })
    OUTDIR.mkdir(parents=True, exist_ok=True)
    j, j0, phi, w = build_operators()
    d = periodic_offset(j, j0, N)
    order = np.argsort(d)
    d, phi_s, w_s = d[order], phi[order], np.abs(w[order])

    # measured per-point decay ratio of the nodal tails (fit away from the
    # center and away from the periodic wrap at |d| ~ N/2)
    sel = (np.abs(d) >= 6) & (np.abs(d) <= 20)
    slope, intercept = np.polyfit(np.abs(d[sel]), np.log(w_s[sel]), 1)
    rho_fit = float(np.exp(slope))
    rho_exact = exact_decay_ratio()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    # --- (a) coefficient-space footprint: strictly finite support ----------
    ax = axes[0]
    win = np.abs(d) <= 14.5
    nz = phi_s > 0.0
    markerline, stemlines, _ = ax.stem(d[win & nz], phi_s[win & nz],
                                       linefmt="-", basefmt=" ",
                                       markerfmt="o")
    markerline.set(color=ACCENT, markersize=5.0, markeredgecolor="white",
                   markeredgewidth=0.8, zorder=4)
    stemlines.set(color=ACCENT, linewidth=1.2)
    ax.plot(d[win & ~nz], np.zeros(np.sum(win & ~nz)), marker="o", ls="none",
            ms=3.4, mfc="white", mec=GREY, mew=0.9, zorder=3)
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)
    n_nz = int(np.sum(nz))
    ax.annotate(rf"exactly $p+1 = {n_nz}$" "\nnonzero weights",
                xy=(0.76, 0.68), xycoords="axes fraction", ha="center",
                fontsize=13.0, color="#2F4858", linespacing=1.25)
    ax.annotate("identical zeros\noutside the support", xy=(-10.2, 0.02),
                xytext=(-13.8, 0.16), fontsize=13.5, color="#555555",
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.03, 0.48)
    ax.set_xlabel(r"coefficient offset $j - x^*/h$")
    ax.set_ylabel(r"read weight on coefficient $j$")
    ax.text(0.03, 0.955, r"(a)",
            transform=ax.transAxes, ha="left", va="top", fontsize=14.0,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    # --- (b) nodal-map column: exponential tails ---------------------------
    ax = axes[1]
    ax.semilogy(d, w_s, marker="o", ms=3.2, lw=0.9, color=ACCENT,
                mec="white", mew=0.5)
    # fitted guide lines, offset just above the data on both wings
    dd = np.linspace(5.0, 25.0, 40)
    guide = 6.0 * np.exp(intercept + slope * dd)
    for sgn in (-1.0, 1.0):
        ax.semilogy(sgn * dd, guide, ls=(0, (4, 2)), lw=1.0, color="0.35",
                    zorder=4)
    ax.set_xlim(-33, 33)
    ax.set_ylim(5e-8, 4.0)
    ax.set_xlabel(r"nodal offset $j - x^*/h$")
    ax.set_ylabel(r"$|$read weight on node $j|$")
    ax.text(0.03, 0.955, r"(b)",
            transform=ax.transAxes, ha="left", va="top", fontsize=14.0,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    fig.tight_layout()
    out = OUTDIR / "fig_support_vs_tails.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  p = {P}, N = {N}, x* = node {N // 2} + {FRAC}h")
    print(f"  nonzero coefficient weights: {n_nz} (= p+1)")
    print(f"  max |phi| outside support:   "
          f"{np.max(np.abs(phi_s[~nz])) if np.any(~nz) else 0.0:.2e}")
    print(f"  measured nodal decay ratio:  {rho_fit:.4f} per point")
    print(f"  exact Euler-Frobenius root:  {rho_exact:.4f}")


if __name__ == "__main__":
    main()
