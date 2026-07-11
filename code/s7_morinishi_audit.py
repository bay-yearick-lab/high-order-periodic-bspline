"""Section 7 sub-capability (4): de-aliased products via Morinishi orthogonality.

Verifies the periodic 2D Morinishi orthogonality identity of
Theorem~4
  ``<C(u, v; phi), phi>_H = 0``
where
  ``C = (1/2) (D_x(u phi) + u D_x phi + D_y(v phi) + v D_y phi)``
on a 2D periodic torus with ``H = h^2 I``. The identity holds
unconditionally (no divergence-free hypothesis is needed for the
inner-product cancellation), but the validator chooses ``(u, v)``
projected to the discrete divergence-free subspace through the
spectral pressure projector that the Taylor--Green driver uses, to
reflect the physical context.

The figure reports the maximum relative inner-product deviation
across many random ``(u, v, phi)`` realizations as a function of
grid size, for the periodic spline first-derivative catalog
``BMC_2`` through ``BMC_8`` and ``BMG_1`` through ``BMG_5``.

Outputs: ``figures_paper/fig_s7_morinishi.pdf``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import plotting_helpers as ph
import project_paths as pp
import spline_analysis as sa


@dataclass(frozen=True)
class MorinishiRecord:
    label: str
    family: str
    degree: int
    npts: int
    worst_relative: float


def _build_first_derivative(label: str, npts: int, h: float) -> np.ndarray:
    return sa.first_derivative_matrix_by_label(label, npts, h=h)


def _first_derivative_symbol(label: str, w: np.ndarray) -> np.ndarray:
    return sa.first_derivative_symbol_by_label(label, w)


def _project_div_free(D1: np.ndarray, label: str, u: np.ndarray, v: np.ndarray, npts: int, h: float):
    omega = 2.0 * np.pi * np.fft.fftfreq(npts, d=1.0)
    kprime = _first_derivative_symbol(label, omega)
    eig = (kprime / h) ** 2
    eig2d = eig[:, None] + eig[None, :]
    null_mask = eig2d <= 1e-14
    eig2d_safe = np.where(null_mask, 1.0, eig2d)
    inv_lap = np.where(null_mask, 0.0, -1.0 / eig2d_safe)
    div = D1 @ u + v @ D1.T
    rhs_hat = np.fft.fft2(div)
    p_hat = inv_lap * rhs_hat
    p = np.real(np.fft.ifft2(p_hat))
    return u - D1 @ p, v - p @ D1.T


def _audit_morinishi(label: str, npts: int, n_trials: int = 6) -> float:
    h = 1.0 / npts
    D1 = _build_first_derivative(label, npts, h)
    rng = np.random.default_rng(20260510)
    worst = 0.0
    for _ in range(n_trials):
        u = rng.standard_normal((npts, npts))
        v = rng.standard_normal((npts, npts))
        u, v = _project_div_free(D1, label, u, v, npts, h)
        phi = rng.standard_normal((npts, npts))
        phi -= phi.mean()
        cx = 0.5 * (D1 @ (u * phi) + u * (D1 @ phi))
        cy = 0.5 * ((v * phi) @ D1.T + v * (phi @ D1.T))
        c = cx + cy
        inner = (h * h) * float(np.sum(c * phi))
        norm_phi = (h * h) * float(np.sum(phi * phi))
        rel = abs(inner) / max(abs(norm_phi), 1e-300)
        worst = max(worst, rel)
    return worst


def _audit_records(grid_sizes: list[int]) -> list[MorinishiRecord]:
    records: list[MorinishiRecord] = []
    for n in grid_sizes:
        for p in range(2, 9):
            r = _audit_morinishi(f"BMC{p}", n)
            records.append(MorinishiRecord(f"BMC{p}", "BMC", p, n, r))
        for p in range(1, 6):
            r = _audit_morinishi(f"BMG{p}", n)
            records.append(MorinishiRecord(f"BMG{p}", "BMG", p, n, r))
        # baselines through the identical machinery: the orthogonality is a
        # consequence of centered antisymmetry, so every centered periodic
        # operator must (and does) pass at round-off
        for label, fam, q in (("C4", "compact", 4), ("C6", "compact", 6),
                              ("C8", "compact", 8), ("E2", "explicit", 2),
                              ("E4", "explicit", 4), ("E6", "explicit", 6),
                              ("DRP", "drp", 4)):
            r = _audit_morinishi(label, n)
            records.append(MorinishiRecord(label, fam, q, n, r))
    return records


def _figure(records: list[MorinishiRecord], grid_sizes: list[int]) -> None:
    ph.apply_classy_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax_bmc, ax_bmg = axes

    bmc_palette = {
        2: "#22517A", 3: "#2F6BA1", 4: "#4385C8", 5: "#5C9DDB",
        6: ph.ROUTE_COLORS["BMC"], 7: "#7AB1E6", 8: "#94C2EE",
    }
    bmg_palette = {
        1: "#1F6F6F", 2: "#2F8A8A", 3: ph.ROUTE_COLORS["BMG"],
        4: "#5DAFAF", 5: "#7DC0C0",
    }

    for p, color in bmc_palette.items():
        x = grid_sizes
        y = [
            max(next((r.worst_relative for r in records if r.label == f"BMC{p}" and r.npts == n), 1e-18), 1e-18)
            for n in grid_sizes
        ]
        ax_bmc.semilogy(x, y, marker="o", color=color, label=f"BMC{p}", lw=1.0)

    for p, color in bmg_palette.items():
        x = grid_sizes
        y = [
            max(next((r.worst_relative for r in records if r.label == f"BMG{p}" and r.npts == n), 1e-18), 1e-18)
            for n in grid_sizes
        ]
        ax_bmg.semilogy(x, y, marker="s", color=color, label=f"BMG{p}", lw=1.0)

    for ax, title in [(ax_bmc, "(a) BMC catalog"), (ax_bmg, "(b) BMG catalog")]:
        ax.axhline(1e-12, color=ph.CLASSY_GUIDE, lw=0.6, ls="--")
        ax.text(grid_sizes[0], 1.5e-12, "machine-zero band", color="#888", fontsize=8)
        ax.set_xlabel("grid size $N$")
        ax.set_ylabel(r"$\max\, |\langle C, \phi\rangle_H| / \langle \phi, \phi\rangle_H$")
        ax.set_ylim(1e-17, 1e-8)
        ax.set_title(title)
        ph.quiet_axis(ax)
        ph.quiet_legend(ax, loc="best")

    fig.tight_layout()
    fig.savefig(pp.figure_path("fig_s7_morinishi.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    grid_sizes = [32, 48, 64]
    records = _audit_records(grid_sizes)
    print("max Morinishi relative deviation across BMC2-8, BMG1-5, C4-C8, E2-E6, DRP:")
    print(f"  {max(r.worst_relative for r in records):.3e}")
    _figure(records, grid_sizes)
    out = {"records": [r.__dict__ for r in records]}
    with open(pp.results_path("s7_morinishi.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
