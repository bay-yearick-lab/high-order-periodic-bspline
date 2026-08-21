"""Section 7 sub-capability (4): gradient-divergence adjointness.

Verifies the periodic discrete duality corollary
  ``<u, D p>_H + <p, D u>_H = 0``
for every periodic grid functions ``u`` and ``p``. The identity is a
direct consequence of the diagonal-norm SBP first-derivative
algebra and holds unconditionally, with no divergence-free
hypothesis. The validator ranges over many random ``(u, p)`` pairs
on grids of size ``N = 32, 48, 64`` for every operator in the BMC
and BMG catalogs.

The companion check sums per-direction contributions on a 2D
divergence-free velocity field and confirms that
  ``sum_i <u_i, D_i p>_H = 0``
to machine precision (this is the discrete statement that the
pressure-gradient force does no net work on incompressible flow,
identity (eq:periodic-pressure-energy-neutrality) of Section 5).

Outputs: ``figures_paper/fig_s7_pressure_work.pdf``.
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
class PressureRecord:
    label: str
    family: str
    degree: int
    npts: int
    pair_residual: float
    incompressible_residual: float


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


def _audit_pair(label: str, npts: int, n_trials: int = 8) -> tuple[float, float]:
    h = 1.0 / npts
    D1 = _build_first_derivative(label, npts, h)
    rng = np.random.default_rng(20260510)
    worst_pair = 0.0
    worst_ns = 0.0
    for _ in range(n_trials):
        u_ = rng.standard_normal(npts)
        p_ = rng.standard_normal(npts)
        inner_pair = h * (
            float(np.dot(u_, D1 @ p_)) + float(np.dot(p_, D1 @ u_))
        )
        scale = h * (np.linalg.norm(u_) * np.linalg.norm(D1 @ p_)
                     + np.linalg.norm(p_) * np.linalg.norm(D1 @ u_))
        worst_pair = max(worst_pair, abs(inner_pair) / max(scale, 1e-300))

        u = rng.standard_normal((npts, npts))
        v = rng.standard_normal((npts, npts))
        u, v = _project_div_free(D1, label, u, v, npts, h)
        p_field = rng.standard_normal((npts, npts))
        inner_ns = (h * h) * (
            float(np.sum(u * (D1 @ p_field))) + float(np.sum(v * (p_field @ D1.T)))
        )
        norm_ns = (h * h) * (
            np.linalg.norm(u) * np.linalg.norm(D1 @ p_field)
            + np.linalg.norm(v) * np.linalg.norm(p_field @ D1.T)
        )
        worst_ns = max(worst_ns, abs(inner_ns) / max(norm_ns, 1e-300))
    return worst_pair, worst_ns


def _audit_records(grid_sizes: list[int]) -> list[PressureRecord]:
    records: list[PressureRecord] = []
    for n in grid_sizes:
        for p in range(2, 9):
            wp, wn = _audit_pair(f"BMC{p}", n)
            records.append(PressureRecord(f"BMC{p}", "BMC", p, n, wp, wn))
        for p in range(1, 6):
            wp, wn = _audit_pair(f"BMG{p}", n)
            records.append(PressureRecord(f"BMG{p}", "BMG", p, n, wp, wn))
        # The centered baselines share the same SBP gradient-divergence duality.
        for label, fam, q in (("C4", "compact", 4), ("C6", "compact", 6),
                              ("C8", "compact", 8), ("E2", "explicit", 2),
                              ("E4", "explicit", 4), ("E6", "explicit", 6),
                              ("DRP", "drp", 4)):
            wp, wn = _audit_pair(label, n)
            records.append(PressureRecord(label, fam, q, n, wp, wn))
    return records


def _figure(records: list[PressureRecord], grid_sizes: list[int]) -> None:
    ph.apply_classy_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax_pair, ax_ns = axes

    bmc_palette = {
        2: "#22517A", 3: "#2F6BA1", 4: "#4385C8", 5: "#5C9DDB",
        6: ph.ROUTE_COLORS["BMC"], 7: "#7AB1E6", 8: "#94C2EE",
    }
    bmg_palette = {
        1: "#1F6F6F", 2: "#2F8A8A", 3: ph.ROUTE_COLORS["BMG"],
        4: "#5DAFAF", 5: "#7DC0C0",
    }

    def _plot(ax, attr, title):
        for p, color in bmc_palette.items():
            y = [
                max(next((getattr(r, attr) for r in records if r.label == f"BMC{p}" and r.npts == n), 1e-18), 1e-18)
                for n in grid_sizes
            ]
            ax.semilogy(grid_sizes, y, marker="o", color=color, label=f"BMC{p}", lw=1.0)
        for p, color in bmg_palette.items():
            y = [
                max(next((getattr(r, attr) for r in records if r.label == f"BMG{p}" and r.npts == n), 1e-18), 1e-18)
                for n in grid_sizes
            ]
            ax.semilogy(grid_sizes, y, marker="s", color=color, label=f"BMG{p}", lw=1.0, ls="--")
        ax.axhline(1e-12, color=ph.CLASSY_GUIDE, lw=0.6, ls="--")
        ax.text(grid_sizes[0], 1.5e-12, "machine-zero band", color="#888", fontsize=8)
        ax.set_xlabel("grid size $N$")
        ax.set_ylim(1e-17, 1e-8)
        ax.set_title(title)
        ph.quiet_axis(ax)

    _plot(ax_pair, "pair_residual", "(a) scalar pair: $|\\langle u, Dp\\rangle_H + \\langle p, Du\\rangle_H|$")
    _plot(ax_ns, "incompressible_residual", "(b) incompressible 2D: $|\\sum_i\\langle u_i, D_i p\\rangle_H|$")
    ax_pair.set_ylabel("relative residual")
    ph.quiet_legend(ax_pair, loc="best")

    fig.tight_layout()
    fig.savefig(pp.figure_path("fig_s7_pressure_work.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    grid_sizes = [32, 48, 64]
    records = _audit_records(grid_sizes)
    print("max scalar gradient-divergence residual:",
          f"{max(r.pair_residual for r in records):.3e}")
    print("max incompressible 2D pressure-energy residual:",
          f"{max(r.incompressible_residual for r in records):.3e}")
    _figure(records, grid_sizes)
    out = {"records": [r.__dict__ for r in records]}
    with open(pp.results_path("s7_pressure_work.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
