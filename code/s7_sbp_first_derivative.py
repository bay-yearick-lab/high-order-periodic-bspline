"""Section 7 sub-capability (3): first-derivative SBP audit.

Verifies the diagonal-norm SBP identity ``H D + D^T H = 0`` at machine
precision for the periodic spline first-derivative catalog
(``BMC_2`` through ``BMC_11``, ``BMG_1`` through ``BMG_5``) and the
periodic finite-difference baselines ``C_4, C_6, C_8`` and
``E_2, E_4, E_6, E_8``. All centered first-derivative operators on
the periodic uniform grid are antisymmetric circulant and therefore
satisfy this identity, so the audit panel reads as a positive
verification rather than a discriminator; the discriminator emerges
in the *compatible* second-derivative subsection that follows.

The companion panel reports the semi-discrete energy drift
``|E(t) - E(0)| / E(0)`` under periodic linear advection with
FFT-exact time integration on the circulant operator. With time
error eliminated, the spatial SBP property gives identically
zero drift to machine precision; this is the structural
``(d/dt) <u, u>_H = 0`` claim of Theorem~1 made visible.

Outputs: ``figures_paper/fig_s7_sbp_residual.pdf``,
``figures_paper/fig_s7_advection_energy.pdf``.
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
class SBPResidualRecord:
    label: str
    family: str
    degree_or_order: int
    residual_inf: float


def _build_first_derivative(label: str, npts: int, h: float) -> np.ndarray:
    if label.startswith("BMC"):
        return sa.collocation_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("BMG"):
        return sa.iga_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("C") and label[1:].isdigit():
        order = int(label[1:])
        mass, advection = sa.compact_scheme_entries(order)
        return sa.compact_derivative_matrix(mass, advection, npts, h=h)
    if label.startswith("E") and label[1:].isdigit():
        order = int(label[1:])
        mass, advection = sa.explicit_centered_fd_entries(order)
        return sa.compact_derivative_matrix(mass, advection, npts, h=h)
    raise ValueError(f"Unknown operator label {label!r}")


def _audit_sbp_residual(npts: int) -> list[SBPResidualRecord]:
    h = 1.0 / npts
    records: list[SBPResidualRecord] = []
    for p in range(2, 12):
        D = sa.collocation_derivative_matrix(p, npts, h=h)
        residual = sa.periodic_sbp_residual(D, h=h)
        records.append(
            SBPResidualRecord(f"BMC{p}", "BMC", p, float(np.max(np.abs(residual))))
        )
    for p in range(1, 6):
        D = sa.iga_derivative_matrix(p, npts, h=h)
        residual = sa.periodic_sbp_residual(D, h=h)
        records.append(
            SBPResidualRecord(f"BMG{p}", "BMG", p, float(np.max(np.abs(residual))))
        )
    for q in (4, 6, 8):
        D = _build_first_derivative(f"C{q}", npts, h)
        residual = sa.periodic_sbp_residual(D, h=h)
        records.append(
            SBPResidualRecord(f"C{q}", "Compact", q, float(np.max(np.abs(residual))))
        )
    for q in (2, 4, 6, 8):
        D = _build_first_derivative(f"E{q}", npts, h)
        residual = sa.periodic_sbp_residual(D, h=h)
        records.append(
            SBPResidualRecord(f"E{q}", "Explicit", q, float(np.max(np.abs(residual))))
        )
    return records


def _exact_advection_history(
    label: str, npts: int, times: np.ndarray
) -> np.ndarray:
    """FFT-exact time integration of u_t = -a D u on the periodic torus.

    Assumes the spatial operator is circulant. Eigenvalues are obtained
    from the column generator via FFT, and the propagator at each time
    is applied as a per-mode complex exponential. This eliminates the
    time-integration error and lets the spatial SBP identity show its
    semi-discrete content directly.
    """
    h = 1.0 / npts
    D = _build_first_derivative(label, npts, h)
    column = D[:, 0]
    eigvals = np.fft.fft(column)
    x = np.arange(npts) * h
    u0 = (
        np.sin(2.0 * np.pi * x)
        + 0.5 * np.sin(4.0 * np.pi * x + 0.7)
        + 0.25 * np.sin(8.0 * np.pi * x + 1.3)
    )
    u0_hat = np.fft.fft(u0)
    energy0 = h * float(np.dot(u0, u0))
    drift = np.zeros_like(times)
    for k, t in enumerate(times):
        u_hat = np.exp(-eigvals * t) * u0_hat
        u = np.real(np.fft.ifft(u_hat))
        energy = h * float(np.dot(u, u))
        drift[k] = abs(energy - energy0) / energy0
    return drift


def _figure_residual(records: list[SBPResidualRecord]) -> None:
    ph.apply_classy_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    panel_a, panel_b = axes

    bmc = [r for r in records if r.family == "BMC"]
    bmg = [r for r in records if r.family == "BMG"]
    panel_a.semilogy(
        [r.degree_or_order for r in bmc],
        [max(r.residual_inf, 1e-18) for r in bmc],
        marker="o",
        color=ph.ROUTE_COLORS["BMC"],
        label=r"$\mathrm{BMC}_p$",
    )
    panel_a.semilogy(
        [r.degree_or_order for r in bmg],
        [max(r.residual_inf, 1e-18) for r in bmg],
        marker="s",
        color=ph.ROUTE_COLORS["BMG"],
        label=r"$\mathrm{BMG}_p$",
    )
    panel_a.axhline(1e-12, color=ph.CLASSY_GUIDE, lw=0.6, ls="--")
    panel_a.text(1, 1.5e-12, "machine-zero band", color="#888", fontsize=8)
    panel_a.set_xlabel("spline degree $p$")
    panel_a.set_ylabel(r"$\|HD + D^{T}H\|_\infty$")
    panel_a.set_ylim(1e-17, 1e-8)
    panel_a.set_title("(a) periodic spline routes")
    ph.quiet_axis(panel_a)
    ph.quiet_legend(panel_a, loc="upper right")

    compact = [r for r in records if r.family == "Compact"]
    explicit = [r for r in records if r.family == "Explicit"]
    panel_b.semilogy(
        [r.degree_or_order for r in compact],
        [max(r.residual_inf, 1e-18) for r in compact],
        marker="^",
        color=ph.ROUTE_COLORS["Cq"],
        label=r"$\mathrm{C}_q$",
    )
    panel_b.semilogy(
        [r.degree_or_order for r in explicit],
        [max(r.residual_inf, 1e-18) for r in explicit],
        marker="d",
        color=ph.ROUTE_COLORS["E6"],
        label=r"$\mathrm{E}_q$",
    )
    panel_b.axhline(1e-12, color=ph.CLASSY_GUIDE, lw=0.6, ls="--")
    panel_b.text(2, 1.5e-12, "machine-zero band", color="#888", fontsize=8)
    panel_b.set_xlabel("formal order $q$")
    panel_b.set_ylim(1e-17, 1e-8)
    panel_b.set_title("(b) periodic finite-difference baselines")
    ph.quiet_axis(panel_b)
    ph.quiet_legend(panel_b, loc="upper right")

    fig.tight_layout()
    out = pp.figure_path("fig_s7_sbp_residual.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _figure_energy(times: np.ndarray, histories: dict[str, np.ndarray]) -> None:
    ph.apply_classy_defaults()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    color_map = {
        "BMC6": ph.ROUTE_COLORS["BMC"],
        "BMG6": ph.ROUTE_COLORS["BMG"],
        "C6": ph.ROUTE_COLORS["Cq"],
        "E6": ph.ROUTE_COLORS["E6"],
    }
    for label, drift in histories.items():
        ax.semilogy(
            times,
            np.clip(drift, 1e-17, None),
            label=label,
            color=color_map.get(label, "#444"),
            lw=1.0,
        )
    ax.axhline(1e-12, color=ph.CLASSY_GUIDE, lw=0.6, ls="--")
    ax.text(times[-1] * 0.55, 1.5e-12, "machine-zero band", color="#888", fontsize=8)
    ax.set_xlabel("time (FFT-exact integration)")
    ax.set_ylabel(r"$|E(t) - E(0)| / E(0)$")
    ax.set_ylim(1e-17, 1e-8)
    ax.set_title("Periodic linear advection: semi-discrete energy drift")
    ph.quiet_axis(ax)
    ph.quiet_legend(ax, loc="lower right")
    fig.tight_layout()
    out = pp.figure_path("fig_s7_advection_energy.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    records = _audit_sbp_residual(npts=64)
    headline = max((r.residual_inf for r in records if r.label in ("BMC6", "BMG6")), default=0.0)
    print(f"max SBP residual across BMC2-11 + BMG1-5 + C4/6/8 + E2/4/6/8: "
          f"{max(r.residual_inf for r in records):.3e}")
    print(f"BMC6 / BMG6 residual = {headline:.3e}")
    _figure_residual(records)

    times = np.linspace(0.0, 50.0, 200)
    histories = {
        label: _exact_advection_history(label, npts=48, times=times)
        for label in ("BMC6", "BMG6", "C6", "E6")
    }
    finals = {label: float(drift[-1]) for label, drift in histories.items()}
    print(f"final FFT-exact energy drift at t=50: {finals}")
    _figure_energy(times, histories)

    out = {
        "sbp_residuals": [r.__dict__ for r in records],
        "energy_drift_finals_at_t50": finals,
    }
    with open(pp.results_path("s7_sbp_first_derivative.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
