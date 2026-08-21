"""Section 7 sub-capability (5b): matched-order modified-wavenumber ladder.

Audits the matched-order collapse identity of Section~6 entrywise
across the BMC and BMG catalogs (``||BMC_{2p+1} - BMG_p||_inf`` and
``||BMC_{2p+1}^{(2)} - BMG_p^{(2)}||_inf``) and prints the worst
relative residuals, then plots the scaled modified-wavenumber ladder
``k' h(omega)`` of the matched pairs against the analytic
``omega`` reference and the pseudospectral limit. Each successive
matched pair sits closer to the spectral limit, so the parity-driven
nodal superconvergence ladder reaches the cardinal-spline spectral
ceiling asymptotically in ``p``.

Output: ``figures_paper/fig_matched_modwave_ladder.pdf``.
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
class CollapseRecord:
    bmc_label: str
    bmg_label: str
    bmc_degree: int
    bmg_degree: int
    npts: int
    first_derivative_residual: float
    second_derivative_residual: float


def _audit_pairs(grid_sizes: list[int]) -> list[CollapseRecord]:
    records: list[CollapseRecord] = []
    for n in grid_sizes:
        h = 1.0 / n
        for p in range(1, 6):
            bmc_p = 2 * p + 1
            D_bmc = sa.collocation_derivative_matrix(bmc_p, n, h=h)
            D_bmg = sa.iga_derivative_matrix(p, n, h=h)
            res_d = float(np.max(np.abs(D_bmc - D_bmg))) / max(np.max(np.abs(D_bmg)), 1e-300)
            D2_bmc = sa.collocation_second_derivative_matrix(bmc_p, n, h=h)
            D2_bmg = sa.iga_second_derivative_matrix(p, n, h=h)
            res_d2 = float(np.max(np.abs(D2_bmc - D2_bmg))) / max(np.max(np.abs(D2_bmg)), 1e-300)
            records.append(CollapseRecord(
                bmc_label=f"BMC{bmc_p}", bmg_label=f"BMG{p}",
                bmc_degree=bmc_p, bmg_degree=p, npts=n,
                first_derivative_residual=res_d,
                second_derivative_residual=res_d2,
            ))
    return records


def _figure(records: list[CollapseRecord], grid_sizes: list[int]) -> None:
    """Modified-wavenumber landscape for the matched parity ladder.

    Two-panel layout. Panel (a) shows the modified wavenumber
    ``k' h(omega)`` for each matched pair against the analytic diagonal,
    rendered at equal aspect so the spectral approach reads visually.
    Panel (b) shows the per-pair phase-error magnitude on a semilog
    axis, where each successive pair sits below the previous one and
    the cardinal-spline ladder visibly approaches the pseudospectral
    floor as ``p`` increases.
    """
    ph.apply_full_width_print_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4),
                              gridspec_kw={"wspace": 0.30,
                                           "width_ratios": [1.0, 1.05]})
    ax_kp, ax_err = axes

    families = [
        (3, 1, r"$\mathrm{BMC3}{=}\mathrm{BMG1}$", ph.method_style("BMC3")),
        (5, 2, r"$\mathrm{BMC5}{=}\mathrm{BMG2}$", ph.method_style("BMC5")),
        (7, 3, r"$\mathrm{BMC7}{=}\mathrm{BMG3}$", ph.method_style("BMC7")),
        (9, 4, r"$\mathrm{BMC9}{=}\mathrm{BMG4}$", ph.method_style("BMC9")),
        (11, 5, r"$\mathrm{BMC11}{=}\mathrm{BMG5}$", ph.method_style("BMC11")),
    ]

    omega = np.linspace(1e-3, np.pi, 400)

    ax_kp.plot(
        omega, sa.kprime_pseudospectral(omega),
        color=ph.SCHEME_COLORS["Analytic"], lw=1.0, ls=(0, (1, 1.5)),
        label=r"analytic $\omega h$ (pseudospectral)",
    )
    for bmc_p, bmg_p, label, style in families:
        kp_bmc = sa.kprime_collocation(omega, bmc_p)
        kp_bmg = sa.kprime_iga(omega, bmg_p)
        if np.max(np.abs(kp_bmc - kp_bmg)) > 1e-10:
            raise ValueError(f"BMC{bmc_p} and BMG{bmg_p} do not match in symbol")
        ax_kp.plot(omega, kp_bmc, color=style["color"], lw=1.8,
                   ls=style["ls"], marker=style["marker"], markevery=55,
                   ms=4.0, label=label)
    ax_kp.set_xlabel(r"$\omega h$")
    ax_kp.set_ylabel(r"$k'h(\omega)$")
    ax_kp.set_xlim(0, np.pi)
    ax_kp.set_ylim(0, np.pi)
    ax_kp.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_kp.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax_kp.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_kp.set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax_kp.set_aspect("equal", adjustable="box")
    ax_kp.text(0.04, 0.96, "(a)",
               transform=ax_kp.transAxes, ha="left", va="top",
               fontsize=14, color=ph.CLASSY_ACCENT, fontstyle="italic")
    ph.quiet_axis(ax_kp)

    err_floor = 1e-9
    for bmc_p, bmg_p, label, style in families:
        kp_bmc = sa.kprime_collocation(omega, bmc_p)
        err = np.maximum(np.abs(kp_bmc - omega) / np.maximum(omega, 1e-12),
                         err_floor)
        ax_err.semilogy(omega, err, color=style["color"], lw=1.8,
                        ls=style["ls"], marker=style["marker"],
                        markevery=55, ms=4.0, label=label)
    ax_err.axhline(1e-2, color="0.45", lw=0.6, ls=(0, (3, 2)))
    ax_err.text(0.97, 1.25e-2, r"$1\%$ band",
                fontsize=12.0, color="0.40", ha="right")
    ax_err.set_xlabel(r"$\omega h$")
    ax_err.set_ylabel("relative phase error")
    ax_err.set_xlim(0, np.pi)
    ax_err.set_ylim(err_floor, 2.0)
    ax_err.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_err.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax_err.text(0.04, 0.96, "(b)",
                transform=ax_err.transAxes, ha="left", va="top",
                fontsize=14, color=ph.CLASSY_ACCENT, fontstyle="italic")
    ph.quiet_axis(ax_err)

    handles, labels = ax_kp.get_legend_handles_labels()
    ph.make_overhead_legend(
        fig, handles, labels, ncol=3,
        bbox=(0.5, 1.02), rect=(0, 0, 1, 0.82), fontsize=12.0,
    )

    fig.savefig(pp.figure_path("fig_matched_modwave_ladder.pdf"),
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    grid_sizes = [32, 48, 64]
    records = _audit_pairs(grid_sizes)
    print("max first-derivative collapse residual:",
          f"{max(r.first_derivative_residual for r in records):.3e}")
    print("max second-derivative collapse residual:",
          f"{max(r.second_derivative_residual for r in records):.3e}")
    _figure(records, grid_sizes)
    out = {"records": [r.__dict__ for r in records]}
    with open(pp.results_path("s7_collapse_audit.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
