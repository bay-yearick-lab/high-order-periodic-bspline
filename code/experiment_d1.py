#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "joblib"]
# ///
"""
Textbook-style modified-wavenumber study for periodic first and second derivatives.

This script generates the paper figures used for the Section 7 cost-versus-accuracy
discussion and the upstream pedagogy.  The off-grid, IBM, and wider-catalog
figures referenced in Section 8 are produced by experiment_offgrid.py and
experiment_ibm.py.

Pedagogy figures:
  fig_wavenumbers.pdf                     : What a discrete wavenumber means
  fig_modified_wavenumber_derivation.pdf  : Continuous wave -> sample -> stencil -> symbol
  fig_collocation_vs_galerkin.pdf         : Pinning at nodes vs L2 projection
  fig_bm_banded_maps.pdf                  : First-derivative BM maps plus periodic SBP cue
  fig_bm_second_derivative_structure.pdf  : Second-derivative BM maps plus energy cue
  fig_label_fairness.pdf                  : How the labels work and what is fair to compare
  fig_equivalence_map.pdf                 : Spaces times formulations diagram
  fig_bspline_ladder.pdf                  : Cardinal splines as repeated box convolutions
  fig_overlap_identity.pdf                : Sliding overlaps produce higher-degree samples
  fig_matrix_identity_bmc5_iga2.pdf       : BMC5 and BMG2 produce the same periodic matrix
  fig_stability_regions.pdf               : Forward Euler vs RK4 stability regions
  fig_bspline_degree_ladder.pdf           : Odd-degree BM family with a cost-tagged PPW ladder
  fig_anisotropy_order6.pdf               : Sixth-order phase-speed anisotropy at a hard PPW
  fig_second_derivative_dispersion.pdf    : Direct second-derivative symbol and error landscape

Results figures:
  fig_matched_order.pdf       : Matched-order comparisons
  fig_same_degree.pdf         : Same-degree spline comparisons (BMC vs BMG)
  fig_cost_tradeoff.pdf       : Resolution versus two cost proxies
  fig_second_derivative_diffusion.pdf : Direct second-derivative diffusion accuracy and cost
  fig_second_derivative_mission_design.pdf : Direct second-derivative mission-design chooser
  fig_phase_stress.pdf        : Same-order phase-lag stress test with packet envelope
  fig_phase_budget.pdf        : Phase-budget curves plus a fixed-mesh decision map
  fig_dimensional_payoff.pdf  : Optimized-grid work and memory payoff in 1D/2D/3D
  fig_mission_design.pdf      : Cheapest admissible sixth-order method over (M, d, Pmax)
  fig_pde_convergence.pdf     : Sixth-order PDE convergence on a smooth Gaussian pulse
  fig_pde_snapshot.pdf        : Representative pulse snapshot at a stress PPW
"""

from dataclasses import dataclass
import os
from fractions import Fraction
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import spline_analysis as sa

THISDIR = Path(__file__).resolve().parent
REPO_ROOT = THISDIR.parent if THISDIR.name == "code" else THISDIR
FIGDIR = REPO_ROOT / "figures_paper"
FIGDIR.mkdir(exist_ok=True)
# Figures the manuscript no longer references go to a side directory, so that a
# run of this script never leaves an orphan PDF in figures_paper (the verifier's
# figure gate counts orphans).
UNUSED_FIGDIR = FIGDIR / "unused"
UNUSED_FIGDIR.mkdir(exist_ok=True)
PARALLEL_JOBS = int(os.environ.get("BSPLINE_JOBS", "1"))


def setup_style() -> None:
    plt.rcParams.update(
        {
            # TrueType (Type 42), never Type 3 -- arXiv rejects Type-3 fonts.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Seamless with the lmodern manuscript: Latin Modern Roman text,
            # Computer Modern math, DejaVu Serif fallback only.
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9.4,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.8,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def setup_full_width_print_style(*, three_panel: bool = False) -> None:
    """Apply source text sizes that remain at least 8 pt in the manuscript."""
    setup_style()
    source_size = 13.5 if three_panel else 12.0
    tick_size = 13.2 if three_panel else 12.0
    plt.rcParams.update(
        {
            "font.size": source_size,
            "axes.titlesize": source_size,
            "axes.labelsize": source_size,
            "legend.fontsize": source_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
        }
    )


def padded_positive_limits(
    values: list[float] | np.ndarray,
    *,
    lower_scale: float = 0.75,
    upper_scale: float = 1.35,
    floor: float | None = None,
    ceiling: float | None = None,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    lo = float(arr.min()) * lower_scale
    hi = float(arr.max()) * upper_scale
    if floor is not None:
        lo = max(lo, floor)
    if ceiling is not None:
        hi = min(hi, ceiling)
    return lo, hi


def add_better_arrow(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    text: str = "better",
) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        fontsize=9.0,
        color="#444444",
        arrowprops=dict(arrowstyle="->", lw=1.25, color="#444444"),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.15),
    )


def lighten(color: str, amount: float = 0.46) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color), dtype=float)
    return tuple(1.0 - amount * (1.0 - rgb))


try:
    from plotting_helpers import (  # type: ignore
        METHOD_STYLES as _METHOD_STYLES,
        SCHEME_COLORS as _SCHEME_COLORS,
        add_slope_reference as _add_slope_reference,
        add_theoretical_reference_line as _add_theoretical_reference_line,
        make_overhead_legend as _make_overhead_legend,
    )
except Exception:  # pragma: no cover - defensive fallback for ad-hoc runs
    _SCHEME_COLORS = {
        "BMC": "#22517a",
        "BMG": "#1c7c34",
        "C": "#b06a2c",
        "E": "#6b6b6b",
        "DRP": "#a83232",
        "BB": "#2c7d4f",
        "PS": "#3a3a3a",
        "Analytic": "#1a1a1a",
        "ParedLagrange": "#7a7a7a",
    }
    _add_slope_reference = None  # type: ignore
    _add_theoretical_reference_line = None  # type: ignore
    _make_overhead_legend = None  # type: ignore
    _METHOD_STYLES = {}

COLORS = {
    "Exact": _SCHEME_COLORS["Analytic"],
    "E2": "#A6CEE3",
    "E4": _METHOD_STYLES.get("E4", {"color": "#bdbdbd"})["color"],
    "E6": _METHOD_STYLES.get("E6", {"color": "#969696"})["color"],
    "E8": _METHOD_STYLES.get("E8", {"color": "#737373"})["color"],
    "E10": _METHOD_STYLES.get("E10", {"color": "#525252"})["color"],
    "E12": _METHOD_STYLES.get("E12", {"color": "#252525"})["color"],
    "C4 = BMC3 = BMG1": _METHOD_STYLES.get("BMC3", {"color": "#9ecae1"})["color"],
    "C6": _METHOD_STYLES.get("C6", {"color": "#e08214"})["color"],
    "C8": _METHOD_STYLES.get("C8", {"color": "#d95f0e"})["color"],
    "C10": _METHOD_STYLES.get("C10", {"color": "#b35806"})["color"],
    "C12": _METHOD_STYLES.get("C12", {"color": "#7f2704"})["color"],
    "BMC5 = BMG2": _METHOD_STYLES.get("BMC5", {"color": "#6baed6"})["color"],
    "BMC7 = BMG3": _METHOD_STYLES.get("BMC7", {"color": "#3182bd"})["color"],
    "BMC9 = BMG4": _METHOD_STYLES.get("BMC9", {"color": "#08519c"})["color"],
    "BMC11 = BMG5": _METHOD_STYLES.get("BMC11", {"color": "#08306b"})["color"],
    "BMC2": _METHOD_STYLES.get("BMC2", {"color": "#c6dbef"})["color"],
    "BMC3": _METHOD_STYLES.get("BMC3", {"color": "#9ecae1"})["color"],
    "BMC5": _METHOD_STYLES.get("BMC5", {"color": "#6baed6"})["color"],
    "BMC6": _METHOD_STYLES.get("BMC6", {"color": "#4292c6"})["color"],
    "BMC8": _METHOD_STYLES.get("BMC8", {"color": "#2171b5"})["color"],
    "BMC10": _METHOD_STYLES.get("BMC10", {"color": "#084594"})["color"],
}

SECTION7_COLOR_OVERRIDES: dict[str, str] = {}


def section7_color(label: str) -> str:
    return SECTION7_COLOR_OVERRIDES.get(label, COLORS[label])


LINESTYLES = {
    "Exact": {"ls": "--", "lw": 1.2, "zorder": 0},
    "E2": {"ls": ":"},
    "E4": {"ls": "-."},
    "E6": {"ls": "-."},
    "E8": {"ls": "-."},
    "E10": {"ls": "-."},
    "E12": {"ls": "-."},
    "C4 = BMC3 = BMG1": {"ls": "-"},
    "C6": {"ls": "-"},
    "C8": {"ls": "-"},
    "C10": {"ls": "-"},
    "C12": {"ls": "-"},
    "BMC5 = BMG2": {"ls": "--"},
    "BMC7 = BMG3": {"ls": "-"},
    "BMC9 = BMG4": {"ls": "-"},
    "BMC11 = BMG5": {"ls": "-"},
    "BMC2": {"ls": "--"},
    "BMC3": {"ls": "--"},
    "BMC5": {"ls": "--"},
    "BMC6": {"ls": "--"},
    "BMC8": {"ls": "--"},
    "BMC10": {"ls": "--"},
}


def formal_order_for_label(label: str) -> int:
    if label.startswith("E") and label[1:].isdigit():
        return sa.formal_order_explicit(int(label[1:]))
    if label.startswith("C") and label[1:].isdigit():
        return sa.formal_order_compact(int(label[1:]))
    if label.startswith("BMC") and label[3:].isdigit():
        return sa.formal_order_collocation(int(label[3:]))
    if label.startswith("BMG") and label[3:].isdigit():
        return sa.formal_order_iga(int(label[3:]))

    aliases = {
        "C4 = BMC3 = BMG1": sa.formal_order_compact(4),
        "BMC5 = BMG2": sa.formal_order_collocation(5),
        "BMC7 = BMG3": sa.formal_order_iga(3),
        "BMC9 = BMG4": sa.formal_order_iga(4),
        "BMC11 = BMG5": sa.formal_order_iga(5),
    }
    try:
        return aliases[label]
    except KeyError as exc:
        raise KeyError(f"Unknown scheme label: {label}") from exc


FORMAL_ORDER = {
    label: formal_order_for_label(label)
    for label in [
        "E2",
        "E4",
        "E6",
        "E8",
        "E10",
        "E12",
        "C4",
        "C6",
        "C8",
        "BMC2",
        "BMC3",
        "BMC5",
        "BMC6",
        "BMC7",
        "BMC8",
        "BMC9",
        "BMC10",
        "BMC11",
        "BMG1",
        "BMG2",
        "BMG3",
        "BMG4",
        "BMG5",
        "C4 = BMC3 = BMG1",
        "BMC5 = BMG2",
        "BMC7 = BMG3",
        "BMC9 = BMG4",
        "BMC11 = BMG5",
    ]
}


NATIVE_WORK_UNITS = {
    "E2": 4,
    "E4": 7,
    "E6": 10,
    "E8": 13,
    "E10": 16,
    "E12": 19,
    "C4 = BMC3 = BMG1": 8,
    "C6": 11,
    "C8": 14,
    "BMC2": 8,
    "BMC5 = BMG2": 15,
    "BMC6": 22,
    "BMC7 = BMG3": 22,
    "BMC8": 29,
    "BMC9 = BMG4": 29,
    "BMC10": 36,
    "BMC11 = BMG5": 36,
}


HALO_WIDTH = {
    "E2": 1,
    "E4": 2,
    "E6": 3,
    "E8": 4,
    "E10": 5,
    "E12": 6,
    "C4 = BMC3 = BMG1": 1,
    "C6": 2,
    "C8": 3,
    "BMC2": 1,
    "BMC5 = BMG2": 2,
    "BMC6": 3,
    "BMC7 = BMG3": 3,
    "BMC8": 4,
    "BMC9 = BMG4": 4,
    "BMC10": 5,
    "BMC11 = BMG5": 5,
}


UNIQUE_METHODS = [
    "E2",
    "E4",
    "E6",
    "E8",
    "E10",
    "E12",
    "BMC2",
    "C4 = BMC3 = BMG1",
    "C6",
    "C8",
    "BMC5 = BMG2",
    "BMC6",
    "BMC7 = BMG3",
    "BMC8",
    "BMC9 = BMG4",
    "BMC10",
    "BMC11 = BMG5",
]


SECOND_DERIVATIVE_UNIQUE_METHODS = [
    "E2",
    "E4",
    "E6",
    "E8",
    "E10",
    "E12",
    "BMC2",
    "C4 = BMC3 = BMG1",
    "C6",
    "BMC5 = BMG2",
    "BMC6",
    "BMC7 = BMG3",
    "BMC8",
    "BMC9 = BMG4",
    "BMC10",
    "BMC11 = BMG5",
]


SECOND_DERIVATIVE_GROUPS = [
    ("(a) Second-order start", ["E2", "BMC2"], "upper left", 1),
    ("(b) First duplicate-collapse block", ["E4", "C4 = BMC3 = BMG1", "BMC5 = BMG2"], "upper left", 1),
    ("(c) Mid-order comparison block", ["E6", "C6", "BMC6", "BMC7 = BMG3"], "upper left", 1),
    (
        "(d) Higher-order tail",
        ["E8", "E10", "E12", "BMC8", "BMC9 = BMG4", "BMC10", "BMC11 = BMG5"],
        "upper left",
        2,
    ),
]


FIXED_RADIUS_GROUPS = [
    ("Stencil radius $R=1$", ["E2", "C4 = BMC3 = BMG1"]),
    ("Stencil radius $R=2$", ["E4", "C6", "BMC5 = BMG2"]),
    ("Stencil radius $R=3$", ["E6", "C8", "BMC7 = BMG3"]),
    ("Stencil radius $R=4$", ["E8", "BMC9 = BMG4"]),
    ("Stencil radius $R=5$", ["E10", "BMC11 = BMG5"]),
    ("Stencil radius $R=6$", ["E12"]),
]


FIXED_ORDER_GROUPS = [
    ("Fourth order", ["E4", "C4 = BMC3 = BMG1"]),
    ("Sixth order", ["E6", "C6", "BMC5 = BMG2"]),
    ("Eighth order", ["E8", "C8", "BMC7 = BMG3"]),
    ("Tenth and twelfth order", ["E10", "C10", "BMC9 = BMG4", "E12", "C12", "BMC11 = BMG5"]),
]


@dataclass(frozen=True)
class SpatialCostModel:
    rhs_pairs: int
    solve_radius: int
    derivative_flops_per_point: int

    def derivative_flops(self, npts: int) -> int:
        return self.derivative_flops_per_point * npts


RepeatedFlopModel = SpatialCostModel


@dataclass
class SpatialOperator:
    label: str
    h: float
    mass_entries: np.ndarray
    advection_entries: np.ndarray
    solve_class: str
    spatial_cost: SpatialCostModel
    setup_time: float
    solver: object | None

    def derivative(self, u: np.ndarray) -> np.ndarray:
        rhs = np.zeros_like(u)
        for offset, coeff in enumerate(self.advection_entries[1:], start=1):
            if abs(coeff) < 1e-15:
                continue
            rhs += coeff * (np.roll(u, -offset) - np.roll(u, offset))
        if self.solver is None:
            return rhs / self.h
        return self.solver.solve(rhs) / self.h


@dataclass
class SecondDerivativeOperator:
    label: str
    h: float
    mass_entries: np.ndarray
    second_derivative_entries: np.ndarray
    solve_class: str
    spatial_cost: SpatialCostModel
    setup_time: float
    solver: object | None

    def second_derivative(self, u: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(self.second_derivative_entries, dtype=float)
        rhs = coeffs[0] * u
        for offset, coeff in enumerate(coeffs[1:], start=1):
            if abs(coeff) < 1e-15:
                continue
            rhs += coeff * (np.roll(u, -offset) + np.roll(u, offset))
        if self.solver is None:
            return rhs / (self.h**2)
        return self.solver.solve(rhs) / (self.h**2)


@dataclass(frozen=True)
class MarchResult:
    label: str
    npts: int
    error: float
    dt: float
    nsteps: int
    repeated_flops: float | None = None
    wall_time: float | None = None
    setup_time: float | None = None


@dataclass(frozen=True)
class SpatialAccuracyResult:
    label: str
    npts: int
    error: float
    derivative_flops: int


@dataclass(frozen=True)
class CarrierRegimeSpec:
    slug: str
    title: str
    mode: int
    sigma_env_scale: float
    time_final: float
    common_cfl: float
    fairness_lens: str


@dataclass(frozen=True)
class ShortwaveCandidateSummary:
    regime: CarrierRegimeSpec
    bspline_beats_compact: int
    bspline_beats_explicit: int
    compact_over_bspline_geomean: float
    explicit_over_bspline_geomean: float
    bspline_time_over_compact_geomean: float


PDE_TRIO_LABELS = ("E6", "C6", "BMC5 = BMG2")
DIRECT_SECOND_DERIVATIVE_TRIO_LABELS = ("E6", "C6", "BMC7 = BMG3")
SHORTWAVE_BASELINE_LABELS = ("E6", "C6", "BMC5 = BMG2")
GAUSSIAN_CONVERGENCE_TIME = 20.0
GAUSSIAN_SNAPSHOT_TIME = 400.0
PHASE_STRESS_PPW = 4.5
PHASE_STRESS_DISTANCE_WAVELENGTHS = 160.0
PHASE_SLIP_TOLERANCE_CYCLES = 0.25
DIRECT_DIFFUSION_TIME = 35.0
DIRECT_DIFFUSION_NU = 8.0e-4
DIRECT_DIFFUSION_SIGMA = 0.06
DIRECT_DIFFUSION_RELATIVE_AMPLITUDE_TOL = 5.0e-2
DIRECT_DIFFUSION_MISSION_RELATIVE_AMPLITUDE_TOL = 2.0e-2
DIRECT_DIFFUSION_MISSION_WINDOW = (1.0e-2, 1.0)
DIRECT_DIFFUSION_MISSION_PPW_BUDGET_WINDOW = (3.0, 8.0)

SHORTWAVE_CANDIDATES = [
    CarrierRegimeSpec(
        slug="current_stress",
        title="Near-threshold short-wave carrier packet",
        mode=40,
        sigma_env_scale=8.0,
        time_final=1.5,
        common_cfl=0.05,
        fairness_lens="fixed packet, fixed grid, common RK4 CFL",
    ),
    CarrierRegimeSpec(
        slug="long_packet",
        title="Long-horizon carrier packet",
        mode=56,
        sigma_env_scale=6.0,
        time_final=8.0,
        common_cfl=0.05,
        fairness_lens="fixed packet, fixed grid, common RK4 CFL",
    ),
    CarrierRegimeSpec(
        slug="long_packet_soft",
        title="Long-horizon broad-envelope packet",
        mode=56,
        sigma_env_scale=8.0,
        time_final=8.0,
        common_cfl=0.05,
        fairness_lens="fixed packet, fixed grid, common RK4 CFL",
    ),
    CarrierRegimeSpec(
        slug="borderline_long",
        title="Borderline long-run packet",
        mode=72,
        sigma_env_scale=8.0,
        time_final=8.0,
        common_cfl=0.05,
        fairness_lens="fixed packet, fixed grid, common RK4 CFL",
    ),
]


def _carrier_regime_by_slug(slug: str) -> CarrierRegimeSpec:
    for regime in SHORTWAVE_CANDIDATES:
        if regime.slug == slug:
            return regime
    raise KeyError(f"Unknown carrier regime slug: {slug}")


PAPER_SHORTWAVE_REGIME = _carrier_regime_by_slug("current_stress")


def _geometric_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def summarize_shortwave_candidate(
    regime: CarrierRegimeSpec,
    results: dict[str, list[MarchResult]],
) -> ShortwaveCandidateSummary:
    e6_results = sorted(results["E6"], key=lambda item: item.npts)
    c6_results = sorted(results["C6"], key=lambda item: item.npts)
    bmc_results = sorted(results["BMC5 = BMG2"], key=lambda item: item.npts)

    if not (
        len(e6_results) == len(c6_results) == len(bmc_results)
        and [item.npts for item in e6_results] == [item.npts for item in c6_results] == [item.npts for item in bmc_results]
    ):
        raise ValueError("Short-wave candidate summaries require matching grid points across methods.")

    compact_over_bspline = []
    explicit_over_bspline = []
    bspline_over_compact_time = []
    bspline_beats_compact = 0
    bspline_beats_explicit = 0

    for e6_item, c6_item, bmc_item in zip(e6_results, c6_results, bmc_results):
        compact_over_bspline.append(c6_item.error / bmc_item.error)
        explicit_over_bspline.append(e6_item.error / bmc_item.error)
        if bmc_item.wall_time is not None and c6_item.wall_time is not None:
            bspline_over_compact_time.append(bmc_item.wall_time / c6_item.wall_time)
        if bmc_item.error < c6_item.error:
            bspline_beats_compact += 1
        if bmc_item.error < e6_item.error:
            bspline_beats_explicit += 1

    return ShortwaveCandidateSummary(
        regime=regime,
        bspline_beats_compact=bspline_beats_compact,
        bspline_beats_explicit=bspline_beats_explicit,
        compact_over_bspline_geomean=_geometric_mean(compact_over_bspline),
        explicit_over_bspline_geomean=_geometric_mean(explicit_over_bspline),
        bspline_time_over_compact_geomean=_geometric_mean(bspline_over_compact_time),
    )


def choose_headline_shortwave_regime(
    summaries: list[ShortwaveCandidateSummary],
) -> ShortwaveCandidateSummary:
    if not summaries:
        raise ValueError("At least one short-wave candidate summary is required.")
    return max(
        summaries,
        key=lambda item: (
            item.bspline_beats_compact,
            item.compact_over_bspline_geomean,
            item.bspline_beats_explicit,
            item.explicit_over_bspline_geomean,
            -item.bspline_time_over_compact_geomean,
        ),
    )


def carrier_sigma_env(regime: CarrierRegimeSpec, length: float = 1.0) -> float:
    return length / regime.sigma_env_scale


def solve_radius_from_mass_entries(mass_entries: np.ndarray) -> int:
    active = np.where(np.abs(np.asarray(mass_entries, dtype=float)) > 1e-14)[0]
    return int(active.max()) if active.size else 0


def rhs_pairs_from_advection_entries(advection_entries: np.ndarray) -> int:
    return int(
        np.count_nonzero(np.abs(np.asarray(advection_entries, dtype=float)[1:]) > 1e-14)
    )


def repeated_flop_model(
    mass_entries: np.ndarray,
    advection_entries: np.ndarray,
) -> RepeatedFlopModel:
    rhs_pairs = rhs_pairs_from_advection_entries(advection_entries)
    solve_radius = solve_radius_from_mass_entries(mass_entries)
    derivative_flops_per_point = 3 * rhs_pairs + 4 * solve_radius + 1
    return RepeatedFlopModel(
        rhs_pairs=rhs_pairs,
        solve_radius=solve_radius,
        derivative_flops_per_point=derivative_flops_per_point,
    )


def solve_class_from_mass_entries(mass_entries: np.ndarray) -> str:
    solve_radius = solve_radius_from_mass_entries(mass_entries)
    if solve_radius == 0:
        return "explicit"
    names = {
        1: "tridiagonal banded",
        2: "pentadiagonal banded",
        3: "heptadiagonal banded",
    }
    return names.get(solve_radius, f"{2 * solve_radius + 1}-diagonal banded")


def unique_scheme_entries(label: str) -> tuple[np.ndarray, np.ndarray]:
    # The exponential-family test comes first: an "EXP:<path>" label also starts
    # with "E", which would otherwise be read as an explicit centered order.
    if sa.is_exp_member_label(label):
        return sa.exp_member_mass_and_advection_entries(label)
    if label.startswith("E"):
        return sa.explicit_centered_fd_entries(int(label[1:]))
    if label == "C4 = BMC3 = BMG1":
        return sa.compact_scheme_entries(4)
    if label == "C6":
        return sa.compact_scheme_entries(6)
    if label == "C8":
        return sa.compact_scheme_entries(8)
    if label == "BMC5 = BMG2":
        return sa.collocation_mass_and_advection_entries(5)
    if label == "BMC7 = BMG3":
        return sa.collocation_mass_and_advection_entries(7)
    if label == "BMC9 = BMG4":
        return sa.collocation_mass_and_advection_entries(9)
    if label == "BMC11 = BMG5":
        return sa.collocation_mass_and_advection_entries(11)
    if label.startswith("BMC") and label[3:].isdigit():
        return sa.collocation_mass_and_advection_entries(int(label[3:]))
    if label.startswith("BMG") and label[3:].isdigit():
        return sa.iga_mass_and_advection_entries(int(label[3:]))
    raise KeyError(f"Unknown scheme label: {label}")


def unique_second_derivative_scheme_entries(label: str) -> tuple[np.ndarray, np.ndarray]:
    if sa.is_exp_member_label(label):
        return sa.exp_member_mass_and_second_derivative_entries(label)
    if label.startswith("E"):
        return sa.explicit_centered_second_derivative_entries(int(label[1:]))
    if label == "C4 = BMC3 = BMG1":
        return sa.compact_second_derivative_scheme_entries(4)
    if label == "C6":
        return sa.compact_second_derivative_scheme_entries(6)
    if label == "BMC5 = BMG2":
        return sa.collocation_mass_and_second_derivative_entries(5)
    if label == "BMC7 = BMG3":
        return sa.collocation_mass_and_second_derivative_entries(7)
    if label == "BMC9 = BMG4":
        return sa.collocation_mass_and_second_derivative_entries(9)
    if label == "BMC11 = BMG5":
        return sa.collocation_mass_and_second_derivative_entries(11)
    if label.startswith("BMC") and label[3:].isdigit():
        return sa.collocation_mass_and_second_derivative_entries(int(label[3:]))
    if label.startswith("BMG") and label[3:].isdigit():
        return sa.iga_mass_and_second_derivative_entries(int(label[3:]))
    raise KeyError(f"Unknown direct second-derivative scheme label: {label}")


def second_derivative_work_units(label: str) -> int:
    mass_entries, second_derivative_entries = unique_second_derivative_scheme_entries(label)
    return repeated_flop_model(mass_entries, second_derivative_entries).derivative_flops_per_point


def build_spatial_operator(label: str, npts: int, length: float = 1.0) -> SpatialOperator:
    h = length / npts
    t0 = perf_counter()
    mass_entries, advection_entries = unique_scheme_entries(label)
    solve_class = solve_class_from_mass_entries(mass_entries)
    cost_model = repeated_flop_model(mass_entries, advection_entries)
    solver = None
    if solve_class != "explicit":
        solver = sa.factorize_cyclic_compact_mass(mass_entries, npts)
    setup_time = perf_counter() - t0
    return SpatialOperator(
        label=label,
        h=h,
        mass_entries=np.asarray(mass_entries, dtype=float),
        advection_entries=np.asarray(advection_entries, dtype=float),
        solve_class=solve_class,
        spatial_cost=cost_model,
        setup_time=setup_time,
        solver=solver,
    )


def build_second_derivative_operator(
    label: str,
    npts: int,
    length: float = 1.0,
) -> SecondDerivativeOperator:
    h = length / npts
    t0 = perf_counter()
    mass_entries, second_derivative_entries = unique_second_derivative_scheme_entries(label)
    solve_class = solve_class_from_mass_entries(mass_entries)
    cost_model = repeated_flop_model(mass_entries, second_derivative_entries)
    solver = None
    if solve_class != "explicit":
        solver = sa.factorize_cyclic_compact_mass(mass_entries, npts)
    setup_time = perf_counter() - t0
    return SecondDerivativeOperator(
        label=label,
        h=h,
        mass_entries=np.asarray(mass_entries, dtype=float),
        second_derivative_entries=np.asarray(second_derivative_entries, dtype=float),
        solve_class=solve_class,
        spatial_cost=cost_model,
        setup_time=setup_time,
        solver=solver,
    )


def grid_points(npts: int, length: float = 1.0) -> np.ndarray:
    return np.linspace(0.0, length, npts, endpoint=False)


def periodic_centered_distance(x: np.ndarray, center: float, length: float) -> np.ndarray:
    return (x - center + 0.5 * length) % length - 0.5 * length


def gaussian_pulse(
    x: np.ndarray,
    *,
    length: float = 1.0,
    sigma: float | None = None,
    center: float | None = None,
) -> np.ndarray:
    sigma = length / 16.0 if sigma is None else sigma
    center = 0.5 * length if center is None else center
    dx = periodic_centered_distance(x, center, length)
    return np.exp(-(dx**2) / (2.0 * sigma**2))


def gaussian_carrier_pulse(
    x: np.ndarray,
    *,
    mode: int,
    length: float = 1.0,
    sigma_env: float | None = None,
    center: float | None = None,
) -> np.ndarray:
    sigma_env = length / 8.0 if sigma_env is None else sigma_env
    center = 0.5 * length if center is None else center
    envelope = np.exp(
        -(periodic_centered_distance(x, center, length) ** 2) / (2.0 * sigma_env**2)
    )
    carrier = np.cos(2.0 * np.pi * mode * x / length)
    return envelope * carrier


def local_packet_strength(u: np.ndarray, points_per_wavelength: float) -> np.ndarray:
    """Estimate the local carrier strength with a one-wavelength moving RMS."""
    window = max(3, int(round(points_per_wavelength)))
    kernel = np.ones(window, dtype=float) / window
    padded = np.r_[u[-window:], u, u[:window]]
    smoothed = np.convolve(padded * padded, kernel, mode="same")[window:-window]
    return np.sqrt(2.0 * smoothed)


def rk4_march_advection(
    operator: SpatialOperator,
    u0: np.ndarray,
    time_final: float,
    cfl: float,
) -> tuple[np.ndarray, float, int]:
    dt_nominal = cfl * operator.h
    nsteps = max(1, int(np.ceil(time_final / dt_nominal)))
    dt = time_final / nsteps
    u = u0.copy()

    for _ in range(nsteps):
        k1 = -operator.derivative(u)
        k2 = -operator.derivative(u + 0.5 * dt * k1)
        k3 = -operator.derivative(u + 0.5 * dt * k2)
        k4 = -operator.derivative(u + dt * k3)
        u += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return u, dt, nsteps


def exact_semidiscrete_advection(
    operator: SpatialOperator,
    u0: np.ndarray,
    time_final: float,
) -> np.ndarray:
    """Exact solution of u_t = -D u for circulant D via FFT eigendecomposition.

    Every spatial operator on the periodic equispaced grid is circulant,
    so its eigenvalues are the DFT of its first column.  Evolving each
    Fourier mode exactly removes all time-integration error; the only
    error in the returned solution comes from the spatial discretization
    (k'(w) != w).
    """
    N = len(u0)
    e0 = np.zeros(N)
    e0[0] = 1.0
    first_col = operator.derivative(e0)
    eigenvalues = np.fft.fft(first_col)
    u_hat = np.fft.fft(u0)
    u_hat *= np.exp(-eigenvalues * time_final)
    return np.real(np.fft.ifft(u_hat))


def exact_continuum_advection_diffusion(
    u0: np.ndarray,
    time_final: float,
    *,
    c: float = 0.0,
    nu: float = 0.0,
    length: float = 1.0,
) -> np.ndarray:
    npts = len(u0)
    h = length / npts
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(npts, d=h)
    u_hat = np.fft.fft(u0)
    u_hat *= np.exp((-1j * c * wave_numbers - nu * wave_numbers**2) * time_final)
    return np.real(np.fft.ifft(u_hat))


def exact_semidiscrete_diffusion(
    operator: SecondDerivativeOperator,
    u0: np.ndarray,
    time_final: float,
    *,
    nu: float,
) -> np.ndarray:
    npts = len(u0)
    e0 = np.zeros(npts)
    e0[0] = 1.0
    first_col = operator.second_derivative(e0)
    eigenvalues = np.fft.fft(first_col)
    u_hat = np.fft.fft(u0)
    u_hat *= np.exp(nu * eigenvalues * time_final)
    return np.real(np.fft.ifft(u_hat))


def circulant_eigenvalues(apply_operator: Callable[[np.ndarray], np.ndarray], npts: int) -> np.ndarray:
    e0 = np.zeros(npts)
    e0[0] = 1.0
    return np.fft.fft(apply_operator(e0))


def exact_continuum_advection_diffusion_2d(
    u0: np.ndarray,
    time_final: float,
    *,
    c: tuple[float, float] = (0.0, 0.0),
    nu: float = 0.0,
    length: float = 1.0,
) -> np.ndarray:
    npts = u0.shape[0]
    h = length / npts
    k = 2.0 * np.pi * np.fft.fftfreq(npts, d=h)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    u_hat = np.fft.fft2(u0)
    exponent = (-1j * (c[0] * kx + c[1] * ky) - nu * (kx**2 + ky**2)) * time_final
    return np.real(np.fft.ifft2(u_hat * np.exp(exponent)))


def exact_semidiscrete_2d_advection(
    operator: SpatialOperator,
    u0: np.ndarray,
    time_final: float,
    *,
    c: tuple[float, float],
) -> np.ndarray:
    npts = u0.shape[0]
    lam = circulant_eigenvalues(operator.derivative, npts)
    lam2d = c[0] * lam[:, None] + c[1] * lam[None, :]
    u_hat = np.fft.fft2(u0)
    return np.real(np.fft.ifft2(u_hat * np.exp(-lam2d * time_final)))


def exact_semidiscrete_2d_diffusion(
    operator: SecondDerivativeOperator,
    u0: np.ndarray,
    time_final: float,
    *,
    nu: float,
) -> np.ndarray:
    npts = u0.shape[0]
    lam = circulant_eigenvalues(operator.second_derivative, npts)
    lam2d = lam[:, None] + lam[None, :]
    u_hat = np.fft.fft2(u0)
    return np.real(np.fft.ifft2(u_hat * np.exp(nu * lam2d * time_final)))


def discrete_energy_1d(u: np.ndarray, *, length: float) -> float:
    return float((length / len(u)) * np.sum(u * u))


def relative_l2_error(u_num: np.ndarray, u_exact: np.ndarray) -> float:
    diff = u_num - u_exact
    return float(np.linalg.norm(diff) / np.linalg.norm(u_exact))


def omega_from_ppw(ppw: float) -> float:
    return 2.0 * np.pi / ppw


def phase_error_curve(w: np.ndarray, kph: np.ndarray) -> np.ndarray:
    return np.abs(kph - w) / np.maximum(w, 1.0e-14)


def phase_error_at_ppw(ppw: float, w: np.ndarray, kph: np.ndarray) -> float:
    omega = omega_from_ppw(ppw)
    kph_value = float(np.interp(omega, w, kph))
    return abs(kph_value / omega - 1.0)


def coherent_distance_limit_curve(
    w: np.ndarray,
    kph: np.ndarray,
    *,
    tol_cycles: float = PHASE_SLIP_TOLERANCE_CYCLES,
) -> np.ndarray:
    return tol_cycles / np.maximum(phase_error_curve(w, kph), 1.0e-14)


def coherent_distance_efficiency_curve(
    label: str,
    w: np.ndarray,
    kph: np.ndarray,
) -> np.ndarray:
    ppw = 2.0 * np.pi / np.maximum(w, 1.0e-14)
    eps_phi = phase_error_curve(w, kph)
    cfl = sa.rk4_cfl_limit(kph)
    return cfl / (NATIVE_WORK_UNITS[label] * ppw**2 * np.maximum(eps_phi, 1.0e-14))


def minimum_ppw_curve(
    ppw_grid: np.ndarray,
    eps_phi_grid: np.ndarray,
    mission_lengths: np.ndarray,
    *,
    tol_cycles: float = PHASE_SLIP_TOLERANCE_CYCLES,
) -> np.ndarray:
    admissible = mission_lengths[:, None] * eps_phi_grid[None, :] <= tol_cycles
    first_true = np.argmax(admissible, axis=1)
    has_solution = np.any(admissible, axis=1)
    result = np.full(mission_lengths.shape, np.nan, dtype=float)
    result[has_solution] = ppw_grid[first_true[has_solution]]
    return result


def first_threshold_crossing(
    x: np.ndarray,
    y: np.ndarray,
    *,
    level: float = 1.0,
) -> float | None:
    below = np.flatnonzero(y <= level)
    if below.size == 0:
        return None
    idx = int(below[0])
    if idx == 0:
        return float(x[0])
    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if abs(y1 - y0) < 1.0e-14:
        return x1
    frac = (level - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)


def diffusion_relative_amplitude_error(
    mission_lengths: np.ndarray,
    relative_symbol_error: np.ndarray,
) -> np.ndarray:
    mission_lengths = np.asarray(mission_lengths, dtype=float)
    relative_symbol_error = np.asarray(relative_symbol_error, dtype=float)
    exponent_mismatch = 4.0 * np.pi**2 * mission_lengths[:, None] * relative_symbol_error[None, :]
    return np.expm1(exponent_mismatch)


def minimum_ppw_curve_from_diffusion_budget(
    ppw_grid: np.ndarray,
    relative_symbol_error: np.ndarray,
    mission_lengths: np.ndarray,
    *,
    tol: float = DIRECT_DIFFUSION_RELATIVE_AMPLITUDE_TOL,
) -> np.ndarray:
    admissible = diffusion_relative_amplitude_error(mission_lengths, relative_symbol_error) <= tol
    first_true = np.argmax(admissible, axis=1)
    has_solution = np.any(admissible, axis=1)
    result = np.full(mission_lengths.shape, np.nan, dtype=float)
    result[has_solution] = ppw_grid[first_true[has_solution]]
    return result


def run_gaussian_convergence_case(
    label: str,
    npts: int,
    *,
    time_final: float,
    length: float = 1.0,
    sigma: float | None = None,
) -> SpatialAccuracyResult:
    x = grid_points(npts, length)
    u0 = gaussian_pulse(x, length=length, sigma=sigma)
    operator = build_spatial_operator(label, npts, length=length)
    u_num = exact_semidiscrete_advection(operator, u0, time_final)
    u_exact = gaussian_pulse((x - time_final) % length, length=length, sigma=sigma)
    return SpatialAccuracyResult(
        label=label,
        npts=npts,
        error=relative_l2_error(u_num, u_exact),
        derivative_flops=operator.spatial_cost.derivative_flops(npts),
    )


def run_gaussian_diffusion_convergence_case(
    label: str,
    npts: int,
    *,
    time_final: float = DIRECT_DIFFUSION_TIME,
    nu: float = DIRECT_DIFFUSION_NU,
    length: float = 1.0,
    sigma: float = DIRECT_DIFFUSION_SIGMA,
) -> SpatialAccuracyResult:
    x = grid_points(npts, length)
    u0 = gaussian_pulse(x, length=length, sigma=sigma)
    operator = build_second_derivative_operator(label, npts, length=length)
    u_num = exact_semidiscrete_diffusion(operator, u0, time_final, nu=nu)
    u_exact = exact_continuum_advection_diffusion(u0, time_final, nu=nu, length=length)
    return SpatialAccuracyResult(
        label=label,
        npts=npts,
        error=relative_l2_error(u_num, u_exact),
        derivative_flops=operator.spatial_cost.derivative_flops(npts),
    )


def run_waveform_case(
    label: str,
    npts: int,
    *,
    time_final: float,
    length: float = 1.0,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = grid_points(npts, length)
    u0 = gaussian_pulse(x, length=length, sigma=sigma)
    operator = build_spatial_operator(label, npts, length=length)
    u_num = exact_semidiscrete_advection(operator, u0, time_final)
    return x, u_num


def run_carrier_case(
    label: str,
    npts: int,
    *,
    mode: int,
    time_final: float,
    length: float = 1.0,
    sigma_env: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = grid_points(npts, length)
    u0 = gaussian_carrier_pulse(
        x, mode=mode, length=length, sigma_env=sigma_env
    )
    operator = build_spatial_operator(label, npts, length=length)
    u_num = exact_semidiscrete_advection(operator, u0, time_final)
    return x, u_num


def run_carrier_convergence_case(
    label: str,
    npts: int,
    *,
    mode: int,
    time_final: float,
    length: float = 1.0,
    sigma_env: float | None = None,
) -> SpatialAccuracyResult:
    x = grid_points(npts, length)
    u0 = gaussian_carrier_pulse(
        x,
        mode=mode,
        length=length,
        sigma_env=sigma_env,
    )
    operator = build_spatial_operator(label, npts, length=length)
    u_num = exact_semidiscrete_advection(operator, u0, time_final)
    u_exact = gaussian_carrier_pulse(
        (x - time_final) % length,
        mode=mode,
        length=length,
        sigma_env=sigma_env,
    )
    return SpatialAccuracyResult(
        label=label,
        npts=npts,
        error=relative_l2_error(u_num, u_exact),
        derivative_flops=operator.spatial_cost.derivative_flops(npts),
    )


def measure_marching_time(
    label: str,
    npts: int,
    *,
    time_final: float,
    cfl: float,
    repeats: int = 10,
    length: float = 1.0,
) -> MarchResult:
    x = grid_points(npts, length)
    u0 = gaussian_pulse(x, length=length)
    operator = build_spatial_operator(label, npts, length=length)

    rk4_march_advection(operator, u0, time_final, cfl)

    timings = []
    dt = 0.0
    nsteps = 0
    for _ in range(repeats):
        t0 = perf_counter()
        _, dt, nsteps = rk4_march_advection(operator, u0, time_final, cfl)
        timings.append(perf_counter() - t0)

    u_exact = gaussian_pulse((x - time_final) % length, length=length)
    u_num, _, _ = rk4_march_advection(operator, u0, time_final, cfl)
    return MarchResult(
        label=label,
        npts=npts,
        error=relative_l2_error(u_num, u_exact),
        dt=dt,
        nsteps=nsteps,
        repeated_flops=operator.spatial_cost.derivative_flops(npts),
        wall_time=float(np.mean(timings)),
        setup_time=operator.setup_time,
    )


def measure_carrier_marching_time(
    label: str,
    npts: int,
    *,
    mode: int,
    time_final: float,
    cfl: float,
    repeats: int = 10,
    length: float = 1.0,
    sigma_env: float | None = None,
) -> MarchResult:
    x = grid_points(npts, length)
    u0 = gaussian_carrier_pulse(
        x,
        mode=mode,
        length=length,
        sigma_env=sigma_env,
    )
    operator = build_spatial_operator(label, npts, length=length)

    rk4_march_advection(operator, u0, time_final, cfl)

    timings = []
    dt = 0.0
    nsteps = 0
    for _ in range(repeats):
        t0 = perf_counter()
        _, dt, nsteps = rk4_march_advection(operator, u0, time_final, cfl)
        timings.append(perf_counter() - t0)

    return MarchResult(
        label=label,
        npts=npts,
        error=float("nan"),
        dt=dt,
        nsteps=nsteps,
        repeated_flops=operator.spatial_cost.derivative_flops(npts),
        wall_time=float(np.mean(timings)),
        setup_time=operator.setup_time,
    )


def build_all_curves(w: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Exact": sa.kprime_exact(w),
        "E2": sa.kprime_centered_fd(w, 2),
        "E4": sa.kprime_centered_fd(w, 4),
        "E6": sa.kprime_centered_fd(w, 6),
        "E8": sa.kprime_centered_fd(w, 8),
        "E10": sa.kprime_centered_fd(w, 10),
        "E12": sa.kprime_centered_fd(w, 12),
        "C4": sa.kprime_compact(w, 4),
        "C6": sa.kprime_compact(w, 6),
        "C8": sa.kprime_compact(w, 8),
        "C10": sa.kprime_compact(w, 10),
        "C12": sa.kprime_compact(w, 12),
        "BMC2": sa.kprime_collocation(w, 2),
        "BMC3": sa.kprime_collocation(w, 3),
        "BMC5": sa.kprime_collocation(w, 5),
        "BMC6": sa.kprime_collocation(w, 6),
        "BMC7": sa.kprime_collocation(w, 7),
        "BMC8": sa.kprime_collocation(w, 8),
        "BMC9": sa.kprime_collocation(w, 9),
        "BMC10": sa.kprime_collocation(w, 10),
        "BMC11": sa.kprime_collocation(w, 11),
        "IGA1": sa.kprime_iga(w, 1),
        "IGA2": sa.kprime_iga(w, 2),
        "IGA3": sa.kprime_iga(w, 3),
        "IGA4": sa.kprime_iga(w, 4),
        "IGA5": sa.kprime_iga(w, 5),
    }


def build_all_second_derivative_curves(w: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Exact": sa.ksecond_exact(w),
        "E2": sa.ksecond_centered_fd(w, 2),
        "E4": sa.ksecond_centered_fd(w, 4),
        "E6": sa.ksecond_centered_fd(w, 6),
        "E8": sa.ksecond_centered_fd(w, 8),
        "E10": sa.ksecond_centered_fd(w, 10),
        "E12": sa.ksecond_centered_fd(w, 12),
        "C4": sa.ksecond_compact(w, 4),
        "C6": sa.ksecond_compact(w, 6),
        "BMC2": sa.ksecond_collocation(w, 2),
        "BMC3": sa.ksecond_collocation(w, 3),
        "BMC5": sa.ksecond_collocation(w, 5),
        "BMC6": sa.ksecond_collocation(w, 6),
        "BMC7": sa.ksecond_collocation(w, 7),
        "BMC8": sa.ksecond_collocation(w, 8),
        "BMC9": sa.ksecond_collocation(w, 9),
        "BMC10": sa.ksecond_collocation(w, 10),
        "BMC11": sa.ksecond_collocation(w, 11),
        "IGA1": sa.ksecond_iga(w, 1),
        "IGA2": sa.ksecond_iga(w, 2),
        "IGA3": sa.ksecond_iga(w, 3),
        "IGA4": sa.ksecond_iga(w, 4),
        "IGA5": sa.ksecond_iga(w, 5),
    }


def build_unique_curves(all_curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "Exact": all_curves["Exact"],
        "E2": all_curves["E2"],
        "E4": all_curves["E4"],
        "E6": all_curves["E6"],
        "E8": all_curves["E8"],
        "E10": all_curves["E10"],
        "E12": all_curves["E12"],
        "BMC2": all_curves["BMC2"],
        "C4 = BMC3 = BMG1": all_curves["C4"],
        "C6": all_curves["C6"],
        "C8": all_curves["C8"],
        "BMC5 = BMG2": all_curves["BMC5"],
        "BMC6": all_curves["BMC6"],
        "BMC7 = BMG3": all_curves["IGA3"],
        "BMC8": all_curves["BMC8"],
        "BMC9 = BMG4": all_curves["IGA4"],
        "BMC10": all_curves["BMC10"],
        "BMC11 = BMG5": all_curves["IGA5"],
    }


def build_unique_second_derivative_curves(
    all_curves: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "Exact": all_curves["Exact"],
        "E2": all_curves["E2"],
        "E4": all_curves["E4"],
        "E6": all_curves["E6"],
        "E8": all_curves["E8"],
        "E10": all_curves["E10"],
        "E12": all_curves["E12"],
        "BMC2": all_curves["BMC2"],
        "C4 = BMC3 = BMG1": all_curves["C4"],
        "C6": all_curves["C6"],
        "BMC5 = BMG2": all_curves["BMC5"],
        "BMC6": all_curves["BMC6"],
        "BMC7 = BMG3": all_curves["IGA3"],
        "BMC8": all_curves["BMC8"],
        "BMC9 = BMG4": all_curves["IGA4"],
        "BMC10": all_curves["BMC10"],
        "BMC11 = BMG5": all_curves["IGA5"],
    }


def plot_wavenumber_intro() -> None:
    setup_style()
    plt.rcParams.update({"font.size": 15.0, "axes.labelsize": 15.5,
                         "xtick.labelsize": 14.5, "ytick.labelsize": 14.5})
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    grid = np.arange(0.0, 32.0, 1.0)
    fine = np.linspace(grid[0], grid[-1], 1200)
    ppw_values = [16.0, 8.0, 4.0]

    for panel, ax, ppw in zip("abc", axes, ppw_values):
        wavelength = ppw
        omega = 2.0 * np.pi / ppw
        wave_fine = np.sin(2.0 * np.pi * fine / wavelength)
        wave_grid = np.sin(2.0 * np.pi * grid / wavelength)

        ax.plot(fine, wave_fine, color="#444444", lw=1.4)
        ax.plot(grid, wave_grid, "o", color="#1F78B4", ms=3.2)
        ax.set_title(f"({panel})", fontsize=16.0, loc="left")
        ax.set_xlabel(r"Grid index $j$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Mode value $\sin(j\omega)$")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    out = FIGDIR / "fig_wavenumbers.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_dispersion_landscape(w: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8))

    for name, kph in curves.items():
        ax1.plot(
            w / np.pi,
            kph / np.pi,
            color=COLORS[name],
            label=name,
            **LINESTYLES[name],
        )
    ax1.set_xlabel(r"$\omega / \pi$")
    ax1.set_ylabel(r"$k'h / \pi$")
    ax1.set_title("(a) Modified wavenumber")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)

    for name, kph in curves.items():
        if name == "Exact":
            continue
        err = sa.relative_dispersion_error(w, kph)
        ax2.semilogy(
            w / np.pi,
            err,
            color=COLORS[name],
            label=name,
            **LINESTYLES[name],
        )
    ax2.set_xlabel(r"$\omega / \pi$")
    ax2.set_ylabel(r"$|k'h / \omega - 1|$")
    ax2.set_title("(b) Relative dispersion error")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(1e-5, 1.2)
    ax2.grid(True, alpha=0.3, which="both")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        framealpha=0.94,
        fontsize=9.2,
        columnspacing=1.1,
        handslcngth=2.4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    out = FIGDIR / "fig_dispersion.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_second_derivative_dispersion_landscape(
    w: np.ndarray, curves: dict[str, np.ndarray]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.1, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (title, labels, legend_loc, legend_ncol) in zip(axes, SECOND_DERIVATIVE_GROUPS):
        for label in labels:
            err = sa.relative_second_derivative_error(w, curves[label])
            ax.semilogy(
                w / np.pi,
                err,
                color=COLORS[label],
                label=label,
                **LINESTYLES[label],
            )
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0e-12, 2.0)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(
            loc=legend_loc,
            ncol=legend_ncol,
            framealpha=0.92,
            fontsize=7.7,
            columnspacing=0.85,
            handslcngth=2.4,
        )

    for ax in axes[2:]:
        ax.set_xlabel(r"$\omega / \pi$")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r"$|k_2 h^2 / \omega^2 - 1|$")

    fig.tight_layout()
    out = FIGDIR / "fig_second_derivative_dispersion.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_eigenvalues_landscape(w: np.ndarray, curves: dict[str, np.ndarray], ngrid: int = 64) -> None:
    families = [
        "E2", "E4", "E6",
        "C4 = BMC3 = BMG1", "C6", "C8",
        "BMC5 = BMG2", "BMC7 = BMG3", "BMC9 = BMG4", "BMC11 = BMG5",
    ]
    discrete_wavenumbers = 2.0 * np.pi * np.arange(ngrid) / ngrid

    fig, ax = plt.subplots(figsize=(11.8, 6.4))

    rk4_imag_limit = 2.828
    ax.axhspan(-rk4_imag_limit, rk4_imag_limit, color="#EAF3EA", alpha=0.7, zorder=0)
    ax.axhline(rk4_imag_limit, color="#C0392B", linestyle="--", lw=1.1, alpha=0.85, zorder=1)
    ax.axhline(-rk4_imag_limit, color="#C0392B", linestyle="--", lw=1.1, alpha=0.85, zorder=1)

    for i, name in enumerate(families):
        kph = curves[name]
        samples = np.interp(discrete_wavenumbers, w, kph)
        imag_vals = np.concatenate([samples, -samples])
        peak = float(np.abs(imag_vals).max())
        cfl = rk4_imag_limit / peak if peak > 0 else float("inf")

        # A light vertical guide makes the imaginary-axis extent legible at a glance.
        ax.vlines(i, -peak, peak, color=COLORS[name], lw=2.1, alpha=0.28, zorder=1)

        count = imag_vals.size
        phase = np.linspace(-1.0, 1.0, count)
        x_pos = i + 0.17 * np.sin(np.pi * phase)
        ax.scatter(
            x_pos, imag_vals,
            s=16, color=COLORS[name], alpha=0.84,
            edgecolor="none", zorder=2,
        )
        ax.annotate(
            f"$\\rho_h={peak:.2f}$\nRK4 CFL {cfl:.2f}",
            xy=(i, peak),
            xytext=(i, peak + 0.22),
            ha="center", va="bottom",
            fontsize=7.6, color=COLORS[name],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
        )

    ax.text(
        len(families) - 1.35,
        rk4_imag_limit + 0.10,
        r"RK4 budget $+2.828$",
        color="#C0392B",
        fontsize=8.2,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.2),
    )
    ax.text(
        len(families) - 1.35,
        -rk4_imag_limit - 0.10,
        r"RK4 budget $-2.828$",
        color="#C0392B",
        fontsize=8.2,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.2),
    )

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=24, ha="right", fontsize=8.6)
    ax.set_ylabel(r"$\mathrm{Im}(\lambda h)$")
    ax.set_title(rf"Discrete spectra by operator family ($N={ngrid}$)")
    ax.set_xlim(-0.55, len(families) + 0.9)
    ax.set_ylim(-3.45, 3.45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = FIGDIR / "fig_eigenvalues.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_matched_order(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    curve_lookup = {
        "E4": all_curves["E4"],
        "C4 = BMC3 = BMG1": all_curves["C4"],
        "E6": all_curves["E6"],
        "C6": all_curves["C6"],
        "BMC5 = BMG2": all_curves["BMC5"],
        "E8": all_curves["E8"],
        "C8": all_curves["C8"],
        "BMC7 = BMG3": all_curves["IGA3"],
        "E10": all_curves["E10"],
        "C10": all_curves["C10"],
        "BMC9 = BMG4": all_curves["IGA4"],
        "E12": all_curves["E12"],
        "C12": all_curves["C12"],
        "BMC11 = BMG5": all_curves["IGA5"],
    }

    # House navy for the spline-only resolved band, amber for the 1% line.
    band_navy = _SCHEME_COLORS["BMC"]
    rule_amber = _SCHEME_COLORS["C"]
    tol = 1e-2

    def omega_pi_at_1pct(label: str) -> float:
        opi, _ = sa.resolution_limit(w, curve_lookup[label], tol=tol)
        return opi

    # Per-panel win plan: the best competitor the spline overtakes at the
    # one-percent dispersion line, the spline champion, and the one-line
    # naming of the band only the banded family reaches. At fourth order the
    # compact and the spline are the same operator, so the honest competitor
    # is the matched explicit stencil.
    band_plans = {
        "Fourth order": {
            "competitor": "E4",
            "champion": "C4 = BMC3 = BMG1",
            "note": "compact and spline coincide;\nband over matched explicit $\\mathrm{E4}$",
        },
        "Sixth order": {
            "competitor": "C6",
            "champion": "BMC5 = BMG2",
            "note": "$\\mathrm{BMC5}$ stays trustworthy\npast where $\\mathrm{C6}$ fails",
        },
        "Eighth order": {
            "competitor": "C8",
            "champion": "BMC7 = BMG3",
            "note": "$\\mathrm{BMC7}$ resolves shorter\nwaves than $\\mathrm{C8}$ reaches",
        },
        "Tenth and twelfth order": {
            "competitor": "C12",
            "champion": "BMC11 = BMG5",
            "note": "$\\mathrm{BMC11}$ resolves shorter waves\nthan $\\mathrm{C12}$ reaches, staying banded",
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (title, labels) in zip(axes, FIXED_ORDER_GROUPS):
        for label in labels:
            kph = curve_lookup[label]
            err = sa.relative_dispersion_error(w, kph)
            ax.semilogy(
                w / np.pi,
                err,
                color=COLORS[label],
                label=label,
                **LINESTYLES[label],
            )
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-12, 2.0)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", framealpha=0.9)

        # Annotation layer: the 1% dispersion line and the spline-only band.
        plan = band_plans[title]
        comp_x = omega_pi_at_1pct(plan["competitor"])
        champ_x = omega_pi_at_1pct(plan["champion"])

        ax.axhline(tol, color=rule_amber, lw=1.1, ls=(0, (5, 2)), zorder=1.5)
        ax.text(
            0.985,
            tol * 1.6,
            "1% dispersion",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.0,
            color=rule_amber,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )

        # Shade the omega interval only the spline crosses the 1% line inside.
        ax.axvspan(comp_x, champ_x, color=band_navy, alpha=0.12, zorder=0.5)
        for cross_x in (comp_x, champ_x):
            ax.plot(
                [cross_x],
                [tol],
                marker="o",
                ms=4.2,
                mfc="white",
                mec=band_navy,
                mew=1.3,
                zorder=6,
                clip_on=False,
            )

        band_mid = 0.5 * (comp_x + champ_x)
        ax.annotate(
            plan["note"],
            xy=(band_mid, tol),
            xytext=(band_mid, 3.5e-6),
            ha="center",
            va="center",
            fontsize=7.8,
            color=band_navy,
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=band_navy,
                lw=0.8,
                alpha=0.92,
            ),
            arrowprops=dict(arrowstyle="-", lw=0.8, color=band_navy, alpha=0.7),
        )

    for ax in axes[2:]:
        ax.set_xlabel(r"$\omega / \pi$")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r"$|k'h / \omega - 1|$")

    fig.tight_layout()
    out = UNUSED_FIGDIR / "fig_matched_order.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_same_degree(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    setup_full_width_print_style(three_panel=True)
    groups = [
        ("(a)", [("BMC2", all_curves["BMC2"]), ("BMC5 = BMG2", all_curves["IGA2"])]),
        ("(b)", [("BMC3", all_curves["BMC3"]), ("BMC7 = BMG3", all_curves["IGA3"])]),
        ("(c)", [("BMC5", all_curves["BMC5"]), ("BMC11 = BMG5", all_curves["IGA5"])]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.9), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (title, entries) in zip(axes, groups):
        for label, kph in entries:
            err = sa.relative_dispersion_error(w, kph)
            ax.semilogy(
                w / np.pi,
                err,
                color=COLORS[label],
                label=label,
                **LINESTYLES[label],
            )
        ax.set_title(title, loc="left", fontsize=14)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-12, 2.0)
        ax.set_xlabel(r"$\omega / \pi$")
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(labelsize=13.5)

    axes[0].set_ylabel(r"$|k'h / \omega - 1|$")
    handles = [line for ax in axes for line in ax.lines]
    labels = [line.get_label() for line in handles]
    unique = dict(zip(labels, handles))
    if _make_overhead_legend is not None:
        _make_overhead_legend(fig, list(unique.values()), list(unique.keys()),
                              ncol=3, fontsize=13.5,
                              rect=(0, 0, 1, 0.78))
    else:
        fig.legend(list(unique.values()), list(unique.keys()), loc="upper center",
                   ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.78))
    out = FIGDIR / "fig_same_degree.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_fixed_halo(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    curve_lookup = {
        "E2": all_curves["E2"],
        "E4": all_curves["E4"],
        "E6": all_curves["E6"],
        "E8": all_curves["E8"],
        "E10": all_curves["E10"],
        "E12": all_curves["E12"],
        "C4 = BMC3 = BMG1": all_curves["C4"],
        "C6": all_curves["C6"],
        "C8": all_curves["C8"],
        "BMC5 = BMG2": all_curves["BMC5"],
        "BMC7 = BMG3": all_curves["IGA3"],
        "BMC9 = BMG4": all_curves["IGA4"],
        "BMC11 = BMG5": all_curves["IGA5"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (title, labels) in zip(axes, FIXED_RADIUS_GROUPS):
        metric_lines = []
        for label in labels:
            kph = curve_lookup[label]
            err = sa.relative_dispersion_error(w, kph)
            metrics = sa.summarize_curve(w, kph)
            metric_lines.append(
                f"{label:<16}  CFL {metrics.rk4_cfl:>4.2f}  PPW {metrics.ppw_1pct:>4.2f}"
            )
            ax.semilogy(
                w / np.pi,
                err,
                color=COLORS[label],
                label=label,
                **LINESTYLES[label],
            )
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-12, 2.0)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", framealpha=0.92, fontsize=8.2)
        ax.text(
            0.97,
            0.08,
            "\n".join(metric_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            family="monospace",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#BBBBBB", alpha=0.9),
        )

    for ax in axes[3:]:
        ax.set_xlabel(r"$\omega / \pi$")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(r"$|k'h / \omega - 1|$")

    add_better_arrow(
        axes[4],
        xy=(0.60, 0.33),
        xytext=(0.82, 0.59),
    )
    fig.tight_layout()
    out = FIGDIR / "fig_fixed_halo.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_cost_tradeoff(
    _convergence_results: dict[str, list[SpatialAccuracyResult]],
    metrics_map: dict[str, sa.CurveMetrics],
    detail_labels: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    costs = np.array([NATIVE_WORK_UNITS[label] for label in detail_labels], dtype=float)
    orders = np.array([FORMAL_ORDER[label] for label in detail_labels], dtype=int)
    ppw_1pct = np.array([metrics_map[label].ppw_1pct for label in detail_labels], dtype=float)
    ppw_5pct = np.array([metrics_map[label].ppw_5pct for label in detail_labels], dtype=float)
    bar_colors = [COLORS[label] for label in detail_labels]
    light_colors = [lighten(COLORS[label], amount=0.28) for label in detail_labels]

    y = np.arange(len(detail_labels), dtype=float)
    axes[0].barh(y, costs, color=bar_colors, height=0.56, alpha=0.95)
    axes[0].set_yticks(y, detail_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, float(costs.max()) + 3.0)
    axes[0].set_xlabel(r"Derivative cost per grid point $F_D/N$")
    axes[0].set_title("(a) Sixth-order cost ladder")
    axes[0].grid(True, axis="x", alpha=0.3)
    for idx, (label, cost, order) in enumerate(zip(detail_labels, costs, orders)):
        axes[0].text(
            max(0.35, cost - 0.35),
            idx,
            f"{int(cost)}",
            va="center",
            ha="right",
            fontsize=8.8,
            color="white",
            fontweight="bold",
        )
        axes[0].text(
            cost + 0.25,
            idx,
            rf"order {order}, {family_of(label)}",
            va="center",
            ha="left",
            fontsize=8.8,
            color="#222222",
        )

    x = np.arange(len(detail_labels), dtype=float)
    width = 0.34
    axes[1].bar(
        x - 0.5 * width,
        ppw_1pct,
        width,
        color=bar_colors,
        label=r"$1\%$ dispersion threshold",
    )
    axes[1].bar(
        x + 0.5 * width,
        ppw_5pct,
        width,
        color=light_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        label=r"$5\%$ dispersion threshold",
    )
    axes[1].set_xticks(
        x,
        ["E6\norder 6", "C6\norder 6", "BMC5 = BMG2\norder 6"],
    )
    axes[1].set_ylabel("Points per wavelength")
    axes[1].set_title("(b) Phase-error tolerance thresholds")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", framealpha=0.94, fontsize=8.8)
    for xpos, y1, y5 in zip(x, ppw_1pct, ppw_5pct):
        axes[1].text(xpos - 0.5 * width, y1 + 0.07, f"{y1:.2f}", ha="center", va="bottom", fontsize=8.1)
        axes[1].text(xpos + 0.5 * width, y5 + 0.07, f"{y5:.2f}", ha="center", va="bottom", fontsize=8.1)

    fig.tight_layout()
    out = FIGDIR / "fig_cost_tradeoff.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_second_derivative_cost_tradeoff(
    metrics_map: dict[str, sa.SecondDerivativeCurveMetrics],
    detail_labels: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    costs = np.array([second_derivative_work_units(label) for label in detail_labels], dtype=float)
    fitted_orders = np.array([metrics_map[label].fitted_order for label in detail_labels], dtype=float)
    ppw_1pct = np.array([metrics_map[label].ppw_1pct for label in detail_labels], dtype=float)
    ppw_5pct = np.array([metrics_map[label].ppw_5pct for label in detail_labels], dtype=float)
    bar_colors = [COLORS[label] for label in detail_labels]
    light_colors = [lighten(COLORS[label], amount=0.28) for label in detail_labels]

    y = np.arange(len(detail_labels), dtype=float)
    axes[0].barh(y, costs, color=bar_colors, height=0.56, alpha=0.95)
    axes[0].set_yticks(y, detail_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, float(costs.max()) + 3.0)
    axes[0].set_xlabel(r"Direct second-derivative cost per grid point $F_{D_2}/N$")
    axes[0].set_title("(a) Sixth-order direct second-derivative cost ladder")
    axes[0].grid(True, axis="x", alpha=0.3)
    for idx, (label, cost, fitted_order) in enumerate(zip(detail_labels, costs, fitted_orders)):
        axes[0].text(
            max(0.35, cost - 0.35),
            idx,
            f"{int(cost)}",
            va="center",
            ha="right",
            fontsize=8.8,
            color="white",
            fontweight="bold",
        )
        axes[0].text(
            cost + 0.25,
            idx,
            rf"fit $\approx {fitted_order:.1f}$, {family_of(label)}",
            va="center",
            ha="left",
            fontsize=8.8,
            color="#222222",
        )

    x = np.arange(len(detail_labels), dtype=float)
    width = 0.34
    axes[1].bar(
        x - 0.5 * width,
        ppw_1pct,
        width,
        color=bar_colors,
        label=r"$1\%$ symbol threshold",
    )
    axes[1].bar(
        x + 0.5 * width,
        ppw_5pct,
        width,
        color=light_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        label=r"$5\%$ symbol threshold",
    )
    axes[1].set_xticks(x, list(detail_labels))
    axes[1].set_ylabel("Points per wavelength")
    axes[1].set_title("(b) Direct-$D_2$ symbol-threshold PPW")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", framealpha=0.94, fontsize=8.8)
    for xpos, y1, y5 in zip(x, ppw_1pct, ppw_5pct):
        axes[1].text(xpos - 0.5 * width, y1 + 0.07, f"{y1:.2f}", ha="center", va="bottom", fontsize=8.1)
        axes[1].text(xpos + 0.5 * width, y5 + 0.07, f"{y5:.2f}", ha="center", va="bottom", fontsize=8.1)

    fig.tight_layout()
    out = FIGDIR / "fig_second_derivative_cost_tradeoff.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_second_derivative_diffusion(
    convergence_results: dict[str, list[SpatialAccuracyResult]],
    detail_labels: tuple[str, ...],
    *,
    time_final: float = DIRECT_DIFFUSION_TIME,
    nu: float = DIRECT_DIFFUSION_NU,
) -> None:
    def _truncate_at_floor(
        x: np.ndarray, y: np.ndarray, *, floor: float = 5.0e-13
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.size <= 2:
            return x, y
        cut = y.size
        seen_floor_band = False
        for i in range(1, y.size):
            if y[i] < 10.0 * floor:
                seen_floor_band = True
            if seen_floor_band and y[i] > y[i - 1]:
                cut = i
                break
        return x[:cut], y[:cut]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"wspace": 0.30})
    all_errors = [
        item.error
        for label in detail_labels
        for item in convergence_results[label]
        if item.error > 0.0
    ]

    bmc_anchor: tuple[np.ndarray, np.ndarray] | None = None
    bmc_anchor_cost: tuple[np.ndarray, np.ndarray] | None = None
    for label in detail_labels:
        data = convergence_results[label]
        npts = np.array([item.npts for item in data], dtype=float)
        errors = np.array([item.error for item in data], dtype=float)
        costs = np.array([float(item.derivative_flops) for item in data], dtype=float)
        n_t, e_t = _truncate_at_floor(npts, errors)
        c_t, e_t_cost = _truncate_at_floor(costs, errors)
        axes[0].loglog(
            n_t, e_t, marker="o", ms=4.3, color=COLORS[label], label=label, **LINESTYLES[label],
        )
        axes[1].loglog(
            c_t, e_t_cost, marker="o", ms=4.3, color=COLORS[label], label=label, **LINESTYLES[label],
        )
        if label == "BMC7 = BMG3":
            bmc_anchor = (n_t, e_t)
            bmc_anchor_cost = (c_t, e_t_cost)

    truncated_errors = []
    for label in detail_labels:
        data = convergence_results[label]
        npts = np.array([item.npts for item in data], dtype=float)
        errors = np.array([item.error for item in data], dtype=float)
        _, e_t = _truncate_at_floor(npts, errors)
        truncated_errors.extend([float(v) for v in e_t if v > 0.0])
    ymin, ymax = padded_positive_limits(truncated_errors or all_errors, lower_scale=0.7, upper_scale=1.6)
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin, ymax)
    axes[0].set_xlabel(r"Grid points $N$")
    axes[0].set_ylabel(
        rf"Relative $L^2$ error for $u_t=\nu u_{{xx}}$ ($\nu={nu:.1e}$, $T={time_final:.0f}$)"
    )
    axes[0].set_title("(a) Direct second-derivative diffusion convergence")
    axes[1].set_xlabel(r"Charged spatial work per application $F_{D_2}$")
    axes[1].set_ylabel(r"Relative $L^2$ error")
    axes[1].set_title("(b) The same error plotted against charged direct-operator work")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    if _add_slope_reference is not None:
        _add_slope_reference(
            axes[0], slope=-6, label=r"$N^{-6}$", corner="lower_left",
        )
        _add_slope_reference(
            axes[1], slope=-6, label=r"$F^{-6}$", corner="lower_left",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if _make_overhead_legend is not None:
        _make_overhead_legend(fig, handles, labels, ncol=len(detail_labels))
    else:
        fig.subplots_adjust(top=0.85)
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.985),
            ncol=len(detail_labels), frameon=False, fontsize=9.0,
        )
    out = FIGDIR / "fig_second_derivative_diffusion.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {out}")
    plt.close(fig)


def plot_second_derivative_mission_design(
    w: np.ndarray,
    curves: dict[str, np.ndarray],
    detail_labels: tuple[str, ...],
    *,
    rel_amp_tol: float = DIRECT_DIFFUSION_MISSION_RELATIVE_AMPLITUDE_TOL,
    mission_window: tuple[float, float] = DIRECT_DIFFUSION_MISSION_WINDOW,
    ppw_budget_window: tuple[float, float] = DIRECT_DIFFUSION_MISSION_PPW_BUDGET_WINDOW,
    ppw_search_window: tuple[float, float] = (2.8, 16.0),
) -> None:
    compare_labels = tuple(label for label in detail_labels if label in DIRECT_SECOND_DERIVATIVE_TRIO_LABELS)
    mission_lengths = np.geomspace(mission_window[0], mission_window[1], 260)
    ppw_budget = np.linspace(ppw_budget_window[0], ppw_budget_window[1], 260)
    ppw_grid = np.linspace(ppw_search_window[0], ppw_search_window[1], 3200)
    omega_grid = omega_from_ppw(ppw_grid)

    relative_symbol_error = {
        label: np.abs(np.interp(omega_grid, w, curves[label]) / np.maximum(omega_grid**2, 1.0e-14) - 1.0)
        for label in compare_labels
    }
    pmin = {
        label: minimum_ppw_curve_from_diffusion_budget(
            ppw_grid,
            relative_symbol_error[label],
            mission_lengths,
            tol=rel_amp_tol,
        )
        for label in compare_labels
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.1, 5.05), gridspec_kw={"wspace": 0.24})
    map_colors = [section7_color(label) for label in compare_labels]
    cmap = ListedColormap(map_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for ax, dim in zip(axes, (1, 2, 3)):
        decision = np.full((ppw_budget.size, mission_lengths.size), -1, dtype=int)
        for i, p_cap in enumerate(ppw_budget):
            for j, mission in enumerate(mission_lengths):
                candidates: list[tuple[float, int]] = []
                for idx, label in enumerate(compare_labels):
                    req = pmin[label][j]
                    if np.isfinite(req) and req <= p_cap:
                        recurring_work = second_derivative_work_units(label) * mission * req ** (dim + 2)
                        candidates.append((recurring_work, idx))
                if candidates:
                    decision[i, j] = min(candidates)[1]

        ax.set_facecolor("#DDDDDD")
        mission_mesh, budget_mesh = np.meshgrid(mission_lengths, ppw_budget)
        masked = np.ma.masked_where(decision < 0, decision)
        ax.pcolormesh(
            mission_mesh,
            budget_mesh,
            masked,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
        ax.set_xscale("log")
        ax.set_xlim(*mission_window)
        ax.set_ylim(*ppw_budget_window)
        ax.grid(False)
        ax.set_title(rf"$d={dim}$", pad=8.0)
        ax.set_xlabel(r"Diffusive mission length $\nu T/\lambda^2$")
        if dim == 1:
            ax.set_ylabel(r"Available PPW budget $P_{\max}$")

    legend_handles = [Patch(facecolor=section7_color(label), label=label) for label in compare_labels]
    legend_handles.append(Patch(facecolor="#DDDDDD", label="no sixth-order scheme"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        framealpha=0.95,
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.72, bottom=0.17, wspace=0.24)
    out = FIGDIR / "fig_second_derivative_mission_design.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_modified_wavenumber_derivation() -> None:
    """Three-stage teaching flow from a continuous wave to its scalar symbol."""
    setup_style()
    plt.rcParams.update({"font.size": 15.5, "axes.labelsize": 16.0,
                         "xtick.labelsize": 14.5, "ytick.labelsize": 14.5})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8))

    # Panel (a): the continuous wave
    ax = axes[0]
    x_cont = np.linspace(0.0, 4.0 * np.pi, 600)
    ax.plot(x_cont, np.cos(x_cont), color="#1F78B4", lw=1.8)
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlim(0, 4.0 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a)", fontsize=16.5, loc="left")
    ax.text(0.5 * np.pi, -1.35, r"$u(x) = e^{ikx}$", fontsize=15.0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Panel (b): sample on a periodic grid and apply a centered stencil
    ax = axes[1]
    grid = np.linspace(0.0, 4.0 * np.pi, 17)
    ax.plot(x_cont, np.cos(x_cont), color="#1F78B4", lw=0.9, alpha=0.4)
    ax.plot(grid, np.cos(grid), "o", color="#D62728", ms=6.5)
    center = 8
    offsets = np.array([-2, -1, 1, 2])
    weights = {-2: 1.0 / 12.0, -1: -8.0 / 12.0, 1: 8.0 / 12.0, 2: -1.0 / 12.0}
    weight_labels = {-2: r"$+1/12$", -1: r"$-2/3$",
                     1: r"$+2/3$", 2: r"$-1/12$"}
    weight_label_y = {-2: -0.20, -1: 0.48, 1: 0.28, 2: -0.20}
    ax.scatter([grid[center]], [np.cos(grid[center])], s=170,
               facecolors="none", edgecolors="#222222", lw=1.5, zorder=4)
    for off in offsets:
        ax.annotate(
            "",
            xy=(grid[center], np.cos(grid[center]) + 0.18),
            xytext=(grid[center + off], np.cos(grid[center + off]) + 0.08),
            arrowprops=dict(arrowstyle="->", color="#1F78B4", lw=1.2, alpha=0.8),
        )
        ax.text(grid[center + off], weight_label_y[off],
                weight_labels[off], ha="center", fontsize=12.5,
                color="#1F78B4")
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlim(0, 4.0 * np.pi)
    ax.set_ylim(-1.55, 1.65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b)", fontsize=16.5, loc="left")
    ax.text(2.0 * np.pi, 1.48,
            r"$(Du)_j=h^{-1}\sum_m a_m u_{j+m}$",
            ha="center", fontsize=14.5)
    ax.text(0.5 * np.pi, -1.42, r"$u_j=e^{ij\omega}$, $\omega=kh$", fontsize=14.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Panel (c): the resulting scalar symbol
    ax = axes[2]
    w = np.linspace(1e-6, np.pi, 600)
    ax.plot(w / np.pi, w / np.pi, "--", color="#222", lw=1.3, label="exact")
    ax.plot(w / np.pi, sa.kprime_centered_fd(w, 4) / np.pi,
            color="#D62728", lw=1.8, label="E4 symbol")
    ax.set_xlabel(r"$\omega / \pi$")
    ax.set_ylabel(r"$k'h / \pi$")
    ax.set_title("(c)", fontsize=16.5, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               frameon=False, fontsize=14.0, bbox_to_anchor=(0.5, 0.99),
               handletextpad=0.5, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.2)
    out = FIGDIR / "fig_modified_wavenumber_derivation.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_collocation_vs_galerkin_cartoon() -> None:
    """Two-panel cartoon: pinning at nodes vs L2 projection against test functions."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # Shared setup: cubic B-splines centered at integer nodes
    centers = np.arange(-1, 7)
    x_dense = np.linspace(centers[0] - 0.5, centers[-1] + 0.5, 1200)
    splines = np.array(
        [sa.centered_cardinal_bspline(3, x_dense - c) for c in centers]
    )

    # Left panel: collocation
    ax = axes[0]
    rng = np.random.default_rng(7)
    coeffs = np.cos(centers * 0.7) + 0.25 * rng.standard_normal(len(centers))
    u_h = (coeffs[:, None] * splines).sum(axis=0)
    nodes = np.arange(0, 6, dtype=float)
    u_h_nodes = np.array(
        [
            float((coeffs * sa.centered_cardinal_bspline(3, n - centers)).sum())
            for n in nodes
        ]
    )
    for i in range(len(centers)):
        ax.plot(x_dense, coeffs[i] * splines[i], color="#999", lw=0.7, alpha=0.45)
    ax.plot(x_dense, u_h, color="#1F78B4", lw=2.1, label=r"$u_h(x)$")
    for n in nodes:
        ax.axvline(n, color="k", lw=0.5, alpha=0.18)
    ax.plot(nodes, u_h_nodes, "o", color="#D62728", ms=8, zorder=5,
            label="constraint: $u_h(x_j) = u_j$")
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(-2.2, 2.5)
    ax.set_xticks(nodes)
    ax.set_xticklabels([rf"$x_{{{int(j)}}}$" for j in nodes])
    ax.set_yticks([])
    ax.set_title("(a) Collocation: pin the spline through the nodes")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.text(2.5, -2.02,
            r"$D = QB^{-1}$:  recover coefficients, differentiate, evaluate",
            ha="center", fontsize=9.5, color="#333")

    # Right panel: Galerkin projection (overlap inner products)
    ax = axes[1]
    i_idx, j_idx = 2, 3
    Ni = sa.centered_cardinal_bspline(3, x_dense - i_idx)
    Nj = sa.centered_cardinal_bspline(3, x_dense - j_idx)
    Nk = sa.centered_cardinal_bspline(3, x_dense - 4)
    ax.plot(x_dense, Nk, color="#888", lw=1.0, alpha=0.7)
    ax.plot(x_dense, sa.centered_cardinal_bspline(3, x_dense - 1),
            color="#888", lw=1.0, alpha=0.7)
    ax.plot(x_dense, Ni, color="#1F78B4", lw=2.2, label=rf"$B_{{{i_idx}}}$")
    ax.plot(x_dense, Nj, color="#D62728", lw=2.2, label=rf"$B_{{{j_idx}}}$")
    overlap = np.minimum(Ni, Nj)
    ax.fill_between(x_dense, 0, overlap, color="#9467BD", alpha=0.45,
                    label=r"$M_{ij}=\int B_i B_j\,dx$")
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.18, 0.92)
    ax.set_xticks(np.arange(0, 7))
    ax.set_xticklabels([rf"$x_{{{j}}}$" for j in range(7)])
    ax.set_yticks([])
    ax.set_title("(b) Galerkin: integrate against test functions")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.text(3.0, -0.13,
            r"$D = M^{-1}K$:  invert overlap mass, apply derivative coupling",
            ha="center", fontsize=9.5, color="#333")

    fig.tight_layout()
    out = UNUSED_FIGDIR / "fig_collocation_vs_galerkin.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)




def plot_label_fairness() -> None:
    """Decode the four scheme labels and the three comparison lenses.

    Restrained two-panel diagram in the boundary-paper construction-routes
    idiom: cream panel, dark accent, route-color row keys, faint horizontal
    rule separators. No rounded coloured cards, no shadow effects.
    """
    accent = "#2F4858"
    panel_bg = "#FBF8F2"
    rule_color = "#D9D2C5"

    fig = plt.figure(figsize=(13.2, 5.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.04, right=0.97, top=0.92, bottom=0.06)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    for ax in (left, right):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_facecolor(panel_bg)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    left.text(0.04, 0.965, "(a) what each numeral means",
              fontsize=10.4, color=accent, va="top", style="italic")
    right.text(0.04, 0.965, "(b) what is held fixed across a comparison",
              fontsize=10.4, color=accent, va="top", style="italic")

    label_rows = [
        (
            "$\\mathrm{E6}$",
            COLORS["E6"],
            "explicit centered FD",
            "numeral = formal order 6",
            "stencil radius 3, no implicit solve",
        ),
        (
            "$\\mathrm{C6}$",
            COLORS["C6"],
            "compact Pad" + "\u00e9" + " FD",
            "numeral = formal order 6",
            "tridiagonal mass solve, RHS radius 2",
        ),
        (
            "$\\mathrm{BMC5}$",
            COLORS["BMC5"],
            "B-spline mimetic collocation",
            "numeral = spline degree 5",
            f"radius 2, first-derivative order {FORMAL_ORDER['BMC5']}",
        ),
        (
            "$\\mathrm{BMG2}$",
            COLORS["BMC5 = BMG2"],
            "B-spline mimetic Galerkin",
            "numeral = spline degree 2",
            f"identical periodic order-{FORMAL_ORDER['BMG2']} matrix as $\\mathrm{{BMC5}}$",
        ),
    ]

    n_rows = len(label_rows)
    row_height = 0.84 / n_rows
    y_top_start = 0.90

    for idx, (latex_label, color, family, numeral_meaning, radius_note) in enumerate(label_rows):
        y_top = y_top_start - idx * row_height
        y_mid = y_top - 0.5 * row_height + 0.02
        left.add_patch(plt.Rectangle((0.03, y_top - row_height + 0.012), 0.018, row_height - 0.024,
                                      facecolor=color, edgecolor="none", alpha=0.95))
        left.text(0.075, y_mid + 0.03, latex_label,
                  fontsize=14.0, color="#111111", va="center", family="serif")
        left.text(0.21, y_mid + 0.05, family,
                  fontsize=10.4, color="#1a1a1a", va="center", family="serif")
        left.text(0.21, y_mid - 0.005, numeral_meaning,
                  fontsize=9.6, color=accent, va="center", family="serif")
        left.text(0.21, y_mid - 0.06, radius_note,
                  fontsize=9.6, color="#444444", va="center", family="serif", style="italic")
        if idx < n_rows - 1:
            left.plot([0.03, 0.97], [y_top - row_height, y_top - row_height],
                      color=rule_color, lw=0.6)

    lens_rows = [
        (
            "fixed formal order",
            r"same Taylor order near $\omega=0$",
            r"$\mathrm{E6}$, $\mathrm{C6}$, $\mathrm{BMC5}{=}\mathrm{BMG2}$",
            "main PDE comparison lens",
            COLORS["BMC5 = BMG2"],
        ),
        (
            "fixed derivative cost",
            "same repeated spatial work per apply",
            r"$\mathrm{E10}$, $\mathrm{C8}$, $\mathrm{BMC5}{=}\mathrm{BMG2}$",
            "supporting cost diagnostic",
            COLORS["C8"],
        ),
        (
            "fixed compact-relation radius",
            "same nominal stencil reach before the solve",
            r"$\mathrm{E4}$, $\mathrm{C6}$, $\mathrm{BMC5}{=}\mathrm{BMG2}$",
            "supporting locality diagnostic",
            COLORS["BMC5"],
        ),
    ]

    n_lens = len(lens_rows)
    lens_height = 0.84 / n_lens
    y_top_start_r = 0.90

    for idx, (title, definition, example, role, color) in enumerate(lens_rows):
        y_top = y_top_start_r - idx * lens_height
        y_mid = y_top - 0.5 * lens_height + 0.02
        right.add_patch(plt.Rectangle((0.03, y_top - lens_height + 0.012), 0.018, lens_height - 0.024,
                                       facecolor=color, edgecolor="none", alpha=0.95))
        right.text(0.075, y_mid + 0.07, title,
                   fontsize=11.0, color="#111111", va="center", family="serif", weight="bold")
        right.text(0.075, y_mid + 0.02, definition,
                   fontsize=9.6, color="#1a1a1a", va="center", family="serif")
        right.text(0.075, y_mid - 0.03, example,
                   fontsize=9.6, color=accent, va="center", family="serif")
        right.text(0.075, y_mid - 0.08, role,
                   fontsize=9.4, color="#444444", va="center", family="serif", style="italic")
        if idx < n_lens - 1:
            right.plot([0.03, 0.97], [y_top - lens_height, y_top - lens_height],
                       color=rule_color, lw=0.6)

    out = UNUSED_FIGDIR / "fig_label_fairness.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"Saved {out}")
    plt.close(fig)

def plot_equivalence_map() -> None:
    """Diagram: spaces times formulations, with the BMG equivalence drawn explicitly."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.6, 6.8))

    def draw_box(xy, w, h, text, fc, ec):
        rect = FancyBboxPatch(
            xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.014",
            fc=fc, ec=ec, lw=1.6,
        )
        ax.add_patch(rect)
        ax.text(xy[0] + w / 2.0, xy[1] + h / 2.0, text,
                ha="center", va="center", fontsize=10.5)

    # Column x positions and widths (three columns: space, collocation, Galerkin)
    x_space, w_space = 0.03, 0.27
    x_coll,  w_coll  = 0.355, 0.25
    x_gal,   w_gal   = 0.66,  0.22
    h_box = 0.115

    # Row bottoms (so y is bottom-left corner of each box)
    rows_y = [0.78, 0.60, 0.42, 0.16]

    # Row 1: polynomial space -> centered FD
    draw_box((x_space, rows_y[0]), w_space, h_box,
             "Polynomial space\n(degree $2r$ through nodes)",
             "#E6F2FF", "#1F78B4")
    draw_box((x_coll, rows_y[0]), w_coll, h_box,
             "Centered FD\n$\\mathrm{E}_{2r}$",
             "#FFF1E6", "#FF7F0E")

    # Row 2: nodal value and slope -> Pade compact FD
    draw_box((x_space, rows_y[1]), w_space, h_box,
             "Nodal value and slope data",
             "#E6F2FF", "#1F78B4")
    draw_box((x_coll, rows_y[1]), w_coll, h_box,
             r"Pad" + "\u00e9" + " compact FD" + "\n$\\mathrm{C}_{2r}$",
             "#FFF1E6", "#FF7F0E")

    # Row 3: spline space S_p -> BMC_p (collocation) and BMG_p (Galerkin)
    draw_box((x_space, rows_y[2]), w_space, h_box,
             "Spline space $S_p$\n(degree $p$, $C^{p-1}$)",
             "#E6F2FF", "#1F78B4")
    draw_box((x_coll, rows_y[2]), w_coll, h_box,
             "BM collocation\n$\\mathrm{BMC}_p$",
             "#FFF1E6", "#FF7F0E")
    draw_box((x_gal, rows_y[2]), w_gal, h_box,
             "BM Galerkin\n$\\mathrm{BMG}_p$",
             "#F0E6FF", "#9467BD")

    # Row 4: spline space S_{2p+1} -> BMC_{2p+1}
    draw_box((x_space, rows_y[3]), w_space, h_box,
             "Spline space $S_{2p+1}$\n(degree $2p+1$)",
             "#E6F2FF", "#1F78B4")
    draw_box((x_coll, rows_y[3]), w_coll, h_box,
             "BM collocation\n$\\mathrm{BMC}_{2p+1}$",
             "#FFF1E6", "#FF7F0E")

    # Horizontal arrows: space -> collocation (every row)
    for y in rows_y:
        cy = y + h_box / 2.0
        ax.annotate(
            "",
            xy=(x_coll, cy),
            xytext=(x_space + w_space, cy),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.3),
        )

    # Horizontal arrow: space S_p -> Galerkin BMG_p (above the BMC_p box)
    cy_iga_in = rows_y[2] + h_box / 2.0
    ax.annotate(
        "",
        xy=(x_gal, cy_iga_in),
        xytext=(x_coll + w_coll, cy_iga_in),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.3),
    )

    # The equivalence arrow.  Route it cleanly to avoid crossing any box:
    # BMG_p box bottom -> down -> across -> up to BMC_{2p+1} box top.
    iga_cx = x_gal + w_gal / 2.0
    iga_by = rows_y[2]
    bmc_cx = x_coll + w_coll / 2.0
    bmc_ty = rows_y[3] + h_box

    elbow_y = 0.305  # midway between row 3 bottom and row 4 top
    eq_color = "#D62728"
    ax.annotate(
        "",
        xy=(iga_cx, iga_by),
        xytext=(iga_cx, elbow_y),
        arrowprops=dict(arrowstyle="-", color=eq_color, lw=2.6),
    )
    ax.annotate(
        "",
        xy=(iga_cx, elbow_y),
        xytext=(bmc_cx, elbow_y),
        arrowprops=dict(arrowstyle="-", color=eq_color, lw=2.6),
    )
    ax.annotate(
        "",
        xy=(bmc_cx, bmc_ty),
        xytext=(bmc_cx, elbow_y),
        arrowprops=dict(arrowstyle="->", color=eq_color, lw=2.6),
    )
    ax.annotate(
        "",
        xy=(iga_cx, iga_by),
        xytext=(iga_cx, iga_by + 0.012),
        arrowprops=dict(arrowstyle="<-", color=eq_color, lw=2.6),
    )

    # Equivalence label, placed to the right of the elbow, well clear of every box.
    ax.text(
        iga_cx + 0.04, elbow_y,
        "exactly equal\n$\\mathrm{BMC}_{2p+1} = \\mathrm{BMG}_p$",
        ha="left", va="center", fontsize=11.0, color=eq_color, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=eq_color, boxstyle="round,pad=0.32"),
    )

    # Column headers
    ax.text(x_space + w_space / 2.0, 0.94, "Approximation space",
            ha="center", fontsize=11.5, fontweight="bold")
    ax.text(x_coll + w_coll / 2.0, 0.94, "Collocation",
            ha="center", fontsize=11.5, fontweight="bold")
    ax.text(x_gal + w_gal / 2.0, 0.94, "Galerkin",
            ha="center", fontsize=11.5, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    out = UNUSED_FIGDIR / "fig_equivalence_map.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def _fraction_string(value: float, max_denominator: int = 10000) -> str:
    if abs(value) < 1e-14:
        return "0"
    frac = Fraction(float(value)).limit_denominator(max_denominator)
    if frac.denominator == 1:
        return f"{frac.numerator}"
    return rf"$\frac{{{frac.numerator}}}{{{frac.denominator}}}$"


def _annotate_matrix(ax: plt.Axes, mat: np.ndarray, im, fmt: str = "float") -> None:
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            value = mat[i, j]
            if fmt == "fraction":
                text = _fraction_string(value)
                fontsize = 9.2
            else:
                text = f"{value:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
                fontsize = 8.4
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.48 else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color)


def bm_banded_map_plot_data(
    *,
    npts: int = 32,
    bmc_degree: int = 5,
    bmg_degree: int = 2,
) -> dict[str, object]:
    b_mass_entries, b_advection_entries = sa.collocation_mass_and_advection_entries(
        bmc_degree
    )
    m_mass_entries, m_advection_entries = sa.iga_mass_and_advection_entries(bmg_degree)

    b_matrix = sa.cyclic_compact_mass_matrix(b_mass_entries, npts)
    q_matrix = sa.cyclic_compact_advection_matrix(b_advection_entries, npts)
    m_matrix = sa.cyclic_compact_mass_matrix(m_mass_entries, npts)
    k_matrix = sa.cyclic_compact_advection_matrix(m_advection_entries, npts)
    d_matrix = sa.collocation_derivative_matrix(bmc_degree, npts)
    eigvals = np.linalg.eigvals(d_matrix)
    eigvals = eigvals[np.argsort(eigvals.imag)]

    return {
        "npts": npts,
        "colors": {"primary": "#22517A", "secondary": "#B06A2C"},
        "bmc": {
            "family_label": rf"$\mathrm{{BMC{bmc_degree}}}$",
            "matrix_order": ("B", "B'"),
            "B": b_matrix,
            "B'": q_matrix,
        },
        "bmg": {
            "family_label": rf"$\mathrm{{BMG{bmg_degree}}}$",
            "matrix_order": ("M", "K"),
            "M": m_matrix,
            "K": k_matrix,
        },
        "structure": {
            "kind": "imaginary_spectrum",
            "label": rf"$\mathrm{{BMC{bmc_degree}}} = \mathrm{{BMG{bmg_degree}}}$",
            "eigenvalues": eigvals,
        },
    }


def bm_second_derivative_plot_data(
    *,
    npts: int = 32,
    bmc_degree: int = 5,
    bmg_degree: int = 2,
) -> dict[str, object]:
    b_mass_entries, b_second_entries = sa.collocation_mass_and_second_derivative_entries(
        bmc_degree
    )
    m_mass_entries, m_second_entries = sa.iga_mass_and_second_derivative_entries(
        bmg_degree
    )

    b_matrix = sa.cyclic_compact_mass_matrix(b_mass_entries, npts)
    q2_matrix = sa.cyclic_compact_mass_matrix(b_second_entries, npts)
    m_matrix = sa.cyclic_compact_mass_matrix(m_mass_entries, npts)
    s_matrix = sa.cyclic_compact_mass_matrix(m_second_entries, npts)
    d2_matrix = sa.collocation_second_derivative_matrix(bmc_degree, npts)
    eigvals = np.linalg.eigvalsh(d2_matrix)

    return {
        "npts": npts,
        "colors": {"primary": "#22517A", "secondary": "#B06A2C"},
        "bmc": {
            "family_label": rf"$\mathrm{{BMC{bmc_degree}}}^{{(2)}}$",
            "matrix_order": ("B", "B''"),
            "B": b_matrix,
            "B''": q2_matrix,
        },
        "bmg": {
            "family_label": rf"$\mathrm{{BMG{bmg_degree}}}^{{(2)}}$",
            "matrix_order": ("M", "S"),
            "M": m_matrix,
            "S": s_matrix,
        },
        "structure": {
            "kind": "nonpositive_spectrum",
            "label": rf"$\mathrm{{BMC{bmc_degree}}}^{{(2)}} = \mathrm{{BMG{bmg_degree}}}^{{(2)}}$",
            "eigenvalues": eigvals,
        },
    }


def _plot_bm_map_panel(
    ax: plt.Axes,
    family: dict[str, object],
    *,
    colors: dict[str, str],
    npts: int,
    title: str,
    xlabel: str,
    ylabel: str,
    tol: float = 1.0e-14,
) -> None:
    primary_name, secondary_name = family["matrix_order"]
    primary = family[primary_name]
    secondary = family[secondary_name]

    columns = np.arange(npts)
    offsets = ((columns + npts // 2) % npts) - npts // 2
    order = np.argsort(offsets)
    offsets = offsets[order]
    primary_row = np.asarray(primary[0, :])[order]
    secondary_row = np.asarray(secondary[0, :])[order]
    active = (np.abs(primary_row) > tol) | (np.abs(secondary_row) > tol)
    radius = int(np.max(np.abs(offsets[active])))

    for values, name, color, marker in (
        (primary_row, primary_name, colors["primary"], "s"),
        (secondary_row, secondary_name, colors["secondary"], "o"),
    ):
        scale = float(np.max(np.abs(values)))
        normalized = values / scale if scale else values
        mask = np.abs(values) > tol
        ax.vlines(offsets[mask], 0.0, normalized[mask], color=color,
                  linewidth=1.5, alpha=0.85, zorder=2)
        ax.scatter(offsets[mask], normalized[mask], s=38, marker=marker,
                   facecolor=color, edgecolor="white", linewidth=0.7,
                   label=rf"${name}$", zorder=3)

    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlim(-radius - 0.7, radius + 0.7)
    ax.set_xticks(np.arange(-radius, radius + 1))
    ax.set_ylim(-1.15, 1.15)
    ax.grid(axis="y", color="#BDBDBD", alpha=0.25, linewidth=0.8)
    ax.set_title(title, fontsize=13.5, pad=8, loc="left")
    ax.set_xlabel(xlabel, fontsize=13.0)
    ax.set_ylabel(ylabel, fontsize=13.0)
    ax.tick_params(labelsize=12.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_bm_structure_panel(
    ax: plt.Axes,
    structure: dict[str, object],
    *,
    title: str,
) -> None:
    kind = structure["kind"]
    eigvals = np.asarray(structure["eigenvalues"])

    if kind == "imaginary_spectrum":
        imag_max = float(np.max(np.abs(eigvals.imag)))
        real_max = float(np.max(np.abs(eigvals.real)))
        x_half = max(0.08 * imag_max, 6.0 * real_max, 1.0e-3)
        ax.axvline(0.0, color="#666666", lw=1.0, ls="--", alpha=0.8)
        ax.scatter(
            eigvals.real,
            eigvals.imag,
            s=26,
            color="#6A3D9A",
            alpha=0.92,
            zorder=3,
        )
        ax.set_xlim(-x_half, x_half)
        ax.set_ylim(-1.08 * imag_max, 1.08 * imag_max)
        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    elif kind == "nonpositive_spectrum":
        eigvals = np.asarray(eigvals, dtype=float)
        x_min = float(eigvals.min())
        x_pad = max(0.05 * abs(x_min), 1.0e-3)
        ax.axvline(0.0, color="#666666", lw=1.0, ls="--", alpha=0.8)
        ax.scatter(
            eigvals,
            np.zeros_like(eigvals),
            s=26,
            color="#2C7FB8",
            alpha=0.92,
            zorder=3,
        )
        ax.set_xlim(x_min - x_pad, x_pad)
        ax.set_ylim(-0.35, 0.35)
        ax.set_yticks([])
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("")
    else:
        raise ValueError(f"Unknown BM structure kind: {kind}")

    ax.set_title(title, fontsize=11.3, pad=8)
    ax.grid(True, color="#BDBDBD", alpha=0.22, linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_bm_structure_figure(
    data: dict[str, object],
    *,
    titles: tuple[str, str],
    labels: tuple[tuple[str, str], tuple[str, str]],
    output_name: str,
) -> None:
    npts = int(data["npts"])
    colors = data["colors"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 4.4))

    _plot_bm_map_panel(
        ax_a,
        data["bmc"],
        colors=colors,
        npts=npts,
        title=titles[0],
        xlabel=labels[0][0],
        ylabel=labels[0][1],
    )

    _plot_bm_map_panel(
        ax_b,
        data["bmg"],
        colors=colors,
        npts=npts,
        title=titles[1],
        xlabel=labels[1][0],
        ylabel=labels[1][1],
    )

    handles = []
    legend_labels = []
    for ax in (ax_a, ax_b):
        hs, ls = ax.get_legend_handles_labels()
        handles.extend(hs)
        legend_labels.extend(ls)
    fig.legend(handles, legend_labels, loc="upper center", ncol=4,
               frameon=False, fontsize=12.0, bbox_to_anchor=(0.5, 0.99),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)

    out = FIGDIR / output_name
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_bm_banded_maps() -> None:
    setup_style()
    data = bm_banded_map_plot_data()
    _plot_bm_structure_figure(
        data,
        titles=(
            r"(a)",
            r"(b)",
        ),
        labels=(("periodic offset", "entry / maximum magnitude"),
                ("periodic offset", "entry / maximum magnitude")),
        output_name="fig_bm_banded_maps.pdf",
    )


def plot_bm_second_derivative_structure() -> None:
    setup_style()
    data = bm_second_derivative_plot_data()
    _plot_bm_structure_figure(
        data,
        titles=(
            r"(a)",
            r"(b)",
        ),
        labels=(("periodic offset", "entry / maximum magnitude"),
                ("periodic offset", "entry / maximum magnitude")),
        output_name="fig_bm_second_derivative_structure.pdf",
    )


def plot_bspline_ladder() -> None:
    setup_style()
    plt.rcParams.update({"font.size": 12.0, "axes.labelsize": 12.5,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.7), sharey=True)
    x = np.linspace(-3.2, 3.2, 1200)
    panels = [
        (0, r"$\beta_0$", r"one box"),
        (2, r"$\beta_2 = \beta_0^{*3}$", r"threefold convolution"),
        (5, r"$\beta_5 = \beta_2 * \beta_2$", r"self-overlap gives degree $2p+1$"),
    ]
    colors = ["#22517A", "#B06A2C", "#2C7D4F"]

    handles = []
    for panel, ax, (degree, title, subtitle), color in zip("abc", axes, panels, colors):
        y = sa.centered_cardinal_bspline(degree, x)
        line, = ax.plot(x, y, color=color, lw=2.2, label=title)
        handles.append(line)
        ax.fill_between(x, 0, y, color=color, alpha=0.18)
        support_radius = 0.5 * (degree + 1)
        ax.axvline(-support_radius, color="#666", lw=0.7, ls="--", alpha=0.6)
        ax.axvline(support_radius, color="#666", lw=0.7, ls="--", alpha=0.6)
        ax.set_title(f"({panel})", fontsize=13.0, loc="left")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel(r"$x$")
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel(r"$\beta_p(x)$")
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=11.5, bbox_to_anchor=(0.5, 0.99),
               handletextpad=0.5, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = FIGDIR / "fig_bspline_ladder.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_overlap_identity() -> None:
    setup_style()
    plt.rcParams.update({"font.size": 16.0, "axes.labelsize": 16.5,
                         "xtick.labelsize": 15.0, "ytick.labelsize": 15.0})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 5.0))
    x = np.linspace(-2.2, 4.2, 1600)
    fixed = sa.centered_cardinal_bspline(2, x)
    shifts = [0.0, 2.0]
    labels = [r"(a)", r"(b)"]

    for ax, shift, label in zip(axes[:2], shifts, labels):
        shifted = sa.centered_cardinal_bspline(2, x - shift)
        product = fixed * shifted
        ax.plot(x, fixed, color="#22517A", lw=2.1, label=r"$\beta_2(t)$")
        ax.plot(x, shifted, color="#B06A2C", lw=2.1, label=rf"$\beta_2(t-{int(shift)})$")
        ax.fill_between(x, 0, product, color="#6F8FAF", alpha=0.32,
                        label=r"integrand $\beta_2(t)\beta_2(t-n)$")
        area = np.trapezoid(product, x)
        sample = sa.centered_cardinal_bspline(5, np.array([shift]))[0]
        ax.text(
            0.97, 0.94,
            rf"area $\approx {area:.6f}$" + "\n" + rf"$\beta_5({int(shift)}) = {sample:.6f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=15.0,
            bbox=dict(facecolor="white", edgecolor="#999", boxstyle="round,pad=0.25"),
        )
        ax.set_title(label, fontsize=17.0, loc="left")
        ax.set_xlim(-1.8, 3.8)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)

    ax = axes[2]
    shift_grid = np.linspace(-3.0, 3.0, 500)
    overlap_curve = np.empty_like(shift_grid)
    x_int = np.linspace(-2.5, 2.5, 2000)
    fixed_int = sa.centered_cardinal_bspline(2, x_int)
    for idx, shift in enumerate(shift_grid):
        overlap_curve[idx] = np.trapezoid(
            fixed_int * sa.centered_cardinal_bspline(2, x_int - shift), x_int
        )
    beta5 = sa.centered_cardinal_bspline(5, shift_grid)
    ax.plot(shift_grid, overlap_curve, color="#22517A", lw=2.4,
            label=r"$m(n)=\int \beta_2(t)\beta_2(t-n)\,dt$")
    ax.plot(shift_grid, beta5, "--", color="#111111", lw=1.6,
            label=r"$\beta_5(n)$")
    sample_shifts = np.array([0.0, 1.0, 2.0])
    ax.plot(
        sample_shifts,
        sa.centered_cardinal_bspline(5, sample_shifts),
        "o",
        color="#B06A2C",
        ms=7.5,
        label=r"integer samples",
    )
    ax.set_title(r"(c)", fontsize=17.0, loc="left")
    ax.set_xlabel(r"shift $n$")
    ax.set_ylabel("value")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 0.78)
    ax.grid(True, alpha=0.25)
    handles = []
    legend_labels = []
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        for handle, text in zip(hs, ls):
            if text not in legend_labels:
                handles.append(handle)
                legend_labels.append(text)
    fig.legend(handles, legend_labels, loc="upper center", ncol=3,
               frameon=False, fontsize=14.0, bbox_to_anchor=(0.5, 0.99),
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.6)
    out = FIGDIR / "fig_overlap_identity.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_matrix_identity_bmc5_iga2() -> None:
    fig = plt.figure(figsize=(14.4, 11.4))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.85, 1.7],
        hspace=0.32,
        wspace=0.32,
        left=0.05,
        right=0.985,
        top=0.93,
        bottom=0.04,
    )
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])

    mass_bmc, adv_bmc = sa.collocation_mass_and_advection_entries(5)
    mass_iga, adv_iga = sa.iga_mass_and_advection_entries(2)
    c8 = sa.cyclic_compact_mass_matrix(mass_bmc, 8)
    a8 = sa.cyclic_compact_advection_matrix(adv_bmc, 8)
    d8 = sa.collocation_derivative_matrix(5, 8)

    n_mass = np.arange(len(mass_bmc))
    n_adv = np.arange(len(adv_bmc))

    ax = axes[0, 0]
    ax.stem(n_mass - 0.06, mass_bmc, linefmt="#1F78B4", markerfmt="o", basefmt=" ")
    ax.stem(n_mass + 0.06, mass_iga, linefmt="#D62728", markerfmt="s", basefmt=" ")
    ax.set_title(r"(a) Mass entries: BMC5 samples = BMG2 overlaps", fontsize=10.5)
    ax.set_xlabel(r"shift $n$")
    ax.set_ylabel(r"$m_n$")
    ax.set_xticks(n_mass)
    ax.set_ylim(0.0, max(mass_bmc) * 1.55)
    ax.grid(True, alpha=0.25)
    ax.legend(
        [Line2D([0], [0], color="#1F78B4", marker="o", lw=0),
         Line2D([0], [0], color="#D62728", marker="s", lw=0)],
        [r"$\beta_5(n)$", r"$\int \beta_2(t)\beta_2(t-n)\,dt$"],
        loc="upper right",
        fontsize=8.4,
        framealpha=0.92,
    )

    ax = axes[0, 1]
    ax.stem(n_adv[1:] - 0.06, adv_bmc[1:], linefmt="#1F78B4", markerfmt="o", basefmt=" ")
    ax.stem(n_adv[1:] + 0.06, adv_iga[1:], linefmt="#D62728", markerfmt="s", basefmt=" ")
    ax.set_title(r"(b) Advection entries: $-\beta_5'(n)$ = overlap derivative",
                 fontsize=10.5)
    ax.set_xlabel(r"shift $n$")
    ax.set_ylabel(r"$k_n$")
    ax.set_xticks(n_adv[1:])
    ax.set_ylim(0.0, max(adv_bmc[1:]) * 1.55)
    ax.grid(True, alpha=0.25)
    ax.legend(
        [Line2D([0], [0], color="#1F78B4", marker="o", lw=0),
         Line2D([0], [0], color="#D62728", marker="s", lw=0)],
        [r"$-\beta_5'(n)$", r"$\int \beta_2(t)\beta_2'(t-n)\,dt$"],
        loc="upper right",
        fontsize=8.4,
        framealpha=0.92,
    )

    ax = axes[0, 2]
    ax.axis("off")
    text = (
        r"$\mathrm{BMC5}: \quad D = C^{-1}A$" "\n\n"
        r"$C_{ij} = \beta_5(i-j)$" "\n"
        r"$A_{ij} = -\beta_5'(i-j)$" "\n\n"
        r"$\mathrm{BMG2}: \quad D = M^{-1}K$" "\n\n"
        r"$M_{ij} = \int \beta_2(t-i)\beta_2(t-j)\,dt$" "\n"
        r"$K_{ij} = \int \beta_2(t-i)\beta_2'(t-j)\,dt$" "\n\n"
        r"These are entrywise identical."
    )
    ax.text(
        0.50, 0.50, text, ha="center", va="center", fontsize=10.8,
        bbox=dict(facecolor="#FAFAFA", edgecolor="#999", boxstyle="round,pad=0.4")
    )
    ax.set_title(r"(c) Two constructions, one matrix", fontsize=10.5)

    matrices = [
        (axes[1, 0], c8, r"(d) Shared compact mass matrix $C_8 = M_8$", "fraction", "PuBu"),
        (axes[1, 1], a8, r"(e) Shared compact advection matrix $A_8 = K_8$", "fraction", "RdBu_r"),
        (axes[1, 2], d8, r"(f) Shared dense derivative matrix $D_8$", "float", "coolwarm"),
    ]
    for ax, mat, title, fmt, cmap in matrices:
        vmax = np.max(np.abs(mat))
        im = ax.imshow(
            mat,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax if vmax > 0 else 1.0,
            aspect="equal",
        )
        _annotate_matrix(ax, mat, im, fmt=fmt)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.tick_params(labelsize=7.4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = UNUSED_FIGDIR / "fig_matrix_identity_bmc5_iga2.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_stability_regions() -> None:
    """Forward Euler vs RK4 stability regions, with the imaginary axis through both."""
    setup_style()
    plt.rcParams.update({"font.size": 14.5, "axes.labelsize": 15.0,
                         "xtick.labelsize": 13.5, "ytick.labelsize": 13.5})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2))

    # Panel (a): Forward Euler region
    ax = axes[0]
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    fe_x = -1.0 + np.cos(theta)
    fe_y = np.sin(theta)
    ax.fill(fe_x, fe_y, color="#A6CEE3", alpha=0.55)
    ax.plot(fe_x, fe_y, color="#1F78B4", lw=1.5, label="FE region")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.plot([0, 0], [-3, 3], color="#D62728", lw=2.3, alpha=0.85,
            label="imaginary axis (advection)")
    ax.plot([0], [1.0], "o", color="#D62728", ms=11, mfc="white", mew=2.0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title("(a)", fontsize=15.5, loc="left")
    ax.set_xlabel(r"$\mathrm{Re}(\Delta t\,\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\Delta t\,\lambda)$")
    ax.grid(True, alpha=0.3)

    # Panel (b): RK4 region
    ax = axes[1]
    rk4 = sa.rk4_stability_boundary(800)
    ax.fill(rk4.real, rk4.imag, color="#C5E0B4", alpha=0.55)
    ax.plot(rk4.real, rk4.imag, color="#2CA02C", lw=1.5, label="RK4 region")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.plot([0, 0], [-3, 3], color="#D62728", lw=2.3, alpha=0.85,
            label="imaginary axis (advection)")
    ax.plot([0, 0], [2.828, -2.828], "s", color="#D62728",
            ms=11, mfc="white", mew=2.0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title("(b)", fontsize=15.5, loc="left")
    ax.set_xlabel(r"$\mathrm{Re}(\Delta t\,\lambda)$")
    ax.grid(True, alpha=0.3)
    handles = []
    legend_labels = []
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        for handle, text in zip(hs, ls):
            if text not in legend_labels:
                handles.append(handle)
                legend_labels.append(text)
    fig.legend(handles, legend_labels, loc="upper center", ncol=3,
               frameon=False, fontsize=13.5, bbox_to_anchor=(0.5, 0.99),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.0)
    out = FIGDIR / "fig_stability_regions.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_other_pdes(w: np.ndarray) -> None:
    """The same modified wavenumber controls every linear constant-coefficient wave PDE."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    kph_e6 = sa.kprime_centered_fd(w, 6)
    kph_c6 = sa.kprime_compact(w, 6)
    kph_bmc5 = sa.kprime_collocation(w, 5)

    panels = [
        {
            "title": "(a) Scalar advection $u_t + u_x = 0$",
            "exact": w / np.pi,
            "e6": kph_e6 / np.pi,
            "c6": kph_c6 / np.pi,
            "spl": kph_bmc5 / np.pi,
            "ylabel": r"$|\lambda|\,h\,/\,\pi$",
            "annotation": r"$\lambda(\omega) = i\,k'h(\omega)/h$",
        },
        {
            "title": "(b) Linear acoustics $(p,u)$",
            "exact": w / np.pi,
            "e6": kph_e6 / np.pi,
            "c6": kph_c6 / np.pi,
            "spl": kph_bmc5 / np.pi,
            "ylabel": r"$|\lambda|\,h\,/\,(c\,\pi)$",
            "annotation": r"$\lambda_{\pm}(\omega) = \pm\,i\,c\,k'h(\omega)/h$",
        },
        {
            "title": r"(c) Dispersive $u_t = i\,u_{xx}$",
            "exact": (w / np.pi) ** 2,
            "e6": (kph_e6 / np.pi) ** 2,
            "c6": (kph_c6 / np.pi) ** 2,
            "spl": (kph_bmc5 / np.pi) ** 2,
            "ylabel": r"$|\lambda|\,h^{2}/\pi^{2}$",
            "annotation": r"$\lambda(\omega) \sim (k'h(\omega)/h)^{2}$",
        },
    ]

    for ax, panel in zip(axes, panels):
        ax.plot(w / np.pi, panel["exact"], "--", color="#222", lw=1.5, label="exact")
        ax.plot(w / np.pi, panel["e6"], color=COLORS["E6"], lw=1.6, label="E6")
        ax.plot(w / np.pi, panel["c6"], color=COLORS["C6"], lw=1.6, ls="-.", label="C6")
        ax.plot(w / np.pi, panel["spl"], color=COLORS["BMC5 = BMG2"], lw=1.8, label=r"BMC5 = BMG2")
        ax.set_title(panel["title"])
        ax.set_xlabel(r"$\omega/\pi$")
        ax.set_ylabel(panel["ylabel"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.04,
            0.95,
            panel["annotation"],
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.3"),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=8.4,
        framealpha=0.92,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    out = FIGDIR / "fig_other_pdes.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def _phase_speed_ratio_2d(kph_lookup, omega: float, theta: np.ndarray) -> np.ndarray:
    return (
        np.cos(theta) * kph_lookup(omega * np.cos(theta))
        + np.sin(theta) * kph_lookup(omega * np.sin(theta))
    ) / omega


def plot_bspline_degree_ladder(w: np.ndarray) -> None:
    """Odd-degree BM family plot tied to a simple derivative-cost bill."""
    setup_style()
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.5),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    labels = [
        "C4 = BMC3 = BMG1",
        "BMC5 = BMG2",
        "BMC7 = BMG3",
        "BMC9 = BMG4",
        "BMC11 = BMG5",
    ]
    short_labels = [
        "BMC3 = BMG1",
        "BMC5 = BMG2",
        "BMC7 = BMG3",
        "BMC9 = BMG4",
        "BMC11 = BMG5",
    ]

    curve_map = {
        "C4 = BMC3 = BMG1": sa.kprime_compact(w, 4),
        "BMC5 = BMG2": sa.kprime_collocation(w, 5),
        "BMC7 = BMG3": sa.kprime_iga(w, 3),
        "BMC9 = BMG4": sa.kprime_iga(w, 4),
        "BMC11 = BMG5": sa.kprime_iga(w, 5),
    }

    ax0.plot(w / np.pi, w / np.pi, "--", color=COLORS["Exact"], lw=1.4, label="exact")
    for label, short in zip(labels, short_labels):
        curve = curve_map[label]
        metrics = sa.summarize_curve(w, curve)
        color = COLORS[label]
        ax0.plot(w / np.pi, curve / np.pi, color=color, lw=1.8, label=short)
        omega_mark = metrics.omega_pi_1pct * np.pi
        y_mark = np.interp(omega_mark, w, curve) / np.pi
        ax0.plot(metrics.omega_pi_1pct, y_mark, marker="o", ms=4.0, color=color)

    ax0.set_title("(a)", loc="left", fontsize=14)
    ax0.set_xlabel(r"$\omega/\pi$")
    ax0.set_ylabel(r"$k'h/\pi$")
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(0.0, 1.04)
    ax0.grid(True, alpha=0.3)
    ax0.text(
        0.03,
        0.05,
        "filled dots = 1% cutoff",
        transform=ax0.transAxes,
        fontsize=11.5,
        bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.25"),
    )

    x = np.arange(len(labels), dtype=float)
    p1 = []
    p5 = []
    colors = []
    for label in labels:
        curve = curve_map[label]
        metrics = sa.summarize_curve(w, curve)
        p1.append(metrics.ppw_1pct)
        p5.append(metrics.ppw_5pct)
        colors.append(COLORS[label])

    for xpos, y1, y5, color in zip(x, p1, p5, colors):
        ax1.plot([xpos, xpos], [y5, y1], color=color, lw=3.0, solid_capstyle="round")
        ax1.plot(xpos, y1, marker="o", ms=6.5, color=color)
        ax1.plot(xpos, y5, marker="o", ms=6.5, mfc="white", mec=color, mew=1.6)
        ax1.text(xpos + 0.07, y1 + 0.05, f"{y1:.2f}", color=color, fontsize=10.5, ha="left", va="bottom")
        ax1.text(xpos + 0.07, y5 - 0.05, f"{y5:.2f}", color=color, fontsize=10.5, ha="left", va="top")

    ax1.set_title("(b)", loc="left", fontsize=14)
    ax1.set_ylabel(r"threshold $P_\varepsilon$ (smaller is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["$3$\n$8$", "$5$\n$15$", "$7$\n$22$",
                         "$9$\n$29$", "$11$\n$36$"])
    ax1.set_xlabel("degree $p$ (top);\nwork per point (bottom)")
    ax1.set_xlim(-0.45, len(labels) - 0.1)
    # Pinned so the two-line x label below cannot change the tick spacing.
    ax1.set_yticks(np.arange(2.0, 6.75, 0.5))
    ax1.set_ylim(2.1, 6.2)
    ax1.grid(True, axis="y", alpha=0.3)
    add_better_arrow(ax1, xy=(0.10, 0.18), xytext=(0.18, 0.25), text="shorter waves")
    legend_handles = [
        Line2D([0], [0], marker="o", color="#555", lw=0, markersize=6.0, label=r"$P_{1\%}$"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#555",
            lw=0,
            markersize=6.0,
            label=r"$P_{5\%}$",
        ),
    ]
    method_handles, method_labels = ax0.get_legend_handles_labels()
    all_handles = method_handles + legend_handles
    all_labels = method_labels + [r"$P_{1\%}$", r"$P_{5\%}$"]
    if _make_overhead_legend is not None:
        _make_overhead_legend(fig, all_handles, all_labels, ncol=4,
                              fontsize=10.8, rect=(0, 0, 1, 0.78))
    else:
        fig.legend(all_handles, all_labels, loc="upper center", ncol=4,
                   frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.78))
    out = FIGDIR / "fig_bspline_degree_ladder.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_anisotropy_order6(w: np.ndarray) -> None:
    """Plot directional dispersion for the matched sixth-order trio."""
    setup_full_width_print_style(three_panel=True)
    del w
    ppw = 4.0
    omega = 2.0 * np.pi / ppw
    theta = np.linspace(0.0, 2.0 * np.pi, 721)

    lookup = {
        "E6": lambda x: sa.kprime_centered_fd(np.asarray(x, dtype=float), 6),
        "C6": lambda x: sa.kprime_compact(np.asarray(x, dtype=float), 6),
        "BMC5 = BMG2": lambda x: sa.kprime_collocation(np.asarray(x, dtype=float), 5),
    }

    curves = {}
    max_err = {}
    for label, func in lookup.items():
        cps = _phase_speed_ratio_2d(func, omega, theta)
        curves[label] = cps
        max_err[label] = float(np.max(np.abs(cps - 1.0)))

    rmin = min(float(np.min(cps)) for cps in curves.values()) - 0.003
    rmax = 1.001

    fig = plt.figure(figsize=(9.2, 3.9))
    axes = [fig.add_subplot(1, 3, idx + 1, projection="polar") for idx in range(3)]
    labels = ["E6", "C6", "BMC5 = BMG2"]

    method_handles = []
    for panel, (ax, label) in enumerate(zip(axes, labels)):
        cps = curves[label]
        color = COLORS[label]
        ax.plot(theta, np.ones_like(theta), "--", color="#444444", lw=1.2)
        method_line, = ax.plot(theta, cps, color=color, lw=2.0)
        method_handles.append(method_line)
        ax.fill_between(theta, cps, 1.0, color=color, alpha=0.12)
        ax.set_rlim(rmin, rmax)
        ax.set_rticks([0.94, 0.96, 0.98, 1.00])
        ax.set_rlabel_position(112.5)
        ax.grid(alpha=0.35)
        ax.set_title(f"({'abc'[panel]})", loc="left", va="bottom",
                     fontsize=14, pad=12)
        ax.tick_params(labelsize=13.2)

    legend_labels = [
        f"{label}, max error {100.0 * max_err[label]:.1f}%"
        for label in labels
    ]
    if _make_overhead_legend is not None:
        _make_overhead_legend(fig, method_handles, legend_labels, ncol=3,
                              fontsize=13.2, rect=(0, 0, 1, 0.80))
    else:
        fig.legend(method_handles, legend_labels, loc="upper center", ncol=3,
                   frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.80))
    out = FIGDIR / "fig_anisotropy_order6.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def collect_gaussian_convergence_results(
    npts_values: list[int] | None = None,
    *,
    time_final: float = 20.0,
    length: float = 1.0,
) -> dict[str, list[SpatialAccuracyResult]]:
    npts_values = [32, 64, 128, 256, 512, 1024] if npts_values is None else npts_values
    tasks = []
    for label in UNIQUE_METHODS:
        for npts in npts_values:
            tasks.append(
                delayed(run_gaussian_convergence_case)(
                    label,
                    npts,
                    time_final=time_final,
                    length=length,
                )
            )

    results = Parallel(n_jobs=PARALLEL_JOBS, verbose=0)(tasks)
    grouped: dict[str, list[SpatialAccuracyResult]] = {label: [] for label in UNIQUE_METHODS}
    for result in results:
        grouped[result.label].append(result)
    for label in grouped:
        grouped[label].sort(key=lambda item: item.npts)
    return grouped


def collect_gaussian_diffusion_convergence_results(
    npts_values: list[int] | None = None,
    *,
    labels: tuple[str, ...] = DIRECT_SECOND_DERIVATIVE_TRIO_LABELS,
    time_final: float = DIRECT_DIFFUSION_TIME,
    nu: float = DIRECT_DIFFUSION_NU,
    length: float = 1.0,
    sigma: float = DIRECT_DIFFUSION_SIGMA,
) -> dict[str, list[SpatialAccuracyResult]]:
    npts_values = [48, 64, 96, 128, 192, 256, 384, 512] if npts_values is None else npts_values
    tasks = []
    for label in labels:
        for npts in npts_values:
            tasks.append(
                delayed(run_gaussian_diffusion_convergence_case)(
                    label,
                    npts,
                    time_final=time_final,
                    nu=nu,
                    length=length,
                    sigma=sigma,
                )
            )

    results = Parallel(n_jobs=PARALLEL_JOBS, verbose=0)(tasks)
    grouped: dict[str, list[SpatialAccuracyResult]] = {label: [] for label in labels}
    for result in results:
        grouped[result.label].append(result)
    for label in grouped:
        grouped[label].sort(key=lambda item: item.npts)
    return grouped


def collect_runtime_results(
    metrics_map: dict[str, sa.CurveMetrics],
    npts_values: list[int] | None = None,
    *,
    labels: tuple[str, ...] | None = None,
    time_final: float = 1.0,
    repeats: int = 10,
    length: float = 1.0,
) -> dict[str, list[MarchResult]]:
    npts_values = [32, 64, 128, 256, 512, 1024, 2048, 4096] if npts_values is None else npts_values
    labels = tuple(UNIQUE_METHODS) if labels is None else labels
    grouped: dict[str, list[MarchResult]] = {label: [] for label in labels}

    # Keep wall-time measurements isolated. Concurrent timing jobs would
    # contend for the same CPU and bias the cost-vs-accuracy story.
    for label in labels:
        cfl = 0.9 * metrics_map[label].rk4_cfl
        for npts in npts_values:
            grouped[label].append(
                measure_marching_time(
                    label,
                    npts,
                    time_final=time_final,
                    cfl=cfl,
                    repeats=repeats,
                    length=length,
                )
            )
    for label in grouped:
        grouped[label].sort(key=lambda item: item.npts)
    return grouped


def collect_carrier_convergence_results(
    regime: CarrierRegimeSpec,
    npts_values: list[int] | None = None,
    *,
    labels: tuple[str, ...] = SHORTWAVE_BASELINE_LABELS,
    length: float = 1.0,
) -> dict[str, list[SpatialAccuracyResult]]:
    npts_values = (
        [192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 896, 1024]
        if npts_values is None
        else npts_values
    )
    sigma_env = carrier_sigma_env(regime, length)
    tasks = []
    for label in labels:
        for npts in npts_values:
            tasks.append(
                delayed(run_carrier_convergence_case)(
                    label,
                    npts,
                    mode=regime.mode,
                    time_final=regime.time_final,
                    length=length,
                    sigma_env=sigma_env,
                )
            )

    results = Parallel(n_jobs=PARALLEL_JOBS, verbose=0)(tasks)
    grouped: dict[str, list[SpatialAccuracyResult]] = {label: [] for label in labels}
    for result in results:
        grouped[result.label].append(result)
    for label in grouped:
        grouped[label].sort(key=lambda item: item.npts)
    return grouped


def collect_carrier_runtime_results(
    regime: CarrierRegimeSpec,
    npts_values: list[int] | None = None,
    *,
    labels: tuple[str, ...] = SHORTWAVE_BASELINE_LABELS,
    repeats: int = 10,
    length: float = 1.0,
) -> dict[str, list[MarchResult]]:
    npts_values = [192, 256, 384, 512, 768] if npts_values is None else npts_values
    sigma_env = carrier_sigma_env(regime, length)
    grouped: dict[str, list[MarchResult]] = {label: [] for label in labels}

    # Keep wall-time measurements isolated for the same reason as above.
    for label in labels:
        for npts in npts_values:
            grouped[label].append(
                measure_carrier_marching_time(
                    label,
                    npts,
                    mode=regime.mode,
                    time_final=regime.time_final,
                    cfl=regime.common_cfl,
                    repeats=repeats,
                    length=length,
                    sigma_env=sigma_env,
                )
            )
    for label in grouped:
        grouped[label].sort(key=lambda item: item.npts)
    return grouped


def collect_shortwave_candidate_summaries(
    npts_values: list[int] | None = None,
    *,
    length: float = 1.0,
) -> list[ShortwaveCandidateSummary]:
    summaries = []
    for regime in SHORTWAVE_CANDIDATES:
        results = collect_carrier_convergence_results(
            regime,
            npts_values=npts_values,
            labels=SHORTWAVE_BASELINE_LABELS,
            length=length,
        )
        summaries.append(summarize_shortwave_candidate(regime, results))
    return summaries


@dataclass(frozen=True)
class FrontierPoint:
    label: str
    npts: int
    derivative_flops: float
    metric_value: float


def family_of(label: str) -> str:
    if label.startswith("E"):
        return "explicit"
    if label.startswith("C"):
        return "compact"
    return "spline"


def trustworthy_mode_count(
    npts: int,
    metrics: sa.CurveMetrics,
    *,
    tol: str = "1pct",
) -> float:
    if tol == "1pct":
        return npts / metrics.ppw_1pct
    if tol == "5pct":
        return npts / metrics.ppw_5pct
    raise ValueError(f"Unknown resolution tolerance label: {tol}")


def pareto_frontier_mask(
    costs: np.ndarray,
    metric_values: np.ndarray,
    *,
    larger_is_better: bool,
) -> np.ndarray:
    order = np.argsort(costs, kind="mergesort")
    keep = np.zeros_like(costs, dtype=bool)
    if larger_is_better:
        best_so_far = -np.inf
        for idx in order:
            value = metric_values[idx]
            if not np.isfinite(best_so_far):
                keep[idx] = True
                best_so_far = value
                continue
            tol = 1e-12 * max(1.0, abs(best_so_far))
            if value > best_so_far + tol:
                keep[idx] = True
                best_so_far = value
    else:
        best_so_far = np.inf
        for idx in order:
            value = metric_values[idx]
            if not np.isfinite(best_so_far):
                keep[idx] = True
                best_so_far = value
                continue
            tol = 1e-12 * max(1.0, abs(best_so_far))
            if value < best_so_far - tol:
                keep[idx] = True
                best_so_far = value
    return keep


def collect_actual_frontier_points(
    convergence_results: dict[str, list[SpatialAccuracyResult]],
) -> list[FrontierPoint]:
    points = [
        FrontierPoint(item.label, item.npts, float(item.derivative_flops), float(item.error))
        for data in convergence_results.values()
        for item in data
        if np.isfinite(item.error) and item.error > 0.0
    ]
    costs = np.array([point.derivative_flops for point in points], dtype=float)
    errors = np.array([point.metric_value for point in points], dtype=float)
    mask = pareto_frontier_mask(costs, errors, larger_is_better=False)
    return [point for point, keep in zip(points, mask) if keep]


def collect_resolution_frontier_points(
    convergence_results: dict[str, list[SpatialAccuracyResult]],
    metrics_map: dict[str, sa.CurveMetrics],
    *,
    tol: str = "1pct",
) -> list[FrontierPoint]:
    points = []
    for label, data in convergence_results.items():
        metrics = metrics_map[label]
        for item in data:
            points.append(
                FrontierPoint(
                    label=label,
                    npts=item.npts,
                    derivative_flops=float(item.derivative_flops),
                    metric_value=trustworthy_mode_count(item.npts, metrics, tol=tol),
                )
            )
    costs = np.array([point.derivative_flops for point in points], dtype=float)
    modes = np.array([point.metric_value for point in points], dtype=float)
    mask = pareto_frontier_mask(costs, modes, larger_is_better=True)
    return [point for point, keep in zip(points, mask) if keep]


def frontier_method_counts(points: list[FrontierPoint]) -> dict[str, int]:
    counts = {label: 0 for label in UNIQUE_METHODS}
    for point in points:
        counts[point.label] += 1
    return counts


def frontier_method_min_cost(points: list[FrontierPoint]) -> dict[str, float]:
    min_cost = {label: float("inf") for label in UNIQUE_METHODS}
    for point in points:
        min_cost[point.label] = min(min_cost[point.label], point.derivative_flops)
    return min_cost


def frontier_series(
    points: list[FrontierPoint],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(points, key=lambda point: (point.derivative_flops, point.metric_value))
    costs = np.array([point.derivative_flops for point in ordered], dtype=float)
    metrics = np.array([point.metric_value for point in ordered], dtype=float)
    return costs, metrics


def lower_envelope(
    costs: np.ndarray,
    values: np.ndarray,
    *,
    larger_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mask = pareto_frontier_mask(costs, values, larger_is_better=larger_is_better)
    env_costs = costs[mask]
    env_values = values[mask]
    order = np.argsort(env_costs)
    return env_costs[order], env_values[order]


def select_detail_labels(
    actual_frontier: list[FrontierPoint],
    resolution_frontier: list[FrontierPoint],
) -> tuple[str, ...]:
    counts = frontier_method_counts(actual_frontier)
    resolution_counts = frontier_method_counts(resolution_frontier)
    min_cost = frontier_method_min_cost(actual_frontier + resolution_frontier)
    scores = {
        label: counts[label] + resolution_counts[label]
        for label in UNIQUE_METHODS
    }

    selected: list[str] = []
    for family in ("explicit", "compact", "spline"):
        family_members = [label for label in UNIQUE_METHODS if family_of(label) == family]
        ranked = sorted(
            family_members,
            key=lambda label: (
                scores[label],
                -min_cost[label],
            ),
            reverse=True,
        )
        if ranked and scores[ranked[0]] > 0:
            selected.append(ranked[0])

    if len(selected) < 3:
        extras = sorted(
            UNIQUE_METHODS,
            key=lambda label: (
                scores[label],
                -min_cost[label],
            ),
            reverse=True,
        )
        for label in extras:
            if label not in selected and scores[label] > 0:
                selected.append(label)
            if len(selected) >= 3:
                break

    return tuple(selected)


def plot_pde_convergence(
    convergence_results: dict[str, list[SpatialAccuracyResult]],
    detail_labels: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(8.9, 5.3))
    all_errors = [
        item.error
        for label in detail_labels
        for item in convergence_results[label]
        if item.error > 0.0
    ]
    error_lookup: dict[str, dict[int, float]] = {}

    for label in detail_labels:
        data = convergence_results[label]
        npts = np.array([item.npts for item in data], dtype=float)
        errors = np.array([item.error for item in data], dtype=float)
        error_lookup[label] = {item.npts: item.error for item in data}
        ax.loglog(
            npts,
            errors,
            marker="o",
            ms=4.3,
            color=COLORS[label],
            label=label,
            **LINESTYLES[label],
        )

    ax.set_xlabel(r"Grid points $N$")
    ax.set_ylabel(rf"Relative $L^2$ error (exact semi-discrete, $T={GAUSSIAN_CONVERGENCE_TIME:.0f}L$)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(28, 1400)
    ax.set_ylim(
        *padded_positive_limits(
            all_errors,
            lower_scale=0.18,
            upper_scale=1.55,
            floor=1.0e-12,
            ceiling=3e-1,
        )
    )

    if _add_theoretical_reference_line is not None:
        _add_theoretical_reference_line(
            ax, slope=-6, label=r"$\propto N^{-6}$",
        )

    handles, labels = ax.get_legend_handles_labels()
    if _make_overhead_legend is not None:
        _make_overhead_legend(fig, handles, labels, ncol=len(detail_labels))
    else:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.985),
            ncol=len(detail_labels), frameon=False, fontsize=9.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGDIR / "fig_pde_convergence.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_pde_waveform_snapshot(
    detail_labels: tuple[str, ...],
) -> None:
    length = 1.0
    npts = 48
    time_final = 800.0
    x = grid_points(npts, length)
    exact = gaussian_pulse((x - time_final) % length, length=length)

    fig, (ax_curve, ax_dev) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11.0, 6.4),
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )
    ax_curve.plot(
        x / length,
        exact,
        color=COLORS["Exact"],
        lw=1.5,
        ls="--",
        label="Exact",
    )
    ax_dev.axhline(0.0, color=COLORS["Exact"], lw=1.0, ls="--", alpha=0.85)

    deviation_max = 0.0
    for label in detail_labels:
        _, u_num = run_waveform_case(
            label,
            npts,
            time_final=time_final,
            length=length,
        )
        ax_curve.plot(
            x / length,
            u_num,
            color=COLORS[label],
            label=label,
            **LINESTYLES[label],
        )
        deviation = u_num - exact
        deviation_max = max(deviation_max, float(np.max(np.abs(deviation))))
        ax_dev.plot(
            x / length,
            deviation,
            color=COLORS[label],
            label=label,
            **LINESTYLES[label],
        )

    ax_curve.set_ylabel(r"$u(x,T)$")
    ax_curve.set_xlim(0.05, 0.95)
    ax_curve.set_ylim(-0.06, 1.06)
    ax_curve.grid(True, alpha=0.3)
    legend_cols = 2 if len(detail_labels) <= 3 else 3
    ax_curve.legend(
        loc="upper left",
        framealpha=0.94,
        ncol=legend_cols,
        fontsize=9.4,
    )

    ax_dev.set_xlabel(r"$x/L$")
    ax_dev.set_ylabel(r"$u_{\mathrm{num}}-u_{\mathrm{exact}}$")
    ax_dev.set_ylim(-1.15 * deviation_max, 1.15 * deviation_max)
    ax_dev.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.10, hspace=0.10)
    out = FIGDIR / "fig_pde_snapshot.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_phase_stress_test(
    w: np.ndarray,
    curves: dict[str, np.ndarray],
    detail_labels: tuple[str, ...],
    *,
    ppw_target: float = PHASE_STRESS_PPW,
    travel_wavelengths: float = PHASE_STRESS_DISTANCE_WAVELENGTHS,
) -> None:
    packet_mode = 88
    packet_npts = int(round(packet_mode * ppw_target))
    packet_sigma_env = 1.0 / 12.0
    packet_time = travel_wavelengths / packet_mode
    x_packet = grid_points(packet_npts, 1.0)

    exact_packet = gaussian_carrier_pulse(
        (x_packet - packet_time) % 1.0,
        mode=packet_mode,
        length=1.0,
        sigma_env=packet_sigma_env,
    )
    exact_env = np.abs(hilbert(exact_packet))

    fig, (ax_env, ax_slip) = plt.subplots(
        2,
        1,
        figsize=(11.2, 6.65),
        gridspec_kw={"height_ratios": [1.75, 1.18]},
    )

    ax_env.plot(
        x_packet,
        exact_env,
        color=COLORS["Exact"],
        lw=1.5,
        ls="--",
        label="Exact envelope",
    )
    peak_positions = [float(x_packet[np.argmax(exact_env)])]

    handles = [Line2D([0], [0], color=COLORS["Exact"], lw=1.5, ls="--", label="Exact envelope")]
    slip_cycles: dict[str, float] = {}

    for label in detail_labels:
        _, u_num = run_carrier_case(
            label,
            packet_npts,
            mode=packet_mode,
            time_final=packet_time,
            length=1.0,
            sigma_env=packet_sigma_env,
        )
        env = np.abs(hilbert(u_num))
        peak_x = float(x_packet[np.argmax(env)])
        peak_positions.append(peak_x)
        ax_env.plot(
            x_packet,
            env,
            color=section7_color(label),
            lw=2.2,
            label=label,
            **LINESTYLES[label],
        )
        ax_env.plot([peak_x], [float(env.max())], marker="o", ms=4.4, color=section7_color(label))
        handles.append(
            Line2D([0], [0], color=section7_color(label), lw=2.2, label=label, **LINESTYLES[label])
        )
        eps_phi = phase_error_at_ppw(ppw_target, w, curves[label])
        slip_cycles[label] = travel_wavelengths * eps_phi

    env_xlo = max(0.06, min(peak_positions) - 0.16)
    env_xhi = min(0.96, max(peak_positions) + 0.16)
    ax_env.set_xlim(env_xlo, env_xhi)
    ax_env.set_ylim(-0.02, 1.05)
    ax_env.set_ylabel("Envelope amplitude")
    ax_env.set_xlabel(r"$x/L$")
    ax_env.grid(True, alpha=0.3)
    ax_env.text(
        0.01,
        0.98,
        "(a)",
        transform=ax_env.transAxes,
        fontsize=10.2,
        fontweight="bold",
        color="#222222",
        ha="left",
        va="top",
    )

    distance = np.linspace(0.0, travel_wavelengths, 240)
    y_focus_max = 1.0
    ax_slip.plot(distance, np.zeros_like(distance), color=COLORS["Exact"], lw=1.5, ls="--")
    for label in detail_labels:
        eps_phi = phase_error_at_ppw(ppw_target, w, curves[label])
        slip_curve = distance * eps_phi
        ax_slip.plot(
            distance,
            np.minimum(slip_curve, y_focus_max),
            color=section7_color(label),
            lw=2.3,
            **LINESTYLES[label],
        )
        if slip_cycles[label] <= y_focus_max:
            ax_slip.plot(
                [travel_wavelengths],
                [slip_cycles[label]],
                marker="o",
                ms=5.2,
                color=section7_color(label),
            )
        else:
            ax_slip.plot(
                [travel_wavelengths],
                [y_focus_max],
                marker="^",
                ms=6.2,
                color=section7_color(label),
                clip_on=False,
            )
    ax_slip.axhline(
        PHASE_SLIP_TOLERANCE_CYCLES,
        color="#666666",
        lw=1.1,
        ls=":",
    )
    ax_slip.set_xlabel(r"Travel distance $D/\lambda$")
    ax_slip.set_ylabel(r"Phase slip $\Delta_\lambda$")
    ax_slip.grid(True, alpha=0.3)
    ax_slip.set_xlim(0.0, travel_wavelengths)
    ax_slip.set_ylim(0.0, 1.02)
    ax_slip.text(
        0.01,
        0.98,
        "(b)",
        transform=ax_slip.transAxes,
        fontsize=10.2,
        fontweight="bold",
        color="#222222",
        ha="left",
        va="top",
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(handles), 4),
        framealpha=0.95,
        fontsize=9.2,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.12, hspace=0.14)
    out = FIGDIR / "fig_phase_stress.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_phase_budget(
    w: np.ndarray,
    curves: dict[str, np.ndarray],
    detail_labels: tuple[str, ...],
    *,
    tol_cycles: float = PHASE_SLIP_TOLERANCE_CYCLES,
    ppw_window: tuple[float, float] = (3.2, 9.0),
    mission_window: tuple[float, float] = (8.0, 4000.0),
) -> None:
    ppw_all = 2.0 * np.pi / np.maximum(w, 1.0e-14)
    order = np.argsort(ppw_all)
    ppw = ppw_all[order]
    mask = (ppw >= ppw_window[0]) & (ppw <= ppw_window[1])
    ppw_grid = np.linspace(ppw_window[0], ppw_window[1], 220)
    omega_grid = omega_from_ppw(ppw_grid)
    mission_grid = np.geomspace(mission_window[0], mission_window[1], 220)

    fig, (ax_limit, ax_map) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.95),
        gridspec_kw={"wspace": 0.28},
    )

    phase_error_grid: dict[str, np.ndarray] = {}
    cfl_map: dict[str, float] = {}
    for label in detail_labels:
        limit = coherent_distance_limit_curve(w, curves[label], tol_cycles=tol_cycles)[order][mask]
        ax_limit.plot(
            ppw[mask],
            limit,
            color=COLORS[label],
            lw=2.35,
            label=label,
            **LINESTYLES[label],
        )
        kph_interp = np.interp(omega_grid, w, curves[label])
        phase_error_grid[label] = np.abs(kph_interp / omega_grid - 1.0)
        cfl_map[label] = sa.rk4_cfl_limit(curves[label])

    decision = np.full((mission_grid.size, ppw_grid.size), -1, dtype=int)
    for i, mission in enumerate(mission_grid):
        for j, ppw_value in enumerate(ppw_grid):
            candidates: list[tuple[float, int]] = []
            for idx, label in enumerate(detail_labels):
                if mission * phase_error_grid[label][j] <= tol_cycles:
                    mission_work = NATIVE_WORK_UNITS[label] * ppw_value**2 / cfl_map[label]
                    candidates.append((mission_work, idx))
            if candidates:
                decision[i, j] = min(candidates)[1]

    ax_limit.axvline(PHASE_STRESS_PPW, color="#BBBBBB", lw=1.0, ls=":")
    ax_limit.axhline(PHASE_STRESS_DISTANCE_WAVELENGTHS, color="#BBBBBB", lw=1.0, ls=":")
    ax_limit.set_xlabel("Points per wavelength $P$")
    ax_limit.set_ylabel(r"Mission length before $\Delta_\lambda=0.25$ [$D/\lambda$]")
    ax_limit.set_title("(a) Quarter-wave coherent-distance budget")
    ax_limit.set_yscale("log")
    ax_limit.grid(True, alpha=0.3, which="both")

    ax_map.set_facecolor("#DDDDDD")
    cmap = ListedColormap([COLORS[label] for label in detail_labels])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ppw_mesh, mission_mesh = np.meshgrid(ppw_grid, mission_grid)
    masked = np.ma.masked_where(decision < 0, decision)
    ax_map.pcolormesh(
        ppw_mesh,
        mission_mesh,
        masked,
        cmap=cmap,
        norm=norm,
        shading="nearest",
    )
    ax_map.set_yscale("log")
    ax_map.set_xlabel("Points per wavelength $P$")
    ax_map.set_ylabel(r"Mission length $M$ [carrier wavelengths]")
    ax_map.set_title("(b) Fixed-mesh decision map")
    ax_map.grid(False)
    ax_map.plot(
        [PHASE_STRESS_PPW],
        [PHASE_STRESS_DISTANCE_WAVELENGTHS],
        marker="o",
        ms=5.6,
        mfc="white",
        mec="#111111",
        mew=1.0,
        linestyle="none",
    )
    legend_handles = [Patch(facecolor=COLORS[label], label=label) for label in detail_labels]
    legend_handles.append(Patch(facecolor="#DDDDDD", label="no sixth-order scheme"))

    ax_limit.set_xlim(*ppw_window)
    ax_map.set_xlim(*ppw_window)
    ax_map.set_ylim(*mission_window)

    line_handles, line_labels = ax_limit.get_legend_handles_labels()
    if _make_overhead_legend is not None:
        _make_overhead_legend(
            fig, [*line_handles, legend_handles[-1]], [*line_labels, "no sixth-order scheme"],
            ncol=4, bbox=(0.5, 1.01), rect=(0, 0, 1, 0.86),
        )
    else:
        fig.legend(
            handles=[*line_handles, legend_handles[-1]],
            labels=[*line_labels, "no sixth-order scheme"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            frameon=False,
            fontsize=9.0,
        )
        fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.14, wspace=0.28)
    out = FIGDIR / "fig_phase_budget.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_dimensional_payoff(
    w: np.ndarray,
    curves: dict[str, np.ndarray],
    detail_labels: tuple[str, ...],
    *,
    tol_cycles: float = PHASE_SLIP_TOLERANCE_CYCLES,
    mission_window: tuple[float, float] = (10.0, 3200.0),
    ppw_search_window: tuple[float, float] = (2.8, 16.0),
) -> None:
    compare_labels = tuple(label for label in detail_labels if label in ("E6", "C6", "BMC5 = BMG2"))
    mission_lengths = np.geomspace(mission_window[0], mission_window[1], 260)
    ppw_grid = np.linspace(ppw_search_window[0], ppw_search_window[1], 3200)
    omega_grid = omega_from_ppw(ppw_grid)

    cfl_map = {label: sa.rk4_cfl_limit(curves[label]) for label in compare_labels}
    phase_error_grid = {
        label: np.abs(np.interp(omega_grid, w, curves[label]) / omega_grid - 1.0)
        for label in compare_labels
    }
    pmin = {
        label: minimum_ppw_curve(ppw_grid, phase_error_grid[label], mission_lengths, tol_cycles=tol_cycles)
        for label in compare_labels
    }

    fig, (ax_req, ax_work, ax_mem) = plt.subplots(
        1,
        3,
        figsize=(14.7, 4.8),
        gridspec_kw={"wspace": 0.29},
    )

    for label in compare_labels:
        ax_req.plot(
            mission_lengths,
            pmin[label],
            color=COLORS[label],
            lw=2.35,
            label=label,
            **LINESTYLES[label],
        )
    ax_req.set_xscale("log")
    ax_req.set_yscale("log")
    ax_req.grid(True, alpha=0.3, which="both")
    ax_req.set_xlabel(r"Mission length $M$ [carrier wavelengths]")
    ax_req.set_ylabel("Minimum points per wavelength $P$")
    ax_req.set_title("(a) Required PPW vs mission length")
    ax_req.legend(loc="upper left", framealpha=0.94, fontsize=8.8)

    work_crossings: dict[int, float | None] = {}
    for dim, ls in ((1, "-"), (2, "--"), (3, ":")):
        ratio = (
            (NATIVE_WORK_UNITS["BMC5 = BMG2"] / cfl_map["BMC5 = BMG2"])
            * pmin["BMC5 = BMG2"] ** (dim + 1)
            / ((NATIVE_WORK_UNITS["C6"] / cfl_map["C6"]) * pmin["C6"] ** (dim + 1))
        )
        work_crossings[dim] = first_threshold_crossing(mission_lengths, ratio, level=1.0)
        ax_work.plot(
            mission_lengths,
            ratio,
            color=COLORS["BMC5 = BMG2"],
            lw=2.35,
            ls=ls,
            label=rf"BMC5 / C6, $d={dim}$",
        )
    ax_work.axhline(1.0, color="#666666", lw=1.0, ls=":")
    ax_work.set_xscale("log")
    ax_work.grid(True, alpha=0.3, which="both")
    ax_work.set_xlabel(r"Mission length $M$ [carrier wavelengths]")
    ax_work.set_ylabel("Recurring work / C6")
    ax_work.set_title("(b) Recurring work, normalized by C6")
    for dim in (2, 3):
        crossing = work_crossings[dim]
        if crossing is not None:
            ax_work.axvline(crossing, color="#BBBBBB", lw=0.9, ls=":")
    ax_work.legend(loc="upper right", framealpha=0.94, fontsize=8.5)

    for dim, ls in ((1, "-"), (2, "--"), (3, ":")):
        mem_ratio = (pmin["BMC5 = BMG2"] / pmin["C6"]) ** dim
        ax_mem.plot(
            mission_lengths,
            mem_ratio,
            color=COLORS["BMC5 = BMG2"],
            lw=2.35,
            ls=ls,
            label=rf"$d={dim}$",
        )
    ax_mem.axhline(1.0, color="#666666", lw=1.0, ls=":")
    ax_mem.set_xscale("log")
    ax_mem.grid(True, alpha=0.3, which="both")
    ax_mem.set_xlabel(r"Mission length $M$ [carrier wavelengths]")
    ax_mem.set_ylabel("State size / C6")
    ax_mem.set_title("(c) Memory scaling")
    ax_mem.legend(loc="upper right", framealpha=0.94, fontsize=8.5)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.85, bottom=0.16, wspace=0.29)
    out = FIGDIR / "fig_dimensional_payoff.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_mission_design(
    w: np.ndarray,
    curves: dict[str, np.ndarray],
    detail_labels: tuple[str, ...],
    *,
    tol_cycles: float = PHASE_SLIP_TOLERANCE_CYCLES,
    mission_window: tuple[float, float] = (10.0, 3200.0),
    ppw_budget_window: tuple[float, float] = (3.0, 12.0),
    ppw_search_window: tuple[float, float] = (2.8, 16.0),
) -> None:
    compare_labels = tuple(label for label in detail_labels if label in ("E6", "C6", "BMC5 = BMG2"))
    mission_lengths = np.geomspace(mission_window[0], mission_window[1], 260)
    ppw_budget = np.linspace(ppw_budget_window[0], ppw_budget_window[1], 260)
    ppw_grid = np.linspace(ppw_search_window[0], ppw_search_window[1], 3200)
    omega_grid = omega_from_ppw(ppw_grid)

    cfl_map = {label: sa.rk4_cfl_limit(curves[label]) for label in compare_labels}
    phase_error_grid = {
        label: np.abs(np.interp(omega_grid, w, curves[label]) / omega_grid - 1.0)
        for label in compare_labels
    }
    pmin = {
        label: minimum_ppw_curve(ppw_grid, phase_error_grid[label], mission_lengths, tol_cycles=tol_cycles)
        for label in compare_labels
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.1, 5.05), gridspec_kw={"wspace": 0.24})
    cmap = ListedColormap([section7_color(label) for label in compare_labels])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for ax, dim in zip(axes, (1, 2, 3)):
        decision = np.full((ppw_budget.size, mission_lengths.size), -1, dtype=int)
        for i, p_cap in enumerate(ppw_budget):
            for j, mission in enumerate(mission_lengths):
                candidates: list[tuple[float, int]] = []
                for idx, label in enumerate(compare_labels):
                    req = pmin[label][j]
                    if np.isfinite(req) and req <= p_cap:
                        recurring_work = (NATIVE_WORK_UNITS[label] / cfl_map[label]) * req ** (dim + 1)
                        candidates.append((recurring_work, idx))
                if candidates:
                    decision[i, j] = min(candidates)[1]

        ax.set_facecolor("#DDDDDD")
        mission_mesh, budget_mesh = np.meshgrid(mission_lengths, ppw_budget)
        masked = np.ma.masked_where(decision < 0, decision)
        ax.pcolormesh(
            mission_mesh,
            budget_mesh,
            masked,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
        ax.set_xscale("log")
        ax.set_xlim(*mission_window)
        ax.set_ylim(*ppw_budget_window)
        ax.grid(False)
        ax.set_title(rf"$d={dim}$", pad=8.0)
        ax.set_xlabel(r"Mission length $M$ [carrier wavelengths]")
        if dim == 1:
            ax.set_ylabel(r"Available PPW budget $P_{\max}$")

    legend_handles = [Patch(facecolor=section7_color(label), label=label) for label in compare_labels]
    legend_handles.append(Patch(facecolor="#DDDDDD", label="no sixth-order scheme"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        framealpha=0.95,
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.72, bottom=0.17, wspace=0.24)
    out = FIGDIR / "fig_mission_design.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_pde_stress_test(
    regime: CarrierRegimeSpec,
    detail_labels: tuple[str, ...],
) -> None:
    length = 1.0
    npts = 256
    time_final = regime.time_final
    mode = regime.mode
    sigma_env = carrier_sigma_env(regime, length)
    ppw = npts / mode
    x = grid_points(npts, length)
    exact = gaussian_carrier_pulse(
        (x - time_final) % length,
        mode=mode,
        length=length,
        sigma_env=sigma_env,
    )
    exact_strength = local_packet_strength(exact, ppw)

    fig, (ax, ax_strength) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11.2, 6.4),
        gridspec_kw={"height_ratios": [2.1, 1.2]},
    )

    cached_curves = {"Exact": exact}
    cached_strengths = {"Exact": exact_strength}
    x_scaled = x / length
    window = (x_scaled >= 0.30) & (x_scaled <= 0.70)

    ax.plot(
        x_scaled[window],
        exact[window],
        color=COLORS["Exact"],
        lw=1.4,
        ls="--",
        label="Exact",
    )
    ax_strength.plot(
        x_scaled[window],
        exact_strength[window],
        color=COLORS["Exact"],
        lw=1.4,
        ls="--",
        label="Exact",
    )

    for label in detail_labels:
        _, u_num = run_carrier_case(
            label,
            npts,
            mode=mode,
            time_final=time_final,
            length=length,
            sigma_env=sigma_env,
        )
        strength = local_packet_strength(u_num, ppw)
        cached_curves[label] = u_num
        cached_strengths[label] = strength
        ax.plot(
            x_scaled[window],
            u_num[window],
            color=COLORS[label],
            label=label,
            **LINESTYLES[label],
        )
        ax_strength.plot(
            x_scaled[window],
            strength[window],
            color=COLORS[label],
            label=label,
            **LINESTYLES[label],
        )

    ax.set_ylabel(r"$u(x,T)$")
    ax.grid(True, alpha=0.3)

    ax_strength.set_xlabel(r"$x/L$")
    ax_strength.set_ylabel("Local packet strength")
    ax_strength.grid(True, alpha=0.3)
    ax_strength.text(
        0.02,
        0.92,
        "One-wavelength moving RMS",
        transform=ax_strength.transAxes,
        fontsize=8.7,
        color="#444444",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="round,pad=0.24"),
    )

    signal_max = max(
        np.max(np.abs(curve[window]))
        for curve in cached_curves.values()
    )
    strength_max = max(
        np.max(strength[window])
        for strength in cached_strengths.values()
    )
    ax.set_xlim(0.30, 0.70)
    ax.set_ylim(-1.05 * signal_max, 1.05 * signal_max)
    ax_strength.set_ylim(-0.02, 1.05 * strength_max)

    handles = [Line2D([0], [0], color=COLORS["Exact"], lw=1.4, ls="--", label="Exact")]
    for label in detail_labels:
        handles.append(
            Line2D([0], [0], color=COLORS[label], label=label, **LINESTYLES[label])
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(handles), 4),
        framealpha=0.95,
        fontsize=9.4,
    )

    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.12, hspace=0.08)
    out = FIGDIR / "fig_pde_stress.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_shortwave_runtime_pareto(
    error_results: dict[str, list[SpatialAccuracyResult]],
    regime: CarrierRegimeSpec,
    labels: tuple[str, ...],
) -> None:
    fig, (ax_n, ax_cost) = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.0),
        gridspec_kw={"wspace": 0.30},
    )
    all_npts: list[float] = []
    all_costs: list[float] = []
    all_errors: list[float] = []

    for label in labels:
        data = sorted(error_results[label], key=lambda item: item.npts)
        npts = np.array([float(item.npts) for item in data], dtype=float)
        costs = np.array([float(item.derivative_flops) for item in data], dtype=float)
        errors = np.array([item.error for item in data], dtype=float)

        all_npts.extend(npts.tolist())
        all_costs.extend(costs.tolist())
        all_errors.extend(errors.tolist())

        ax_n.loglog(
            npts,
            errors,
            marker="o",
            ms=5.0,
            color=COLORS[label],
            label=label,
            lw=2.0,
            alpha=0.96,
            **LINESTYLES[label],
        )

        ax_cost.loglog(
            costs,
            errors,
            marker="o",
            ms=5.2,
            mfc="white",
            mec=COLORS[label],
            mew=1.2,
            color=COLORS[label],
            alpha=0.25,
            lw=0.9,
            label=None,
            zorder=1,
            **LINESTYLES[label],
        )
        env_costs, env_errors = lower_envelope(costs, errors, larger_is_better=False)
        ax_cost.loglog(
            env_costs,
            env_errors,
            marker="o",
            ms=5.0,
            color=COLORS[label],
            label=label,
            lw=2.1,
            alpha=0.98,
            zorder=2,
            **LINESTYLES[label],
        )

    title_suffix = (
        rf"mode $m={regime.mode}$, $\sigma=L/{regime.sigma_env_scale:.0f}$, "
        rf"$T={regime.time_final:.1f}L$"
    )

    ax_n.set_xlabel(r"Grid points $N$")
    ax_n.set_ylabel(r"Relative $L^2$ error (exact semi-discrete)")
    ax_n.set_title("(a) Error vs grid points")
    ax_n.grid(True, alpha=0.3, which="both")
    add_better_arrow(
        ax_n,
        xy=(0.30, 0.18),
        xytext=(0.55, 0.50),
    )
    ax_n.legend(loc="lower left", framealpha=0.92, fontsize=9.2)
    ax_n.set_xlim(
        *padded_positive_limits(
            all_npts,
            lower_scale=0.85,
            upper_scale=1.18,
            floor=1.0,
        )
    )
    ax_n.set_ylim(
        *padded_positive_limits(
            all_errors,
            lower_scale=0.55,
            upper_scale=2.20,
            floor=5e-6,
            ceiling=4.0,
        )
    )

    ax_cost.set_xlabel(r"Derivative cost $F_D(N)=N\,c_{\mathrm{D}}$ [flops]")
    ax_cost.set_ylabel(r"Relative $L^2$ error (exact semi-discrete)")
    ax_cost.set_title("(b) Error vs derivative-cost work")
    ax_cost.grid(True, alpha=0.3, which="both")
    add_better_arrow(
        ax_cost,
        xy=(0.30, 0.18),
        xytext=(0.55, 0.50),
    )
    ax_cost.text(
        0.03,
        0.05,
        "hollow markers: all tested grids\nsolid polyline: per-method Pareto envelope",
        transform=ax_cost.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#333333",
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.9, boxstyle="round,pad=0.25"),
    )
    ax_cost.legend(loc="upper right", framealpha=0.92, fontsize=9.2)
    ax_cost.set_xlim(
        *padded_positive_limits(
            all_costs,
            lower_scale=0.72,
            upper_scale=1.55,
            floor=1.0e2,
        )
    )
    ax_cost.set_ylim(
        *padded_positive_limits(
            all_errors,
            lower_scale=0.55,
            upper_scale=2.20,
            floor=5e-6,
            ceiling=4.0,
        )
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.13, wspace=0.30)
    out = FIGDIR / "fig_shortwave_pareto.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_runtime_scaling(
    runtime_results: dict[str, list[MarchResult]],
    detail_labels: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    all_times = [
        item.wall_time
        for label in detail_labels
        for item in runtime_results[label]
        if item.npts >= 64 and item.wall_time is not None and item.wall_time > 0.0
    ]

    for label in detail_labels:
        data = [item for item in runtime_results[label] if item.npts >= 64]
        npts = np.array([item.npts for item in data], dtype=float)
        wall_time = np.array([item.wall_time for item in data], dtype=float)
        ax.loglog(
            npts,
            wall_time,
            marker="o",
            ms=4.0,
            color=COLORS[label],
            label=label,
            **LINESTYLES[label],
        )

    common_npts = sorted(
        set.intersection(
            *[
                {
                    int(item.npts)
                    for item in runtime_results[label]
                    if item.npts >= 64 and item.wall_time is not None and item.wall_time > 0.0
                }
                for label in detail_labels
            ]
        )
    )

    x_ref = 420.0
    y_ref = min(all_times) * 4.4
    x0 = x_ref / 2.0
    y0 = y_ref / (x_ref / x0) ** 2
    ax.plot([x0, x_ref], [y0, y0], color="#444444", lw=1.0)
    ax.plot([x_ref, x_ref], [y0, y_ref], color="#444444", lw=1.0)
    ax.plot([x0, x_ref], [y0, y_ref], color="#444444", lw=1.2)
    ax.text(
        x_ref * 1.04,
        y_ref,
        r"slope $2$",
        fontsize=8.6,
        color="#444444",
        va="bottom",
    )
    ax.set_xlabel(r"Grid points $N$")
    ax.set_ylabel(r"Measured marching time for $T=L$ (s)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9.2)
    ax.set_xlim(56, 4600)
    ax.set_ylim(
        *padded_positive_limits(
            all_times,
            lower_scale=0.85,
            upper_scale=1.25,
            floor=7e-4,
            ceiling=3.5,
        )
    )
    fig.tight_layout()
    out = FIGDIR / "fig_runtime_scaling.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_runtime_pareto(
    convergence_results: dict[str, list[SpatialAccuracyResult]],
    detail_labels: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    frontier_points = collect_actual_frontier_points(convergence_results)
    all_costs = []
    all_errors = []

    for label in UNIQUE_METHODS:
        data = convergence_results[label]
        costs = np.array([float(item.derivative_flops) for item in data], dtype=float)
        errors = np.array([item.error for item in data], dtype=float)
        order = np.argsort(costs)
        costs = costs[order]
        errors = errors[order]
        all_costs.extend(costs.tolist())
        all_errors.extend(errors.tolist())
        ax.scatter(
            costs,
            errors,
            s=16,
            color="#BFC7D2",
            alpha=0.48,
            edgecolors="none",
            zorder=1,
        )

    frontier_costs, frontier_errors = frontier_series(frontier_points)
    ax.loglog(
        frontier_costs,
        frontier_errors,
        color="#222222",
        lw=1.25,
        ls=(0, (3.0, 2.2)),
        marker="o",
        ms=3.6,
        mfc="white",
        mec="#222222",
        zorder=2,
    )
    for label in detail_labels:
        data = convergence_results[label]
        costs = np.array([float(item.derivative_flops) for item in data], dtype=float)
        errors = np.array([item.error for item in data], dtype=float)
        order = np.argsort(costs)
        ax.loglog(
            costs[order],
            errors[order],
            marker="o",
            ms=4.6,
            color=COLORS[label],
            lw=2.0,
            alpha=0.98,
            label=label,
            zorder=3,
            **LINESTYLES[label],
        )

    ax.set_xlabel(r"Derivative cost $F_D(N)$ [flops]")
    ax.set_ylabel(r"Relative $L^2$ error (exact semi-discrete)")
    ax.grid(True, alpha=0.3, which="both")
    add_better_arrow(
        ax,
        xy=(0.25, 0.30),
        xytext=(0.46, 0.54),
    )
    all_handle = Line2D([0], [0], marker="o", color="none", markerfacecolor="#BFC7D2",
                        markeredgecolor="none", markersize=6.5, alpha=0.85)
    frontier_handle = Line2D([0], [0], color="#222222", lw=1.25, ls=(0, (3.0, 2.2)),
                             marker="o", markerfacecolor="white", markeredgecolor="#222222",
                             markersize=4.5)
    rep_handles = [
        Line2D([0], [0], color=COLORS[label], lw=2.0, marker="o", markersize=5.0)
        for label in detail_labels
    ]
    ax.legend(
        [all_handle, frontier_handle, *rep_handles],
        ["all screened methods", "global frontier", *detail_labels],
        loc="upper right",
        framealpha=0.92,
        fontsize=8.5,
        title="Screening view",
        title_fontsize=8.8,
    )
    ax.set_xlim(
        *padded_positive_limits(
            all_costs,
            lower_scale=0.72,
            upper_scale=1.55,
            floor=1.0e2,
            ceiling=1.0e10,
        )
    )
    ax.set_ylim(
        *padded_positive_limits(
            all_errors,
            lower_scale=0.72,
            upper_scale=1.65,
            floor=1.5e-6,
            ceiling=2e-1,
        )
    )
    fig.tight_layout()
    out = FIGDIR / "fig_runtime_pareto.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def print_identity_checks(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    checks = [
        ("C4 vs BMC3", all_curves["C4"], all_curves["BMC3"]),
        ("C4 vs BMG1", all_curves["C4"], all_curves["IGA1"]),
        ("BMC5 vs BMG2", all_curves["BMC5"], all_curves["IGA2"]),
        ("BMC7 vs BMG3", all_curves["BMC7"], all_curves["IGA3"]),
        ("BMC9 vs BMG4", all_curves["BMC9"], all_curves["IGA4"]),
        ("BMC11 vs BMG5", all_curves["BMC11"], all_curves["IGA5"]),
    ]

    print("Exact operator identities on the periodic uniform grid:")
    tol = 1e-10
    failed = []
    for label, left, right in checks:
        diff = np.max(np.abs(left - right))
        marker = "ok" if diff < tol else "FAIL"
        print(f"  {label:14s}  max diff = {diff:.2e}  [{marker}]")
        if diff >= tol:
            failed.append((label, diff))
    if failed:
        details = ", ".join(f"{lbl} ({d:.2e})" for lbl, d in failed)
        raise AssertionError(
            "BMG equals BMC degree-(2p+1) identity violated: " + details
        )

    print("\nBMG entry verification against direct quadrature:")
    for degree in [1, 2, 3, 4, 5]:
        mass_exact, adv_exact = sa.iga_mass_and_advection_entries(degree)
        mass_quad, adv_quad = sa.quadrature_iga_entries(degree)
        mass_err = np.max(np.abs(mass_exact - mass_quad))
        adv_err = np.max(np.abs(adv_exact - adv_quad))
        print(
            f"  BMG p={degree:<2d}  mass err = {mass_err:.2e}   "
            f"advection err = {adv_err:.2e}"
        )


def print_summary_table(w: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    print("\nUnique operator summary:")
    header = "{:20s} {:>7s} {:>10s} {:>10s} {:>10s}".format(
        "Method", "order", "max|k'h|", "RK4 CFL", "PPW@1%"
    )
    print("  " + header)
    print("  " + "-" * 62)
    for name, kph in curves.items():
        if name == "Exact":
            continue
        metrics = sa.summarize_curve(w, kph)
        print(
            f"  {name:20s} {FORMAL_ORDER[name]:7d} "
            f"{metrics.spectral_radius:10.4f} {metrics.rk4_cfl:10.4f} "
            f"{metrics.ppw_1pct:10.2f}"
        )


def print_core_metric_table(w: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    print("\nCore metric table for the unique operators:")
    header = "{:20s} {:>5s} {:>5s} {:>6s} {:>8s} {:>9s} {:>9s}".format(
        "Method", "rad.", "work", "order", "RK4 CFL", "PPW@1%", "PPW@5%"
    )
    print("  " + header)
    print("  " + "-" * 76)
    for name, kph in curves.items():
        if name == "Exact":
            continue
        metrics = sa.summarize_curve(w, kph)
        print(
            f"  {name:20s} {HALO_WIDTH[name]:5d} {NATIVE_WORK_UNITS[name]:5d} "
            f"{FORMAL_ORDER[name]:6d} {metrics.rk4_cfl:8.2f} "
            f"{metrics.ppw_1pct:9.2f} {metrics.ppw_5pct:9.2f}"
        )


def print_fixed_halo_table(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    print("\nFixed stencil-radius comparison:")
    header = "{:>4s} {:20s} {:>20s} {:>5s} {:>6s} {:>8s} {:>9s} {:>9s}".format(
        "rad.", "method", "solve class", "work", "order", "RK4 CFL", "PPW@1%", "PPW@5%"
    )
    print("  " + header)
    print("  " + "-" * 109)
    for halo, labels in [(str(idx + 1), labels) for idx, (_, labels) in enumerate(FIXED_RADIUS_GROUPS)]:
        methods = []
        for label in labels:
            source = label
            if label == "C4 = BMC3 = BMG1":
                source = "C4"
            elif label == "BMC5 = BMG2":
                source = "BMC5"
            elif label == "BMC7 = BMG3":
                source = "IGA3"
            elif label == "BMC9 = BMG4":
                source = "IGA4"
            elif label == "BMC11 = BMG5":
                source = "IGA5"
            methods.append((label, all_curves[source]))
        for idx, (name, kph) in enumerate(methods):
            metrics = sa.summarize_curve(w, kph)
            halo_label = halo if idx == 0 else ""
            solve_class = solve_class_from_mass_entries(unique_scheme_entries(name)[0])
            print(
                f"  {halo_label:>4s} {name:20s} {solve_class:>20s} {NATIVE_WORK_UNITS[name]:5d} "
                f"{FORMAL_ORDER[name]:6d} {metrics.rk4_cfl:8.2f} "
                f"{metrics.ppw_1pct:9.2f} {metrics.ppw_5pct:9.2f}"
            )


def print_first_principles_notes() -> None:
    print("\nReading guide:")
    print("  Order: higher is better.")
    print("  max|k'h|: lower is better for explicit time stepping.")
    print("  RK4 CFL: higher is better.")
    print("  PPW@1%: lower is better, because fewer points per wavelength means")
    print("           the method resolves shorter waves accurately on the same grid.")


def print_fixed_order_table(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    groups = [
        ("2", [("E2", all_curves["E2"])]),
        ("4", [("E4", all_curves["E4"]), ("C4 = BMC3 = BMG1", all_curves["C4"])]),
        ("6", [("E6", all_curves["E6"]), ("C6", all_curves["C6"]), ("BMC5 = BMG2", all_curves["BMC5"])]),
        ("8", [("E8", all_curves["E8"]), ("C8", all_curves["C8"]), ("BMC7 = BMG3", all_curves["IGA3"])]),
        ("10", [("E10", all_curves["E10"]), ("BMC9 = BMG4", all_curves["IGA4"])]),
        ("12", [("E12", all_curves["E12"]), ("BMC11 = BMG5", all_curves["IGA5"])]),
    ]

    print("\nFixed-order comparison:")
    print("  order  best CFL          best PPW@1%       methods compared")
    print("  " + "-" * 78)
    for order, methods in groups:
        summary = []
        for name, kph in methods:
            metrics = sa.summarize_curve(w, kph)
            summary.append((name, metrics))
        best_cfl = max(summary, key=lambda item: item[1].rk4_cfl)
        best_ppw = min(summary, key=lambda item: item[1].ppw_1pct)
        compared = ", ".join(f"{name} ({metrics.ppw_1pct:.2f}, {metrics.rk4_cfl:.2f})" for name, metrics in summary)
        print(
            f"  {order:>5s}  {best_cfl[0]:16s}  {best_ppw[0]:16s}  {compared}"
        )


def print_same_degree_table(w: np.ndarray, all_curves: dict[str, np.ndarray]) -> None:
    rows = [
        ("2", "BMC2", all_curves["BMC2"], "BMC5 = BMG2", all_curves["IGA2"]),
        ("3", "BMC3", all_curves["BMC3"], "BMC7 = BMG3", all_curves["IGA3"]),
        ("5", "BMC5", all_curves["BMC5"], "BMC11 = BMG5", all_curves["IGA5"]),
    ]

    print("\nSame-degree BM comparison:")
    print("  p   BMC order/PPW@1%            BMG order/PPW@1%         PPW gain")
    print("  " + "-" * 78)
    for degree, cname, ccurve, iname, icurve in rows:
        cm = sa.summarize_curve(w, ccurve)
        im = sa.summarize_curve(w, icurve)
        gain = cm.ppw_1pct / im.ppw_1pct
        print(
            f"  {degree:>1s}   {cname:8s} {cm.fitted_order:4.1f} / {cm.ppw_1pct:5.2f}      "
            f"{iname:13s} {im.fitted_order:4.1f} / {im.ppw_1pct:5.2f}      "
            f"{gain:5.2f}x"
        )


def print_cost_tables(w: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    print("\nCost proxies for one-wavelength advection over one wavelength:")
    print("  mesh-time proxy      = PPW^2 / CFL")
    print("  native local-work    = work_units * PPW^2 / CFL")
    print("  work units use a simple banded model: 3 per RHS pair, 4 per solve band.")
    print("  ")
    header = "{:20s} {:>5s} {:>6s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Method", "ord", "work", "mesh@1%", "mesh@5%", "local@1%", "local@5%"
    )
    print("  " + header)
    print("  " + "-" * 86)
    for name, kph in curves.items():
        if name == "Exact":
            continue
        metrics = sa.summarize_curve(w, kph)
        work = NATIVE_WORK_UNITS[name]
        mesh1 = metrics.ppw_1pct**2 / metrics.rk4_cfl
        mesh5 = metrics.ppw_5pct**2 / metrics.rk4_cfl
        local1 = work * mesh1
        local5 = work * mesh5
        print(
            f"  {name:20s} {FORMAL_ORDER[name]:5d} {work:6d} "
            f"{mesh1:10.2f} {mesh5:10.2f} {local1:10.2f} {local5:10.2f}"
        )


def print_repeated_flop_table() -> None:
    print("\nSpatial derivative cost (setup excluded):")
    header = "{:20s} {:>7s} {:>9s} {:>12s}".format(
        "Method", "rhs", "solve", "F_D/N"
    )
    print("  " + header)
    print("  " + "-" * 50)
    for label in UNIQUE_METHODS:
        mass_entries, advection_entries = unique_scheme_entries(label)
        cost = repeated_flop_model(mass_entries, advection_entries)
        print(
            f"  {label:20s} {cost.rhs_pairs:7d} {cost.solve_radius:9d} "
            f"{cost.derivative_flops_per_point:12d}"
        )


def print_frontier_summary(
    actual_frontier: list[FrontierPoint],
    resolution_frontier: list[FrontierPoint],
    detail_labels: tuple[str, ...],
) -> None:
    actual_methods = sorted({point.label for point in actual_frontier})
    resolution_methods = sorted({point.label for point in resolution_frontier})
    print("\nFrontier screening summary:")
    print("  Actual-error frontier methods:")
    print("   ", ", ".join(actual_methods) if actual_methods else "(none)")
    print("  Resolution frontier methods:")
    print("   ", ", ".join(resolution_methods) if resolution_methods else "(none)")
    print("  Detailed waveform representatives:")
    print("   ", ", ".join(detail_labels) if detail_labels else "(none)")


def print_bspline_win_audit(metrics_map: dict[str, sa.CurveMetrics]) -> None:
    e6 = metrics_map["E6"]
    c6 = metrics_map["C6"]
    bmc = metrics_map["BMC5 = BMG2"]
    print("\nSixth-order spline audit:")
    print("  Same formal order q=6:")
    print(
        f"    PPW@1%: E6 {e6.ppw_1pct:.2f}, C6 {c6.ppw_1pct:.2f}, BMC5 = BMG2 {bmc.ppw_1pct:.2f}"
    )
    print(
        f"    PPW@5%: E6 {e6.ppw_5pct:.2f}, C6 {c6.ppw_5pct:.2f}, BMC5 = BMG2 {bmc.ppw_5pct:.2f}"
    )
    print("    Interpretation: each extra layer of implicit structure buys better short-wave phase fidelity.")
    print("  Time-step budget:")
    print(
        f"    RK4 CFL: E6 {e6.rk4_cfl:.2f}, C6 {c6.rk4_cfl:.2f}, BMC5 = BMG2 {bmc.rk4_cfl:.2f}"
    )
    print("    Interpretation: explicit E6 keeps the largest step; B-splines pay a modest CFL penalty.")
    mesh_rank = sorted(
        [("E6", e6.ppw_1pct**2 / e6.rk4_cfl), ("C6", c6.ppw_1pct**2 / c6.rk4_cfl), ("BMC5 = BMG2", bmc.ppw_1pct**2 / bmc.rk4_cfl)],
        key=lambda item: item[1],
    )
    native_rank = sorted(
        [
            ("E6", NATIVE_WORK_UNITS["E6"] * e6.ppw_1pct**2 / e6.rk4_cfl),
            ("C6", NATIVE_WORK_UNITS["C6"] * c6.ppw_1pct**2 / c6.rk4_cfl),
            ("BMC5 = BMG2", NATIVE_WORK_UNITS["BMC5 = BMG2"] * bmc.ppw_1pct**2 / bmc.rk4_cfl),
        ],
        key=lambda item: item[1],
    )
    print(
        "  Cost lens at 1% phase error:"
        f" mesh-time best = {mesh_rank[0][0]}, solve-aware best = {native_rank[0][0]}"
    )
    print("    Interpretation: the fair B-spline-forward story is resolution-hungry transport, not blanket cheapest-runtime claims.")


def print_shortwave_candidate_table(
    summaries: list[ShortwaveCandidateSummary],
    selected: ShortwaveCandidateSummary,
) -> None:
    print("\nShort-wave candidate screening:")
    print("  slug                BMC<C6  BMC<E6  geom(C6/BMC)  geom(E6/BMC)  geom(time_BMC/C6)")
    print("  " + "-" * 84)
    for summary in summaries:
        marker = "*" if summary.regime.slug == selected.regime.slug else " "
        print(
            f"{marker} {summary.regime.slug:18s} {summary.bspline_beats_compact:6d} {summary.bspline_beats_explicit:7d} "
            f"{summary.compact_over_bspline_geomean:13.2f} {summary.explicit_over_bspline_geomean:13.2f} "
            f"{summary.bspline_time_over_compact_geomean:18.2f}"
        )
    print(
        f"  Selected headline short-wave regime: {selected.regime.slug}"
        f" ({selected.regime.title.lower()}, {selected.regime.fairness_lens})"
    )


def observed_order(
    results: list[SpatialAccuracyResult]
) -> float:
    if len(results) < 2:
        return float("nan")
    n = np.array([item.npts for item in results], dtype=float)
    e = np.array([item.error for item in results], dtype=float)
    mask = np.isfinite(e) & (e > 0.0)
    if int(mask.sum()) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(n[mask]), np.log(e[mask]), 1)
    return float(-slope)


def latex_float(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1.0e-2 <= magnitude < 1.0e2:
        return f"{value:.{digits}f}"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def plot_phase_race() -> None:
    """Where the spline shines: long-haul wave fidelity at aggressive resolution.
    Panel (a) reports how far each operator carries a single mode before its phase
    error reaches 1%, as a function of points per wavelength. Panel (b) propagates
    a Gaussian wave packet at a fixed 6 points per wavelength and tracks its
    relative L2 error with distance. Both use exact semi-discrete evolution, so the
    spatial operator is the only error source."""
    setup_full_width_print_style()
    import numpy as _np
    phase_styles = {
        key: _METHOD_STYLES.get(key, {"color": COLORS.get(key, "#2F4858"), "ls": "-"})
        for key in ("E6", "C6", "DRP", "BMC5", "BMC7", "BMC9")
    }
    cols = {key: style["color"] for key, style in phase_styles.items()}
    lss = {key: style["ls"] for key, style in phase_styles.items()}
    labels = {"E6": "E6 (order 6)", "C6": "C6 (order 6)", "DRP": "DRP",
              "BMC5": "BMC5 (order 6)", "BMC7": "BMC7 (order 8)",
              "BMC9": "BMC9 (order 10)"}
    order = ["E6", "C6", "DRP", "BMC5", "BMC7", "BMC9"]

    def kp(s, w):
        return {"E6": sa.kprime_centered_fd(w, 6), "C6": sa.kprime_compact(w, 6),
                "DRP": sa.kprime_drp_tam_webb(w), "BMC5": sa.kprime_collocation(w, 5),
                "BMC7": sa.kprime_collocation(w, 7),
                "BMC9": sa.kprime_collocation(w, 9)}[s]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4),
                             gridspec_kw={"wspace": 0.28})

    # Panel (a): distance (in wavelengths) to 1% phase error vs PPW. For a single
    # mode the L2 error after D wavelengths is 2|sin(pi D |k'h/omega - 1|)|.
    ax = axes[0]
    ppw = _np.linspace(3.0, 12.0, 250)
    wv = 2 * _np.pi / ppw
    for s in order:
        d = _np.abs(kp(s, wv) / wv - 1.0)
        D = _np.arcsin(0.005) / (_np.pi * _np.maximum(d, 1e-16))
        ax.semilogy(ppw, D, color=cols[s], ls=lss[s], lw=1.9, label=labels[s])
    # opaque label backgrounds, drawn ABOVE the curves so nothing cuts through text
    _lblbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=1.0)
    ax.axvline(6.0, color="0.5", ls=(0, (2, 2)), lw=0.9)
    ax.text(6.2, 6.0e4, "6 PPW", fontsize=12.0, color="0.45", bbox=_lblbox, zorder=10)
    ax.set_xlabel("points per wavelength")
    ax.set_ylabel("wavelengths carried before 1% phase error")
    ax.set_xlim(3, 12)
    ax.set_ylim(1e-1, 1e6)
    ax.text(0.03, 0.97, "(a)",
            transform=ax.transAxes, va="top", fontsize=14, fontstyle="italic",
            color="#2F4858", bbox=_lblbox, zorder=10)

    # Panel (b): a Gaussian packet at 6 PPW, relative L2 error vs distance.
    ax = axes[1]
    N = 4096
    L = 2 * _np.pi
    h = L / N
    x = _np.arange(N) * h
    ppw0 = 6.0
    k0 = 2 * _np.pi / (ppw0 * h)
    x0 = L / 2
    sig = L * 0.05
    u0 = _np.exp(-((x - x0) / sig) ** 2) * _np.cos(k0 * (x - x0))
    kf = _np.fft.fftfreq(N, d=h) * 2 * _np.pi
    om = _np.abs(kf) * h
    U0 = _np.fft.fft(u0)
    E0 = _np.linalg.norm(u0)
    Dlist = _np.unique(_np.round(_np.logspace(0, 3.5, 70)).astype(int))
    cross = {}
    for s in order:
        r = _np.ones_like(om)
        nz = om > 1e-12
        r[nz] = kp(s, om[nz]) / om[nz]
        keff = kf * r
        errs = []
        for D in Dlist:
            sh = D * (2 * _np.pi / k0)
            un = _np.real(_np.fft.ifft(U0 * _np.exp(-1j * keff * sh)))
            ue = _np.real(_np.fft.ifft(U0 * _np.exp(-1j * kf * sh)))
            errs.append(_np.linalg.norm(un - ue) / E0)
        errs = _np.array(errs)
        ax.loglog(Dlist, errs, color=cols[s], ls=lss[s], lw=1.9)
        below = Dlist[errs < 0.01]
        cross[s] = float(below.max()) if below.size else float("nan")
    ax.axhline(0.01, color="0.4", ls=(0, (4, 2)), lw=1.0)
    ax.text(1.25, 0.013, "1% error", fontsize=12.0, color="0.4")
    ax.set_xlabel("distance traveled (wavelengths)")
    ax.set_ylabel(r"relative $L^2$ error of the packet")
    ax.set_xlim(1, 3000)
    ax.set_ylim(1e-7, 2)
    ax.text(0.03, 0.97, "(b)",
            transform=ax.transAxes, va="top", ha="left", fontsize=14,
            fontstyle="italic", color="#2F4858", bbox=_lblbox, zorder=10)

    handles, labs = axes[0].get_legend_handles_labels()
    if _make_overhead_legend is not None:
        _make_overhead_legend(fig, handles, labs, ncol=3, fontsize=12.0,
                              bbox=(0.5, 1.03), rect=(0, 0, 1, 0.70))
    else:
        fig.legend(handles, labs, loc="upper center", ncol=6,
                   bbox_to_anchor=(0.5, 1.02), frameon=False)
    out = FIGDIR / "fig_phase_race.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out}")
    print("  phase race, packet at 6 PPW, distance to 1% (wavelengths): "
          + ", ".join(f"{s}={cross[s]:.0f}" for s in order))


def main() -> None:
    setup_style()
    w = np.linspace(1e-12, np.pi, 4000)

    all_curves = build_all_curves(w)
    unique_curves = build_unique_curves(all_curves)
    all_second_derivative_curves = build_all_second_derivative_curves(w)
    unique_second_derivative_curves = build_unique_second_derivative_curves(
        all_second_derivative_curves
    )
    metrics_map = {label: sa.summarize_curve(w, unique_curves[label]) for label in UNIQUE_METHODS}
    second_derivative_metrics_map = {
        label: sa.summarize_second_derivative_curve(w, unique_second_derivative_curves[label])
        for label in SECOND_DERIVATIVE_UNIQUE_METHODS
    }

    print_identity_checks(w, all_curves)
    print_summary_table(w, unique_curves)
    print_core_metric_table(w, unique_curves)
    print_fixed_halo_table(w, all_curves)
    print_fixed_order_table(w, all_curves)
    print_same_degree_table(w, all_curves)
    print_repeated_flop_table()
    print_first_principles_notes()

    detail_labels = PDE_TRIO_LABELS
    print("\nSelected PDE trio (same formal order q=6):")
    for label in detail_labels:
        metrics = metrics_map[label]
        print(
            f"  {label:20s} F_D/N = {NATIVE_WORK_UNITS[label]:2d}  "
            f"order = {FORMAL_ORDER[label]:2d}  PPW@1% = {metrics.ppw_1pct:4.2f}"
        )
    print_bspline_win_audit(metrics_map)

    # The one-dimensional transport, phase-budget, mission-design, and direct
    # second-derivative figures were retired when Section 7 moved its evidence to
    # the 3D Navier-Stokes DNS comparisons (see experiment_dns.py). The remaining
    # figures here support Sections 2-6 and the appendices.
    print("\nGenerating figures...")
    plot_wavenumber_intro()
    plot_modified_wavenumber_derivation()
    plot_collocation_vs_galerkin_cartoon()
    plot_bm_banded_maps()
    plot_bm_second_derivative_structure()
    plot_label_fairness()
    plot_equivalence_map()
    plot_bspline_ladder()
    plot_overlap_identity()
    plot_matrix_identity_bmc5_iga2()
    plot_stability_regions()
    plot_bspline_degree_ladder(w)
    plot_anisotropy_order6(w)
    plot_matched_order(w, all_curves)
    plot_same_degree(w, all_curves)
    plot_phase_race()
    print("Done.")


if __name__ == "__main__":
    main()
