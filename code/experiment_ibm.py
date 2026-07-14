#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Periodic 2D immersed-boundary showcase for the B-spline mimetic family.

Solves the periodic 2D Helmholtz problem
    -Delta u + kappa^2 u = f          on the torus T^2 = [0, 2 pi]^2
            u(x_b) = g(x_b)            for every marker x_b on each obstacle
on a uniform N x N grid using direct-forcing immersed-boundary multipliers.

The key showcase: the same B-spline basis carries both the discrete
differentiation matrix and the off-grid evaluator at obstacle markers, so the
boundary-trace error decays at the differentiation order with no auxiliary
interpolant. Compact and explicit finite-difference schemes need a separate
Lagrange interpolant at the wall, and any order mismatch caps boundary
accuracy.

Outputs
-------
fig_ibm_obstacles.pdf
    Three immersed obstacles inside a periodic box, with stream contours and a
    |grad u| heat map drawn from the BMG2/BMC5 solution.
fig_ibm_convergence.pdf
    Boundary-trace L^infty error vs mesh spacing for E6, C6, BMC5=BMG2, and
    BMC7=BMG3. Demonstrates that BM operators reach their differentiation
    order at the wall while paired-order Lagrange caps the FD operators.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    CLASSY_ACCENT,
    CLASSY_PANEL_BG,
    SCHEME_COLORS,
    add_slope_reference,
    add_theoretical_reference_line,
    apply_classy_defaults,
    make_overhead_legend,
    quiet_axis,
)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


def setup_style() -> None:
    apply_classy_defaults()


# ---------------------------------------------------------------------------
# Obstacles. Each is a closed simple curve given by a 2 pi-periodic theta map.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Obstacle:
    name: str
    parametrization: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def disk(center: tuple[float, float], radius: float) -> Obstacle:
    cx, cy = center

    def parametrization(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cx + radius * np.cos(theta), cy + radius * np.sin(theta)

    return Obstacle(name=f"disk@{center}", parametrization=parametrization)


def trefoil(center: tuple[float, float], scale: float = 0.55) -> Obstacle:
    cx, cy = center

    def parametrization(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = scale * (1.0 + 0.55 * np.cos(3.0 * theta))
        return cx + r * np.cos(theta), cy + r * np.sin(theta)

    return Obstacle(name=f"trefoil@{center}", parametrization=parametrization)


def kite(center: tuple[float, float], scale: float = 0.6) -> Obstacle:
    cx, cy = center

    def parametrization(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Asymmetric blob with a hint of cusp on one side.
        x = cx + scale * (np.cos(theta) + 0.65 * np.cos(2.0 * theta) - 0.65)
        y = cy + scale * 1.5 * np.sin(theta)
        return x, y

    return Obstacle(name=f"kite@{center}", parametrization=parametrization)


def standard_obstacles() -> list[Obstacle]:
    box = 2.0 * np.pi
    return [
        disk((0.55 * box, 0.78 * box), radius=0.45),
        trefoil((0.30 * box, 0.42 * box), scale=0.62),
        kite((0.70 * box, 0.34 * box), scale=0.55),
    ]


_OBSTACLE_LABELS = ("disk", "trefoil", "kite")
_OBSTACLE_LABEL_OFFSETS = ((0.0, 0.85), (0.0, 1.05), (0.0, 1.15))


def sample_obstacle_markers(obstacle: Obstacle, n_markers: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_markers, endpoint=False)
    x, y = obstacle.parametrization(theta)
    return np.stack([x, y], axis=1)


def _marker_arclength_weights(obstacles: list[Obstacle], n_markers: int) -> np.ndarray:
    """Per-marker trapezoidal arclength quadrature weight, matching the exact
    adjoint convention documented in
    ``results/ibm_frontier/coupling_adjointness.md`` (Sec. 2): on each closed
    obstacle curve, ``W_m = 1/2(|x_{m+1}-x_m| + |x_m-x_{m-1}|)``, so that
    ``sum_m W_m phi(x_m)`` approximates the boundary integral ``int phi ds``.
    Concatenated in the same obstacle order used by ``_markers_and_queries``."""
    chunks = []
    for obs in obstacles:
        pts = sample_obstacle_markers(obs, n_markers)
        seg_next = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        chunks.append(0.5 * (seg_next + np.roll(seg_next, 1)))
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Periodic Fourier symbols for each scheme. The 2D periodic Helmholtz operator
# is diagonal in the discrete Fourier basis with eigenvalue
#     mu(omega_x, omega_y) = (kp_x / h)^2 + (kp_y / h)^2 + kappa^2,
# where kp(omega) is the modified-wavenumber attached to that scheme.
# ---------------------------------------------------------------------------


def _scheme_root(label: str) -> str:
    """Strip the off-grid evaluator suffix from a scheme label, returning only the
    raw discrete-derivative family name."""
    for prefix in ("E6", "C6", "BMC5 = BMG2", "BMC7 = BMG3", "Fourier pseudospectral"):
        if label.startswith(prefix):
            return prefix
    raise ValueError(f"Unknown scheme label: {label!r}")


def first_derivative_symbol(label: str, w: np.ndarray) -> np.ndarray:
    root = _scheme_root(label)
    if root == "E6":
        return sa.kprime_centered_fd(w, order=6)
    if root == "C6":
        return sa.kprime_compact(w, order=6)
    if root == "BMC5 = BMG2":
        return sa.kprime_collocation(w, degree=5)
    if root == "BMC7 = BMG3":
        return sa.kprime_collocation(w, degree=7)
    if root == "Fourier pseudospectral":
        return sa.kprime_pseudospectral(w)
    raise ValueError(label)


def second_derivative_symbol(label: str, w: np.ndarray) -> np.ndarray:
    """For first-derivative-based Helmholtz solves, the discrete Laplacian
    inherits |D_h|^2 = (kp/h)^2 in each direction. This routine returns
    (kp/h)^2 directly, so callers can build the 2D Helmholtz symbol by
    summing across the two directions."""
    kp = first_derivative_symbol(label, w)
    return kp * kp


def periodic_helmholtz_solver(
    label: str,
    npts: int,
    h: float,
    kappa: float,
):
    """Return a callable that solves (-Delta_h + kappa^2) u = rhs in Fourier space."""
    omega = np.fft.fftfreq(npts, d=1.0) * 2.0 * np.pi  # raw discrete-frequency in [-pi, pi)
    omega = np.where(omega >= 0, omega, omega + 2.0 * np.pi)
    omega = np.where(omega <= np.pi, omega, omega - 2.0 * np.pi)
    omega = np.abs(omega)
    sym1d = second_derivative_symbol(label, omega) / (h * h)
    sym2d = sym1d[:, None] + sym1d[None, :]
    denom = sym2d + kappa * kappa
    # Pin the operator kernel for kappa=0 by a RELATIVE cut, not exact
    # equality: symbol "zeros" computed numerically carry round-off, and a
    # composed first-derivative-squared symbol also vanishes at the Nyquist
    # pairs, not just at the mean (2026-07-13, same defect class as the C++
    # pressure-kernel guard).
    if kappa == 0.0:
        null_modes = np.abs(denom) <= 1e-10 * np.max(np.abs(denom))
        denom_safe = np.where(null_modes, 1.0, denom)
    else:
        null_modes = None
        denom_safe = denom

    def solve(rhs: np.ndarray) -> np.ndarray:
        rhs_hat = np.fft.fft2(rhs)
        if null_modes is not None:
            rhs_hat[null_modes] = 0.0
        u_hat = rhs_hat / denom_safe
        return np.real(np.fft.ifft2(u_hat))

    return solve


# ---------------------------------------------------------------------------
# Off-grid evaluation: same approximation space for the spline schemes,
# Lagrange for the finite-difference schemes.
# ---------------------------------------------------------------------------


def _lagrange_eval_2d(grid_values: np.ndarray, h: float, points: np.ndarray, half_width: int) -> np.ndarray:
    """Periodic 2D Lagrange interpolation of order 2*half_width."""
    npts = grid_values.shape[0]
    px = (points[:, 0] / h) % npts
    py = (points[:, 1] / h) % npts
    base_x = np.floor(px).astype(int)
    base_y = np.floor(py).astype(int)
    offsets = np.arange(-half_width + 1, half_width + 1)

    # Precompute 1D Lagrange weights per offset along x and y.
    wx = np.ones((points.shape[0], len(offsets)))
    wy = np.ones((points.shape[0], len(offsets)))
    for s, off in enumerate(offsets):
        for o2 in offsets:
            if o2 == off:
                continue
            wx[:, s] *= (px - (base_x + o2)) / (off - o2)
            wy[:, s] *= (py - (base_y + o2)) / (off - o2)

    output = np.zeros(points.shape[0])
    for sx, ox in enumerate(offsets):
        ix = (base_x + ox) % npts
        for sy, oy in enumerate(offsets):
            iy = (base_y + oy) % npts
            output += wx[:, sx] * wy[:, sy] * grid_values[ix, iy]
    return output


def _spline_eval_2d(grid_values: np.ndarray, h: float, points: np.ndarray, degree: int) -> np.ndarray:
    """Tensor-product periodic spline evaluation."""
    npts = grid_values.shape[0]
    # Recover spline coefficients on each row, then on each column.
    alpha = np.empty_like(grid_values)
    for j in range(npts):
        alpha[:, j] = sa.coefficients_from_nodal_values(grid_values[:, j], degree)
    coeffs = np.empty_like(grid_values)
    for i in range(npts):
        coeffs[i, :] = sa.coefficients_from_nodal_values(alpha[i, :], degree)

    radius = (degree + 1) / 2.0
    px = (points[:, 0] / h) % npts
    py = (points[:, 1] / h) % npts
    out = np.zeros(points.shape[0])
    span = int(2 * radius + 2)
    base_x = np.floor(px - radius).astype(int)
    base_y = np.floor(py - radius).astype(int)
    half_npts = npts / 2.0
    for sx in range(span):
        jx = base_x + sx
        delta_x = px - jx
        delta_x = np.where(delta_x > half_npts, delta_x - npts, delta_x)
        delta_x = np.where(delta_x < -half_npts, delta_x + npts, delta_x)
        beta_x = sa.centered_cardinal_bspline(degree, delta_x)
        if not np.any(beta_x):
            continue
        ix = jx % npts
        for sy in range(span):
            jy = base_y + sy
            delta_y = py - jy
            delta_y = np.where(delta_y > half_npts, delta_y - npts, delta_y)
            delta_y = np.where(delta_y < -half_npts, delta_y + npts, delta_y)
            beta_y = sa.centered_cardinal_bspline(degree, delta_y)
            if not np.any(beta_y):
                continue
            iy = jy % npts
            out += beta_x * beta_y * coeffs[ix, iy]
    return out


def offgrid_evaluator(label: str, h: float, points: np.ndarray):
    """Return a callable mapping grid-shaped values to marker-point values."""
    if label == "BMC5 = BMG2":
        return lambda u: _spline_eval_2d(u, h, points, degree=5)
    if label == "BMC7 = BMG3":
        return lambda u: _spline_eval_2d(u, h, points, degree=7)
    if label == "C6 paired Lagrange p6":
        return lambda u: _lagrange_eval_2d(u, h, points, half_width=3)
    if label == "C6 paired Lagrange p2":
        return lambda u: _lagrange_eval_2d(u, h, points, half_width=1)
    if label == "E6 paired Lagrange p6":
        return lambda u: _lagrange_eval_2d(u, h, points, half_width=3)
    if label == "E6 paired Lagrange p2":
        return lambda u: _lagrange_eval_2d(u, h, points, half_width=1)
    if label == "Fourier pseudospectral":
        # Truncated trigonometric interpolation
        npts = int(round(2.0 * np.pi / h))

        def evaluator(u: np.ndarray) -> np.ndarray:
            uhat = np.fft.fft2(u)
            kx = np.fft.fftfreq(npts, d=1.0) * npts
            ky = np.fft.fftfreq(npts, d=1.0) * npts
            kx, ky = kx[:, None], ky[None, :]
            x_eval = points[:, 0] / h
            y_eval = points[:, 1] / h
            phase = np.exp(2.0j * np.pi * (kx[None, :, :] * x_eval[:, None, None] / npts
                                         + ky[None, :, :] * y_eval[:, None, None] / npts))
            return np.real(np.sum(uhat[None, :, :] * phase, axis=(1, 2))) / (npts * npts)

        return evaluator
    raise ValueError(label)


# ---------------------------------------------------------------------------
# Spreading: place a Dirac delta at marker x_b in the spline / Lagrange basis.
# By symmetry, the spreading matrix is simply the transpose of the evaluation
# matrix when we think of both as linear maps in matching norms.
# ---------------------------------------------------------------------------


def _spline_spread_2d(values_at_points: np.ndarray, h: float, points: np.ndarray, npts: int, degree: int) -> np.ndarray:
    """Adjoint of _spline_eval_2d. Spreads marker forces back to grid coefficient space.

    Conceptually this is the Galerkin-style spreading: for each marker we add
    a contribution proportional to the tensor-product spline kernel centered
    at the marker, evaluated on the grid.
    """
    out = np.zeros((npts, npts))
    radius = (degree + 1) / 2.0
    px = (points[:, 0] / h) % npts
    py = (points[:, 1] / h) % npts
    span = int(2 * radius + 2)
    base_x = np.floor(px - radius).astype(int)
    base_y = np.floor(py - radius).astype(int)
    half_npts = npts / 2.0
    for m in range(points.shape[0]):
        v = values_at_points[m]
        if v == 0.0:
            continue
        for sx in range(span):
            jx = base_x[m] + sx
            delta_x = px[m] - jx
            if delta_x > half_npts:
                delta_x -= npts
            if delta_x < -half_npts:
                delta_x += npts
            beta_x = sa.centered_cardinal_bspline(degree, np.array([delta_x]))[0]
            if abs(beta_x) < 1e-15:
                continue
            ix = int(jx) % npts
            for sy in range(span):
                jy = base_y[m] + sy
                delta_y = py[m] - jy
                if delta_y > half_npts:
                    delta_y -= npts
                if delta_y < -half_npts:
                    delta_y += npts
                beta_y = sa.centered_cardinal_bspline(degree, np.array([delta_y]))[0]
                if abs(beta_y) < 1e-15:
                    continue
                iy = int(jy) % npts
                out[ix, iy] += v * beta_x * beta_y
    return out


def _lagrange_spread_2d(values_at_points: np.ndarray, h: float, points: np.ndarray, npts: int, half_width: int) -> np.ndarray:
    out = np.zeros((npts, npts))
    px = (points[:, 0] / h) % npts
    py = (points[:, 1] / h) % npts
    base_x = np.floor(px).astype(int)
    base_y = np.floor(py).astype(int)
    offsets = np.arange(-half_width + 1, half_width + 1)
    for m in range(points.shape[0]):
        v = values_at_points[m]
        if v == 0.0:
            continue
        wx = np.ones(len(offsets))
        wy = np.ones(len(offsets))
        for s, off in enumerate(offsets):
            for o2 in offsets:
                if o2 == off:
                    continue
                wx[s] *= (px[m] - (base_x[m] + o2)) / (off - o2)
                wy[s] *= (py[m] - (base_y[m] + o2)) / (off - o2)
        for sx, ox in enumerate(offsets):
            ix = (base_x[m] + ox) % npts
            for sy, oy in enumerate(offsets):
                iy = (base_y[m] + oy) % npts
                out[ix, iy] += v * wx[sx] * wy[sy]
    return out


def _apply_binv_2d(grid_values: np.ndarray, degree: int) -> np.ndarray:
    """Separable periodic B^{-1} coefficient recovery on a 2D grid: solve
    ``B alpha = u`` row-wise then column-wise, exactly mirroring the forward
    pass used by ``_spline_eval_2d``. Because the tensor-product basis matrix
    ``B = B_1d (x) B_1d`` is circulant and symmetric, ``B^{-1}`` is separable
    and self-adjoint, so this same routine both recovers coefficients from
    nodal samples (the interpolation direction) and completes the spline
    adjoint spread ``S = B^{-1} Phi^T diag(W)`` (the spreading direction);
    see results/ibm_frontier/coupling_adjointness.md Sec. 6."""
    npts = grid_values.shape[0]
    stage = np.empty_like(grid_values)
    for j in range(npts):
        stage[:, j] = sa.coefficients_from_nodal_values(grid_values[:, j], degree)
    out = np.empty_like(grid_values)
    for i in range(npts):
        out[i, :] = sa.coefficients_from_nodal_values(stage[i, :], degree)
    return out


def spread_operator(label: str, h: float, points: np.ndarray, npts: int, weights: np.ndarray):
    """Return the exact discrete adjoint spreader ``S = E^T diag(W)`` of the
    matching ``offgrid_evaluator``, with ``W`` the per-marker arclength
    quadrature weight (see results/ibm_frontier/coupling_adjointness.md).

    For the spline routes, ``E = Phi B^{-1}`` (a coefficient recovery ``B^{-1}``
    followed by the local cardinal-spline read ``Phi``), so the adjoint is
    ``S = B^{-1} Phi^T diag(W)``: the coefficient-recovery factor and the marker
    weights are both required, not just the raw kernel transpose ``Phi^T``.
    The Lagrange routes are self-adjoint nodal reads (``E = L``), so
    ``S = L^T diag(W)`` needs only the marker weighting.
    """
    w = np.asarray(weights, dtype=float)
    if label == "BMC5 = BMG2":
        return lambda v: _apply_binv_2d(_spline_spread_2d(w * v, h, points, npts, degree=5), degree=5)
    if label == "BMC7 = BMG3":
        return lambda v: _apply_binv_2d(_spline_spread_2d(w * v, h, points, npts, degree=7), degree=7)
    if label == "C6 paired Lagrange p6":
        return lambda v: _lagrange_spread_2d(w * v, h, points, npts, half_width=3)
    if label == "C6 paired Lagrange p2":
        return lambda v: _lagrange_spread_2d(w * v, h, points, npts, half_width=1)
    if label == "E6 paired Lagrange p6":
        return lambda v: _lagrange_spread_2d(w * v, h, points, npts, half_width=3)
    if label == "E6 paired Lagrange p2":
        return lambda v: _lagrange_spread_2d(w * v, h, points, npts, half_width=1)
    raise ValueError(label)


# ---------------------------------------------------------------------------
# IBM solve assembled around the periodic Helmholtz Fourier solver.
# ---------------------------------------------------------------------------


def manufactured_solution(npts: int) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """A smooth periodic field used to manufacture both f and g.

    u_exact(x, y) = sin(y) + 0.30 sin(x) cos(y) + 0.18 cos(2x + y)
                   + 0.10 sin(x - 2y)

    The leading sin(y) creates a horizontal-flow-like character: when u_exact is
    interpreted as a stream function, the velocity is (partial_y u, -partial_x u),
    which is approximately (cos(y), 0) plus small perturbations. Streamlines
    therefore flow predominantly left-to-right and bend around obstacles.
    """

    def u(points: np.ndarray) -> np.ndarray:
        x = points[..., 0]
        y = points[..., 1]
        return (
            np.sin(y)
            + 0.30 * np.sin(x) * np.cos(y)
            + 0.18 * np.cos(2.0 * x + y)
            + 0.10 * np.sin(x - 2.0 * y)
        )

    def laplacian_u(points: np.ndarray) -> np.ndarray:
        x = points[..., 0]
        y = points[..., 1]
        # term1 = sin(y)         -> Lap = -sin(y)
        term1 = -np.sin(y)
        # term2 = 0.30 sin(x) cos(y) -> Lap = -2 * 0.30 sin(x) cos(y)
        term2 = -0.60 * np.sin(x) * np.cos(y)
        # term3 = 0.18 cos(2x + y) -> Lap = -(4 + 1) * 0.18 cos(2x + y)
        term3 = -5.0 * 0.18 * np.cos(2.0 * x + y)
        # term4 = 0.10 sin(x - 2y) -> Lap = -(1 + 4) * 0.10 sin(x - 2y)
        term4 = -5.0 * 0.10 * np.sin(x - 2.0 * y)
        return term1 + term2 + term3 + term4

    h = 2.0 * np.pi / npts
    grid_x, grid_y = np.meshgrid(np.arange(npts) * h, np.arange(npts) * h, indexing="ij")
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(npts, npts, 2)
    u_grid = u(grid_points)
    f_grid = -laplacian_u(grid_points)  # since solving -Delta u = f
    return u_grid, f_grid, u, laplacian_u


def _markers_and_queries(
    obstacles: list[Obstacle], n_markers: int, n_queries: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return marker and query point arrays. Queries are interleaved between markers
    so that no query coincides with any marker (marker phase = 0, query phase = 0.5)."""
    marker_chunks = []
    query_chunks = []
    for obs in obstacles:
        theta_m = np.linspace(0.0, 2.0 * np.pi, n_markers, endpoint=False)
        theta_q = np.linspace(
            np.pi / n_queries,
            2.0 * np.pi + np.pi / n_queries,
            n_queries,
            endpoint=False,
        )
        # Shift queries by half the marker spacing to guarantee distinct points.
        theta_q = (theta_m[0] + np.pi / n_markers) + np.linspace(
            0.0, 2.0 * np.pi, n_queries, endpoint=False
        )
        bx_m, by_m = obs.parametrization(theta_m)
        bx_q, by_q = obs.parametrization(theta_q)
        marker_chunks.append(np.stack([bx_m, by_m], axis=1))
        query_chunks.append(np.stack([bx_q, by_q], axis=1))
    markers = np.concatenate(marker_chunks, axis=0)
    queries = np.concatenate(query_chunks, axis=0)
    return markers, queries


def solve_ibm(
    label: str,
    npts: int,
    obstacles: list[Obstacle],
    n_markers: int,
    kappa: float = 0.5,
    n_queries: int | None = None,
):
    """Run a single IBM Helmholtz solve, returning the grid solution and trace error.

    The Schur complement is built around the periodic Helmholtz Fourier solver
    using each scheme's modified wavenumber. The boundary-trace error is
    measured at query points that are interleaved between markers, so the
    measurement reflects the off-grid evaluator rather than the trivially zero
    error at the marker locations themselves.
    """
    h = 2.0 * np.pi / npts
    u_exact_grid, f_grid, u_func, _ = manufactured_solution(npts)

    if n_queries is None:
        n_queries = max(64, 4 * n_markers)
    markers, queries = _markers_and_queries(obstacles, n_markers, n_queries)

    g_target_marker = u_func(markers)
    g_target_query = u_func(queries)

    rhs_grid = f_grid.copy()
    if kappa != 0.0:
        rhs_grid = rhs_grid + kappa * kappa * u_exact_grid

    helmholtz = periodic_helmholtz_solver(label, npts, h, kappa)
    u_periodic = helmholtz(rhs_grid)

    eval_marker = offgrid_evaluator(label, h, markers)
    eval_query = offgrid_evaluator(label, h, queries)
    marker_weights = _marker_arclength_weights(obstacles, n_markers)
    spread = spread_operator(label, h, markers, npts, marker_weights)

    n_total = markers.shape[0]
    schur = np.zeros((n_total, n_total))
    e = np.zeros(n_total)
    for k in range(n_total):
        e[:] = 0.0
        e[k] = 1.0
        spread_field = spread(e)
        sol = helmholtz(spread_field)
        schur[:, k] = eval_marker(sol)
    rhs_for_lambda = eval_marker(u_periodic) - g_target_marker
    lam = np.linalg.solve(schur, rhs_for_lambda)
    correction = helmholtz(spread(lam))
    u_h = u_periodic - correction

    trace_error_query = float(np.max(np.abs(eval_query(u_h) - g_target_query)))
    grid_points = np.stack(
        np.meshgrid(np.arange(npts) * h, np.arange(npts) * h, indexing="ij"), axis=-1
    )
    interior_mask = ~_obstacle_mask(grid_points, obstacles)
    grid_error = float(
        np.sqrt(np.mean((u_h[interior_mask] - u_exact_grid[interior_mask]) ** 2))
        / np.sqrt(np.mean(u_exact_grid[interior_mask] ** 2))
    )
    return {
        "u_h": u_h,
        "u_exact": u_exact_grid,
        "trace_error": trace_error_query,
        "grid_error": grid_error,
        "markers": markers,
        "queries": queries,
    }


def _obstacle_mask(grid_points: np.ndarray, obstacles: list[Obstacle], oversample: int = 64) -> np.ndarray:
    """Mark grid cells that fall inside at least one obstacle."""
    mask = np.zeros(grid_points.shape[:2], dtype=bool)
    for obs in obstacles:
        theta = np.linspace(0.0, 2.0 * np.pi, oversample, endpoint=False)
        bx, by = obs.parametrization(theta)
        # Ray casting (not robust at concavities, but obstacles are convex-ish here).
        from matplotlib.path import Path
        polygon = Path(np.stack([bx, by], axis=1))
        flat = grid_points.reshape(-1, 2)
        inside = polygon.contains_points(flat)
        mask |= inside.reshape(grid_points.shape[:2])
    return mask


# ---------------------------------------------------------------------------
# Convergence figure.
# ---------------------------------------------------------------------------


def figure_ibm_convergence() -> None:
    setup_style()
    obstacles = standard_obstacles()
    grids = [48, 64, 96, 128]
    schemes = [
        ("BMC5 = BMG2", SCHEME_COLORS["BMC"], "-", "o"),
        ("BMC7 = BMG3", SCHEME_COLORS["BMG"], "-", "s"),
        ("C6 paired Lagrange p6", SCHEME_COLORS["C"], "--", "^"),
        ("C6 paired Lagrange p2", SCHEME_COLORS["ParedLagrange"], (0, (4, 2)), "v"),
        ("E6 paired Lagrange p2", SCHEME_COLORS["E"], (0, (1, 1.5)), "x"),
    ]

    trace_table: dict[str, list[float]] = {label: [] for label, *_ in schemes}
    grid_table: dict[str, list[float]] = {label: [] for label, *_ in schemes}
    h_list: list[float] = []
    for npts in grids:
        h = 2.0 * np.pi / npts
        h_list.append(h)
        n_markers = max(48, int(round(np.pi / h)))
        for label, *_ in schemes:
            print(f"  solving {label} on {npts}x{npts} ...", flush=True)
            res = solve_ibm(label, npts, obstacles, n_markers, kappa=0.5)
            trace_table[label].append(res["trace_error"])
            grid_table[label].append(res["grid_error"])
            print(
                f"    trace_err={res['trace_error']:.3e}, grid_err={res['grid_error']:.3e}"
            )

    import convergence_style as cs
    formal = {"BMC5 = BMG2": 6, "BMC7 = BMG3": 8, "C6 paired Lagrange p6": 6,
              "C6 paired Lagrange p2": 2, "E6 paired Lagrange p2": 2}
    short = {"BMC5 = BMG2": "BMC5", "BMC7 = BMG3": "BMC7",
             "C6 paired Lagrange p6": "C6+L6", "C6 paired Lagrange p2": "C6+L2",
             "E6 paired Lagrange p2": "E6+L2"}
    hh = np.asarray(h_list)

    fig = plt.figure(figsize=(13.0, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.62])
    ax1, ax2, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    for ax, table, title, ylabel in (
            (ax1, trace_table, r"boundary trace  $\|u_h-g\|_\infty$", "max-norm error"),
            (ax2, grid_table, r"interior  $L^2$ relative error", "relative error")):
        for label, color, ls, marker in schemes:
            e = np.asarray(table[label])
            ax.loglog(hh, e, ls=ls, marker=marker, color=color, ms=4.6, lw=1.4,
                      mec="white", mew=0.5)
            ap = cs.anchor_point(hh, e)
            if ap:
                cs.slope_triangle(ax, ap[0], ap[1] * 0.30, formal[label], color)
        cs.style_axis(ax, r"mesh size  $h$", ylabel, title)

    # Only four grids are run here and the spline curves plunge to the round-off
    # floor, so a fitted slope is not meaningful; the table reports the design
    # order and the slope guides on the curves confirm it.
    rows = []
    for label, color, ls, marker in schemes:
        rows.append((short[label], color, [str(formal[label]), str(formal[label])]))
    cs.order_table(axt, "design order at the wall", ["trace", "interior"], rows,
                   footnote="BMC reaches the differentiation order at the wall from\n"
                            "one banded source (slope guides confirm it); a paired\n"
                            "low-order Lagrange interpolant caps the compact and\n"
                            "explicit schemes at the interpolant order.")
    fig.suptitle(r"Periodic 2D immersed-boundary Helmholtz: convergence at the wall",
                 fontsize=11.5, x=0.42)
    fig.savefig(FIGDIR / "fig_ibm_convergence.pdf",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hero figure: stream contours + |grad u| around three obstacles.
# ---------------------------------------------------------------------------


def solve_ibm_for_hero(label: str, npts: int, obstacles: list[Obstacle], n_markers: int):
    """Variant of solve_ibm that forces psi = 0 on obstacle markers for the
    visualization, while the periodic forcing creates an underlying horizontal
    flow. The obstacles then act as streamlines and the velocity field bends
    visibly around them. A small Helmholtz mass kappa^2 keeps the operator
    invertible on the torus and avoids tugging at the constant null space."""
    h = 2.0 * np.pi / npts
    _u_exact_grid, f_grid, _u_func, _ = manufactured_solution(npts)

    marker_chunks = [sample_obstacle_markers(obs, n_markers) for obs in obstacles]
    markers = np.concatenate(marker_chunks, axis=0)

    kappa = 0.4
    rhs_grid = f_grid.copy()
    helmholtz = periodic_helmholtz_solver(label, npts, h, kappa=kappa)
    u_periodic = helmholtz(rhs_grid)

    eval_marker = offgrid_evaluator(label, h, markers)
    marker_weights = _marker_arclength_weights(obstacles, n_markers)
    spread = spread_operator(label, h, markers, npts, marker_weights)

    n_total = markers.shape[0]
    schur = np.zeros((n_total, n_total))
    e = np.zeros(n_total)
    for k in range(n_total):
        e[:] = 0.0
        e[k] = 1.0
        sol = helmholtz(spread(e))
        schur[:, k] = eval_marker(sol)
    target = np.zeros(n_total)
    rhs_for_lambda = eval_marker(u_periodic) - target
    lam = np.linalg.solve(schur, rhs_for_lambda)
    correction = helmholtz(spread(lam))
    return u_periodic - correction


def figure_ibm_obstacles() -> None:
    """Periodic IBM Helmholtz solve threading three immersed obstacles.

    Cream-anchored sequential colormap built around the muted-blue
    BMC accent at the dark end. Streamlines are thin charcoal so the
    speed field reads as the dominant idiom. Each obstacle silhouette
    carries a small italic side label in a free margin.
    """
    setup_style()
    obstacles = standard_obstacles()
    npts = 192
    h = 2.0 * np.pi / npts
    n_markers = 80

    u_h = solve_ibm_for_hero("BMC5 = BMG2", npts, obstacles, n_markers)

    grad_y, grad_x = np.gradient(u_h, h, h, edge_order=2)
    speed = np.sqrt(grad_x ** 2 + grad_y ** 2)

    grid_points = np.stack(
        np.meshgrid(np.arange(npts) * h, np.arange(npts) * h, indexing="ij"), axis=-1
    )
    # This mask is used only for the colorbar percentile and to stop streamlines
    # before the wall; it never controls where the field pixels are drawn (see
    # below), so its polygon resolution cannot introduce a rendering mismatch.
    interior_mask = ~_obstacle_mask(grid_points, obstacles, oversample=256)

    cream_to_navy = LinearSegmentedColormap.from_list(
        "cream_to_navy",
        [
            (0.00, "#FBF8F2"),
            (0.18, "#E9E0CB"),
            (0.45, "#9CB1C2"),
            (0.75, "#3A5468"),
            (1.00, SCHEME_COLORS["BMC"]),
        ],
    )

    fig = plt.figure(figsize=(7.4, 7.6), facecolor="white")
    ax = fig.add_axes([0.10, 0.18, 0.82, 0.78])
    extent = (0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi)

    vmax = float(np.percentile(speed[interior_mask], 98)) if np.any(interior_mask) else 1.0

    # Single high-resolution boundary curve per obstacle, shared verbatim by the
    # field clip path and the silhouette fill below, so the two coincide exactly
    # and no antialiasing/resolution-mismatch halo can appear between them
    # (the earlier version NaN-masked the field on a coarser polygon and drew a
    # separately-parametrized silhouette on top, which left a visible ring).
    box = 2.0 * np.pi
    theta_fine = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    obstacle_curves = [obs.parametrization(theta_fine) for obs in obstacles]

    pcm = ax.imshow(
        speed.T,
        extent=extent,
        origin="lower",
        cmap=cream_to_navy,
        vmin=0.0,
        vmax=vmax,
        alpha=1.0,
        rasterized=True,
        interpolation="bilinear",
    )

    # Clip the field to the domain box minus the obstacle interiors, using the
    # exact same curves as the silhouette fill: the outer rectangle is wound
    # counter-clockwise and each obstacle loop is wound clockwise (reversed),
    # which the Agg nonzero-winding-rule renderer reads as a hole. This is a
    # vector clip (no raster interpolation at the boundary), so it renders a
    # crisp edge instead of the earlier fuzzy NaN/bilinear-interpolation ring.
    from matplotlib.path import Path as _MplPath

    clip_verts: list[tuple[float, float]] = [
        (0.0, 0.0), (box, 0.0), (box, box), (0.0, box), (0.0, 0.0)
    ]
    clip_codes: list[int] = (
        [_MplPath.MOVETO] + [_MplPath.LINETO] * 3 + [_MplPath.CLOSEPOLY]
    )
    for bx, by in obstacle_curves:
        loop = list(zip(bx[::-1], by[::-1]))
        loop.append(loop[0])
        clip_verts.extend(loop)
        clip_codes.extend(
            [_MplPath.MOVETO] + [_MplPath.LINETO] * (len(loop) - 2) + [_MplPath.CLOSEPOLY]
        )
    field_clip_path = _MplPath(clip_verts, clip_codes)
    pcm.set_clip_path(field_clip_path, transform=ax.transData)

    grid_x = np.arange(npts) * h
    grid_y = np.arange(npts) * h
    velocity_x = grad_y.copy()
    velocity_y = -grad_x.copy()
    velocity_x[~interior_mask] = np.nan
    velocity_y[~interior_mask] = np.nan
    strm = ax.streamplot(
        grid_x,
        grid_y,
        velocity_x.T,
        velocity_y.T,
        color="#1a1a1a",
        linewidth=0.40,
        density=2.0,
        arrowsize=0.0,
    )
    strm.lines.set_clip_path(field_clip_path, transform=ax.transData)

    obstacle_labels = (
        ("disk", (0.55 * box, 0.78 * box), 0.55),
        ("trefoil", (0.30 * box, 0.42 * box), 0.85),
        ("kite", (0.70 * box, 0.34 * box), 1.10),
    )
    for (bx, by) in obstacle_curves:
        # A matching-color edge stroke (not just an unstroked fill) covers any
        # sub-pixel antialiasing seam between this patch and the image clip
        # above, since both are drawn from the identical boundary curve.
        ax.fill(bx, by, facecolor="#1a1a1a", edgecolor="#1a1a1a",
                linewidth=0.6, zorder=4)
    for name, (cx, cy), label_radius in obstacle_labels:
        ax.text(
            cx, cy + label_radius, name,
            ha="center", va="bottom",
            fontsize=10.0, color="#1a1a1a", fontstyle="italic",
            zorder=6,
            bbox=dict(facecolor="#FBF8F2", edgecolor="none",
                      alpha=0.85, pad=1.6),
        )

    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_aspect("equal")
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.4)

    cax = fig.add_axes([0.16, 0.10, 0.70, 0.022])
    cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$|\nabla \psi_h|$", color=SCHEME_COLORS["BMC"], labelpad=4)
    cbar.outline.set_linewidth(0.3)
    cbar.outline.set_edgecolor("#444444")
    cbar.ax.tick_params(colors="#444444", width=0.3, length=2.0)

    fig.savefig(FIGDIR / "fig_ibm_obstacles.pdf", dpi=400,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    figure_ibm_obstacles()
    figure_ibm_convergence()
    print("Wrote IBM figures to", FIGDIR)


if __name__ == "__main__":
    main()
