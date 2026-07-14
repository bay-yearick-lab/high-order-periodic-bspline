#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Matched immersed-boundary head-to-head for the periodic B-spline family.

This is the frontier study for Section 8.8.2/8.8.3: it replaces the qualitative
"identical cost" claims of the earlier IBM showcase with an adjoint-consistent,
matched-order, total-cost comparison.

What changed relative to code/experiment_ibm.py
-----------------------------------------------
1. Adjoint coupling enforced. The spreading operator S is the exact weighted
   transpose of the interpolation operator E, S = E^T diag(W), with W the marker
   arclength quadrature weights. The earlier code spread splines through Phi^T
   only (missing the coefficient-recovery factor B^{-1}), so its spline spreader
   was NOT the adjoint of its spline interpolation. Here every route uses the
   exact discrete adjoint, verified to machine precision, so the discrete
   coupling-power identity (S lambda, u)_grid = (lambda, E u)_W holds by
   construction and the Schur complement is symmetric-definite up to the W
   scaling.

2. Matched pairs. BMC5 vs C6+L6, BMC7 vs C8+L8, BMC9 vs C10+L10, where Cq is the
   maximal-order tridiagonal compact first derivative and Lq is a periodic
   Lagrange off-grid interpolant of formal order q (half-width q/2). The bulk
   Laplacian is the same composed div-grad convention (each direction the square
   of the route's own first derivative) for every route.

3. The Chen-Peskin Fourier spectral immersed-boundary route (Chen & Peskin, JCP
   509:113048, 2024): the fluid solve is Fourier pseudospectral and the coupling
   evaluates the global trigonometric interpolant at the markers (the NUFFT-type
   read), with the exact adjoint spreader. Fielded on the same geometry and the
   same diagnostics.

4. A near-singular fixed-width source (harder than the analytic MMS) keeps the
   asymptotic window above the round-off floor at design orders 6, 8, and 10, so
   at least three consecutive monotone refinements are available for every route
   without extended precision.

5. A total-cost ledger charges setup, bulk solve, Schur assembly, Schur solve,
   and marker evaluation separately, and reports scaling with the marker count.

Outputs (all under results/ibm_frontier/)
-----------------------------------------
results.json          every number the manuscript will quote
cost_ledger.csv       per-run timing splits + marker-count scaling
fig_ibm_setup_receipt.pdf
fig_ibm_error_walltime.pdf
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation

THISDIR = Path(__file__).resolve().parent
REPO_ROOT = THISDIR.parent if THISDIR.name == "code" else THISDIR
OUTDIR = REPO_ROOT / "results" / "ibm_frontier"
OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(THISDIR))

import experiment_ibm as ibm  # noqa: E402
import spline_analysis as sa  # noqa: E402
from plotting_helpers import SCHEME_COLORS, apply_classy_defaults  # noqa: E402

TWO_PI = 2.0 * np.pi
BOX = TWO_PI
KAPPA = 2.0

# Near-singular fixed-width source: one interior peak per obstacle. The width is
# tuned so the interior peak is under-resolved on the coarse grids (no free
# smoothness win) yet the exterior asymptotic window sits above the round-off
# floor at design order ten over at least three consecutive grids.
SOURCES = np.array([
    [0.55 * BOX, 0.78 * BOX],   # inside the disk
    [0.30 * BOX, 0.42 * BOX],   # inside the trefoil
    [0.70 * BOX, 0.34 * BOX],   # inside the kite
])
COEFFS = np.array([1.0, 0.8, 1.3])
SIGMA = 0.22
_IMAGES = np.array([(mx, my) for mx in range(-2, 3) for my in range(-2, 3)],
                   dtype=float)
NREF = 768

# --------------------------------------------------------------------------
# Route registry: matched pairs + Fourier spectral IB.
# --------------------------------------------------------------------------
ROUTES = {
    "BMC5":   dict(kind="spline", degree=5, order=6,  pair="C6+L6",   color=SCHEME_COLORS["BMC"]),
    "BMC7":   dict(kind="spline", degree=7, order=8,  pair="C8+L8",   color=SCHEME_COLORS["BMG"]),
    "BMC9":   dict(kind="spline", degree=9, order=10, pair="C10+L10", color="#7a2d2d"),
    "C6+L6":  dict(kind="fdlag",  fd_order=6,  hw=3, order=6,  color=SCHEME_COLORS["C"]),
    "C8+L8":  dict(kind="fdlag",  fd_order=8,  hw=4, order=8,  color="#c98a3c"),
    "C10+L10":dict(kind="fdlag",  fd_order=10, hw=5, order=10, color="#e0a95e"),
    "Fourier-PS": dict(kind="fourier", order=None, color=SCHEME_COLORS["PS"]),
}
MATCHED_PAIRS = [("BMC5", "C6+L6"), ("BMC7", "C8+L8"), ("BMC9", "C10+L10")]


# --------------------------------------------------------------------------
# Source and exact solution (near-singular fixed-width).
# --------------------------------------------------------------------------
def source_forcing(points: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    shape = pts.shape[:-1]
    flat = pts.reshape(-1, 2)
    out = np.zeros(flat.shape[0])
    inv = 1.0 / (2.0 * np.pi * sigma ** 2)
    for c, z in zip(COEFFS, SOURCES):
        for m in _IMAGES:
            d = flat - (z + TWO_PI * m)[None, :]
            r2 = d[:, 0] ** 2 + d[:, 1] ** 2
            out += c * inv * np.exp(-r2 / (2.0 * sigma ** 2))
    return out.reshape(shape)


def build_exact(sigma: float = SIGMA, kappa: float = KAPPA):
    """Machine-accurate exact solution: spectral periodic Helmholtz response to
    the smooth forcing, returned as a degree-five RectBivariateSpline sampler."""
    h = TWO_PI / NREF
    grid = np.stack(np.meshgrid(np.arange(NREF) * h, np.arange(NREF) * h,
                                indexing="ij"), axis=-1)
    f = source_forcing(grid, sigma)
    kk = np.fft.fftfreq(NREF, d=1.0) * NREF
    kx, ky = np.meshgrid(kk, kk, indexing="ij")
    denom = (kx ** 2 + ky ** 2) + kappa ** 2
    uref = np.real(np.fft.ifft2(np.fft.fft2(f) / denom))
    xs = np.arange(NREF) * h
    spl = RectBivariateSpline(xs, xs, uref, kx=5, ky=5)

    def u_exact(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        shape = pts.shape[:-1]
        flat = (pts.reshape(-1, 2)) % TWO_PI
        return spl.ev(flat[:, 0], flat[:, 1]).reshape(shape)

    return u_exact


# --------------------------------------------------------------------------
# Bulk Helmholtz solver: composed div-grad Laplacian, route-specific symbol.
# --------------------------------------------------------------------------
def _first_deriv_symbol(route: str, w: np.ndarray) -> np.ndarray:
    r = ROUTES[route]
    if r["kind"] == "spline":
        return sa.kprime_collocation(w, r["degree"])
    if r["kind"] == "fdlag":
        return sa.kprime_compact(w, r["fd_order"])
    if r["kind"] == "fourier":
        return sa.kprime_pseudospectral(w)
    raise ValueError(route)


def _bulk_denom(route: str, npts: int, h: float, kappa: float) -> np.ndarray:
    """The fft2-ordered symbol of the composed div-grad Helmholtz operator A_h."""
    omega = np.fft.fftfreq(npts, d=1.0) * TWO_PI
    omega = np.where(omega >= 0, omega, omega + TWO_PI)
    omega = np.where(omega <= np.pi, omega, omega - TWO_PI)
    omega = np.abs(omega)
    kp = _first_deriv_symbol(route, omega)
    sym1d = (kp * kp) / (h * h)
    return sym1d[:, None] + sym1d[None, :] + kappa * kappa


def bulk_solver(route: str, npts: int, h: float, kappa: float):
    denom = _bulk_denom(route, npts, h, kappa)

    def solve(rhs: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))

    return solve


def _bspline_symbol_2d(degree: int, npts: int) -> np.ndarray:
    """The fft2-ordered symbol b(kx) b(ky) of the periodic basis matrix B.

    B is circulant on the periodic grid, so B^{-1} is a Fourier multiplier: the
    coefficient recovery folds into the Fourier solve, and the whole spline
    grid operator B^{-1} A_h^{-1} B^{-1} becomes one FFT-diagonal solve. This is
    what a production periodic B-spline solver does, and it makes the spline
    coupling cost one bulk solve plus local reads per marker, matching the
    compact and Fourier routes."""
    radius = (degree + 1) // 2
    col = np.zeros(npts)
    offsets = np.arange(-radius, radius + 1)
    vals = sa.centered_cardinal_bspline(degree, offsets.astype(float))
    for off, v in zip(offsets, vals):
        col[off % npts] += v
    b1d = np.real(np.fft.fft(col))
    return b1d[:, None] * b1d[None, :]


# --------------------------------------------------------------------------
# Coupling operators E (interpolation) and S (adjoint spreading, S = E^T diag(W)).
# --------------------------------------------------------------------------
def _spline_footprint(px, py, degree, npts):
    """Local tensor-product cardinal-spline weights at a marker: return integer
    row/col offsets and weight matrix pieces (bx, by)."""
    radius = (degree + 1) / 2.0
    span = int(2 * radius + 2)
    half = npts / 2.0
    bx = np.empty(span)
    by = np.empty(span)
    jx = np.empty(span, dtype=int)
    jy = np.empty(span, dtype=int)
    bx0 = int(np.floor(px - radius))
    by0 = int(np.floor(py - radius))
    for s in range(span):
        gx = bx0 + s
        dx = px - gx
        dx = dx - npts if dx > half else (dx + npts if dx < -half else dx)
        bx[s] = sa.centered_cardinal_bspline(degree, np.array([dx]))[0]
        jx[s] = gx % npts
        gy = by0 + s
        dy = py - gy
        dy = dy - npts if dy > half else (dy + npts if dy < -half else dy)
        by[s] = sa.centered_cardinal_bspline(degree, np.array([dy]))[0]
        jy[s] = gy % npts
    return jx, jy, bx, by


def _lagrange_footprint(px, py, hw, npts):
    offsets = np.arange(-hw + 1, hw + 1)
    bx0 = int(np.floor(px))
    by0 = int(np.floor(py))
    wx = np.ones(len(offsets))
    wy = np.ones(len(offsets))
    for s, off in enumerate(offsets):
        for o2 in offsets:
            if o2 == off:
                continue
            wx[s] *= (px - (bx0 + o2)) / (off - o2)
            wy[s] *= (py - (by0 + o2)) / (off - o2)
    jx = (bx0 + offsets) % npts
    jy = (by0 + offsets) % npts
    return jx, jy, wx, wy


def build_phi_sparse(kind: str, param: int, npts: int, h: float,
                     markers: np.ndarray):
    """Sparse interpolation matrix Phi (n_b x N^2) of the local reads: row m holds
    the tensor-product spline (kind='spline', param=degree) or Lagrange
    (kind='lagrange', param=half_width) weights at marker m. E and S are built
    from Phi so the whole coupling is one bulk solve plus sparse matvecs, and the
    spline and compact routes assemble at identical FFT count."""
    from scipy.sparse import csr_matrix
    n_b = markers.shape[0]
    px = (markers[:, 0] / h) % npts
    py = (markers[:, 1] / h) % npts
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for m in range(n_b):
        if kind == "spline":
            jx, jy, wx, wy = _spline_footprint(px[m], py[m], param, npts)
        else:
            jx, jy, wx, wy = _lagrange_footprint(px[m], py[m], param, npts)
        for a in range(len(jx)):
            if wx[a] == 0.0:
                continue
            for b in range(len(jy)):
                w = wx[a] * wy[b]
                if w == 0.0:
                    continue
                rows.append(m)
                cols.append(jx[a] * npts + jy[b])
                data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(n_b, npts * npts))


def make_fourier_phase(markers: np.ndarray, npts: int, h: float):
    kk = np.fft.fftfreq(npts, d=1.0) * npts   # integer wavenumbers
    xg = markers[:, 0] / h
    yg = markers[:, 1] / h
    Ex = np.exp(2.0j * np.pi * np.outer(kk, xg) / npts)   # (N, n_b)
    Ey = np.exp(2.0j * np.pi * np.outer(kk, yg) / npts)   # (N, n_b)
    return Ex, Ey


def coupling_ops(route: str, npts: int, h: float, markers: np.ndarray,
                 weights: np.ndarray, kappa: float = KAPPA):
    """Return a dict of callables for a route:

        E(u)        markers <- grid: interpolation.
        S(v)        grid <- markers: the exact weighted adjoint,
                    <S v, u>_euclid = sum_m W_m v_m (E u)_m.
        schur_col(v)  markers <- markers: E A_h^{-1} S v (fused where possible).
        corr(lam)     grid <- markers: A_h^{-1} S lam (subtracted to form u_h).
        solve(rhs)    grid <- grid: A_h^{-1} rhs (bulk particular solve).

    The spline route fuses B^{-1} A_h^{-1} B^{-1} into one FFT-diagonal solve, so
    its Schur column costs one bulk solve plus local reads, matching the compact
    and Fourier routes exactly."""
    r = ROUTES[route]
    W = np.asarray(weights, dtype=float)
    solve = bulk_solver(route, npts, h, kappa)

    npts2 = npts * npts

    if r["kind"] == "spline":
        d = r["degree"]
        a_denom = _bulk_denom(route, npts, h, kappa)
        b2 = _bspline_symbol_2d(d, npts)          # b(kx) b(ky)
        g_denom = b2 * b2 * a_denom                # symbol of (B^{-1} A^{-1} B^{-1})^{-1}
        ab_denom = b2 * a_denom                    # symbol of (A^{-1} B^{-1})^{-1}
        Phi = build_phi_sparse("spline", d, npts, h, markers)

        def apply_Binv(u):
            return np.real(np.fft.ifft2(np.fft.fft2(u) / b2))

        def apply_G(flat):
            return np.real(np.fft.ifft2(np.fft.fft2(flat.reshape(npts, npts)) / g_denom)).ravel()

        def E(u):
            return Phi @ apply_Binv(u).ravel()

        def S(v):
            return apply_Binv((Phi.T @ (W * v)).reshape(npts, npts))

        def schur_col(v):
            return Phi @ apply_G(Phi.T @ (W * v))

        def corr(lam):
            spread = (Phi.T @ (W * lam)).reshape(npts, npts)
            return np.real(np.fft.ifft2(np.fft.fft2(spread) / ab_denom))

        return dict(E=E, S=S, schur_col=schur_col, corr=corr, solve=solve, Phi=Phi)

    if r["kind"] == "fdlag":
        hw = r["hw"]
        a_denom = _bulk_denom(route, npts, h, kappa)
        Phi = build_phi_sparse("lagrange", hw, npts, h, markers)

        def apply_Ainv(flat):
            return np.real(np.fft.ifft2(np.fft.fft2(flat.reshape(npts, npts)) / a_denom)).ravel()

        def E(u):
            return Phi @ u.ravel()

        def S(v):
            return (Phi.T @ (W * v)).reshape(npts, npts)

        def schur_col(v):
            return Phi @ apply_Ainv(Phi.T @ (W * v))

        def corr(lam):
            return solve(S(lam))

        return dict(E=E, S=S, schur_col=schur_col, corr=corr, solve=solve, Phi=Phi)

    if r["kind"] == "fourier":
        Ex, Ey = make_fourier_phase(markers, npts, h)
        n2 = float(npts * npts)

        def E(u):
            uhat = np.fft.fft2(u)
            tmp = Ex.T @ uhat
            return np.real(np.einsum("mk,km->m", tmp, Ey)) / n2

        def S(v):
            wv = W * v
            c = (Ex * wv[None, :]) @ Ey.T
            return np.real(np.fft.fft2(c)) / n2

        def schur_col(v):
            return E(solve(S(v)))

        def corr(lam):
            return solve(S(lam))

        return dict(E=E, S=S, schur_col=schur_col, corr=corr, solve=solve)

    raise ValueError(route)


# --------------------------------------------------------------------------
# Marker geometry + quadrature weights.
# --------------------------------------------------------------------------
def markers_weights_queries(obstacles, n_markers):
    markers, queries = ibm._markers_and_queries(obstacles, n_markers, n_markers)
    n_obs = len(obstacles)
    pts = markers.reshape(n_obs, n_markers, 2)
    weights = np.empty((n_obs, n_markers))
    for k in range(n_obs):
        p = pts[k]
        seg_next = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        weights[k] = 0.5 * (seg_next + np.roll(seg_next, 1))
    return markers, weights.reshape(-1), queries


# --------------------------------------------------------------------------
# Adjointness self-test.
# --------------------------------------------------------------------------
def adjointness_residual(route, npts=48, n_markers=40, seed=0):
    obstacles = ibm.standard_obstacles()
    markers, weights, _ = markers_weights_queries(obstacles, n_markers)
    h = TWO_PI / npts
    ops = coupling_ops(route, npts, h, markers, weights)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((npts, npts))
    v = rng.standard_normal(markers.shape[0])
    lhs = float(np.sum(ops["S"](v) * u))                 # <S v, u>_euclid
    rhs = float(np.sum(weights * v * ops["E"](u)))       # <v, E u>_W
    denom = max(abs(lhs), abs(rhs), 1e-300)
    return abs(lhs - rhs) / denom, lhs, rhs


# --------------------------------------------------------------------------
# One immersed-boundary solve (near-singular exterior field), timed.
# --------------------------------------------------------------------------
def solve_frontier(route, npts, obstacles, n_markers, u_exact, kappa=KAPPA,
                   want_field=True):
    h = TWO_PI / npts
    t = {}

    t0 = time.perf_counter()
    markers, weights, queries = markers_weights_queries(obstacles, n_markers)
    grid = np.stack(np.meshgrid(np.arange(npts) * h, np.arange(npts) * h,
                                indexing="ij"), axis=-1)
    f_grid = source_forcing(grid)
    ops = coupling_ops(route, npts, h, markers, weights, kappa)
    solve = ops["solve"]
    g_marker = u_exact(markers)
    g_query = u_exact(queries)
    t["setup"] = time.perf_counter() - t0

    # bulk particular solve
    t0 = time.perf_counter()
    u_particular = solve(f_grid)
    t["bulk_solve"] = time.perf_counter() - t0

    # Schur assembly: one fused coupling solve per marker column
    n_total = markers.shape[0]
    t0 = time.perf_counter()
    schur = np.empty((n_total, n_total))
    e = np.zeros(n_total)
    for k in range(n_total):
        e[k] = 1.0
        schur[:, k] = ops["schur_col"](e)
        e[k] = 0.0
    t["schur_assembly"] = time.perf_counter() - t0

    # Schur solve
    t0 = time.perf_counter()
    rhs = ops["E"](u_particular) - g_marker
    lam = np.linalg.solve(schur, rhs)
    t["schur_solve"] = time.perf_counter() - t0

    # reconstruct the grid solution
    t0 = time.perf_counter()
    u_h = u_particular - ops["corr"](lam)
    t["reconstruct"] = time.perf_counter() - t0

    # a standalone marker-evaluation (E) cost for the ledger
    t0 = time.perf_counter()
    _ = ops["E"](u_h)
    t["marker_eval"] = time.perf_counter() - t0

    # boundary-trace error at OFF-marker queries
    ops_q = coupling_ops(route, npts, h, queries, np.ones(queries.shape[0]), kappa)
    trace_err = float(np.max(np.abs(ops_q["E"](u_h) - g_query)))

    # Schur conditioning (symmetrized by the W scaling: cond of W^{1/2} M_sym W^{1/2})
    ws = np.sqrt(np.abs(weights))
    # schur = (E A^{-1} E^T) diag(W) is COLUMN-scaled by the weights, so the
    # symmetric PSD factor is recovered by unscaling the columns (fix:
    # row-unscaling inflated the reported condition number ~4x for the
    # non-uniform marker weights, max/min 4.2 across the three obstacles).
    m_sym = (schur / weights[None, :])           # E A^{-1} E^T  (symmetric PSD)
    m_sym = 0.5 * (m_sym + m_sym.T)
    schur_cond = float(np.linalg.cond(ws[:, None] * m_sym * ws[None, :]))

    result = dict(trace_error=trace_err, schur_cond=schur_cond, n_markers=n_total,
                  timings=t, total_time=sum(t.values()))

    if want_field:
        inside = ibm._obstacle_mask(grid, obstacles)
        near = binary_dilation(inside, iterations=1)
        meas = (~inside) & (~near)
        u_ex = u_exact(grid)
        err = np.abs(u_h - u_ex)[meas]
        ref = np.sqrt(np.mean(u_ex[meas] ** 2))
        result["field_rel_l2"] = float(np.sqrt(np.mean(err ** 2)) / ref)
        result["field_linf"] = float(np.max(err))
        result["u_h"] = u_h
        result["inside"] = inside
        result["markers"] = markers
        result["weights"] = weights
    return result


# --------------------------------------------------------------------------
# Consecutive-triplet order (never a global least-squares fit).
# --------------------------------------------------------------------------
def triplet_orders(h_list, err_list, floor=2e-13):
    h = np.asarray(h_list, float)
    e = np.asarray(err_list, float)
    out = []
    for i in range(len(e) - 2):
        w = e[i:i + 3]
        if np.all(w > floor) and w[0] > w[1] > w[2]:
            p = np.polyfit(np.log(h[i:i + 3]), np.log(w), 1)[0]
            out.append(dict(grids=[i, i + 1, i + 2], order=float(p)))
    return out


def best_triplet_order(h_list, err_list, floor=2e-13):
    trips = triplet_orders(h_list, err_list, floor)
    if not trips:
        return None
    # the coarsest clean monotone triplet is the honest asymptotic rate
    return trips[0]["order"]


COMPONENTS = ["setup", "bulk_solve", "schur_assembly", "schur_solve",
              "reconstruct", "marker_eval"]


def timed_solve(route, npts, obstacles, n_markers, u_exact, repeats=3):
    """Run once for errors, then repeat for stable timing (median split)."""
    res = solve_frontier(route, npts, obstacles, n_markers, u_exact)
    totals = [res["total_time"]]
    splits = {c: [res["timings"][c]] for c in COMPONENTS}
    for _ in range(repeats - 1):
        r = solve_frontier(route, npts, obstacles, n_markers, u_exact,
                           want_field=False)
        totals.append(r["total_time"])
        for c in COMPONENTS:
            splits[c].append(r["timings"][c])
    res["total_time"] = float(np.median(totals))
    res["timings"] = {c: float(np.median(splits[c])) for c in COMPONENTS}
    res["time_repeats"] = totals
    return res


def marker_scaling(route, npts, obstacles, u_exact, marker_counts, repeats=3):
    """Fix the grid, vary the per-obstacle marker count, time Schur assembly and
    Schur solve, and fit the marker-count scaling exponents."""
    n_b, t_asm, t_slv = [], [], []
    for nm in marker_counts:
        asm, slv, tot = [], [], []
        for _ in range(repeats):
            r = solve_frontier(route, npts, obstacles, nm, u_exact,
                               want_field=False)
            asm.append(r["timings"]["schur_assembly"])
            slv.append(r["timings"]["schur_solve"])
        n_b.append(r["n_markers"])
        t_asm.append(float(np.median(asm)))
        t_slv.append(float(np.median(slv)))
    lg = np.log(np.asarray(n_b, float))
    p_asm = float(np.polyfit(lg, np.log(np.maximum(t_asm, 1e-9)), 1)[0])
    p_slv = float(np.polyfit(lg, np.log(np.maximum(t_slv, 1e-9)), 1)[0])
    return dict(route=route, npts=npts, n_b=n_b, t_assembly=t_asm,
                t_solve=t_slv, exp_assembly=p_asm, exp_solve=p_slv)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grids", type=str, default="")
    args = ap.parse_args()

    obstacles = ibm.standard_obstacles()
    u_exact = build_exact()

    if args.quick:
        grids = [32, 48, 64]
    elif args.grids:
        grids = [int(x) for x in args.grids.split(",")]
    else:
        grids = [48, 64, 80, 96, 112, 128, 144, 160, 192]

    route_names = ["BMC5", "C6+L6", "BMC7", "C8+L8", "BMC9", "C10+L10", "Fourier-PS"]

    # 1. adjointness verification
    adj = {}
    for rname in route_names:
        res, lhs, rhs = adjointness_residual(rname)
        adj[rname] = dict(rel_residual=res, lhs=lhs, rhs=rhs)
        print(f"[adjoint] {rname:10s} rel_residual={res:.3e}")

    # 2. convergence sweep (median-of-3 timing for stable wall time)
    conv = {r: dict(h=[], trace=[], field=[], field_linf=[], cond=[],
                    n_markers=[], total_time=[], timings=[]) for r in route_names}
    for npts in grids:
        h = TWO_PI / npts
        n_markers = max(48, int(round(np.pi / h)))
        for rname in route_names:
            res = timed_solve(rname, npts, obstacles, n_markers, u_exact)
            conv[rname]["h"].append(h)
            conv[rname]["trace"].append(res["trace_error"])
            conv[rname]["field"].append(res["field_rel_l2"])
            conv[rname]["field_linf"].append(res["field_linf"])
            conv[rname]["cond"].append(res["schur_cond"])
            conv[rname]["n_markers"].append(res["n_markers"])
            conv[rname]["total_time"].append(res["total_time"])
            conv[rname]["timings"].append(res["timings"])
            print(f"  N={npts:4d} {rname:10s} trace={res['trace_error']:.3e} "
                  f"field={res['field_rel_l2']:.3e} cond={res['schur_cond']:.2e} "
                  f"t={res['total_time']:.3f}s")

    # 3. observed orders per route (all clean consecutive triplets)
    orders = {}
    for rname in route_names:
        orders[rname] = dict(
            formal=ROUTES[rname]["order"],
            trace_triplets=triplet_orders(conv[rname]["h"], conv[rname]["trace"]),
            field_triplets=triplet_orders(conv[rname]["h"], conv[rname]["field"]),
            trace_order=best_triplet_order(conv[rname]["h"], conv[rname]["trace"]),
            field_order=best_triplet_order(conv[rname]["h"], conv[rname]["field"]),
        )

    # 4. total-cost ledger at two representative grids
    ledger = {}
    for npts in (128, 192):
        n_markers = max(48, int(round(np.pi / (TWO_PI / npts))))
        for rname in route_names:
            r = timed_solve(rname, npts, obstacles, n_markers, u_exact, repeats=5)
            ledger[f"{rname}@N{npts}"] = dict(
                route=rname, npts=npts, n_markers=r["n_markers"],
                total_time=r["total_time"], components=r["timings"],
                time_repeats=r["time_repeats"])

    # 5. marker-count scaling (fixed grid)
    mscale = {}
    for rname in ["BMC5", "C6+L6", "Fourier-PS"]:
        mscale[rname] = marker_scaling(rname, 96, obstacles, u_exact,
                                       marker_counts=[40, 80, 160, 240])

    # 6. timing-stability check: rerun one grid twice, independent processes' worth
    stab_route, stab_N = "BMC7", 128
    nm = max(48, int(round(np.pi / (TWO_PI / stab_N))))
    stab_a = timed_solve(stab_route, stab_N, obstacles, nm, u_exact, repeats=5)
    stab_b = timed_solve(stab_route, stab_N, obstacles, nm, u_exact, repeats=5)
    stability = dict(route=stab_route, npts=stab_N,
                     run_a=stab_a["total_time"], run_b=stab_b["total_time"],
                     repeats_a=stab_a["time_repeats"], repeats_b=stab_b["time_repeats"])
    print(f"[stability] {stab_route}@N{stab_N} run_a={stab_a['total_time']:.3f}s "
          f"run_b={stab_b['total_time']:.3f}s")

    # write cost_ledger.csv
    with open(OUTDIR / "cost_ledger.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["route", "N", "n_markers", "total_time_s"] + COMPONENTS)
        for key, d in ledger.items():
            w.writerow([d["route"], d["npts"], d["n_markers"],
                        f"{d['total_time']:.4f}"]
                       + [f"{d['components'][c]:.4f}" for c in COMPONENTS])
        w.writerow([])
        w.writerow(["marker-scaling (fixed grid): schur_assembly and schur_solve vs n_b"])
        w.writerow(["route", "N", "n_b", "t_assembly_s", "t_solve_s"])
        for rname, d in mscale.items():
            for i, nb in enumerate(d["n_b"]):
                w.writerow([rname, d["npts"], nb, f"{d['t_assembly'][i]:.4f}",
                            f"{d['t_solve'][i]:.4f}"])
        w.writerow(["exponents", "", "", "assembly", "solve"])
        for rname, d in mscale.items():
            w.writerow([rname, "", "", f"{d['exp_assembly']:.2f}", f"{d['exp_solve']:.2f}"])
    print("Wrote", OUTDIR / "cost_ledger.csv")

    payload = dict(
        machine=dict(cpu=_cpu_model(),
                     cpu_count=os.cpu_count(),
                     omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
                     platform=platform.platform()),
        source=dict(kind="near-singular fixed-width Gaussian", sigma=SIGMA,
                    kappa=KAPPA, sources=SOURCES.tolist(), coeffs=COEFFS.tolist()),
        grids=grids,
        routes={r: {k: v for k, v in ROUTES[r].items() if k != "color"}
                for r in route_names},
        matched_pairs=MATCHED_PAIRS,
        adjointness=adj,
        convergence=conv,
        orders=orders,
        cost_ledger=ledger,
        marker_scaling=mscale,
        timing_stability=stability,
        chen_peskin=dict(
            authors="Chen, Zhe and Peskin, Charles S.",
            title=("A Fourier spectral immersed boundary method with exact "
                   "translation invariance, improved boundary resolution, and a "
                   "divergence-free velocity field"),
            journal="Journal of Computational Physics", volume=509, article=113048,
            year=2024, doi="10.1016/j.jcp.2024.113048", arxiv="2302.08694",
            verified=True),
    )
    with open(OUTDIR / "results.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print("Wrote", OUTDIR / "results.json")
    return payload


if __name__ == "__main__":
    main()
