"""Frozen Gate 0 metrics for periodic first and second derivative symbols.

Every quantity follows results/exponential_family/gate0/contract.md. A scheme
is described by its symbols kappa_1 and kappa_2 in the normalization

    D1 symbol = (i/h) kappa_1(theta),   D2 symbol = (-1/h^2) kappa_2(theta),

together with the symbol L of its symmetric positive definite left side. The
relative errors are e_1 = kappa_1/theta - 1 and e_2 = kappa_2/theta^2 - 1.

Resolved conventions that the contract leaves open:

* The theta grid is theta_j = j pi / (P+1), j = 1..P with P = 20001, so the
  grid lies in (0, pi] and ends exactly at pi.
* Band membership uses theta <= band edge with a 1e-12 tolerance so that the
  edge point is never dropped by rounding.
* A reach is the first contiguous crossing with linear interpolation between
  the bracketing grid points, per the contract. Because that value inherits a
  grid dependence of order (grid spacing)^2 times the curvature of the error,
  a Brent refinement of the same bracket is also reported; the refined value is
  the one used against the frozen side research receipts.
* The formal order fit uses a log spaced window. The contract window is
  theta in [1e-3, 1e-1], but for schemes of order eight and above the relative
  error inside that window falls below the double precision floor, so the
  window is widened geometrically until at least eight sample points carry an
  error above 1e-13. The window that was actually used is reported with the
  fit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

THETA_POINTS = 20001
BANDS = {"B0": 2.0 * np.pi / 3.0, "B1": 0.75 * np.pi, "B2": 0.8 * np.pi}
DISCRETE_N = (64, 127, 128)
REACH_TOLS = (1e-2, 1e-3, 1e-4)
GV_TOL = 1e-2
GV_STEP = 1e-5
BAND_EPS = 1e-12
ORDER_WINDOW = (1e-3, 1e-1)
ORDER_FLOOR = 1e-13


def theta_grid(points: int = THETA_POINTS) -> np.ndarray:
    """Return the uniform grid of `points` angles on (0, pi] ending at pi."""
    return np.linspace(0.0, np.pi, points + 1)[1:]


def band_mask(theta: np.ndarray, edge: float) -> np.ndarray:
    """Return the mask of grid angles inside (0, edge]."""
    return theta <= edge + BAND_EPS


def band_max(theta: np.ndarray, err: np.ndarray, edge: float) -> float:
    """Return the band supremum of |e| on the grid."""
    return float(np.max(np.abs(err[band_mask(theta, edge)])))


def band_rms(theta: np.ndarray, err: np.ndarray, edge: float) -> float:
    """Return the band root mean square of e on the grid."""
    sel = err[band_mask(theta, edge)]
    return float(np.sqrt(np.mean(sel * sel)))


def reach(theta: np.ndarray, err: np.ndarray, tol: float, err_fn=None) -> dict:
    """Return the first contiguous crossing of |e| = tol on the grid.

    The linear value interpolates between the bracketing grid points. When an
    error callable is supplied the same bracket is refined by Brent's method,
    which removes the grid dependence of the crossing.
    """
    mag = np.abs(err)
    bad = np.flatnonzero(mag > tol)
    if bad.size == 0:
        return {"linear": float(theta[-1]), "refined": float(theta[-1]), "saturated": True}
    j = int(bad[0])
    if j == 0:
        return {"linear": 0.0, "refined": 0.0, "saturated": False}
    t0, t1 = float(theta[j - 1]), float(theta[j])
    e0, e1 = float(mag[j - 1]), float(mag[j])
    linear = t0 + (tol - e0) * (t1 - t0) / (e1 - e0)
    refined = linear
    if err_fn is not None:
        f = lambda t: float(np.abs(np.atleast_1d(err_fn(np.array([t])))[0])) - tol
        if f(t0) < 0.0 < f(t1):
            refined = float(brentq(f, t0, t1, xtol=1e-15, rtol=8.9e-16, maxiter=200))
    return {"linear": float(linear), "refined": float(refined), "saturated": False}


def fitted_order(theta: np.ndarray, err: np.ndarray,
                 window=ORDER_WINDOW, floor: float = ORDER_FLOOR) -> dict:
    """Return the log-log slope of |e| against theta on the fit window.

    Two effects can hide the leading truncation term inside the contract
    window. Published coefficients quoted to seven or eight digits do not
    satisfy their own order conditions exactly, which leaves a constant
    relative error plateau as theta goes to zero, and a compact second
    derivative symbol whose numerator vanishes like theta^2 loses relative
    accuracy like theta^-2 to cancellation. Both show up as the value of |e|
    at the lower edge of the window, so the fit mask discards everything below
    ten times that value. When the contract window then holds fewer than eight
    usable points the fit moves to the half decade that starts where |e| first
    rises above the mask, and the window that was used is reported.
    """
    lo, hi = window
    mag = np.abs(np.asarray(err, dtype=float))
    edge = int(np.argmin(np.abs(theta - lo)))
    plateau = float(mag[edge])
    thr = max(floor, 10.0 * plateau)
    usable = np.isfinite(mag) & (mag > thr)
    mask = (theta >= lo) & (theta <= hi) & usable
    if np.count_nonzero(mask) >= 8:
        slope = float(np.polyfit(np.log(theta[mask]), np.log(mag[mask]), 1)[0])
        return {"order": slope, "window": [lo, hi], "points": int(np.count_nonzero(mask)),
                "plateau": plateau}
    rising = np.flatnonzero(usable & (theta > lo))
    if rising.size == 0:
        return {"order": None, "window": [lo, hi], "points": 0, "plateau": plateau}
    lo = float(theta[rising[0]])
    hi = float(min(np.pi, 3.0 * lo))
    mask = (theta >= lo) & (theta <= hi) & usable
    if np.count_nonzero(mask) < 8:
        return {"order": None, "window": [lo, hi], "points": int(np.count_nonzero(mask)),
                "plateau": plateau}
    slope = float(np.polyfit(np.log(theta[mask]), np.log(mag[mask]), 1)[0])
    return {"order": slope, "window": [lo, hi], "points": int(np.count_nonzero(mask)),
            "plateau": plateau}


def symbol_cond(left, theta_points: int = THETA_POINTS) -> dict:
    """Return the continuous and discrete condition numbers of a left side."""
    theta = np.linspace(0.0, np.pi, theta_points)
    L = np.asarray(left(theta), dtype=float)
    out = {"cond": float(L.max() / L.min()), "min_left": float(L.min()),
           "max_left": float(L.max())}
    for N in DISCRETE_N:
        angles = 2.0 * np.pi * np.arange(N) / N
        Ln = np.asarray(left(angles), dtype=float)
        out[f"cond_N{N}"] = float(Ln.max() / Ln.min())
    return out


def kappa2_sign_check(kappa2) -> dict:
    """Return whether kappa_2(theta_l) >= 0 with equality only at l = 0."""
    out = {}
    for N in DISCRETE_N:
        angles = 2.0 * np.pi * np.arange(N) / N
        values = np.asarray(kappa2(angles), dtype=float)
        ok = bool(values[0] == 0.0 or abs(values[0]) < 1e-13)
        ok = ok and bool(np.all(values[1:] > 0.0))
        out[f"sign_N{N}"] = ok
        out[f"kappa2_min_nonzero_N{N}"] = float(values[1:].min())
    return out


def group_velocity(kappa1, theta: np.ndarray, analytic=None) -> tuple[np.ndarray, float]:
    """Return d kappa_1 / d theta and the stated error of the finite difference.

    The centered difference uses the step GV_STEP. Every symbol here is a
    smooth ratio of trigonometric polynomials with a positive denominator, so
    the difference may reach across theta = 0 and theta = pi without clipping.
    When an analytic derivative is supplied the returned error is the measured
    discrepancy between the two, otherwise it is the truncation estimate
    (GV_STEP^2/6) max |kappa_1'''| with the third derivative taken on the same
    grid by repeated centered differences.
    """
    h = GV_STEP
    fd = (np.asarray(kappa1(theta + h), dtype=float)
          - np.asarray(kappa1(theta - h), dtype=float)) / (2.0 * h)
    if analytic is not None:
        exact = np.asarray(analytic(theta), dtype=float)
        return exact, float(np.max(np.abs(exact - fd)))
    second = (np.asarray(kappa1(theta + 2.0 * h), dtype=float)
              - 2.0 * np.asarray(kappa1(theta), dtype=float)
              + np.asarray(kappa1(theta - 2.0 * h), dtype=float)) / (4.0 * h * h)
    third = float(np.max(np.abs(np.gradient(second, theta))))
    return fd, float(h * h * third / 6.0)


def evaluate(name: str, kappa1, kappa2, left, *,
             analytic_gv=None, coupled=None,
             algebraic_order1=None, algebraic_order2=None,
             bandwidth_left=None, bandwidth_d1=None, bandwidth_d2=None,
             solves=None, extra=None, theta_points: int = THETA_POINTS) -> dict:
    """Return every frozen Gate 0 metric for one scheme."""
    theta = theta_grid(theta_points)
    row: dict = {"name": name, "theta_points": int(theta_points)}
    if extra:
        row.update(extra)

    e1_fn = lambda t: np.asarray(kappa1(t), dtype=float) / np.asarray(t, dtype=float) - 1.0
    e1 = e1_fn(theta)
    for key, edge in BANDS.items():
        row[f"e1_max_{key}"] = band_max(theta, e1, edge)
        row[f"e1_rms_{key}"] = band_rms(theta, e1, edge)
    for tol in REACH_TOLS:
        res = reach(theta, e1, tol, err_fn=e1_fn)
        row[f"reach_d1_{tol:g}"] = res["linear"] / np.pi
        row[f"reach_d1_{tol:g}_refined"] = res["refined"] / np.pi
    row["rho1"] = float(np.max(np.abs(np.asarray(kappa1(theta), dtype=float))))
    row["rk4_hyperbolic"] = 2.828 / row["rho1"]
    fit1 = fitted_order(theta, e1)
    row["order_d1_fitted"] = fit1["order"]
    row["order_d1_window"] = fit1["window"]
    row["order_d1_points"] = fit1["points"]
    row["order_d1_plateau"] = fit1["plateau"]
    row["order_d1_algebraic"] = algebraic_order1

    gv, gv_err = group_velocity(kappa1, theta, analytic=analytic_gv)
    gv_e = gv - 1.0
    res = reach(theta, gv_e, GV_TOL)
    row["gv_reach_1e-2"] = res["linear"] / np.pi
    row["gv_error"] = gv_err

    if kappa2 is None:
        for key in BANDS:
            row[f"e2_max_{key}"] = None
            row[f"e2_rms_{key}"] = None
        for tol in REACH_TOLS:
            row[f"reach_d2_{tol:g}"] = None
            row[f"reach_d2_{tol:g}_refined"] = None
        row["rho2"] = None
        row["rk4_parabolic"] = None
        row["order_d2_fitted"] = None
        row["order_d2_window"] = None
        row["order_d2_points"] = None
        row["order_d2_plateau"] = None
        row["order_d2_algebraic"] = None
        for N in DISCRETE_N:
            row[f"sign_N{N}"] = None
    else:
        e2_fn = lambda t: (np.asarray(kappa2(t), dtype=float)
                           / np.asarray(t, dtype=float) ** 2 - 1.0)
        e2 = e2_fn(theta)
        for key, edge in BANDS.items():
            row[f"e2_max_{key}"] = band_max(theta, e2, edge)
            row[f"e2_rms_{key}"] = band_rms(theta, e2, edge)
        for tol in REACH_TOLS:
            res = reach(theta, e2, tol, err_fn=e2_fn)
            row[f"reach_d2_{tol:g}"] = res["linear"] / np.pi
            row[f"reach_d2_{tol:g}_refined"] = res["refined"] / np.pi
        row["rho2"] = float(np.max(np.asarray(kappa2(theta), dtype=float)))
        row["rk4_parabolic"] = 2.785 / row["rho2"]
        fit2 = fitted_order(theta, e2)
        row["order_d2_fitted"] = fit2["order"]
        row["order_d2_window"] = fit2["window"]
        row["order_d2_points"] = fit2["points"]
        row["order_d2_plateau"] = fit2["plateau"]
        row["order_d2_algebraic"] = algebraic_order2
        row.update(kappa2_sign_check(kappa2))

    row.update(symbol_cond(left, theta_points))

    if coupled is None:
        row["coupled"] = "no field"
        for key in BANDS:
            row[f"ec_max_{key}"] = None
        for tol in REACH_TOLS:
            row[f"reach_ec_{tol:g}"] = None
            row[f"reach_ec_{tol:g}_refined"] = None
    else:
        row["coupled"] = "field"
        ec = np.asarray(coupled(theta), dtype=float)
        for key, edge in BANDS.items():
            row[f"ec_max_{key}"] = band_max(theta, ec, edge)
        for tol in REACH_TOLS:
            res = reach(theta, ec, tol, err_fn=coupled)
            row[f"reach_ec_{tol:g}"] = res["linear"] / np.pi
            row[f"reach_ec_{tol:g}_refined"] = res["refined"] / np.pi

    row["bandwidth_left"] = bandwidth_left
    row["bandwidth_d1"] = bandwidth_d1
    row["bandwidth_d2"] = bandwidth_d2
    row["solves"] = solves
    return row
