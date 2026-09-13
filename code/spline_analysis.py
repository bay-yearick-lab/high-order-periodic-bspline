from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.interpolate import BSpline
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

RK4_IMAGINARY_AXIS_EXTENT = 2.828
RK4_NEGATIVE_REAL_AXIS_EXTENT = 2.785


@dataclass(frozen=True)
class CurveMetrics:
    fitted_order: float
    spectral_radius: float
    rk4_cfl: float
    omega_pi_1pct: float
    ppw_1pct: float
    omega_pi_5pct: float
    ppw_5pct: float


@dataclass(frozen=True)
class SecondDerivativeCurveMetrics:
    fitted_order: float
    spectral_radius: float
    rk4_parabolic_cfl: float
    omega_pi_1pct: float
    ppw_1pct: float
    omega_pi_5pct: float
    ppw_5pct: float


@lru_cache(maxsize=None)
def _basis_element(degree: int) -> BSpline:
    knots = np.arange(degree + 2, dtype=float)
    return BSpline.basis_element(knots, extrapolate=False)


@lru_cache(maxsize=None)
def _basis_element_derivative(degree: int) -> BSpline:
    return _basis_element(degree).derivative()


@lru_cache(maxsize=None)
def _basis_element_second_derivative(degree: int) -> BSpline:
    if degree < 2:
        raise ValueError("Centered cardinal B-spline second derivative requires degree >= 2.")
    return _basis_element(degree).derivative(2)


def centered_cardinal_bspline(degree: int, x: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    shifted = x + 0.5 * (degree + 1)
    return np.nan_to_num(_basis_element(degree)(shifted))


def centered_cardinal_bspline_derivative(
    degree: int, x: np.ndarray | list[float]
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    shifted = x + 0.5 * (degree + 1)
    return np.nan_to_num(_basis_element_derivative(degree)(shifted))


def centered_cardinal_bspline_second_derivative(
    degree: int, x: np.ndarray | list[float]
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    shifted = x + 0.5 * (degree + 1)
    return np.nan_to_num(_basis_element_second_derivative(degree)(shifted))


def explicit_centered_fd_weights(order: int) -> np.ndarray:
    if order < 2 or order % 2:
        raise ValueError("Centered FD order must be a positive even integer.")

    radius = order // 2
    offsets = np.arange(1, radius + 1, dtype=float)
    powers = 2 * np.arange(1, radius + 1) - 1
    system = np.zeros((radius, radius), dtype=float)
    rhs = np.zeros(radius, dtype=float)
    rhs[0] = 1.0

    for row, power in enumerate(powers):
        system[row, :] = 2.0 * offsets**power

    return np.linalg.solve(system, rhs)


def explicit_centered_fd_entries(order: int) -> tuple[np.ndarray, np.ndarray]:
    weights = explicit_centered_fd_weights(order)
    mass = np.array([1.0], dtype=float)
    advection = np.concatenate(([0.0], weights.astype(float)))
    return mass, advection


def explicit_centered_second_derivative_weights(order: int) -> np.ndarray:
    if order < 2 or order % 2:
        raise ValueError("Centered FD order must be a positive even integer.")

    radius = order // 2
    offsets = np.arange(1, radius + 1, dtype=float)
    powers = 2 * np.arange(1, radius + 1)
    system = np.zeros((radius, radius), dtype=float)
    rhs = np.zeros(radius, dtype=float)
    rhs[0] = 1.0

    for row, power in enumerate(powers):
        system[row, :] = offsets**power

    return np.linalg.solve(system, rhs)


def explicit_centered_second_derivative_entries(order: int) -> tuple[np.ndarray, np.ndarray]:
    weights = explicit_centered_second_derivative_weights(order)
    center = -2.0 * np.sum(weights)
    mass = np.array([1.0], dtype=float)
    second_derivative = np.concatenate(([center], weights.astype(float)))
    return mass, second_derivative


def _trim_trailing_zeros(entries: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    entries = np.asarray(entries, dtype=float)
    stop = len(entries)
    while stop > 1 and abs(entries[stop - 1]) <= tol:
        stop -= 1
    return entries[:stop].copy()


def _even_compact_symbol(w: np.ndarray, entries: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    entries = np.asarray(entries, dtype=float)
    symbol = np.full_like(w, entries[0], dtype=float)
    for offset in range(1, len(entries)):
        symbol += 2.0 * entries[offset] * np.cos(offset * w)
    return symbol


def _odd_compact_symbol(w: np.ndarray, entries: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    entries = np.asarray(entries, dtype=float)
    symbol = np.zeros_like(w)
    for offset in range(1, len(entries)):
        symbol += 2.0 * entries[offset] * np.sin(offset * w)
    return symbol


def _first_derivative_symbol_from_entries(
    w: np.ndarray, mass_entries: np.ndarray, advection_entries: np.ndarray
) -> np.ndarray:
    numerator = _odd_compact_symbol(w, advection_entries)
    denominator = _even_compact_symbol(w, mass_entries)
    return numerator / denominator


def _second_derivative_symbol_from_entries(
    w: np.ndarray, mass_entries: np.ndarray, second_derivative_entries: np.ndarray
) -> np.ndarray:
    numerator = _even_compact_symbol(w, second_derivative_entries)
    denominator = _even_compact_symbol(w, mass_entries)
    return -numerator / denominator


def kprime_exact(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, dtype=float)


def kprime_centered_fd(w: np.ndarray, order: int) -> np.ndarray:
    mass, advection = explicit_centered_fd_entries(order)
    return _first_derivative_symbol_from_entries(w, mass, advection)


def compact_scheme_entries(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order == 4:
        return (
            np.array([1.0, 1.0 / 4.0], dtype=float),
            np.array([0.0, 3.0 / 4.0], dtype=float),
        )
    if order == 6:
        return (
            np.array([1.0, 1.0 / 3.0], dtype=float),
            np.array([0.0, 7.0 / 9.0, 1.0 / 36.0], dtype=float),
        )
    if order == 8:
        return (
            np.array([1.0, 3.0 / 8.0], dtype=float),
            np.array([0.0, 25.0 / 32.0, 1.0 / 20.0, -1.0 / 480.0], dtype=float),
        )
    # Orders 10 and 12: the maximal-order tridiagonal members with minimal
    # right-hand-side radius (r = order/2 - 1), coefficients solved exactly
    # from the Taylor order conditions; the same exact solve reproduces the
    # built-in C4/C6/C8 fractions above.
    if order == 10:
        return (
            np.array([1.0, 2.0 / 5.0], dtype=float),
            np.array([0.0, 39.0 / 50.0, 1.0 / 15.0, -1.0 / 210.0,
                      1.0 / 4200.0], dtype=float),
        )
    if order == 12:
        return (
            np.array([1.0, 5.0 / 12.0], dtype=float),
            np.array([0.0, 7.0 / 9.0, 5.0 / 63.0, -5.0 / 672.0,
                      1.0 / 1512.0, -1.0 / 30240.0], dtype=float),
        )
    raise ValueError(
        "Tridiagonal compact first-derivative schemes are built in for "
        "orders 4, 6, 8, 10, and 12."
    )


def compact_second_derivative_scheme_entries(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order == 4:
        alpha = 1.0 / 10.0
        a = 6.0 / 5.0
        b = 0.0
        c = 0.0
    elif order == 6:
        alpha = 2.0 / 11.0
        a = 12.0 / 11.0
        b = 3.0 / 11.0
        c = 0.0
    else:
        raise ValueError(
            "Only the fourth- and sixth-order tridiagonal SLC second-derivative schemes are built in."
        )

    mass = np.array([1.0, alpha], dtype=float)
    second_derivative = _trim_trailing_zeros(
        np.array(
            [
                -(2.0 * a + 0.5 * b + (2.0 / 9.0) * c),
                a,
                0.25 * b,
                c / 9.0,
            ],
            dtype=float,
        )
    )
    return mass, second_derivative


def kprime_compact(w: np.ndarray, order: int) -> np.ndarray:
    mass, advection = compact_scheme_entries(order)
    return _first_derivative_symbol_from_entries(w, mass, advection)


def drp_tam_webb_entries() -> tuple[np.ndarray, np.ndarray]:
    """Seven-point Tam-Webb 1993 dispersion-relation-preserving stencil.

    Antisymmetric explicit scheme. Coefficients a_1..a_3 are the values
    that minimize integrated dispersion error from
    Tam and Webb, JCP 107, 1993.
    """
    a1 = 0.79926643
    a2 = -0.18941314
    a3 = 0.02651995
    mass = np.array([1.0], dtype=float)
    advection = np.array([0.0, a1, a2, a3], dtype=float)
    return mass, advection


def kprime_drp_tam_webb(w: np.ndarray) -> np.ndarray:
    mass, advection = drp_tam_webb_entries()
    return _first_derivative_symbol_from_entries(w, mass, advection)


def bogey_bailly_entries() -> tuple[np.ndarray, np.ndarray]:
    """Eleven-point optimized explicit stencil from Bogey and Bailly, JCP 194, 2004.

    The "FDo11p" coefficients tuned for low dispersion at high wavenumbers.
    """
    a1 = 0.872756993
    a2 = -0.286511173
    a3 = 0.090320001
    a4 = -0.020779405
    a5 = 0.002484594
    mass = np.array([1.0], dtype=float)
    advection = np.array([0.0, a1, a2, a3, a4, a5], dtype=float)
    return mass, advection


def kprime_bogey_bailly(w: np.ndarray) -> np.ndarray:
    mass, advection = bogey_bailly_entries()
    return _first_derivative_symbol_from_entries(w, mass, advection)


def kprime_pseudospectral(w: np.ndarray) -> np.ndarray:
    """Periodic Fourier pseudospectral first-derivative symbol.

    On the periodic uniform grid the FFT-based derivative is exact for every
    discrete wavenumber up to the Nyquist limit, so the modified wavenumber
    coincides with the continuous wavenumber.
    """
    return np.asarray(w, dtype=float)


def ksecond_pseudospectral(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w * w


def formal_order_drp_tam_webb() -> int:
    return 4


def formal_order_bogey_bailly() -> int:
    return 4


def formal_order_pseudospectral() -> int:
    return 0


def native_work_units(label: str) -> int:
    """Return a per-point flop estimate for the operator labelled `label`.

    Uses the antisymmetric/symmetric stencil counts that the periodic paper
    charges throughout. Implicit (Pade-type) operators count both the
    right-hand-side stencil and the band-solve work.
    """
    if label.startswith("DRP") and label[3:] in ("TamWebb", "TW"):
        return 7
    if label.startswith("BBp"):
        return 11
    if label.startswith("FFT") or label.startswith("PS"):
        return 0
    raise KeyError(f"native_work_units does not know label {label!r}")


def ksecond_exact(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w * w


def ksecond_centered_fd(w: np.ndarray, order: int) -> np.ndarray:
    mass, second_derivative = explicit_centered_second_derivative_entries(order)
    return _second_derivative_symbol_from_entries(w, mass, second_derivative)


def ksecond_compact(w: np.ndarray, order: int) -> np.ndarray:
    mass, second_derivative = compact_second_derivative_scheme_entries(order)
    return _second_derivative_symbol_from_entries(w, mass, second_derivative)


def collocation_integer_samples(degree: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(0, degree + 1, dtype=float)
    values = centered_cardinal_bspline(degree, offsets)
    derivs = centered_cardinal_bspline_derivative(degree, offsets)
    return values, derivs


def collocation_integer_second_derivative_samples(
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(0, degree + 1, dtype=float)
    values = centered_cardinal_bspline(degree, offsets)
    second_derivs = centered_cardinal_bspline_second_derivative(degree, offsets)
    return values, second_derivs


def compact_stencil_radius(values: np.ndarray, derivs: np.ndarray, tol: float = 1e-14) -> int:
    values = np.asarray(values, dtype=float)
    derivs = np.asarray(derivs, dtype=float)
    active = np.where((np.abs(values) > tol) | (np.abs(derivs) > tol))[0]
    if active.size == 0:
        return 0
    return int(active.max())


def collocation_mass_and_advection_entries(degree: int) -> tuple[np.ndarray, np.ndarray]:
    if degree < 2:
        raise ValueError(
            "Nodal B-spline collocation first derivatives require degree >= 2; "
            "the degree-1 spline derivative is not defined at the knots."
        )
    values, derivs = collocation_integer_samples(degree)
    radius = compact_stencil_radius(values, derivs)
    mass = values[: radius + 1].copy()
    advection = (-derivs[: radius + 1]).copy()
    advection[0] = 0.0
    return mass, advection


def collocation_mass_and_second_derivative_entries(
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    if degree < 2:
        raise ValueError("Direct collocation second derivatives require degree >= 2.")
    values, second_derivs = collocation_integer_second_derivative_samples(degree)
    radius = compact_stencil_radius(values, second_derivs)
    mass = values[: radius + 1].copy()
    second_derivs = second_derivs[: radius + 1].copy()
    return mass, second_derivs


def cyclic_compact_mass_matrix(entries: np.ndarray, npts: int) -> np.ndarray:
    entries = np.asarray(entries, dtype=float)
    radius = len(entries) - 1
    mat = np.zeros((npts, npts), dtype=float)
    for i in range(npts):
        mat[i, i] += entries[0]
        for offset in range(1, radius + 1):
            value = entries[offset]
            mat[i, (i + offset) % npts] += value
            mat[i, (i - offset) % npts] += value
    return mat


def cyclic_compact_advection_matrix(entries: np.ndarray, npts: int) -> np.ndarray:
    entries = np.asarray(entries, dtype=float)
    radius = len(entries) - 1
    mat = np.zeros((npts, npts), dtype=float)
    for i in range(npts):
        for offset in range(1, radius + 1):
            value = entries[offset]
            mat[i, (i + offset) % npts] += value
            mat[i, (i - offset) % npts] -= value
    return mat


def compact_derivative_matrix(
    mass_entries: np.ndarray, advection_entries: np.ndarray, npts: int, h: float = 1.0
) -> np.ndarray:
    mass = cyclic_compact_mass_matrix(mass_entries, npts)
    advection = cyclic_compact_advection_matrix(advection_entries, npts)
    return np.linalg.solve(mass, advection) / h


def compact_second_derivative_matrix(
    mass_entries: np.ndarray,
    second_derivative_entries: np.ndarray,
    npts: int,
    h: float = 1.0,
) -> np.ndarray:
    mass = cyclic_compact_mass_matrix(mass_entries, npts)
    second_derivative = cyclic_compact_mass_matrix(second_derivative_entries, npts)
    return np.linalg.solve(mass, second_derivative) / (h * h)


def collocation_derivative_matrix(degree: int, npts: int, h: float = 1.0) -> np.ndarray:
    mass, advection = collocation_mass_and_advection_entries(degree)
    return compact_derivative_matrix(mass, advection, npts, h=h)


def collocation_second_derivative_matrix(
    degree: int, npts: int, h: float = 1.0
) -> np.ndarray:
    mass, second_derivs = collocation_mass_and_second_derivative_entries(degree)
    return compact_second_derivative_matrix(mass, second_derivs, npts, h=h)


def kprime_collocation(w: np.ndarray, degree: int) -> np.ndarray:
    mass, advection = collocation_mass_and_advection_entries(degree)
    return _first_derivative_symbol_from_entries(w, mass, advection)


def ksecond_collocation(w: np.ndarray, degree: int) -> np.ndarray:
    mass, second_derivative = collocation_mass_and_second_derivative_entries(degree)
    return _second_derivative_symbol_from_entries(w, mass, second_derivative)


def iga_mass_and_advection_entries(degree: int) -> tuple[np.ndarray, np.ndarray]:
    collocation_degree = 2 * degree + 1
    offsets = np.arange(0, degree + 1, dtype=float)
    mass = centered_cardinal_bspline(collocation_degree, offsets)
    advection = -centered_cardinal_bspline_derivative(collocation_degree, offsets)
    advection[0] = 0.0
    return mass, advection


def iga_mass_and_second_derivative_entries(degree: int) -> tuple[np.ndarray, np.ndarray]:
    if degree < 1:
        raise ValueError("Direct IGA second derivatives require degree >= 1.")
    collocation_degree = 2 * degree + 1
    offsets = np.arange(0, degree + 1, dtype=float)
    mass = centered_cardinal_bspline(collocation_degree, offsets)
    second_derivs = centered_cardinal_bspline_second_derivative(
        collocation_degree, offsets
    )
    return mass, second_derivs


def quadrature_iga_entries(degree: int) -> tuple[np.ndarray, np.ndarray]:
    radius = degree
    mass = np.zeros(radius + 1)
    advection = np.zeros(radius + 1)
    lo = -(degree + 2)
    hi = degree + 2
    breakpoints = np.arange(lo, hi + 0.5, 0.5)

    for offset in range(radius + 1):
        def _mass(x: float, shift: int = offset) -> float:
            return (
                centered_cardinal_bspline(degree, np.array([x]))[0]
                * centered_cardinal_bspline(degree, np.array([x - shift]))[0]
            )

        def _advection(x: float, shift: int = offset) -> float:
            return (
                centered_cardinal_bspline(degree, np.array([x]))[0]
                * centered_cardinal_bspline_derivative(degree, np.array([x - shift]))[0]
            )

        mass[offset], _ = integrate.quad(
            _mass, lo, hi, limit=400, points=breakpoints
        )
        advection[offset], _ = integrate.quad(
            _advection, lo, hi, limit=400, points=breakpoints
        )

    return mass, advection


def kprime_iga(w: np.ndarray, degree: int) -> np.ndarray:
    mass, advection = iga_mass_and_advection_entries(degree)
    return _first_derivative_symbol_from_entries(w, mass, advection)


def ksecond_iga(w: np.ndarray, degree: int) -> np.ndarray:
    mass, second_derivative = iga_mass_and_second_derivative_entries(degree)
    return _second_derivative_symbol_from_entries(w, mass, second_derivative)


def iga_derivative_matrix(degree: int, npts: int, h: float = 1.0) -> np.ndarray:
    mass, advection = iga_mass_and_advection_entries(degree)
    return compact_derivative_matrix(mass, advection, npts, h=h)


def iga_second_derivative_matrix(degree: int, npts: int, h: float = 1.0) -> np.ndarray:
    mass, second_derivs = iga_mass_and_second_derivative_entries(degree)
    return compact_second_derivative_matrix(mass, second_derivs, npts, h=h)


def periodic_sbp_norm(npts: int, h: float = 1.0) -> np.ndarray:
    return h * np.eye(npts)


def periodic_sbp_residual(D: np.ndarray, h: float = 1.0) -> np.ndarray:
    derivative = np.asarray(D, dtype=float)
    norm = periodic_sbp_norm(derivative.shape[0], h=h)
    return norm @ derivative + derivative.T @ norm


def _relative_symbol_error(exact: np.ndarray, numerical: np.ndarray) -> np.ndarray:
    exact = np.asarray(exact, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    return np.abs(numerical / (exact + 1e-30) - 1.0)


def relative_dispersion_error(w: np.ndarray, kph: np.ndarray) -> np.ndarray:
    return _relative_symbol_error(kprime_exact(w), kph)


def relative_second_derivative_error(w: np.ndarray, k2h2: np.ndarray) -> np.ndarray:
    return _relative_symbol_error(ksecond_exact(w), k2h2)


def _estimate_order_from_error(
    w: np.ndarray, err: np.ndarray, wmin: float = 0.05, wmax: float = 0.8
) -> float:
    w = np.asarray(w, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = (w >= wmin) & (w <= wmax) & (err > 1e-20)
    if np.count_nonzero(mask) < 3:
        mask = (w >= 1e-4) & (w <= max(wmax, 0.3)) & (err > 1e-22)
    if np.count_nonzero(mask) < 3:
        mask = err > 0.0
    coeffs = np.polyfit(np.log(w[mask]), np.log(err[mask]), 1)
    return float(coeffs[0])


def estimate_order(w: np.ndarray, kph: np.ndarray, wmin: float = 0.05, wmax: float = 0.8) -> float:
    err = relative_dispersion_error(w, kph)
    return _estimate_order_from_error(w, err, wmin=wmin, wmax=wmax)


def estimate_second_derivative_order(
    w: np.ndarray, k2h2: np.ndarray, wmin: float = 0.05, wmax: float = 0.8
) -> float:
    err = relative_second_derivative_error(w, k2h2)
    return _estimate_order_from_error(w, err, wmin=wmin, wmax=wmax)


def spectral_radius(kph: np.ndarray) -> float:
    return float(np.max(np.abs(kph)))


def rk4_cfl_limit(kph: np.ndarray) -> float:
    return RK4_IMAGINARY_AXIS_EXTENT / spectral_radius(kph)


def rk4_parabolic_cfl_limit(k2h2: np.ndarray) -> float:
    return RK4_NEGATIVE_REAL_AXIS_EXTENT / spectral_radius(k2h2)


def _resolution_limit_from_error(
    w: np.ndarray, err: np.ndarray, tol: float = 1e-2
) -> tuple[float, float]:
    valid = np.where(err <= tol)[0]
    idx = valid[-1]
    omega = float(w[idx])
    return omega / np.pi, 2.0 * np.pi / omega


def resolution_limit(
    w: np.ndarray, kph: np.ndarray, tol: float = 1e-2
) -> tuple[float, float]:
    err = relative_dispersion_error(w, kph)
    return _resolution_limit_from_error(w, err, tol=tol)


def second_derivative_resolution_limit(
    w: np.ndarray, k2h2: np.ndarray, tol: float = 1e-2
) -> tuple[float, float]:
    err = relative_second_derivative_error(w, k2h2)
    return _resolution_limit_from_error(w, err, tol=tol)


def summarize_curve(w: np.ndarray, kph: np.ndarray) -> CurveMetrics:
    omega_pi_1pct, ppw_1pct = resolution_limit(w, kph, tol=1e-2)
    omega_pi_5pct, ppw_5pct = resolution_limit(w, kph, tol=5e-2)
    return CurveMetrics(
        fitted_order=estimate_order(w, kph),
        spectral_radius=spectral_radius(kph),
        rk4_cfl=rk4_cfl_limit(kph),
        omega_pi_1pct=omega_pi_1pct,
        ppw_1pct=ppw_1pct,
        omega_pi_5pct=omega_pi_5pct,
        ppw_5pct=ppw_5pct,
    )


def summarize_second_derivative_curve(
    w: np.ndarray, k2h2: np.ndarray
) -> SecondDerivativeCurveMetrics:
    omega_pi_1pct, ppw_1pct = second_derivative_resolution_limit(w, k2h2, tol=1e-2)
    omega_pi_5pct, ppw_5pct = second_derivative_resolution_limit(w, k2h2, tol=5e-2)
    return SecondDerivativeCurveMetrics(
        fitted_order=estimate_second_derivative_order(w, k2h2),
        spectral_radius=spectral_radius(k2h2),
        rk4_parabolic_cfl=rk4_parabolic_cfl_limit(k2h2),
        omega_pi_1pct=omega_pi_1pct,
        ppw_1pct=ppw_1pct,
        omega_pi_5pct=omega_pi_5pct,
        ppw_5pct=ppw_5pct,
    )


def rk4_stability_boundary(npts: int = 500) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, npts)
    boundary = np.zeros(npts, dtype=complex)

    for idx, angle in enumerate(theta):
        direction = np.exp(1j * angle)
        radius = 2.0
        for _ in range(200):
            z = radius * direction
            g = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
            gp = direction + radius * direction**2 + radius**2 * direction**3 / 2.0
            gp += radius**3 * direction**4 / 6.0
            f = np.abs(g) ** 2 - 1.0
            fp = 2.0 * np.real(np.conj(g) * gp)
            if abs(fp) < 1e-15:
                break
            radius -= f / fp
            if radius < 0.0:
                radius = 0.01
        boundary[idx] = radius * direction

    return boundary


def formal_order_explicit(order: int) -> int:
    return order


def formal_order_compact(order: int) -> int:
    return order


def formal_order_collocation(degree: int) -> int:
    if degree < 2:
        raise ValueError(
            "The BMC first-derivative order is defined only for degree >= 2."
        )
    return degree if degree % 2 == 0 else degree + 1


def formal_order_iga(degree: int) -> int:
    return 2 * degree + 2


def cyclic_sparse_compact_mass_matrix(entries: np.ndarray, npts: int) -> csc_matrix:
    entries = np.asarray(entries, dtype=float)
    radius = len(entries) - 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(npts):
        rows.append(i)
        cols.append(i)
        data.append(float(entries[0]))
        for offset in range(1, radius + 1):
            value = float(entries[offset])
            if abs(value) < 1e-15:
                continue
            rows.extend([i, i])
            cols.extend([(i + offset) % npts, (i - offset) % npts])
            data.extend([value, value])

    return csc_matrix((data, (rows, cols)), shape=(npts, npts))


def cyclic_sparse_compact_advection_matrix(entries: np.ndarray, npts: int) -> csc_matrix:
    entries = np.asarray(entries, dtype=float)
    radius = len(entries) - 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(npts):
        for offset in range(1, radius + 1):
            value = float(entries[offset])
            if abs(value) < 1e-15:
                continue
            rows.extend([i, i])
            cols.extend([(i + offset) % npts, (i - offset) % npts])
            data.extend([value, -value])

    return csc_matrix((data, (rows, cols)), shape=(npts, npts))


def factorize_cyclic_compact_mass(entries: np.ndarray, npts: int):
    return splu(cyclic_sparse_compact_mass_matrix(entries, npts))


def cyclic_collocation_basis_matrix(degree: int, npts: int) -> csc_matrix:
    """Return the periodic basis matrix B with B[i,j] = beta_p(i - j).

    On a periodic uniform grid with spacing h = 1, the spline expansion
    u(x_i) = sum_j alpha_j * beta_p(x_i - j) collapses to u = B alpha, where B
    is circulant with bandwidth 2*ceil((degree+1)/2)+1 because the centered
    cardinal B-spline has support [-(p+1)/2, (p+1)/2].
    """
    radius = (degree + 1) // 2
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = centered_cardinal_bspline(degree, offsets)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(npts):
        for off, w in zip(offsets, weights):
            if abs(w) < 1e-15:
                continue
            j = (i + int(off)) % npts
            rows.append(i)
            cols.append(j)
            data.append(float(w))
    return csc_matrix((data, (rows, cols)), shape=(npts, npts))


def factorize_cyclic_collocation_basis(degree: int, npts: int):
    return splu(cyclic_collocation_basis_matrix(degree, npts))


def evaluate_periodic_spline(
    coefficients: np.ndarray,
    x_eval: np.ndarray,
    h: float,
    degree: int,
    *,
    derivative: int = 0,
) -> np.ndarray:
    """Evaluate the periodic spline u(x) = sum_j alpha_j * beta_p((x - jh)/h).

    The kernel is the centered cardinal B-spline, so basis function j is
    centered on the grid node x_j = j*h, on a domain of length L = npts*h.
    Evaluation handles arbitrary off-grid `x_eval` via periodic wrap-around.
    `derivative` selects between value, first, or second derivative.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    npts = coefficients.shape[0]
    L = npts * h
    radius = (degree + 1) / 2.0

    if derivative == 0:
        kernel = centered_cardinal_bspline
        scale = 1.0
    elif derivative == 1:
        kernel = centered_cardinal_bspline_derivative
        scale = 1.0 / h
    elif derivative == 2:
        kernel = centered_cardinal_bspline_second_derivative
        scale = 1.0 / (h * h)
    else:
        raise ValueError("Only derivatives 0, 1, 2 are supported.")

    x_local = (x_eval / h) % npts
    j_lo = np.floor(x_local - radius).astype(int)
    j_hi = np.floor(x_local + radius).astype(int) + 1
    output = np.zeros_like(x_local, dtype=float)

    half_npts = npts / 2.0
    for offset in range(int(radius * 2 + 2)):
        j = j_lo + offset
        mask = j <= j_hi
        if not np.any(mask):
            continue
        delta = x_local - j
        delta = np.where(delta > half_npts, delta - npts, delta)
        delta = np.where(delta < -half_npts, delta + npts, delta)
        weights = kernel(degree, delta)
        idx = j % npts
        output += np.where(mask, coefficients[idx] * weights, 0.0) * scale
    return output


def coefficients_from_nodal_values(
    values: np.ndarray, degree: int
) -> np.ndarray:
    """Solve B alpha = u for the periodic spline coefficient vector alpha.

    Uses a banded periodic solve. For odd degrees the cardinal sample
    centered on a node carries B[i,i] = beta_p(0) > 0 and the rest is
    diagonally dominant; for even degrees the symmetric stencil is the same
    matrix that appears in the BMC and BMG mass operators.
    """
    values = np.asarray(values, dtype=float)
    npts = values.shape[0]
    solver = factorize_cyclic_collocation_basis(degree, npts)
    return solver.solve(values)


# ---------------------------------------------------------------------------
# Exponential-family member routes.
#
# A member file holds the one-sided lattice samples beta(n), beta'(n),
# beta''(n) for n = 0..R of one generator E_{m,r}(omega). beta is even, beta'
# is odd and beta'' is even, so those arrays are exactly the compact-scheme
# entries this module already speaks:
#
#   mass entries              = beta(n),
#   advection entries         = -beta'(n)  with the center forced to zero,
#   second-derivative entries = beta''(n).
#
# The advection sign matches collocation_mass_and_advection_entries: the
# right-side matrix places +entries[o] at column i+o, while the stencil built
# from the kernel places beta'(-o) = -beta'(o) there.
#
# A member is addressed either by the literal label "EXP:<path>" or by a short
# name previously bound with register_exp_member(name, path). Every existing
# label is untouched; the "EXP:" test runs before the explicit-FD "E<order>"
# test, which would otherwise capture it.
# ---------------------------------------------------------------------------

EXP_LABEL_PREFIX = "EXP:"
_EXP_MEMBER_REGISTRY: dict[str, str] = {}


def register_exp_member(label: str, json_path) -> str:
    """Bind a short label to an exponential-family member JSON file."""
    if label.startswith(EXP_LABEL_PREFIX):
        raise ValueError(
            f"A registered label must not use the reserved prefix {EXP_LABEL_PREFIX!r}."
        )
    _EXP_MEMBER_REGISTRY[label] = str(Path(json_path))
    return label


def registered_exp_members() -> dict[str, str]:
    """Return a copy of the label to path bindings."""
    return dict(_EXP_MEMBER_REGISTRY)


def is_exp_member_label(label: str) -> bool:
    """Return whether the label addresses an exponential-family member."""
    return label.startswith(EXP_LABEL_PREFIX) or label in _EXP_MEMBER_REGISTRY


def exp_member_path(label: str) -> str:
    """Resolve a member label to the path of its JSON file."""
    if label.startswith(EXP_LABEL_PREFIX):
        return label[len(EXP_LABEL_PREFIX):]
    if label in _EXP_MEMBER_REGISTRY:
        return _EXP_MEMBER_REGISTRY[label]
    raise ValueError(f"Unknown exponential-family member label {label!r}")


@lru_cache(maxsize=None)
def _load_exp_member(path: str) -> tuple:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    radius = int(record["radius"])
    beta = np.asarray(record["beta"], dtype=float)
    dbeta = np.asarray(record["dbeta"], dtype=float)
    ddbeta = np.asarray(record["ddbeta"], dtype=float)
    for name, array in (("beta", beta), ("dbeta", dbeta), ("ddbeta", ddbeta)):
        if array.shape != (radius + 1,):
            raise ValueError(
                f"Member {path!r} field {name!r} must hold radius+1 one-sided samples."
            )
    if beta[0] <= 0.0:
        raise ValueError(f"Member {path!r} must have beta(0) > 0.")
    return (
        str(record.get("name", Path(path).stem)),
        int(record.get("p", 2 * radius - 1)),
        radius,
        tuple(beta),
        tuple(dbeta),
        tuple(ddbeta),
    )


def exp_member_record(label: str) -> dict:
    """Return the member samples addressed by the label."""
    name, degree, radius, beta, dbeta, ddbeta = _load_exp_member(exp_member_path(label))
    return {
        "name": name,
        "p": degree,
        "radius": radius,
        "beta": np.asarray(beta, dtype=float),
        "dbeta": np.asarray(dbeta, dtype=float),
        "ddbeta": np.asarray(ddbeta, dtype=float),
    }


def exp_member_mass_and_advection_entries(label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the left-side and first-derivative right-side one-sided entries."""
    record = exp_member_record(label)
    mass = record["beta"].copy()
    advection = (-record["dbeta"]).copy()
    advection[0] = 0.0
    return mass, advection


def exp_member_mass_and_second_derivative_entries(
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the left-side and second-derivative right-side one-sided entries."""
    record = exp_member_record(label)
    return record["beta"].copy(), record["ddbeta"].copy()


def exp_member_derivative_matrix(label: str, npts: int, h: float = 1.0) -> np.ndarray:
    mass, advection = exp_member_mass_and_advection_entries(label)
    return compact_derivative_matrix(mass, advection, npts, h=h)


def exp_member_second_derivative_matrix(
    label: str, npts: int, h: float = 1.0
) -> np.ndarray:
    mass, second = exp_member_mass_and_second_derivative_entries(label)
    return compact_second_derivative_matrix(mass, second, npts, h=h)


def kprime_exp_member(w: np.ndarray, label: str) -> np.ndarray:
    mass, advection = exp_member_mass_and_advection_entries(label)
    return _first_derivative_symbol_from_entries(w, mass, advection)


def ksecond_exp_member(w: np.ndarray, label: str) -> np.ndarray:
    mass, second = exp_member_mass_and_second_derivative_entries(label)
    return _second_derivative_symbol_from_entries(w, mass, second)


def first_derivative_matrix_by_label(label: str, npts: int, h: float = 1.0) -> np.ndarray:
    """Assembled periodic first derivative for any catalog or baseline label.

    Labels: BMC<p>, BMG<p>, C<order> (compact Pade), E<order> (explicit
    centered), DRP (Tam--Webb), and exponential-family members addressed as
    "EXP:<path>" or by a name bound with register_exp_member. One dispatch
    shared by every audit so the baselines run through the identical machinery
    as the spline routes.
    """
    if is_exp_member_label(label):
        return exp_member_derivative_matrix(label, npts, h=h)
    if label.startswith("BMC"):
        return collocation_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("BMG"):
        return iga_derivative_matrix(int(label[3:]), npts, h=h)
    if label == "DRP":
        mass, advection = drp_tam_webb_entries()
        return compact_derivative_matrix(mass, advection, npts, h=h)
    if label.startswith("C"):
        mass, advection = compact_scheme_entries(int(label[1:]))
        return compact_derivative_matrix(mass, advection, npts, h=h)
    if label.startswith("E"):
        mass, advection = explicit_centered_fd_entries(int(label[1:]))
        return compact_derivative_matrix(mass, advection, npts, h=h)
    raise ValueError(f"Unknown scheme label {label!r}")


def first_derivative_symbol_by_label(label: str, w: np.ndarray) -> np.ndarray:
    """Modified-wavenumber symbol matching first_derivative_matrix_by_label."""
    if is_exp_member_label(label):
        return kprime_exp_member(w, label)
    if label.startswith("BMC"):
        return kprime_collocation(w, int(label[3:]))
    if label.startswith("BMG"):
        return kprime_iga(w, int(label[3:]))
    if label == "DRP":
        return kprime_drp_tam_webb(w)
    if label.startswith("C"):
        return kprime_compact(w, int(label[1:]))
    if label.startswith("E"):
        return kprime_centered_fd(w, int(label[1:]))
    raise ValueError(f"Unknown scheme label {label!r}")
