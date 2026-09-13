"""Pointwise evaluator for the exponential cardinal B-spline generator.

The member E_{m,r}(omega) carries the exponent multiset

    Lambda = (0^m, +i omega_1, -i omega_1, ..., +i omega_r, -i omega_r),

support [-n/2, n/2] with n = m + 2 r, and degree p = n - 1. This module
produces phi(x), phi'(x) and phi''(x) at arbitrary real x by the localized
Green-function construction of the Gate 5 contract.

Construction
------------

Let L = prod_{lambda in Lambda} (D - lambda) = D^m prod_j (D^2 + omega_j^2)
and let q be its causal Green function, so L q = delta and q(t) = 0 for
t < 0. Write D(s) = s^m prod_j (s^2 + omega_j^2) for the symbol of L. The
partial fraction expansion of 1/D(s) gives, for t >= 0,

    q(t) = P(t) + sum_j 2 Re[ exp(i omega_j t) sum_{a<mu_j} kappa_{j,a} t^a ],

with P a polynomial of degree m - 1 and mu_j the multiplicity of the root
i omega_j. For a root rho of multiplicity mu the coefficients come from the
truncated series of u^mu / D(rho + u), that is from inverting the shifted
denominator, which covers the simple and the confluent case with the same
code path.

The causal centered factor of exponent lambda is b_lambda(x - 1/2) with
b_lambda(x) = c_lambda exp(lambda x) 1_{[-1/2,1/2]}(x) and
c_lambda = lambda / (2 sinh(lambda/2)), and it equals
(1 - exp(lambda) S) applied to the Green function of D - lambda, where S is
the unit right shift. Convolving the n factors gives

    phi(x) = nu * sum_{l=0}^{n} A_l q(x + n/2 - l),
    A(z)   = (1 - z)^m prod_j (1 - 2 cos(omega_j) z + z^2),
    nu     = prod_j 1 / sinc(omega_j / 2)^2,

which is the localization polynomial and the normalization of the contract.
The normalization is the product of the c_lambda over the multiset, in which
a conjugate pair contributes omega^2 / (4 sin^2(omega/2)) and every zero
exponent contributes one. It reproduces phi_hat(0) = 1, hence the integer
samples of family.py.

Cancellation-safe cell form
---------------------------

A(S) annihilates every polynomial of degree below m and every sequence
exp(+- i omega_j t), so A(S) q vanishes identically when q is replaced by its
analytic continuation Q(t) = P(t) + oscillatory part, valid on all of R.
Therefore the truncated sum over the terms with x + n/2 - l >= 0 equals minus
the truncated sum over the terms with x + n/2 - l < 0, and the evaluator uses
whichever of the two carries fewer terms. On the cell y = x + n/2 in
[J, J+1), with xi = y - J, this yields the exponential polynomial

    phi(x) / nu = sum_k a_k xi^k
                + sum_{j,b} xi^b [ C_{j,b} cos(omega_j xi) + S_{j,b} sin(omega_j xi) ],

whose coefficients depend only on the cell index J. The cubic Taylor jet of
each oscillatory term at the cell anchor xi = 0 is folded into the polynomial
part, which leaves the stable remainders

    C_2(z) = 1 - cos z,        S_3(z) = z - sin z,
    C_4(z) = cos z - 1 + z^2/2, S_5(z) = sin z - z + z^3/6,

evaluated by their series near zero, and the identities dS_5/dz = C_4,
dC_4/dz = S_3, dS_3/dz = C_2, dC_2/dz = sin z give the first and second
derivatives with the same four functions:

    phi   / nu = poly(xi)   + sum xi^b [ C C_4(w xi) + S S_5(w xi) ],
    phi'  / nu = poly'(xi)  + sum w xi^b [ C S_3(w xi) + S C_4(w xi) ] + (b terms),
    phi'' / nu = poly''(xi) + sum w^2 xi^b [ C C_2(w xi) + S S_3(w xi) ] + (b terms),

the b terms being the Leibniz contributions of the factor xi^b, which are
absent for a simple pair.

Conditioning and the two documented tolerances
----------------------------------------------

The cell coefficients are assembled once, in extended precision
(numpy.longdouble), because the truncated sum over l cancels the polynomial
growth of Q against itself. The ratio of the largest term entering that
assembly to max |phi| is reported as `conditioning`; it grows like
omega^{-(m+1)} as a frequency approaches zero, which is the same statement as
the loss of independence of {1, xi, ..., xi^{m-1}, cos(omega xi),
sin(omega xi)} in that limit.

Two limits are handled by a documented tolerance rather than by a separate
formula, exactly as the contract allows:

* a frequency with omega <= ZERO_FREQUENCY_TOLERANCE is replaced by two extra
  zero exponents, that is E_{m,r}(..., omega, ...) is evaluated as
  E_{m+2,r-1}(...). This is the exact omega -> 0 limit of the member and the
  substitution perturbs the generator by O(omega^2);
* frequencies within CONFLUENT_TOLERANCE of each other are merged into one
  frequency of higher multiplicity, taken at their mean. The confluent
  multiplicity is carried through the residue series, so the exactly repeated
  pair is evaluated by its own formula and not by a difference of nearly
  equal simple residues.

The wider multisets of Gate 7 (real pairs +-alpha and quads +-alpha +- i omega)
and the polynomial dial of Gate 8 live in the sibling module
general_evaluator.py, which carries the same construction through complex
exponents. `as_evaluator` dispatches to it on the generator type; the Gate 0
path of this module is untouched.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

ZERO_FREQUENCY_TOLERANCE = 1e-7
CONFLUENT_TOLERANCE = 1e-7
SERIES_LIMIT = 4.0
SERIES_TERMS = 26

_LD = np.longdouble
_CLD = np.clongdouble


def _series_C4(z: np.ndarray) -> np.ndarray:
    """Return cos z - 1 + z^2/2 by the series that starts at z^4/24."""
    z2 = z * z
    term = z2 * z2 / 24.0
    total = term.copy()
    for k in range(2, SERIES_TERMS):
        term = -term * z2 / ((2 * k + 1) * (2 * k + 2))
        total = total + term
    return total


def _series_S5(z: np.ndarray) -> np.ndarray:
    """Return sin z - z + z^3/6 by the series that starts at z^5/120."""
    z2 = z * z
    term = z2 * z2 * z / 120.0
    total = term.copy()
    for k in range(2, SERIES_TERMS):
        term = -term * z2 / ((2 * k + 2) * (2 * k + 3))
        total = total + term
    return total


def _series_S3(z: np.ndarray) -> np.ndarray:
    """Return z - sin z by the series that starts at z^3/6."""
    z2 = z * z
    term = z2 * z / 6.0
    total = term.copy()
    for k in range(1, SERIES_TERMS):
        term = -term * z2 / ((2 * k + 2) * (2 * k + 3))
        total = total + term
    return total


def remainder_C2(z: np.ndarray) -> np.ndarray:
    """Return 1 - cos z as 2 sin^2(z/2), which never cancels."""
    s = np.sin(0.5 * np.asarray(z, dtype=float))
    return 2.0 * s * s


def remainder_S3(z: np.ndarray) -> np.ndarray:
    """Return z - sin z, by series below SERIES_LIMIT and directly above it."""
    z = np.asarray(z, dtype=float)
    out = z - np.sin(z)
    small = np.abs(z) <= SERIES_LIMIT
    if np.any(small):
        out = np.where(small, _series_S3(np.where(small, z, 0.0)), out)
    return out


def remainder_C4(z: np.ndarray) -> np.ndarray:
    """Return cos z - 1 + z^2/2, by series below SERIES_LIMIT."""
    z = np.asarray(z, dtype=float)
    out = 0.5 * z * z - remainder_C2(z)
    small = np.abs(z) <= SERIES_LIMIT
    if np.any(small):
        out = np.where(small, _series_C4(np.where(small, z, 0.0)), out)
    return out


def remainder_S5(z: np.ndarray) -> np.ndarray:
    """Return sin z - z + z^3/6, by series below SERIES_LIMIT."""
    z = np.asarray(z, dtype=float)
    out = np.sin(z) - z + z * z * z / 6.0
    small = np.abs(z) <= SERIES_LIMIT
    if np.any(small):
        out = np.where(small, _series_S5(np.where(small, z, 0.0)), out)
    return out


def _remainder_chain(z: np.ndarray, base: int, order: int) -> np.ndarray:
    """Return the order-th derivative in z of C_4 (base 0) or S_5 (base 1).

    The chain of derivatives is S_5 -> C_4 -> S_3 -> C_2 -> sin -> cos ->
    -sin -> -cos -> sin, so index j = order + base counted from S_5 selects
    the function, and beyond index four the plain circular functions apply.
    """
    j = order + (1 - base)
    if j == 0:
        return remainder_S5(z)
    if j == 1:
        return remainder_C4(z)
    if j == 2:
        return remainder_S3(z)
    if j == 3:
        return remainder_C2(z)
    phase = (j - 4) % 4
    if phase == 0:
        return np.sin(z)
    if phase == 1:
        return np.cos(z)
    if phase == 2:
        return -np.sin(z)
    return -np.cos(z)


def _poly_shift_series(den: np.ndarray, rho) -> np.ndarray:
    """Return the ascending Taylor coefficients of den(rho + u) in u."""
    out = np.zeros(len(den), dtype=_CLD)
    for k, c in enumerate(den):
        if c == 0:
            continue
        for i in range(k + 1):
            out[i] = out[i] + c * _CLD(comb(k, i)) * _CLD(rho) ** (k - i)
    return out


def _series_inverse(coeffs: np.ndarray, order: int) -> np.ndarray:
    """Return the ascending series of 1 / sum_k coeffs[k] u^k up to u^order."""
    c = np.concatenate([np.asarray(coeffs, dtype=_CLD),
                        np.zeros(order + 1, dtype=_CLD)])
    out = np.zeros(order + 1, dtype=_CLD)
    out[0] = _CLD(1.0) / c[0]
    for k in range(1, order + 1):
        acc = _CLD(0.0)
        for i in range(1, k + 1):
            acc = acc + c[i] * out[k - i]
        out[k] = -acc / c[0]
    return out


@dataclass(frozen=True)
class FrequencyGroup:
    """One distinct pair frequency of the member and its multiplicity."""

    omega: float
    multiplicity: int


class ExponentialEvaluator:
    """Pointwise phi, phi' and phi'' of the member E_{m,r}(omega)."""

    ZERO_FREQUENCY_TOLERANCE = ZERO_FREQUENCY_TOLERANCE
    CONFLUENT_TOLERANCE = CONFLUENT_TOLERANCE

    def __init__(self, m: int, omegas=(), *, zero_tolerance: float | None = None,
                 confluent_tolerance: float | None = None) -> None:
        m = int(m)
        if m < 1:
            raise ValueError("The polynomial core multiplicity m must be positive.")
        self.zero_tolerance = (ZERO_FREQUENCY_TOLERANCE if zero_tolerance is None
                               else float(zero_tolerance))
        self.confluent_tolerance = (CONFLUENT_TOLERANCE if confluent_tolerance is None
                                    else float(confluent_tolerance))
        omegas = tuple(float(w) for w in np.ravel(np.asarray(omegas, dtype=float)))
        self.requested_m = m
        self.requested_omegas = omegas

        kept = []
        collapsed = 0
        for w in omegas:
            if abs(w) <= self.zero_tolerance:
                collapsed += 1
            else:
                kept.append(abs(w))
        self.collapsed_pairs = collapsed
        self.m = m + 2 * collapsed
        kept.sort()
        self.omegas = tuple(kept)
        self.r = len(kept)

        groups: list[FrequencyGroup] = []
        for w in kept:
            if groups and abs(w - groups[-1].omega) <= self.confluent_tolerance:
                last = groups.pop()
                merged = (last.omega * last.multiplicity + w) / (last.multiplicity + 1)
                groups.append(FrequencyGroup(merged, last.multiplicity + 1))
            else:
                groups.append(FrequencyGroup(w, 1))
        self.groups = tuple(groups)

        self.n = self.m + 2 * self.r
        self.p = self.n - 1
        self.support = (-0.5 * self.n, 0.5 * self.n)
        self.poly_degree = max(self.m - 1, 3 + max([g.multiplicity for g in self.groups],
                                                   default=1) - 1)
        self._build()

    # ---------------- construction ----------------

    def label(self) -> str:
        if self.r == 0:
            return f"E_{{{self.m},0}}"
        inner = ",".join(f"{w:.6f}" for w in self.omegas)
        return f"E_{{{self.m},{self.r}}}({inner})"

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"ExponentialEvaluator(m={self.requested_m}, omegas={self.requested_omegas})"

    def _localization(self) -> np.ndarray:
        """Return the ascending coefficients of A(z) of the contract."""
        A = np.array([_LD(1.0)])
        for _ in range(self.m):
            A = np.convolve(A, np.array([_LD(1.0), _LD(-1.0)]))
        for g in self.groups:
            factor = np.array([_LD(1.0), _LD(-2.0) * np.cos(_LD(g.omega)), _LD(1.0)])
            for _ in range(g.multiplicity):
                A = np.convolve(A, factor)
        return A

    def _denominator(self) -> np.ndarray:
        """Return the ascending coefficients of D(s) = s^m prod (s^2+omega^2)."""
        D = np.array([_LD(1.0)])
        for g in self.groups:
            factor = np.array([_LD(g.omega) ** 2, _LD(0.0), _LD(1.0)])
            for _ in range(g.multiplicity):
                D = np.convolve(D, factor)
        return np.concatenate([np.zeros(self.m, dtype=_LD), D])

    def _build(self) -> None:
        A = self._localization()
        D = self._denominator()
        n = self.n

        # Polynomial part of the causal Green function: the principal part at
        # s = 0 of 1/D(s), inverse Laplace transformed term by term.
        inv0 = _series_inverse(D[self.m:], self.m - 1)
        poly_kappa = np.array([inv0[self.m - 1 - a].real / _LD(factorial(a))
                               for a in range(self.m)], dtype=_LD)

        # Oscillatory part: for each distinct root i omega of multiplicity mu,
        # the confluent residue series of u^mu / D(i omega + u).
        group_kappa = []
        for g in self.groups:
            mu = g.multiplicity
            shifted = _poly_shift_series(D, _CLD(1j) * _CLD(g.omega))
            inv = _series_inverse(shifted[mu:], mu - 1)
            kappa = np.array([inv[mu - 1 - a] / _CLD(factorial(a)) for a in range(mu)],
                             dtype=_CLD)
            group_kappa.append(kappa)

        nu = _LD(1.0)
        for g in self.groups:
            half = _LD(g.omega) / 2
            sinc = np.sin(half) / half
            for _ in range(g.multiplicity):
                nu = nu / (sinc * sinc)
        self.normalization = float(nu)

        total = sum(g.multiplicity for g in self.groups)
        deg = self.poly_degree
        poly = np.zeros((n, deg + 1), dtype=_LD)
        cosc = np.zeros((n, total), dtype=_LD)
        sinc_c = np.zeros((n, total), dtype=_LD)
        scale = _LD(0.0)

        for J in range(n):
            left = list(range(0, J + 1))
            right = list(range(J - n, 0))
            if len(left) <= len(right):
                use, sign = left, _LD(1.0)
            else:
                use, sign = right, _LD(-1.0)
            row = np.zeros(deg + 1, dtype=_LD)
            worst = _LD(0.0)
            for s in use:
                weight = sign * A[J - s]
                for a, ka in enumerate(poly_kappa):
                    if ka == 0:
                        continue
                    for b in range(a + 1):
                        term = weight * ka * _LD(comb(a, b)) * _LD(s) ** (a - b)
                        row[b] = row[b] + term
                        worst = max(worst, abs(term))
            slot = 0
            for gi, g in enumerate(self.groups):
                mu = g.multiplicity
                kappa = group_kappa[gi]
                w = _LD(g.omega)
                moments = np.zeros(mu, dtype=_CLD)
                for s in use:
                    phase = np.exp(_CLD(1j) * _CLD(w) * _CLD(s))
                    base = _CLD(sign) * _CLD(A[J - s]) * phase
                    for c in range(mu):
                        moments[c] = moments[c] + base * _CLD(s) ** c
                for b in range(mu):
                    Z = _CLD(0.0)
                    for a in range(b, mu):
                        Z = Z + _CLD(comb(a, b)) * kappa[a] * moments[a - b]
                    Cc = 2.0 * Z.real
                    Ss = -2.0 * Z.imag
                    cosc[J, slot + b] = Cc
                    sinc_c[J, slot + b] = Ss
                    worst = max(worst, abs(Cc), abs(Ss))
                    row[b] = row[b] + Cc
                    row[b + 1] = row[b + 1] + Ss * w
                    row[b + 2] = row[b + 2] - Cc * w * w / 2
                    row[b + 3] = row[b + 3] - Ss * w ** 3 / 6
                slot += mu
            poly[J] = row
            scale = max(scale, worst)

        self._slots = []
        slot = 0
        for g in self.groups:
            self._slots.append((g.omega, g.multiplicity, slot))
            slot += g.multiplicity
        self.term_count = total

        self.cell_poly = np.asarray(poly * nu, dtype=float)
        self.cell_cos = np.asarray(cosc * nu, dtype=float)
        self.cell_sin = np.asarray(sinc_c * nu, dtype=float)
        peak = float(np.max(np.abs(self.cell_poly[:, 0]))) or 1.0
        self.conditioning = float(scale * nu) / peak

    # ---------------- evaluation ----------------

    def _cell_index(self, x: np.ndarray):
        y = np.asarray(x, dtype=float) + 0.5 * self.n
        inside = (y > 0.0) & (y < float(self.n))
        J = np.clip(np.floor(y), 0.0, self.n - 1).astype(np.intp)
        xi = y - J
        return inside, J, xi

    def derivative(self, x, order: int = 0) -> np.ndarray:
        """Return the order-th derivative of phi at x, zero outside the support."""
        order = int(order)
        if order < 0:
            raise ValueError("The derivative order must be nonnegative.")
        x = np.asarray(x, dtype=float)
        shape = x.shape
        flat = np.atleast_1d(x).ravel()
        inside, J, xi = self._cell_index(flat)
        out = np.zeros(flat.shape, dtype=float)
        if not np.any(inside):
            return out.reshape(shape)
        Ji = J[inside]
        xii = xi[inside]
        value = np.zeros(xii.shape, dtype=float)

        coeffs = self.cell_poly[Ji]
        for k in range(order, self.poly_degree + 1):
            fall = 1.0
            for t in range(order):
                fall *= (k - t)
            value += fall * coeffs[:, k] * xii ** (k - order)

        for gi, (w, mu, slot) in enumerate(self._slots):
            z = w * xii
            for b in range(mu):
                Cc = self.cell_cos[Ji, slot + b]
                Ss = self.cell_sin[Ji, slot + b]
                for i in range(0, min(order, b) + 1):
                    fall = 1.0
                    for t in range(i):
                        fall *= (b - t)
                    if fall == 0.0:
                        continue
                    weight = comb(order, i) * fall * w ** (order - i)
                    term = (Cc * _remainder_chain(z, 0, order - i)
                            + Ss * _remainder_chain(z, 1, order - i))
                    value += weight * xii ** (b - i) * term
        out[inside] = value
        return out.reshape(shape)

    def value(self, x) -> np.ndarray:
        """Return phi(x), exactly zero outside [-n/2, n/2]."""
        return self.derivative(x, 0)

    def d1(self, x) -> np.ndarray:
        """Return phi'(x), exactly zero outside [-n/2, n/2]."""
        return self.derivative(x, 1)

    def d2(self, x) -> np.ndarray:
        """Return phi''(x), exactly zero outside [-n/2, n/2]."""
        return self.derivative(x, 2)

    def __call__(self, x, derivative: int = 0) -> np.ndarray:
        return self.derivative(x, derivative)

    def one_sided(self, knot: int, order: int, side: str) -> float:
        """Return the one-sided derivative limit at an integer knot.

        The knot is given in the support coordinate, so knot lies in
        [-n/2, n/2]. The left limit is the cell to the left evaluated at
        xi = 1 and the right limit is the cell to the right evaluated at
        xi = 0, both taken from the cell representation itself, so no finite
        difference or offset enters the comparison.
        """
        y = float(knot) + 0.5 * self.n
        cell = int(round(y)) - 1 if side == "left" else int(round(y))
        xi = 1.0 if side == "left" else 0.0
        if cell < 0 or cell >= self.n:
            return 0.0
        value = 0.0
        coeffs = self.cell_poly[cell]
        for k in range(order, self.poly_degree + 1):
            fall = 1.0
            for t in range(order):
                fall *= (k - t)
            value += fall * coeffs[k] * xi ** (k - order)
        for (w, mu, slot) in self._slots:
            z = np.array([w * xi])
            for b in range(mu):
                Cc = self.cell_cos[cell, slot + b]
                Ss = self.cell_sin[cell, slot + b]
                for i in range(0, min(order, b) + 1):
                    fall = 1.0
                    for t in range(i):
                        fall *= (b - t)
                    if fall == 0.0:
                        continue
                    weight = comb(order, i) * fall * w ** (order - i)
                    term = float(Cc * _remainder_chain(z, 0, order - i)[0]
                                 + Ss * _remainder_chain(z, 1, order - i)[0])
                    value += weight * xi ** (b - i) * term
        return float(value)

    def offgrid_samples(self, xi, order: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return the lattice n and phi^{(order)}(n + xi) for scalar xi."""
        radius = (self.n + 1) // 2 + 1
        lattice = np.arange(-radius, radius + 1)
        xi = float(xi)
        return lattice, self.derivative(lattice.astype(float) + xi, order)

    def cell_table(self) -> dict:
        """Return the flat cell tables that the C++ evaluator reads."""
        return {
            "eval_support_n": int(self.n),
            "eval_cells": int(self.n),
            "eval_poly_degree": int(self.poly_degree),
            "eval_group_count": int(len(self.groups)),
            "eval_term_count": int(self.term_count),
            "eval_group_omega": [float(g.omega) for g in self.groups],
            "eval_group_multiplicity": [float(g.multiplicity) for g in self.groups],
            "eval_normalization": float(self.normalization),
            "eval_conditioning": float(self.conditioning),
            "eval_poly": [float(v) for v in self.cell_poly.reshape(-1)],
            "eval_cos": [float(v) for v in self.cell_cos.reshape(-1)],
            "eval_sin": [float(v) for v in self.cell_sin.reshape(-1)],
        }


@lru_cache(maxsize=None)
def evaluator_for(m: int, omegas: tuple) -> ExponentialEvaluator:
    """Return a cached evaluator for the member E_{m,r}(omega)."""
    return ExponentialEvaluator(m, omegas)


def as_evaluator(generator):
    """Return an evaluator for an evaluator, a generator, or an (m, omega) pair.

    The dispatch is on the generator type. A Gate 7 or Gate 8 generator carries
    a `slots` attribute, and possibly a polynomial dial `poly`, so it is sent to
    general_evaluator.GeneralEvaluator, which covers real pairs, quads and the
    factor P(xi^2). Everything else keeps the ExponentialEvaluator path of this
    module, unchanged for the Gate 0 multiset (0^m, +-i omega_j).
    """
    if isinstance(generator, ExponentialEvaluator):
        return generator
    from general_evaluator import GeneralEvaluator, evaluator_for_generator
    if isinstance(generator, GeneralEvaluator):
        return generator
    if hasattr(generator, "slots"):
        return evaluator_for_generator(generator)
    m = getattr(generator, "m", None)
    if m is not None:
        return evaluator_for(int(m), tuple(getattr(generator, "omegas", ())))
    m, omegas = generator
    return evaluator_for(int(m), tuple(omegas))


def evaluate_periodic_field(coefficients: np.ndarray, x_query: np.ndarray, h: float,
                            generator, derivative: int = 0) -> np.ndarray:
    """Evaluate u(x) = sum_j c_j phi((x - j h)/h) and its derivatives.

    The convention is the one of spline_analysis.evaluate_periodic_spline:
    basis function j is centered on the node x_j = j h, the domain has length
    L = N h with N the length of the coefficient vector, queries wrap
    periodically, and the derivative carries the factor h^{-derivative}.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    ev = as_evaluator(generator)
    npts = coefficients.shape[0]
    radius = 0.5 * ev.n
    if npts < ev.n + 1:
        raise ValueError("The periodic grid must carry more nodes than the support.")
    if derivative not in (0, 1, 2):
        raise ValueError("Only derivatives 0, 1, 2 are supported.")
    scale = h ** (-derivative)

    local = (x_query / h) % npts
    j_lo = np.floor(local - radius).astype(np.intp)
    half = npts / 2.0
    out = np.zeros(local.shape, dtype=float)
    for offset in range(ev.n + 2):
        j = j_lo + offset
        delta = local - j
        delta = np.where(delta > half, delta - npts, delta)
        delta = np.where(delta < -half, delta + npts, delta)
        weights = ev.derivative(delta, derivative)
        out += coefficients[j % npts] * weights
    return out * scale


def periodic_symbol(generator, npts: int) -> np.ndarray:
    """Return b(theta_l) on the N discrete angles from the lattice samples."""
    ev = as_evaluator(generator)
    radius = (ev.n + 1) // 2
    lattice = np.arange(-radius, radius + 1).astype(float)
    samples = ev.value(lattice)
    theta = 2.0 * np.pi * np.arange(npts) / npts
    return np.array([float(np.sum(samples * np.cos(lattice * t))) for t in theta])


def periodic_coefficients(values: np.ndarray, generator) -> np.ndarray:
    """Return the collocation coefficients c with sum_j c_j phi(i - j) = u_i.

    The sample matrix is circulant with symbol b(theta_l), so the solve is one
    forward and one inverse discrete Fourier transform.
    """
    values = np.asarray(values, dtype=float)
    npts = values.shape[0]
    b = periodic_symbol(generator, npts)
    if np.min(b) <= 0.0:
        raise ValueError("The sample symbol is not positive on this grid.")
    return np.real(np.fft.ifft(np.fft.fft(values) / b))
