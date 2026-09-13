"""Centered exponential cardinal B-spline generators and their sampled symbols.

The member E_{m,r}(omega) carries the exponent multiset

    (0^m, +i omega_1, -i omega_1, ..., +i omega_r, -i omega_r),

degree p = m + 2 r - 1, and generator transform

    phi_hat(xi) = sinc(xi/2)^m
                  prod_j sinc((xi - omega_j)/2) sinc((xi + omega_j)/2)
                          / sinc(omega_j/2)^2,      sinc(z) = sin(z)/z.

Sampling on the unit integer lattice uses Poisson summation,

    b(theta)       = sum_k phi_hat(theta + 2 pi k),
    kappa_1(theta) = sum_k (theta + 2 pi k) phi_hat(theta + 2 pi k) / b(theta),
    kappa_2(theta) = sum_k (theta + 2 pi k)^2 phi_hat(theta + 2 pi k) / b(theta),

so that the nodal first derivative has symbol (i/h) kappa_1 and the literal
nodal second derivative has symbol (-1/h^2) kappa_2.

Truncation of the alias sum is controlled by a factorwise bound. For real xi
with |xi| >= X > max_j omega_j,

    |sinc(xi/2)| <= 2/|xi|,
    |sinc((xi -+ omega_j)/2)| <= 2/(|xi| - omega_j) <= 2/(|xi| (1 - omega_j/X)),

hence

    |phi_hat(xi)| <= C |xi|^{-(m + 2 r)},
    C = 2^m 4^r / prod_j [ sinc(omega_j/2)^2 (1 - omega_j/X)^2 ].

The neglected terms of the q-th moment sum have |theta + 2 pi k| >= 2 pi K for
every |k| > K and theta in [0, 2 pi), and successive neglected abscissae are
spaced by 2 pi, so with s = m + 2 r - q >= 2 and X0 = 2 pi K,

    T_q(K) <= 2 C [ X0^{-s} + X0^{1-s} / (2 pi (s - 1)) ].

K is chosen as the smallest integer making that bound at most the requested
tolerance, which is an absolute bound and therefore also a relative bound
against the unit normalization phi_hat(0) = b(0) = sum_n beta(n) = 1.

The sampled symbols are trigonometric polynomials of degree
nmax = floor((p+1)/2), because the generator is supported on [-(p+1)/2,
(p+1)/2]. The integer samples beta(n), beta'(n), beta''(n) are therefore
recovered from the symbols by the trapezoidal rule on M uniform points, which
is exact whenever M > 2 nmax: the rule integrates exp(i n theta) exactly for
|n| < M, and the quadrature of b(theta) exp(i n theta) only aliases
beta(n +- M) into beta(n). This module uses M = 4 nmax + 8.

For m + 2 r <= 5 the second moment sum converges too slowly for a double
precision tail bound at 1e-14. Those members use the exact zero factor
removal identities

    beta'(x)  = g_{m-1}(x + 1/2) - g_{m-1}(x - 1/2),
    beta''(x) = g_{m-2}(x + 1) - 2 g_{m-2}(x) + g_{m-2}(x - 1),

where g_{m-j} is the generator with j zero exponents dropped. They follow from
(i xi) sinc(xi/2) = exp(i xi/2) - exp(-i xi/2). With r = 0 the reduced
generators are centered cardinal polynomial B-splines and their lattice values
are taken from spline_analysis.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import integrate

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import spline_analysis as sa  # noqa: E402

K_MAX = 65536
DEFAULT_TOL = 1e-14
CHUNK = 1 << 21


def sinc(z: np.ndarray) -> np.ndarray:
    """Return sin(z)/z with the removable value 1 at z = 0."""
    z = np.asarray(z, dtype=float)
    out = np.ones(z.shape, dtype=float)
    nz = z != 0.0
    out[nz] = np.sin(z[nz]) / z[nz]
    return out


@dataclass(frozen=True)
class IntegerSamples:
    """Lattice samples of the generator and of its first two derivatives."""

    n: np.ndarray
    beta: np.ndarray
    dbeta: np.ndarray
    ddbeta: np.ndarray
    truncation: dict


class ExponentialGenerator:
    """A centered exponential cardinal B-spline generator E_{m,r}(omega)."""

    def __init__(self, m: int, omegas=()) -> None:
        m = int(m)
        if m < 1:
            raise ValueError("The polynomial core multiplicity m must be positive.")
        omegas = tuple(float(w) for w in np.ravel(np.asarray(omegas, dtype=float)))
        self.m = m
        self.omegas = omegas
        self.r = len(omegas)
        self._cache: dict = {}

    @property
    def p(self) -> int:
        """Return the generator degree p = m + 2 r - 1."""
        return self.m + 2 * self.r - 1

    @property
    def nmax(self) -> int:
        """Return the lattice half width floor((p+1)/2) of the sample stencil."""
        return (self.p + 1) // 2

    def label(self) -> str:
        if self.r == 0:
            return f"E_{{{self.m},0}}"
        inner = ",".join(f"{w:.6f}" for w in self.omegas)
        return f"E_{{{self.m},{self.r}}}({inner})"

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"ExponentialGenerator(m={self.m}, omegas={self.omegas})"

    def fourier(self, xi) -> np.ndarray:
        """Return phi_hat(xi) for the contract normalization phi_hat(0) = 1."""
        xi = np.asarray(xi, dtype=float)
        value = sinc(xi / 2.0) ** self.m
        for w in self.omegas:
            scale = sinc(np.array(w / 2.0)) ** 2
            value = value * sinc((xi - w) / 2.0) * sinc((xi + w) / 2.0) / scale
        return value

    def admissible(self, theta_points: int = 2001) -> tuple[bool, str]:
        """Return whether the frequencies increase inside (0, pi) and b > 0."""
        for w in self.omegas:
            if not (0.0 < w < np.pi):
                return False, "frequency outside (0, pi)"
        if any(self.omegas[j] >= self.omegas[j + 1] for j in range(self.r - 1)):
            return False, "frequencies not strictly increasing"
        theta = np.linspace(0.0, np.pi, theta_points)
        b = self.symbols(theta)[0]
        if not np.all(np.isfinite(b)) or float(b.min()) <= 0.0:
            return False, "sample symbol not positive"
        return True, "admissible"

    def tail_constant(self, X: float) -> float:
        """Return the constant C of |phi_hat(xi)| <= C |xi|^{-(m+2r)} for |xi| >= X."""
        C = 2.0 ** self.m * 4.0 ** self.r
        for w in self.omegas:
            s = float(sinc(np.array(w / 2.0)))
            shrink = 1.0 - w / X
            if shrink <= 0.0:
                return np.inf
            C /= (s * s) * shrink * shrink
        return C

    def tail_bound(self, q: int, K: int) -> float:
        """Return the bound on the neglected q-th moment alias tail beyond |k| = K."""
        s = self.m + 2 * self.r - q
        if s < 2:
            return np.inf
        X0 = 2.0 * np.pi * K
        if X0 <= 2.0 * max(self.omegas, default=0.0):
            return np.inf
        C = self.tail_constant(X0)
        if not np.isfinite(C):
            return np.inf
        return 2.0 * C * (X0 ** (-s) + X0 ** (1 - s) / (2.0 * np.pi * (s - 1)))

    def required_K(self, q: int, tol: float = DEFAULT_TOL) -> int:
        """Return the smallest K whose tail bound for the q-th moment is <= tol."""
        if self.tail_bound(q, K_MAX) > tol:
            return K_MAX + 1
        lo, hi = 1, K_MAX
        while lo < hi:
            mid = (lo + hi) // 2
            if self.tail_bound(q, mid) <= tol:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _alias_moments(self, theta: np.ndarray, K: int) -> tuple[np.ndarray, ...]:
        """Return the three alias moment sums on theta, summed in blocks over k."""
        theta = np.asarray(theta, dtype=float)
        S0 = np.zeros(theta.shape)
        S1 = np.zeros(theta.shape)
        S2 = np.zeros(theta.shape)
        block = max(1, CHUNK // max(1, theta.size))
        k0 = -K
        while k0 <= K:
            k1 = min(K, k0 + block - 1)
            ks = np.arange(k0, k1 + 1, dtype=float)
            X = theta[:, None] + 2.0 * np.pi * ks[None, :]
            P = self.fourier(X)
            S0 += P.sum(axis=1)
            S1 += (X * P).sum(axis=1)
            S2 += (X * X * P).sum(axis=1)
            k0 = k1 + 1
        return S0, S1, S2

    def sample_symbols(self, theta, K: int | None = None, tol: float = DEFAULT_TOL):
        """Return b, kappa_1, kappa_2 from the truncated alias sums of the contract."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if K is None:
            K = max(self.required_K(q, tol) for q in (0, 1, 2))
            if K > K_MAX:
                raise ValueError(
                    "The alias tail bound cannot reach the requested tolerance "
                    f"within K <= {K_MAX} for {self.label()}."
                )
        S0, S1, S2 = self._alias_moments(theta, K)
        return S0, S1 / S0, S2 / S0

    def _quadrature_points(self) -> int:
        """Return the trapezoidal point count M > 2 nmax used for sample recovery."""
        return 4 * self.nmax + 8

    def _samples_from_alias(self, tol: float) -> IntegerSamples:
        M = self._quadrature_points()
        theta = 2.0 * np.pi * np.arange(M) / M
        K = max(self.required_K(q, tol) for q in (0, 1, 2))
        S0, S1, S2 = self._alias_moments(theta, K)
        n = np.arange(-self.nmax, self.nmax + 1)
        phase = np.exp(1j * np.outer(n, theta)) / M
        beta = (phase @ S0).real
        dbeta = (phase @ (1j * S1)).real
        ddbeta = (phase @ (-S2)).real
        truncation = {
            "route": "alias",
            "K": int(K),
            "quadrature_points": int(M),
            "tail_bound": {str(q): self.tail_bound(q, K) for q in (0, 1, 2)},
        }
        return IntegerSamples(n, beta, dbeta, ddbeta, truncation)

    def _value_samples(self, m: int, shifts, index: np.ndarray, tol: float):
        """Return g_m(index + shift) for each shift, g_m the generator with core m."""
        index = np.asarray(index, dtype=float)
        shifts = np.atleast_1d(np.asarray(shifts, dtype=float))
        reduced = ExponentialGenerator(m, self.omegas)
        if m <= 1 and self.r == 0:
            raise ValueError("A reduced polynomial core below degree one has no samples.")
        K = reduced.required_K(0, tol)
        if K > K_MAX:
            if self.r != 0:
                raise ValueError(
                    "The value alias tail cannot reach the tolerance for "
                    f"{reduced.label()} within K <= {K_MAX}."
                )
            out = np.empty((shifts.size, index.size))
            for j, a in enumerate(shifts):
                out[j] = sa.centered_cardinal_bspline(m - 1, index + a)
            return out, {"route": "cardinal_bspline", "degree": m - 1}
        M = 4 * ((reduced.p + 1) // 2) + 8 + 8
        theta = 2.0 * np.pi * np.arange(M) / M
        out = np.empty((shifts.size, index.size))
        block = max(1, CHUNK // M)
        acc = np.zeros((shifts.size, M), dtype=complex)
        k0 = -K
        while k0 <= K:
            k1 = min(K, k0 + block - 1)
            ks = np.arange(k0, k1 + 1, dtype=float)
            X = theta[:, None] + 2.0 * np.pi * ks[None, :]
            P = reduced.fourier(X)
            acc += np.exp(2j * np.pi * np.outer(shifts, ks)) @ P.T
            k0 = k1 + 1
        for j, a in enumerate(shifts):
            series = acc[j] * np.exp(1j * a * theta)
            out[j] = np.array(
                [float((series * np.exp(1j * nn * theta)).sum().real / M) for nn in index]
            )
        return out, {"route": "alias_value", "K": int(K), "quadrature_points": int(M)}

    def _samples_from_differences(self, tol: float) -> IntegerSamples:
        n = np.arange(-self.nmax, self.nmax + 1)
        beta_vals, info0 = self._value_samples(self.m, [0.0], n, tol)
        wide = np.arange(-self.nmax - 1, self.nmax + 2)
        half, info1 = self._value_samples(self.m - 1, [0.5, -0.5], wide, tol)
        full, info2 = self._value_samples(self.m - 2, [1.0, 0.0, -1.0], wide, tol)
        pick = {int(v): i for i, v in enumerate(wide)}
        idx = np.array([pick[int(v)] for v in n])
        dbeta = half[0][idx] - half[1][idx]
        ddbeta = full[0][idx] - 2.0 * full[1][idx] + full[2][idx]
        truncation = {
            "route": "zero_factor_differences",
            "value": info0,
            "first": info1,
            "second": info2,
        }
        return IntegerSamples(n, beta_vals[0], dbeta, ddbeta, truncation)

    def integer_samples(self, tol: float = DEFAULT_TOL) -> IntegerSamples:
        """Return beta(n), beta'(n), beta''(n) for |n| <= floor((p+1)/2)."""
        key = ("samples", float(tol))
        if key in self._cache:
            return self._cache[key]
        if max(self.required_K(q, tol) for q in (0, 1, 2)) <= K_MAX:
            samples = self._samples_from_alias(tol)
        else:
            samples = self._samples_from_differences(tol)
        self._cache[key] = samples
        return samples

    def symbols(self, theta, tol: float = DEFAULT_TOL):
        """Return b, kappa_1, kappa_2 as the exact trigonometric polynomials.

        The forms below remove the leading cancellation analytically using
        sum_n beta'(n) = 0 and sum_n beta''(n) = sum_n n beta''(n) = 0, so the
        relative accuracy of kappa_1 and kappa_2 does not degrade as theta
        approaches zero.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        s = self.integer_samples(tol)
        pos = s.n > 0
        nn = s.n[pos].astype(float)
        arg = np.outer(theta, nn)
        b = s.beta[s.n == 0][0] + 2.0 * (np.cos(arg) @ s.beta[pos])
        k1b = -2.0 * (np.sin(arg) @ s.dbeta[pos])
        k2b = 4.0 * (np.sin(0.5 * arg) ** 2 @ s.ddbeta[pos])
        return b, k1b / b, k2b / b

    def symbol_derivative(self, theta, tol: float = DEFAULT_TOL) -> np.ndarray:
        """Return d kappa_1 / d theta from the analytic derivative of the symbols."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        s = self.integer_samples(tol)
        pos = s.n > 0
        nn = s.n[pos].astype(float)
        arg = np.outer(theta, nn)
        b = s.beta[s.n == 0][0] + 2.0 * (np.cos(arg) @ s.beta[pos])
        db = -2.0 * (np.sin(arg) @ (s.beta[pos] * nn))
        a = -2.0 * (np.sin(arg) @ s.dbeta[pos])
        da = -2.0 * (np.cos(arg) @ (s.dbeta[pos] * nn))
        return (da * b - a * db) / (b * b)

    def cond(self, theta_points: int = 20001, discrete=(64, 127, 128)) -> dict:
        """Return the continuous and discrete left side condition numbers."""
        theta = np.linspace(0.0, np.pi, theta_points)
        b = self.symbols(theta)[0]
        out = {
            "cond": float(b.max() / b.min()),
            "one_over_min_b": float(1.0 / b.min()),
            "max_b": float(b.max()),
            "min_b": float(b.min()),
        }
        for N in discrete:
            angles = 2.0 * np.pi * np.arange(N) / N
            bn = self.symbols(angles)[0]
            out[f"cond_N{N}"] = float(bn.max() / bn.min())
        return out

    def offgrid_transfers(self, theta, xi, K: int | None = None, tol: float = DEFAULT_TOL):
        """Return T_0, T_1, T_2 of the contract by direct alias summation."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        if K is None:
            K = max(self.required_K(q, tol) for q in (0, 1, 2))
            if K > K_MAX:
                raise ValueError("The alias tail bound is unreachable for this member.")
        S = [np.zeros((theta.size, xi.size), dtype=complex) for _ in range(3)]
        b = self.symbols(theta)[0]
        block = max(1, CHUNK // max(1, theta.size * xi.size))
        k0 = -K
        while k0 <= K:
            k1 = min(K, k0 + block - 1)
            ks = np.arange(k0, k1 + 1, dtype=float)
            X = theta[:, None] + 2.0 * np.pi * ks[None, :]
            P = self.fourier(X)
            E = np.exp(2j * np.pi * np.outer(ks, xi))
            S[0] += P @ E
            S[1] += (X * P) @ E
            S[2] += (X * X * P) @ E
            k0 = k1 + 1
        T0 = S[0] / b[:, None]
        T1 = S[1] / (theta[:, None] * b[:, None])
        T2 = S[2] / (theta[:, None] ** 2 * b[:, None])
        return T0, T1, T2

    def offgrid_samples(self, xi, tol: float = DEFAULT_TOL):
        """Return beta(n+xi), beta'(n+xi), beta''(n+xi) on the shifted lattice.

        The same Poisson identity that recovers the integer samples recovers
        the shifted samples, because

            sum_k (theta + 2 pi k)^q phi_hat(theta + 2 pi k) exp(2 pi i k xi)
                = i^{-q} exp(-i xi theta) sum_n beta^{(q)}(n + xi) exp(-i n theta).
        """
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        key = ("offgrid", xi.tobytes(), float(tol))
        if key in self._cache:
            return self._cache[key]
        K = max(self.required_K(q, tol) for q in (0, 1, 2))
        if K > K_MAX:
            raise ValueError("The alias tail bound is unreachable for this member.")
        nmax = self.nmax + 1
        n = np.arange(-nmax, nmax + 1)
        M = 4 * nmax + 8
        theta = 2.0 * np.pi * np.arange(M) / M
        acc = [np.zeros((M, xi.size), dtype=complex) for _ in range(3)]
        block = max(1, CHUNK // max(1, M * xi.size))
        k0 = -K
        while k0 <= K:
            k1 = min(K, k0 + block - 1)
            ks = np.arange(k0, k1 + 1, dtype=float)
            X = theta[:, None] + 2.0 * np.pi * ks[None, :]
            P = self.fourier(X)
            E = np.exp(2j * np.pi * np.outer(ks, xi))
            acc[0] += P @ E
            acc[1] += (X * P) @ E
            acc[2] += (X * X * P) @ E
            k0 = k1 + 1
        phase = np.exp(1j * np.outer(n, theta)) / M
        out = []
        for q in range(3):
            series = (1j ** q) * acc[q] * np.exp(1j * np.outer(theta, xi))
            out.append((phase @ series).real)
        result = (n, out[0], out[1], out[2])
        self._cache[key] = result
        return result

    def coupled_error(self, theta, xi, tol: float = DEFAULT_TOL) -> np.ndarray:
        """Return e_c(theta) = max_xi max(|T_0-1|, |T_1-1|, |T_2-1|).

        The transfers are evaluated from the shifted lattice samples using the
        cancellation free forms implied by sum_n beta(n+xi) = 1,
        sum_n beta'(n+xi) = 0 and sum_n beta''(n+xi) = sum_n n beta''(n+xi) = 0.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        n, s0, s1, s2 = self.offgrid_samples(xi, tol)
        b = self.symbols(theta, tol)[0]
        nn = n.astype(float)
        arg = np.outer(theta, nn)
        expo = np.exp(-1j * arg)
        F0 = expo @ s0
        F1 = (expo - 1.0) @ s1
        F2 = (expo - 1.0 + 1j * arg) @ s2
        shift = np.exp(-1j * np.outer(theta, xi))
        T0 = shift * F0 / b[:, None]
        T1 = shift * (-1j) * F1 / (theta[:, None] * b[:, None])
        T2 = shift * (-1.0) * F2 / (theta[:, None] ** 2 * b[:, None])
        err = np.maximum(np.abs(T0 - 1.0), np.abs(T1 - 1.0))
        err = np.maximum(err, np.abs(T2 - 1.0))
        return err.max(axis=1)


def pair_kernel(w: float, x) -> np.ndarray:
    """Return the conjugate pair kernel P_w(x) = w sin(w(1-|x|)) / (4 sin^2(w/2))."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape)
    inside = np.abs(x) < 1.0
    out[inside] = w * np.sin(w * (1.0 - np.abs(x[inside]))) / (4.0 * np.sin(w / 2.0) ** 2)
    return out


def pair_kernel_mass(w: float) -> float:
    """Return the integral of P_w over its support, which must be one."""
    total = 0.0
    for lo, hi in ((-1.0, 0.0), (0.0, 1.0)):
        value, _ = integrate.quad(lambda x: float(pair_kernel(w, np.array(x))), lo, hi,
                                  limit=200, epsabs=1e-14, epsrel=1e-14)
        total += value
    return float(total)


def _bspline_value(degree: int, x: float, derivative: int) -> float:
    if derivative == 0:
        return float(sa.centered_cardinal_bspline(degree, np.array([x]))[0])
    if derivative == 1:
        return float(sa.centered_cardinal_bspline_derivative(degree, np.array([x]))[0])
    return float(sa.centered_cardinal_bspline_second_derivative(degree, np.array([x]))[0])


def convolution_samples(m: int, omegas, points, derivative: int = 0) -> np.ndarray:
    """Return beta^{(q)}(x) by real pair kernel convolution with quadrature.

    The generator factors as beta = B_{m-1} * P_{omega_1} * ... * P_{omega_r},
    where B_{m-1} is the centered cardinal polynomial B-spline of degree m-1.
    Derivatives act on the polynomial factor. Every integrand is piecewise
    smooth with cusps at the pair kernel centers and at the B-spline knots, and
    all of those abscissae are supplied to the adaptive rule as breakpoints.
    """
    omegas = tuple(float(w) for w in omegas)
    degree = m - 1
    knot = 0.5 * (degree + 1)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    out = np.empty(points.shape)

    def core(x: float) -> float:
        return _bspline_value(degree, x, derivative)

    if len(omegas) == 0:
        for i, x in enumerate(points):
            out[i] = core(float(x))
        return out

    if len(omegas) == 1:
        w = omegas[0]
        for i, x in enumerate(points):
            cuts = sorted({-1.0, 0.0, 1.0} | {float(x) - k for k in np.arange(-knot, knot + 1)})
            cuts = [c for c in cuts if -1.0 <= c <= 1.0]
            total = 0.0
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                value, _ = integrate.quad(
                    lambda t, x=float(x): float(pair_kernel(w, np.array(t))) * core(x - t),
                    lo, hi, limit=200, epsabs=1e-14, epsrel=1e-14)
                total += value
            out[i] = total
        return out

    if len(omegas) == 2:
        w1, w2 = omegas

        def inner(x: float, t: float) -> float:
            cuts = sorted({-1.0, 0.0, 1.0}
                          | {float(x) - t - k for k in np.arange(-knot, knot + 1)})
            cuts = [c for c in cuts if -1.0 <= c <= 1.0]
            total = 0.0
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                value, _ = integrate.quad(
                    lambda u: float(pair_kernel(w2, np.array(u))) * core(x - t - u),
                    lo, hi, limit=200, epsabs=1e-13, epsrel=1e-13)
                total += value
            return total

        for i, x in enumerate(points):
            cuts = sorted({-1.0, 0.0, 1.0}
                          | {float(x) - k - s for k in np.arange(-knot, knot + 1)
                             for s in (-1.0, 0.0, 1.0)})
            cuts = [c for c in cuts if -1.0 <= c <= 1.0]
            total = 0.0
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                value, _ = integrate.quad(
                    lambda t, x=float(x): float(pair_kernel(w1, np.array(t))) * inner(x, t),
                    lo, hi, limit=200, epsabs=1e-13, epsrel=1e-13)
                total += value
            out[i] = total
        return out

    raise NotImplementedError("The quadrature cross check covers r <= 2.")


@lru_cache(maxsize=None)
def anchor_generator() -> ExponentialGenerator:
    """Return the frozen side research anchor E_{4,2}(0.1, 2.36)."""
    return ExponentialGenerator(4, (0.1, 2.36))
