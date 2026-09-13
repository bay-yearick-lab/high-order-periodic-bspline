"""Gate 8: polynomial-augmented exponential cardinal B-spline generators.

A Gate 7 member phi (core m, slots s_j = lambda_j^2, width n = m + 2 L with L
the number of slots, a quad occupying two conjugate slots) is multiplied on
the transform side by a real polynomial in xi^2 with unit constant term,

    psi_hat(xi) = P(xi^2) phi_hat(xi),    P(t) = 1 + p_1 t + ... + p_q t^q,

which on the x side is the differential operator P(-D^2) applied to phi,

    psi = sum_l (-1)^l p_l phi^{(2 l)},

so psi keeps the support [-n/2, n/2] of phi, is real and even, and is of
class C^{n-2-2q} (each application of D^2 costs two orders of smoothness at
the knots, which for even n are the integers). The literal second derivative
and the off-grid channels need psi in C^2, hence q <= (n - 4)/2.

Sampled symbols. Poisson summation gives, with xi_k = theta + 2 pi k,

    S_j(theta) = sum_k xi_k^j P(xi_k^2) phi_hat(xi_k),   j = 0, 1, 2,
    b = S_0,  kappa_1 = S_1 / S_0,  kappa_2 = S_2 / S_0,

and Lemma D of Gate 7 factors every alias as phi_hat(xi_k) = kfree(theta)
R(xi_k) with kfree(theta) = (2 sin(theta/2))^m prod_j c(s_j, theta) and the
rational part R(xi) = xi^{-m} prod_j (xi^2 + s_j)^{-1}. Hence

    S_j(theta) = kfree(theta) sum_k f_j(xi_k),
    f_j(z) = z^{j} P(z^2) / (z^m prod_j (z^2 + s_j)),

a rational function of degree j + 2q - n <= -2 whenever psi is C^2.

Alias route with an exact tail. The head |k| <= K is summed term by term
from the sinc products (as in family.py); the tail |k| > K is summed exactly
through the convergent expansion of f_j in inverse powers of xi,

    f_j(xi) = sum_{i=0}^{q} sum_{a>=0} p_i d_a xi^{j - n + 2 i - 2 a},
    prod_j (1 + s_j u)^{-1} = sum_a d_a u^a,   u = xi^{-2},

whose row sums over |k| > K are Hurwitz zeta values,

    sum_{k>K} xi_k^{-w} = (2 pi)^{-w} zeta(w, K + 1 + theta/(2 pi)),
    sum_{k<-K} xi_k^{-w} = (-1)^w (2 pi)^{-w} zeta(w, K + 1 - theta/(2 pi)),

with w = n - j - 2 i + 2 a >= 2. The expansion converges for xi^2 > max |s_j|
and its truncation after A terms is bounded, with x = max|s_j| / (2 pi K)^2
and L slots, by binom(A + L - 1, L - 1) x^A / (1 - x)^L times the absolute
row sums; A is chosen so that this bound is below TAIL_TOL. Without the tail
the C^2 members would need K of order 1e14 for the second moment.

Residue route (used for the off-grid channel and as an independent check of
the samples). For xi' in [0, 1) the kernel

    h(z) = exp(i (z - theta)(xi' - 1/2)) / (2 sin((z - theta)/2))

has simple poles at z = theta + 2 pi k with residue exp(2 pi i k xi') and is
bounded on large square contours, so for a rational f decaying like z^{-2}

    sum_k f(theta + 2 pi k) exp(2 pi i k xi') = - sum_{poles a of f} Res_{z=a} [f(z) h(z)].

The poles of f_j are z = 0 of order m - j, where the residue is a Taylor
coefficient of P(z^2)/prod(z^2 + s_j) times h(z), and the simple poles
z = +-sqrt(-s_j) with residue a^{j-m-1} P(-s_j) / (2 prod_{l != j} (s_l - s_j)).
For an imaginary pair the poles +-omega lie on the real axis and the factor
c(-omega^2, theta) of kfree cancels the pole of h at theta = omega; the
product is evaluated in its cancelled form. This gives the transfers
T_q(theta, xi) = S_q(theta, xi) / (theta^q b(theta)) of the Gate 0 contract
without any alias truncation.

Decimal route. The alias route with the zeta tail is repeated in 60-digit
decimal arithmetic (Hurwitz zeta by Euler-Maclaurin with exact Bernoulli
numbers) with a different head length, and the two must agree to 1e-13.
"""

from __future__ import annotations

import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

import numpy as np
from scipy.special import zeta as hurwitz_zeta

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import family as fam  # noqa: E402
import gate7_family as g7  # noqa: E402
from gate1_champions import DecimalTrig, DECIMAL_PREC  # noqa: E402

K_HEAD = 64
TAIL_TOL = 1e-18
ROUTE_TOL = 1e-13
RESIDUE_TOL = 1e-12
MIN_SMOOTHNESS = 2
POLE_SEPARATION = 1e-9


# ---------------------------------------------------------------------------
# Helpers on the polynomial
# ---------------------------------------------------------------------------

def poly_value(poly, t):
    """Return P(t) = 1 + sum_l poly[l-1] t^l by Horner's rule."""
    t = np.asarray(t, dtype=float)
    out = np.zeros(t.shape)
    for p in reversed(poly):
        out = out * t + p
    return out * t + 1.0


def poly_roots_xi(poly) -> dict:
    """Return the roots of P in t = xi^2 sorted into the classes of the theory.

    Positive real roots t_0 = xi_0^2 are the real zeros of psi_hat on the
    line; those with xi_0 <= pi lie inside the band and make the recovery
    singular; those in (pi, 2 pi) are hit by the k = -1 alias at
    theta_0 = 2 pi - xi_0 (a soft contact). Negative or complex roots keep
    P > 0 on the real line.
    """
    poly = [float(p) for p in poly]
    if not poly or all(p == 0.0 for p in poly):
        return {"real_positive_xi": [], "soft_contacts_theta": [], "inside_band": [],
                "negative_t": [], "complex_t": [], "positive_on_line": True}
    coeffs = [1.0] + poly
    roots = np.roots(coeffs[::-1])
    out = {"real_positive_xi": [], "soft_contacts_theta": [], "inside_band": [],
           "negative_t": [], "complex_t": []}
    for rt in roots:
        if abs(rt.imag) <= 1e-10 * max(1.0, abs(rt)):
            t = float(rt.real)
            if t > 0.0:
                xi0 = float(np.sqrt(t))
                out["real_positive_xi"].append(xi0)
                if xi0 <= np.pi:
                    out["inside_band"].append(xi0)
                elif xi0 < 2.0 * np.pi:
                    out["soft_contacts_theta"].append(2.0 * np.pi - xi0)
            else:
                out["negative_t"].append(t)
        else:
            out["complex_t"].append([float(rt.real), float(rt.imag)])
    out["positive_on_line"] = not out["real_positive_xi"]
    out["real_positive_xi"].sort()
    out["soft_contacts_theta"].sort()
    return out


def poly_min_on_band(poly) -> float:
    """Return min of P(t) on t in [0, pi^2] (exact through the stationary points)."""
    poly = [float(p) for p in poly]
    grid = [0.0, np.pi ** 2]
    if len(poly) >= 2:
        deriv = [(l + 1) * poly[l] for l in range(len(poly))]
        crit = np.roots(deriv[::-1])
        grid += [float(c.real) for c in crit if abs(c.imag) < 1e-12 and 0.0 < c.real < np.pi ** 2]
    return float(min(poly_value(poly, np.array(grid))))


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class PolynomialGenerator:
    """psi_hat = P(xi^2) phi_hat for a Gate 7 member phi and P(0) = 1."""

    def __init__(self, m: int, slots, poly=()) -> None:
        self.base = g7.ExtendedGenerator(m, slots)
        self.poly = tuple(float(p) for p in poly)
        if self.smoothness < MIN_SMOOTHNESS:
            raise ValueError(
                f"degree {self.q} leaves psi in C^{self.smoothness}; the literal second "
                f"derivative needs C^{MIN_SMOOTHNESS} (q <= {(self.n - 4) // 2} at width {self.n})")
        self._cache: dict = {}

    # ---- bookkeeping ------------------------------------------------------

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def slots(self):
        return self.base.slots

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def p(self) -> int:
        return self.base.p

    @property
    def nmax(self) -> int:
        return self.base.nmax

    @property
    def q(self) -> int:
        """Return the degree of P in xi^2 (the length of the coefficient list)."""
        return len(self.poly)

    @property
    def smoothness(self) -> int:
        """Return the smoothness class C^{n-2-2q} of psi."""
        return self.n - 2 - 2 * self.q

    @property
    def parameters(self) -> int:
        return self.base.parameters + self.q

    @property
    def omegas(self):
        return self.base.omegas

    @property
    def real_pairs(self):
        return self.base.real_pairs

    @property
    def quads(self):
        return self.base.quads

    @property
    def r(self) -> int:
        return self.base.r

    def label(self) -> str:
        inner = ",".join(f"{p:.6g}" for p in self.poly) if self.poly else "1"
        return f"{self.base.label()}xP[{inner}]"

    def s_values(self) -> np.ndarray:
        """Return every slot parameter s (a quad contributes s and conj s)."""
        vals = []
        for slot in self.slots:
            s = slot.s
            if slot.kind == "quad":
                vals += [s, s.conjugate()]
            else:
                vals.append(complex(s.real, 0.0))
        return np.array(vals, dtype=complex)

    def polynomial(self, t):
        return poly_value(self.poly, t)

    def fourier(self, xi) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        return self.polynomial(xi * xi) * self.base.fourier(xi)

    def k_free_part(self, theta) -> np.ndarray:
        return self.base.k_free_part(theta)

    # ---- alias route with the exact tail ---------------------------------

    def _tail_series(self, A: int) -> np.ndarray:
        """Return d_0..d_{A-1} of prod_j (1 + s_j u)^{-1} (real by conjugate closure)."""
        d = np.zeros(A, dtype=complex)
        d[0] = 1.0
        for s in self.s_values():
            geo = (-s) ** np.arange(A)
            d = np.convolve(d, geo)[:A]
        return d.real

    def tail_plan(self, K: int = K_HEAD, tol: float = TAIL_TOL) -> dict:
        """Return the head length K, the expansion length A and the remainder bound."""
        key = ("plan", int(K), float(tol))
        if key in self._cache:
            return self._cache[key]
        svals = self.s_values()
        L = svals.size
        smax = float(np.max(np.abs(svals))) if L else 0.0
        K = int(K)
        while smax / (2.0 * np.pi * K) ** 2 >= 0.25:
            K *= 2
        x = smax / (2.0 * np.pi * K) ** 2
        kfree_max = 2.0 ** self.m
        for slot in self.slots:
            G, _ = g7.slot_tail_constant(slot)
            kfree_max *= G ** (2 if slot.kind == "quad" else 1)
        coeffs = [1.0] + list(self.poly)
        A = 8
        while True:
            if L == 0:
                rem = 0.0
            else:
                rem = comb(A + L - 1, L - 1) * x ** A / (1.0 - x) ** L
            bound = 0.0
            for j in range(3):
                row = 0.0
                for i, p in enumerate(coeffs):
                    w = self.n - j - 2 * i
                    row += abs(p) * 2.0 * (2.0 * np.pi) ** (-w) * float(hurwitz_zeta(w, K))
                bound = max(bound, kfree_max * rem * row)
            if bound <= tol or A >= 512:
                break
            A *= 2
        plan = {"K": K, "A": A, "x": x, "remainder_factor": rem, "tail_bound": bound,
                "kfree_bound": kfree_max}
        self._cache[key] = plan
        return plan

    def _alias_moments(self, theta: np.ndarray, K: int, A: int):
        """Return S_0, S_1, S_2 on theta: head |k| <= K plus the exact zeta tail."""
        theta = np.asarray(theta, dtype=float)
        ks = np.arange(-K, K + 1, dtype=float)
        X = theta[:, None] + 2.0 * np.pi * ks[None, :]
        Phat = self.fourier(X)
        S = [Phat.sum(axis=1), (X * Phat).sum(axis=1), (X * X * Phat).sum(axis=1)]
        d = self._tail_series(A)
        coeffs = [1.0] + list(self.poly)
        kf = self.k_free_part(theta)
        qp = K + 1.0 + theta / (2.0 * np.pi)
        qm = K + 1.0 - theta / (2.0 * np.pi)
        for j in range(3):
            weights: dict = {}
            for i, p in enumerate(coeffs):
                for a in range(A):
                    w = self.n - j - 2 * i + 2 * a
                    weights[w] = weights.get(w, 0.0) + p * d[a]
            tail = np.zeros(theta.shape)
            for w, c in weights.items():
                if c == 0.0:
                    continue
                zp = hurwitz_zeta(w, qp)
                zm = hurwitz_zeta(w, qm)
                tail += c * (2.0 * np.pi) ** (-w) * (zp + (-1.0) ** w * zm)
            S[j] = S[j] + kf * tail
        return S

    def sample_symbols(self, theta, K: int | None = None, tol: float = TAIL_TOL):
        """Return b, kappa_1, kappa_2 by the alias route with the exact tail."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        plan = self.tail_plan(K_HEAD if K is None else K, tol)
        S0, S1, S2 = self._alias_moments(theta, plan["K"], plan["A"])
        return S0, S1 / S0, S2 / S0

    def _quadrature_points(self) -> int:
        return 4 * self.nmax + 8

    def integer_samples(self, tol: float = TAIL_TOL) -> fam.IntegerSamples:
        """Return psi(n), psi'(n), psi''(n) for |n| <= nmax by the trapezoid inverse."""
        key = ("samples", float(tol))
        if key in self._cache:
            return self._cache[key]
        M = self._quadrature_points()
        theta = 2.0 * np.pi * np.arange(M) / M
        plan = self.tail_plan(K_HEAD, tol)
        S0, S1, S2 = self._alias_moments(theta, plan["K"], plan["A"])
        n = np.arange(-self.nmax, self.nmax + 1)
        phase = np.exp(1j * np.outer(n, theta)) / M
        beta = (phase @ S0).real
        dbeta = (phase @ (1j * S1)).real
        ddbeta = (phase @ (-S2)).real
        truncation = {"route": "alias_plus_zeta_tail", "K": int(plan["K"]), "A": int(plan["A"]),
                      "quadrature_points": int(M), "tail_bound": float(plan["tail_bound"])}
        samples = fam.IntegerSamples(n, beta, dbeta, ddbeta, truncation)
        self._cache[key] = samples
        return samples

    # ---- symbols from the samples (cancellation-free forms) --------------

    def symbols(self, theta, tol: float = TAIL_TOL):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        s = self.integer_samples(tol)
        pos = s.n > 0
        nn = s.n[pos].astype(float)
        arg = np.outer(theta, nn)
        b = s.beta[s.n == 0][0] + 2.0 * (np.cos(arg) @ s.beta[pos])
        k1b = -2.0 * (np.sin(arg) @ s.dbeta[pos])
        k2b = 4.0 * (np.sin(0.5 * arg) ** 2 @ s.ddbeta[pos])
        return b, k1b / b, k2b / b

    def symbol_derivative(self, theta, tol: float = TAIL_TOL) -> np.ndarray:
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
        theta = np.linspace(0.0, np.pi, theta_points)
        b = self.symbols(theta)[0]
        out = {"cond": float(b.max() / b.min()), "max_b": float(b.max()),
               "min_b": float(b.min()), "positive": bool(b.min() > 0.0),
               "argmax_b_over_pi": float(theta[int(np.argmax(b))] / np.pi),
               "argmin_b_over_pi": float(theta[int(np.argmin(b))] / np.pi)}
        for N in discrete:
            angles = 2.0 * np.pi * np.arange(N) / N
            bn = self.symbols(angles)[0]
            out[f"cond_N{N}"] = float(bn.max() / bn.min())
        return out

    def admissible(self, theta_points: int = 4001) -> tuple[bool, str]:
        for slot in self.slots:
            if slot.kind != "real" and not (0.0 < slot.omega < np.pi):
                return False, "frequency outside (0, pi)"
        if poly_min_on_band(self.poly) <= 0.0:
            return False, "P not positive on [0, pi^2]"
        theta = np.linspace(0.0, np.pi, theta_points)
        b, _, k2 = self.symbols(theta)
        if not np.all(np.isfinite(b)) or float(b.min()) <= 0.0:
            return False, "sample symbol not positive"
        if float(k2[1:].min()) <= 0.0:
            return False, "kappa_2 not positive"
        return True, "admissible"

    # ---- residue route ----------------------------------------------------

    def _g_series(self, order: int) -> np.ndarray:
        """Return the Taylor coefficients in z (even powers) of P(z^2)/prod(z^2 + s_j)."""
        svals = self.s_values()
        n_even = order // 2 + 1
        series = np.zeros(n_even, dtype=complex)
        series[0] = 1.0
        for s in svals:
            geo = (-1.0 / s) ** np.arange(n_even) / s
            series = np.convolve(series, geo)[:n_even]
        pc = np.zeros(n_even, dtype=complex)
        for i, p in enumerate([1.0] + list(self.poly)):
            if i < n_even:
                pc[i] = p
        series = np.convolve(series, pc)[:n_even]
        out = np.zeros(order + 1, dtype=complex)
        out[::2] = series
        return out

    def offgrid_moments(self, theta, xi):
        """Return S_j(theta, xi) = sum_k xi_k^j psi_hat(xi_k) e^{2 pi i k xi}, j = 0, 1, 2.

        Shape (3, theta.size, xi.size), complex. Computed by the residue closed
        form of the module docstring; theta = 0 (mod 2 pi) is the exact value
        (1, 0, 0) because psi_hat(2 pi k) = 0 for k != 0.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        xip = xi - np.floor(xi)
        T, X = theta.size, xi.size
        out = np.zeros((3, T, X), dtype=complex)
        red = np.mod(theta, 2.0 * np.pi)
        zero = np.abs(red) < 1e-300
        out[0, zero, :] = 1.0
        if np.all(zero):
            return out
        th = theta[~zero]
        m = self.m
        svals = self.s_values()
        if svals.size > 1:
            sep = min(abs(svals[i] - svals[j]) for i in range(svals.size) for j in range(i))
            if sep < POLE_SEPARATION:
                raise ValueError("confluent slots: the residue route needs distinct s_j")
        # k-free part, with the imaginary factors kept separate
        base = (2.0 * np.sin(th / 2.0)) ** m
        factors = [g7.slot_k_free(slot, th) for slot in self.slots]
        kf_all = base.copy()
        for f in factors:
            kf_all = kf_all * f
        phase = xip - 0.5
        E_th = np.exp(-1j * np.outer(th, phase))  # exp(-i theta (xi' - 1/2))
        acc = np.zeros((3, th.size, X), dtype=complex)
        # simple poles
        idx = 0
        for slot, f_l in zip(self.slots, factors):
            count = 2 if slot.kind == "quad" else 1
            for c in range(count):
                s = svals[idx]
                others = np.prod([svals[i] - s for i in range(svals.size) if i != idx]) if svals.size > 1 else 1.0
                Ps = poly_value(self.poly, np.array([(-s).real]))[0] if abs(s.imag) == 0.0 else self._poly_complex(-s)
                for sign in (1.0, -1.0):
                    a = sign * np.sqrt(-s + 0j)
                    if slot.kind == "imaginary":
                        w = slot.omega
                        a = sign * w
                        # c(-w^2, theta) h(a) in the cancelled form
                        kf_other = base.copy()
                        for f2 in factors:
                            if f2 is not f_l:
                                kf_other = kf_other * f2
                        pref = -w * w / (1.0 - np.cos(w))
                        if sign > 0:
                            trig = np.sin((th + w) / 2.0)
                        else:
                            trig = np.sin((th - w) / 2.0)
                        kern = np.exp(1j * np.outer(a - th, phase))
                        combined = (kf_other * pref * trig)[:, None] * kern
                    else:
                        kern = np.exp(1j * np.outer(a - th, phase)) / (2.0 * np.sin((a - th) / 2.0))[:, None]
                        combined = kf_all[:, None] * kern
                    for j in range(3):
                        res = a ** (j - m - 1) * Ps / (2.0 * others)
                        acc[j] -= res * combined
                idx += 1
        # pole of order m - j at zero: series of h and of G
        order = m - 1
        u0 = -th / 2.0
        cot = np.zeros((order + 1, th.size))
        csc = np.zeros((order + 1, th.size))
        cot[0] = 1.0 / np.tan(u0)
        csc[0] = 1.0 / np.sin(u0)
        for nn in range(order):
            conv_c = sum(cot[i] * cot[nn - i] for i in range(nn + 1))
            cot[nn + 1] = -((1.0 if nn == 0 else 0.0) + conv_c) / (nn + 1)
            conv_d = sum(csc[i] * cot[nn - i] for i in range(nn + 1))
            csc[nn + 1] = -conv_d / (nn + 1)
        # coefficients of 1/(2 sin((z - theta)/2)) in z: csc_n / 2^{n+1}
        dz = np.array([csc[nn] / 2.0 ** (nn + 1) for nn in range(order + 1)])
        # exp(i (z - theta)(xi' - 1/2)) = E_th * sum (i phase)^n z^n / n!
        ipow = np.array([(1j * phase) ** nn / factorial(nn) for nn in range(order + 1)])  # (order+1, X)
        h = np.zeros((order + 1, th.size, X), dtype=complex)
        for nn in range(order + 1):
            for i in range(nn + 1):
                h[nn] += dz[nn - i][:, None] * ipow[i][None, :]
            h[nn] *= E_th
        g = self._g_series(order)
        for j in range(3):
            target = m - j - 1
            res0 = np.zeros((th.size, X), dtype=complex)
            for a2 in range(0, target + 1, 2):
                res0 += g[a2] * h[target - a2]
            acc[j] -= kf_all[:, None] * res0
        out[:, ~zero, :] = acc
        return out

    def _poly_complex(self, t: complex) -> complex:
        out = 0.0 + 0j
        for p in reversed(self.poly):
            out = out * t + p
        return out * t + 1.0

    def residue_symbols(self, theta):
        """Return b, kappa_1, kappa_2 from the residue closed form (nodal, xi = 0)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        S = self.offgrid_moments(theta, np.array([0.0]))[:, :, 0].real
        return S[0], S[1] / S[0], S[2] / S[0]

    def offgrid_transfers(self, theta, xi):
        """Return T_0, T_1, T_2 of the Gate 0 contract, shape (theta.size, xi.size)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        S = self.offgrid_moments(theta, xi)
        b = self.symbols(theta)[0][:, None]
        T0 = S[0] / b
        T1 = S[1] / (theta[:, None] * b)
        T2 = S[2] / (theta[:, None] ** 2 * b)
        return T0, T1, T2

    def coupled_error(self, theta, xi, tol: float = TAIL_TOL) -> np.ndarray:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        T0, T1, T2 = self.offgrid_transfers(theta, xi)
        err = np.maximum(np.abs(T0 - 1.0), np.abs(T1 - 1.0))
        err = np.maximum(err, np.abs(T2 - 1.0))
        return err.max(axis=1)

    # ---- records ----------------------------------------------------------

    def member_record(self, name: str) -> dict:
        """Return the member file: E62 fields, the Gate 7 slot fields, and P.

        "polynomial" holds [p_1, ..., p_q] with psi_hat = (1 + sum_l p_l xi^{2l})
        phi_hat; "polynomial_scaled" holds p_l pi^{2l}; "smoothness" is the class
        C^{n-2-2q} of psi. The samples are the alias route values with the exact
        zeta tail, verified against the residue and decimal routes.
        """
        samples = self.integer_samples()
        radius = self.nmax
        center = int(np.searchsorted(samples.n, 0))
        one = slice(center, center + radius + 1)
        rec = self.base.member_record(name)
        rec.update({
            "cond": float(self.cond()["cond"]),
            "beta": [float(v) for v in samples.beta[one]],
            "dbeta": [float(v) for v in samples.dbeta[one]],
            "ddbeta": [float(v) for v in samples.ddbeta[one]],
            "polynomial": [float(p) for p in self.poly],
            "polynomial_scaled": [float(p * np.pi ** (2 * (l + 1))) for l, p in enumerate(self.poly)],
            "polynomial_degree": int(self.q),
            "polynomial_roots": poly_roots_xi(self.poly),
            "smoothness": int(self.smoothness),
            "parameters": int(self.parameters),
            "truncation": samples.truncation,
            "schema": "E62_champion.json fields plus real_pairs, quads, exponents, parameters "
                      "(gate7_family.py) plus polynomial: psi_hat(xi) = P(xi^2) phi_hat(xi) with "
                      "P(t) = 1 + sum_l polynomial[l-1] t^l; psi = P(-D^2) phi (gate8_family.py)",
        })
        return rec


def from_record(record: dict) -> PolynomialGenerator:
    base = g7.from_record(record)
    return PolynomialGenerator(base.m, base.slots, record.get("polynomial", []))


# ---------------------------------------------------------------------------
# Sufficient conditions and certificates
# ---------------------------------------------------------------------------

def S_polynomial(m: int, poly, w: float, terms: int = 200000) -> float:
    """Return the Gate 8 S-series 2 sum_{k>=1} (w/(2 pi k - w))^m |P|((2 pi k - w)^2) / P_min.

    |P|(t) = sum_l |p_l| t^l bounds P on the aliases and P_min = min P on
    [0, pi^2] bounds the k = 0 weight from below; the series converges when
    m > 2 q. Together with Lemma E' this is the hypothesis of Theorem P''.
    """
    k = np.arange(1, terms + 1, dtype=float)
    x = 2.0 * np.pi * k - w
    absP = poly_value([abs(p) for p in poly], x * x)
    pmin = poly_min_on_band(poly)
    if pmin <= 0.0 or m <= 2 * len(poly):
        return np.inf
    return float(2.0 * np.sum((w / x) ** m * absP) / pmin)


def sufficient_conditions(gen: PolynomialGenerator) -> dict:
    """Return the Gate 8 sufficient conditions (Theorems P'' and N'') and flags."""
    roots = poly_roots_xi(gen.poly)
    base = g7.sufficient_conditions(gen.base)
    out = {"omega_below_pi": base["omega_below_pi"], "has_imaginary": base["has_imaginary"],
           "P_positive_on_line": roots["positive_on_line"],
           "P_min_on_band": poly_min_on_band(gen.poly),
           "roots": roots, "m_minus_2q": gen.m - 2 * gen.q}
    if base["has_imaginary"]:
        w = max(gen.omegas)
        out["S_m_P"] = S_polynomial(gen.m, gen.poly, w)
        out["S_m_minus_2_P"] = S_polynomial(gen.m - 2, gen.poly, w) if gen.m >= 4 else np.inf
    else:
        out["S_m_P"] = 0.0
        out["S_m_minus_2_P"] = 0.0
    out["theorem_P"] = bool(out["omega_below_pi"] and roots["positive_on_line"]
                            and out["P_min_on_band"] > 0.0 and out["S_m_P"] < 1.0)
    out["theorem_N"] = bool(gen.m >= 4 and out["omega_below_pi"] and roots["positive_on_line"]
                            and out["P_min_on_band"] > 0.0 and out["S_m_minus_2_P"] < 1.0)
    return out


def certify_positivity(gen: PolynomialGenerator, points: int = 200001) -> dict:
    """Certify b > 0 and kappa_2 b > 0 on (0, pi] (Gate 7 grid-plus-Lipschitz argument)."""
    return g7.certify_positivity(gen, points=points)


# ---------------------------------------------------------------------------
# The 60-digit decimal route
# ---------------------------------------------------------------------------

def _bernoulli_numbers(count: int) -> list:
    """Return B_0..B_count as Fractions (B_1 = -1/2)."""
    B = [Fraction(0)] * (count + 1)
    for mm in range(count + 1):
        acc = Fraction(0)
        for kk in range(mm):
            acc += Fraction(comb(mm + 1, kk)) * B[kk]
        B[mm] = -acc / (mm + 1) if mm > 0 else Fraction(1)
    return B


def decimal_hurwitz_zeta(w: int, a: Decimal, head: int = 24, terms: int = 30) -> Decimal:
    """Return zeta(w, a) for integer w >= 2 by Euler-Maclaurin at the current precision.

    zeta(w, a) = sum_{k<N} (a+k)^{-w} + (a+N)^{1-w}/(w-1) + (a+N)^{-w}/2
                 + sum_{j=1}^{J} B_{2j}/(2j)! (w)_{2j-1} (a+N)^{-w-2j+1} + R,
    with (w)_r the rising factorial and |R| below the first omitted term.
    """
    B = _bernoulli_numbers(2 * terms + 2)
    w = int(w)
    total = Decimal(0)
    for k in range(head):
        total += (a + k) ** (-w)
    aN = a + head
    total += aN ** (1 - w) / (w - 1)
    total += aN ** (-w) / 2
    rising = Decimal(w)
    for j in range(1, terms + 1):
        # (w)_{2j-1} = w (w+1) ... (w + 2j - 2)
        if j > 1:
            rising *= (w + 2 * j - 3) * (w + 2 * j - 2)
        coeff = Decimal(B[2 * j].numerator) / Decimal(B[2 * j].denominator) / Decimal(factorial(2 * j))
        total += coeff * rising * aN ** (-w - 2 * j + 1)
    return total


def decimal_integer_samples(gen: PolynomialGenerator, K: int, A: int,
                            prec: int = DECIMAL_PREC) -> dict:
    """Return beta, beta', beta'' in decimal by the alias route with the zeta tail.

    The head sums the factored aliases kfree(theta) xi^j P(xi^2) R(xi) for
    |k| <= K exactly as gate7_family.decimal_integer_samples does, and the
    tail |k| > K is the Hurwitz zeta expansion with A terms. K and A may
    differ from the double route's, so the split itself is checked.
    """
    M = gen._quadrature_points()
    nmax = gen.nmax
    with localcontext() as ctx:
        ctx.prec = int(prec)
        trig = DecimalTrig(int(prec))
        pi = trig.pi
        two_pi = trig.two_pi
        slots = []
        for slot in gen.slots:
            a = Decimal(float(slot.alpha))
            w = Decimal(float(slot.omega))
            if slot.kind == "imaginary":
                slots.append(("imaginary", w * w, trig.cos(w)))
            elif slot.kind == "real":
                ca, _ = g7._dec_cosh_sinh(a)
                slots.append(("real", a * a, ca))
            else:
                ca, sa = g7._dec_cosh_sinh(a)
                cw, sw = trig.cos(w), trig.sin(w)
                slots.append(("quad", (a, w), (ca * cw, sa * sw)))
        poly = [Decimal(1)] + [Decimal(float(p)) for p in gen.poly]
        # tail series d_a of prod (1 + s u)^{-1}, real recurrences per slot
        d = [Decimal(0)] * A
        d[0] = Decimal(1)
        for kind, par, aux in slots:
            if kind == "imaginary":
                s = -par
                geo = [(-s) ** i for i in range(A)]
            elif kind == "real":
                s = par
                geo = [(-s) ** i for i in range(A)]
            else:
                a_, w_ = par
                bsum = 2 * (a_ * a_ - w_ * w_)
                cprod = (a_ * a_ + w_ * w_) ** 2
                geo = [Decimal(0)] * A
                geo[0] = Decimal(1)
                if A > 1:
                    geo[1] = -bsum
                for i in range(2, A):
                    geo[i] = -bsum * geo[i - 1] - cprod * geo[i - 2]
            new = [Decimal(0)] * A
            for i in range(A):
                if d[i] == 0:
                    continue
                for jj in range(A - i):
                    new[i + jj] += d[i] * geo[jj]
            d = new
        n_exp = gen.n
        S = [[Decimal(0)] * M for _ in range(3)]
        for j_ang in range(M):
            theta = two_pi * j_ang / M
            half = pi * j_ang / M
            sin_half = trig.sin(half)
            cos_theta = trig.cos(theta)
            kfree = (2 * sin_half) ** gen.m
            for kind, par, aux in slots:
                if kind == "imaginary":
                    w2, cw = par, aux
                    kfree *= w2 * (cw - cos_theta) / (1 - cw)
                elif kind == "real":
                    a2, ca = par, aux
                    kfree *= a2 * (ca - cos_theta) / (ca - 1)
                else:
                    a_, w_ = par
                    cc, ss = aux
                    kfree *= ((a_ * a_ + w_ * w_) ** 2 * ((cc - cos_theta) ** 2 + ss * ss)
                              / ((cc - 1) ** 2 + ss * ss))
            s0 = s1 = s2 = Decimal(0)
            for k in range(-K, K + 1):
                xi = theta + two_pi * k
                if xi == 0:
                    P = Decimal(1)
                else:
                    xi2 = xi * xi
                    rat = 1 / (xi ** gen.m)
                    for kind, par, aux in slots:
                        if kind == "imaginary":
                            rat /= (xi2 - par)
                        elif kind == "real":
                            rat /= (xi2 + par)
                        else:
                            a_, w_ = par
                            rat /= ((xi2 + a_ * a_ - w_ * w_) ** 2 + 4 * a_ * a_ * w_ * w_)
                    pv = Decimal(0)
                    for p in reversed(poly):
                        pv = pv * xi2 + p
                    P = kfree * rat * pv
                s0 += P
                s1 += xi * P
                s2 += xi * xi * P
            # tail
            qp = K + 1 + theta / two_pi
            qm = K + 1 - theta / two_pi
            for j in range(3):
                weights: dict = {}
                for i, p in enumerate(poly):
                    for a in range(A):
                        w = n_exp - j - 2 * i + 2 * a
                        weights[w] = weights.get(w, Decimal(0)) + p * d[a]
                tail = Decimal(0)
                for w, c in weights.items():
                    if c == 0:
                        continue
                    zp = decimal_hurwitz_zeta(w, qp)
                    zm = decimal_hurwitz_zeta(w, qm)
                    tail += c * two_pi ** (-w) * (zp + (-1) ** w * zm)
                if j == 0:
                    s0 += kfree * tail
                elif j == 1:
                    s1 += kfree * tail
                else:
                    s2 += kfree * tail
            S[0][j_ang], S[1][j_ang], S[2][j_ang] = s0, s1, s2
        cos_table, sin_table = {}, {}
        for residue in range(M):
            angle = two_pi * residue / M
            cos_table[residue] = trig.cos(angle)
            sin_table[residue] = trig.sin(angle)
        beta, dbeta, ddbeta = [], [], []
        for nn in range(-nmax, nmax + 1):
            a0 = a1 = a2 = Decimal(0)
            for j_ang in range(M):
                residue = (nn * j_ang) % M
                a0 += S[0][j_ang] * cos_table[residue]
                a1 += S[1][j_ang] * sin_table[residue]
                a2 += S[2][j_ang] * cos_table[residue]
            beta.append(a0 / M)
            dbeta.append(-a1 / M)
            ddbeta.append(-a2 / M)
        series_bound = float(trig.bound)
    return {"n": list(range(-nmax, nmax + 1)),
            "beta": [float(v) for v in beta], "dbeta": [float(v) for v in dbeta],
            "ddbeta": [float(v) for v in ddbeta], "beta_digits": [str(v) for v in beta],
            "K": int(K), "A": int(A), "quadrature_points": int(M), "working_digits": int(prec),
            "series_truncation_bound": series_bound}


def verify_routes(gen: PolynomialGenerator, route_tol: float = ROUTE_TOL,
                  residue_tol: float = RESIDUE_TOL) -> dict:
    """Return the agreement of the alias route, the residue route, the decimal route,
    and (for P = 1) the Gate 7 route on the integer samples and symbols."""
    alias = gen.integer_samples()
    M = gen._quadrature_points()
    theta = 2.0 * np.pi * np.arange(M) / M
    plan = gen.tail_plan()
    S_alias = gen._alias_moments(theta, plan["K"], plan["A"])
    S_res = gen.offgrid_moments(theta, np.array([0.0]))[:, :, 0]
    n = np.arange(-gen.nmax, gen.nmax + 1)
    phase = np.exp(1j * np.outer(n, theta)) / M
    res_samples = [(phase @ S_res[0]).real, (phase @ (1j * S_res[1])).real, (phase @ (-S_res[2])).real]
    residue_dev = max(float(np.max(np.abs(res_samples[0] - alias.beta))),
                      float(np.max(np.abs(res_samples[1] - alias.dbeta))),
                      float(np.max(np.abs(res_samples[2] - alias.ddbeta))))
    residue_symbol_dev = max(float(np.max(np.abs(S_res[j].real - S_alias[j])) / max(1.0, float(np.max(np.abs(S_alias[j])))))
                             for j in range(3))
    dec = decimal_integer_samples(gen, 2 * plan["K"], plan["A"] + 8)
    dec_dev = max(float(np.max(np.abs(np.asarray(dec[key]) - arr)))
                  for key, arr in (("beta", alias.beta), ("dbeta", alias.dbeta), ("ddbeta", alias.ddbeta)))
    probe = np.linspace(0.05, np.pi, 41)
    direct = gen.sample_symbols(probe)
    from_samples = gen.symbols(probe)
    gap = max(float(np.max(np.abs(direct[j] - from_samples[j])) / max(1.0, float(np.max(np.abs(from_samples[j])))))
              for j in range(3))
    endpoint = max(abs(float(alias.beta[0])), abs(float(alias.beta[-1])), abs(float(alias.dbeta[0])),
                   abs(float(alias.dbeta[-1])), abs(float(alias.ddbeta[0])), abs(float(alias.ddbeta[-1])))
    gate7_dev = None
    if gen.q == 0 or all(p == 0.0 for p in gen.poly):
        s7 = gen.base.integer_samples()
        gate7_dev = max(float(np.max(np.abs(a - b))) for a, b in
                        ((s7.beta, alias.beta), (s7.dbeta, alias.dbeta), (s7.ddbeta, alias.ddbeta)))
    return {"decimal_deviation": dec_dev, "decimal_ok": bool(dec_dev <= route_tol),
            "residue_deviation": residue_dev, "residue_ok": bool(residue_dev <= residue_tol),
            "residue_symbol_deviation": residue_symbol_dev,
            "symbols_from_samples_versus_alias": gap, "symbols_ok": bool(gap <= 1e-12),
            "endpoint_samples": endpoint, "endpoint_ok": bool(endpoint <= 1e-13),
            "gate7_deviation": gate7_dev, "gate7_ok": bool(gate7_dev is None or gate7_dev <= route_tol),
            "K": int(plan["K"]), "A": int(plan["A"]), "tail_bound": float(plan["tail_bound"]),
            "decimal_K": int(dec["K"]), "decimal_A": int(dec["A"]),
            "decimal_series_bound": dec["series_truncation_bound"], "beta_digits": dec["beta_digits"]}


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

def parameter_counts(n: int) -> dict:
    """Return the dimension of the augmented family against the compact class.

    At width n the compact class of type (n/2 - 1, n/2 - 1) has n - 3 free
    parameters (Proposition 7 of Gate 7); requiring formal order at least m
    (even) costs (m - 2)/2 further conditions, so the compact class of order
    >= m has n - 2 - m/2 parameters. The augmented family with core m and
    polynomial degree q has (n - m)/2 + q parameters; at the C^2 limit
    q = (n - 4)/2 this is n - 2 - m/2, the same number, and at the C^4
    restriction q = (n - 6)/2 it is one less.
    """
    L = R = n // 2 - 1
    out = {"n": int(n), "p": int(n - 1), "L": int(L), "R": int(R),
           "compact_free_parameters": int(L + R - 1), "q_C2": int((n - 4) // 2),
           "q_C4": int((n - 6) // 2), "family": {}}
    for m in range(4, n + 1, 2):
        compact_order_m = L + R - 1 - (m - 2) // 2
        fam_c2 = (n - m) // 2 + (n - 4) // 2
        fam_c4 = (n - m) // 2 + max(0, (n - 6) // 2)
        out["family"][str(m)] = {"gate7_parameters": int((n - m) // 2),
                                 "compact_parameters_of_order_m": int(compact_order_m),
                                 "parameters_C2": int(fam_c2), "gap_C2": int(compact_order_m - fam_c2),
                                 "parameters_C4": int(fam_c4), "gap_C4": int(compact_order_m - fam_c4)}
    return out
