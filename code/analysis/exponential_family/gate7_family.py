"""Gate 7: the exponential cardinal B-spline family with real parts.

The exponent multiset is closed under negation and conjugation and consists of

    zeros            0 with multiplicity m (m even, m >= 2),
    imaginary pairs  +- i omega,                 one real parameter omega,
    real pairs       +- alpha,                   one real parameter alpha,
    quads            +- alpha +- i omega,        two real parameters.

Every pair {+lambda, -lambda} is one slot with the real or complex parameter
s = lambda^2; a quad occupies the two conjugate slots s and conj(s). The
generator is the convolution of the normalized centered first-order factors

    b_lambda(x) = c_lambda exp(lambda x) 1_{[-1/2, 1/2]}(x),
    c_lambda    = lambda / (2 sinh(lambda/2)),

whose transform is c_lambda 2 sinh((lambda - i xi)/2) / (lambda - i xi), equal
to one at xi = 0. The two factors of a slot multiply to the slot factor

    g(s, xi) = s (cosh sqrt(s) - cos xi) / ((cosh sqrt(s) - 1) (xi^2 + s)),

an entire function of s (both sqrt(s) branches give the same value), equal to
one at xi = 0, and

    phi_hat(xi) = sinc(xi/2)^m prod_slots g(s_j, xi).

For s = -omega^2 the slot factor is the Gate 0 pair factor
sinc((xi - omega)/2) sinc((xi + omega)/2) / sinc(omega/2)^2; for s = alpha^2 it
is alpha^2 (cosh alpha - cos xi) / ((cosh alpha - 1)(xi^2 + alpha^2)); a quad
contributes |g(s, xi)|^2 with s = (alpha + i omega)^2, which is evaluated in
real arithmetic as

    |s|^2 |cosh(lambda) - cos xi|^2 / (|cosh(lambda) - 1|^2 |xi^2 + s|^2),
    |cosh(lambda) - z|^2 = (cosh(alpha) cos(omega) - z)^2 + (sinh(alpha) sin(omega))^2,
    |xi^2 + s|^2        = (xi^2 + alpha^2 - omega^2)^2 + 4 alpha^2 omega^2.

Support is [-n/2, n/2] with n = m + 2 q exponents (q slots), the degree is
p = n - 1, the generator is real, even, and C^{n-2}. Since cos(theta + 2 pi k)
= cos(theta), every alias of the transform factors as a k-independent part
times a rational function of xi = theta + 2 pi k (Lemma D of Gate 7),

    phi_hat(theta + 2 pi k) = (2 sin(theta/2))^m prod_j c(s_j, theta)
                              / [ xi^m prod_j (xi^2 + s_j) ],
    c(s, theta) = s (cosh sqrt(s) - cos theta) / (cosh sqrt(s) - 1),

for even m. The tail bound used to truncate the alias sums is factorwise: for
|xi| >= X with X^2 > max |s_j|,

    |g(s, xi)| <= G(s) / (xi^2 (1 - |s|/X^2)),
    G(s) = |s| (|cosh sqrt(s)| + 1) / |cosh sqrt(s) - 1|,

so |phi_hat(xi)| <= C |xi|^{-n} with C = 2^m prod_j G(s_j) / (1 - |s_j|/X^2),
and the neglected q-th moment tail beyond |k| = K is bounded exactly as in
family.py. The rest of the sampling calculus (trapezoidal inverse on M > 2 nmax
angles, cancellation-free symbol forms, off-grid transfers) is the Gate 0 one.

A 60-digit decimal route recomputes the integer samples from the alias sums
in the factored form above, where the only k-dependence is rational, with the
same truncation index K as the double route; the two routes must agree to
1e-13 for every member that is reported.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import family as fam  # noqa: E402
from gate1_champions import DecimalTrig, DECIMAL_PREC  # noqa: E402

K_MAX = 65536
DEFAULT_TOL = 1e-14
CHUNK = 1 << 21
ALPHA_MAX = 40.0
ROUTE_TOL = 1e-13
SYMBOL_ROUTE_TOL = 1e-12
SLOT_KIND = ("imaginary", "real", "quad")


# ---------------------------------------------------------------------------
# Slots and the slot factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    """One pair {+lambda, -lambda} with lambda = alpha + i omega, or a quad.

    kind is "imaginary" (alpha = 0, omega > 0), "real" (alpha > 0, omega = 0)
    or "quad" (alpha > 0, omega > 0, counting for two conjugate slots).
    """

    alpha: float
    omega: float

    @property
    def kind(self) -> str:
        if self.alpha == 0.0 and self.omega > 0.0:
            return "imaginary"
        if self.alpha > 0.0 and self.omega == 0.0:
            return "real"
        if self.alpha > 0.0 and self.omega > 0.0:
            return "quad"
        raise ValueError(f"degenerate slot alpha={self.alpha} omega={self.omega}")

    @property
    def s(self) -> complex:
        lam = complex(self.alpha, self.omega)
        return lam * lam

    @property
    def exponents(self) -> int:
        return 4 if self.kind == "quad" else 2

    @property
    def parameters(self) -> int:
        return 2 if self.kind == "quad" else 1

    def label(self) -> str:
        if self.kind == "imaginary":
            return f"i{self.omega:.6f}"
        if self.kind == "real":
            return f"r{self.alpha:.6f}"
        return f"q({self.alpha:.6f},{self.omega:.6f})"


def slot_factor(slot: Slot, xi: np.ndarray) -> np.ndarray:
    """Return the slot factor of the transform at real xi, in real arithmetic."""
    xi = np.asarray(xi, dtype=float)
    if slot.kind == "imaginary":
        w = slot.omega
        scale = float(fam.sinc(np.array(w / 2.0))) ** 2
        return fam.sinc((xi - w) / 2.0) * fam.sinc((xi + w) / 2.0) / scale
    if slot.kind == "real":
        a = slot.alpha
        ch = np.cosh(a)
        return a * a * (ch - np.cos(xi)) / ((ch - 1.0) * (xi * xi + a * a))
    a, w = slot.alpha, slot.omega
    ca, sa = np.cosh(a), np.sinh(a)
    cw, sw = np.cos(w), np.sin(w)
    s_abs2 = (a * a + w * w) ** 2
    num = (ca * cw - np.cos(xi)) ** 2 + (sa * sw) ** 2
    den = ((ca * cw - 1.0) ** 2 + (sa * sw) ** 2) * ((xi * xi + a * a - w * w) ** 2
                                                     + 4.0 * a * a * w * w)
    return s_abs2 * num / den


def slot_k_free(slot: Slot, theta: np.ndarray) -> np.ndarray:
    """Return the k-independent part c(s, theta) of the alias factorization.

    For a quad the returned value is |c(s, theta)|^2, the product over its two
    conjugate slots.
    """
    theta = np.asarray(theta, dtype=float)
    if slot.kind == "imaginary":
        w = slot.omega
        return w * w * (np.cos(w) - np.cos(theta)) / (1.0 - np.cos(w))
    if slot.kind == "real":
        a = slot.alpha
        ch = np.cosh(a)
        return a * a * (ch - np.cos(theta)) / (ch - 1.0)
    a, w = slot.alpha, slot.omega
    ca, sa = np.cosh(a), np.sinh(a)
    cw, sw = np.cos(w), np.sin(w)
    num = (ca * cw - np.cos(theta)) ** 2 + (sa * sw) ** 2
    den = (ca * cw - 1.0) ** 2 + (sa * sw) ** 2
    return (a * a + w * w) ** 2 * num / den


def slot_rational(slot: Slot, xi: np.ndarray) -> np.ndarray:
    """Return the rational part 1/(xi^2 + s) (or 1/|xi^2 + s|^2 for a quad)."""
    xi = np.asarray(xi, dtype=float)
    if slot.kind == "imaginary":
        return 1.0 / (xi * xi - slot.omega ** 2)
    if slot.kind == "real":
        return 1.0 / (xi * xi + slot.alpha ** 2)
    a, w = slot.alpha, slot.omega
    return 1.0 / ((xi * xi + a * a - w * w) ** 2 + 4.0 * a * a * w * w)


def slot_tail_constant(slot: Slot) -> tuple[float, float]:
    """Return (G, |s|) of the factorwise bound |g(s, xi)| <= G/(xi^2 (1 - |s|/X^2))."""
    s = slot.s
    ch = np.cosh(np.sqrt(s))
    G = abs(s) * (abs(ch) + 1.0) / abs(ch - 1.0)
    return float(G), float(abs(s))


# ---------------------------------------------------------------------------
# The extended generator
# ---------------------------------------------------------------------------

class ExtendedGenerator:
    """A centered exponential cardinal B-spline with the extended multiset."""

    def __init__(self, m: int, slots) -> None:
        m = int(m)
        if m < 2 or m % 2:
            raise ValueError("Gate 7 members use an even polynomial core m >= 2.")
        parsed = []
        for item in slots:
            if isinstance(item, Slot):
                parsed.append(item)
            else:
                a, w = item
                parsed.append(Slot(float(a), float(w)))
        for slot in parsed:
            slot.kind  # validates
            if slot.alpha > ALPHA_MAX:
                raise ValueError("real part above the documented range")
        self.m = m
        self.slots = tuple(parsed)
        self._cache: dict = {}

    # ---- bookkeeping ------------------------------------------------------

    @property
    def n(self) -> int:
        """Return the number of exponents, the support width."""
        return self.m + sum(slot.exponents for slot in self.slots)

    @property
    def p(self) -> int:
        return self.n - 1

    @property
    def nmax(self) -> int:
        return (self.p + 1) // 2

    @property
    def q(self) -> int:
        """Return the number of pair slots (a quad counts two)."""
        return (self.n - self.m) // 2

    @property
    def parameters(self) -> int:
        """Return the number of continuous real parameters of the member."""
        return sum(slot.parameters for slot in self.slots)

    @property
    def omegas(self) -> tuple:
        """Return the imaginary pair frequencies, increasing."""
        return tuple(sorted(s.omega for s in self.slots if s.kind == "imaginary"))

    @property
    def real_pairs(self) -> tuple:
        return tuple(sorted(s.alpha for s in self.slots if s.kind == "real"))

    @property
    def quads(self) -> tuple:
        return tuple(sorted((s.alpha, s.omega) for s in self.slots if s.kind == "quad"))

    @property
    def r(self) -> int:
        return len(self.omegas)

    def kinds(self) -> str:
        return "".join(sorted(s.kind[0].upper() for s in self.slots))

    def label(self) -> str:
        inner = ",".join(s.label() for s in sorted(self.slots, key=lambda s: (s.kind, s.omega, s.alpha)))
        return f"X_{{{self.m}}}({inner})"

    def gate0_generator(self):
        """Return the Gate 0 generator when every slot is an imaginary pair."""
        if any(s.kind != "imaginary" for s in self.slots):
            return None
        return fam.ExponentialGenerator(self.m, self.omegas)

    # ---- transform and alias sums ----------------------------------------

    def fourier(self, xi) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        value = fam.sinc(xi / 2.0) ** self.m
        for slot in self.slots:
            value = value * slot_factor(slot, xi)
        return value

    def k_free_part(self, theta) -> np.ndarray:
        """Return (2 sin(theta/2))^m prod_j c(s_j, theta) of Lemma D."""
        theta = np.asarray(theta, dtype=float)
        value = (2.0 * np.sin(theta / 2.0)) ** self.m
        for slot in self.slots:
            value = value * slot_k_free(slot, theta)
        return value

    def rational_part(self, xi) -> np.ndarray:
        """Return xi^{-m} prod_j 1/(xi^2 + s_j) of Lemma D."""
        xi = np.asarray(xi, dtype=float)
        value = xi ** (-float(self.m))
        for slot in self.slots:
            value = value * slot_rational(slot, xi)
        return value

    def tail_constant(self, X: float) -> float:
        C = 2.0 ** self.m
        for slot in self.slots:
            G, s_abs = slot_tail_constant(slot)
            shrink = 1.0 - s_abs / (X * X)
            if shrink <= 0.0:
                return np.inf
            power = 2 if slot.kind == "quad" else 1
            C *= (G / shrink) ** power
        return C

    def tail_bound(self, q: int, K: int) -> float:
        s = self.n - q
        if s < 2:
            return np.inf
        X0 = 2.0 * np.pi * K
        C = self.tail_constant(X0)
        if not np.isfinite(C):
            return np.inf
        return 2.0 * C * (X0 ** (-s) + X0 ** (1 - s) / (2.0 * np.pi * (s - 1)))

    def required_K(self, q: int, tol: float = DEFAULT_TOL) -> int:
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

    def alias_K(self, tol: float = DEFAULT_TOL) -> int:
        K = max(self.required_K(q, tol) for q in (0, 1, 2))
        if K > K_MAX:
            raise ValueError(f"alias tail unreachable at tol {tol} for {self.label()}")
        return K

    def _alias_moments(self, theta: np.ndarray, K: int):
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
        """Return b, kappa_1, kappa_2 from the truncated alias sums."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if K is None:
            K = self.alias_K(tol)
        S0, S1, S2 = self._alias_moments(theta, K)
        return S0, S1 / S0, S2 / S0

    def _quadrature_points(self) -> int:
        return 4 * self.nmax + 8

    def integer_samples(self, tol: float = DEFAULT_TOL) -> fam.IntegerSamples:
        key = ("samples", float(tol))
        if key in self._cache:
            return self._cache[key]
        M = self._quadrature_points()
        theta = 2.0 * np.pi * np.arange(M) / M
        K = self.alias_K(tol)
        S0, S1, S2 = self._alias_moments(theta, K)
        n = np.arange(-self.nmax, self.nmax + 1)
        phase = np.exp(1j * np.outer(n, theta)) / M
        beta = (phase @ S0).real
        dbeta = (phase @ (1j * S1)).real
        ddbeta = (phase @ (-S2)).real
        truncation = {"route": "alias", "K": int(K), "quadrature_points": int(M),
                      "tail_bound": {str(q): self.tail_bound(q, K) for q in (0, 1, 2)}}
        samples = fam.IntegerSamples(n, beta, dbeta, ddbeta, truncation)
        self._cache[key] = samples
        return samples

    # ---- symbols from the samples ----------------------------------------

    def symbols(self, theta, tol: float = DEFAULT_TOL):
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
               "min_b": float(b.min()), "positive": bool(b.min() > 0.0)}
        for N in discrete:
            angles = 2.0 * np.pi * np.arange(N) / N
            bn = self.symbols(angles)[0]
            out[f"cond_N{N}"] = float(bn.max() / bn.min())
        return out

    def admissible(self, theta_points: int = 4001) -> tuple[bool, str]:
        """Return whether b > 0 and kappa_2 > 0 on (0, pi] on a fine grid."""
        for slot in self.slots:
            if slot.kind != "real" and not (0.0 < slot.omega < np.pi):
                return False, "frequency outside (0, pi)"
        theta = np.linspace(0.0, np.pi, theta_points)
        b, _, k2 = self.symbols(theta)
        if not np.all(np.isfinite(b)) or float(b.min()) <= 0.0:
            return False, "sample symbol not positive"
        if float(k2[1:].min()) <= 0.0:
            return False, "kappa_2 not positive"
        return True, "admissible"

    # ---- off-grid --------------------------------------------------------

    def offgrid_samples(self, xi, tol: float = DEFAULT_TOL):
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        key = ("offgrid", xi.tobytes(), float(tol))
        if key in self._cache:
            return self._cache[key]
        K = self.alias_K(tol)
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
        for qq in range(3):
            series = (1j ** qq) * acc[qq] * np.exp(1j * np.outer(theta, xi))
            out.append((phase @ series).real)
        result = (n, out[0], out[1], out[2])
        self._cache[key] = result
        return result

    def coupled_error(self, theta, xi, tol: float = DEFAULT_TOL) -> np.ndarray:
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

    # ---- records ---------------------------------------------------------

    def member_record(self, name: str) -> dict:
        """Return the member file in the E62_champion.json schema, extended.

        The Gate 5 fields m, r, omegas, p, radius, cond, beta, dbeta, ddbeta
        keep their meaning (r and omegas describe the imaginary pairs). The
        extension adds real_pairs (the alpha of each real pair), quads (the
        [alpha, omega] of each quad), exponents (n), parameters (the count of
        continuous real parameters) and the alias truncation receipt.
        """
        samples = self.integer_samples()
        radius = self.nmax
        center = int(np.searchsorted(samples.n, 0))
        one = slice(center, center + radius + 1)
        return {
            "name": name,
            "m": int(self.m),
            "r": int(self.r),
            "omegas": [float(w) for w in self.omegas],
            "real_pairs": [float(a) for a in self.real_pairs],
            "quads": [[float(a), float(w)] for a, w in self.quads],
            "exponents": int(self.n),
            "parameters": int(self.parameters),
            "p": int(self.p),
            "radius": int(radius),
            "cond": float(self.cond()["cond"]),
            "beta": [float(v) for v in samples.beta[one]],
            "dbeta": [float(v) for v in samples.dbeta[one]],
            "ddbeta": [float(v) for v in samples.ddbeta[one]],
            "schema": "E62_champion.json fields plus real_pairs, quads, exponents, "
                      "parameters; the generator transform is "
                      "sinc(xi/2)^m prod_slots g(s_j, xi) of gate7_family.py",
        }


def from_record(record: dict) -> ExtendedGenerator:
    slots = [(0.0, w) for w in record.get("omegas", [])]
    slots += [(a, 0.0) for a in record.get("real_pairs", [])]
    slots += [(a, w) for a, w in record.get("quads", [])]
    return ExtendedGenerator(int(record["m"]), slots)


# ---------------------------------------------------------------------------
# Sufficient conditions (Theorems P and N of Gate 7)
# ---------------------------------------------------------------------------

def S_series(s: int, w: float, terms: int = 200000) -> float:
    """Return S_s(w) = sum_{k>=1} (w/(2 pi k - w))^s + (w/(2 pi k + w))^s."""
    k = np.arange(1, terms + 1, dtype=float)
    return float(np.sum((w / (2.0 * np.pi * k - w)) ** s + (w / (2.0 * np.pi * k + w)) ** s))


def sufficient_conditions(gen: ExtendedGenerator) -> dict:
    """Return the Gate 7 Theorem P and N hypotheses and whether they hold.

    Real pairs impose nothing. Imaginary pairs and quads need omega < pi, which
    also gives Re s = alpha^2 - omega^2 > -pi^2 for the quad (Lemma E). The odd
    intervals of the sign argument exist only below the largest imaginary
    frequency, so the S conditions are evaluated there and are vacuous when no
    imaginary pair is present.
    """
    omegas = gen.omegas
    quad_ok = all(w < np.pi for _, w in gen.quads)
    imag_ok = all(0.0 < w < np.pi for w in omegas)
    out = {"omega_below_pi": bool(quad_ok and imag_ok),
           "re_s_above_minus_pi2": bool(all(a * a - w * w > -np.pi ** 2 for a, w in gen.quads)),
           "has_imaginary": bool(omegas)}
    if omegas:
        w = max(omegas)
        out["S_m"] = S_series(gen.m, w)
        out["S_m_minus_2"] = S_series(gen.m - 2, w) if gen.m >= 4 else None
    else:
        out["S_m"] = 0.0
        out["S_m_minus_2"] = 0.0
    out["theorem_P"] = bool(out["omega_below_pi"] and out["S_m"] < 1.0)
    out["theorem_N"] = bool(gen.m >= 4 and out["omega_below_pi"]
                            and out["S_m_minus_2"] is not None and out["S_m_minus_2"] < 1.0)
    return out


def certify_positivity(gen: ExtendedGenerator, points: int = 200001) -> dict:
    """Certify b > 0 and kappa_2 b > 0 on (0, pi] by a Lipschitz grid argument.

    b and c = kappa_2 b are trigonometric polynomials of degree nmax with
    coefficients beta(n) and beta''(n); their derivatives are bounded by
    2 sum_n n |coefficient|, so a positive minimum on a grid of spacing h with
    margin exceeding h/2 times that bound is a proof of positivity between the
    grid points. kappa_2 b = c vanishes at theta = 0 like theta^2: with
    c(theta) = 4 sum_{n>=1} beta''(n) sin^2(n theta/2) and
    sum_{n>=1} n^2 beta''(n) = 1 (second-order consistency of the direct
    second derivative), the bounds x^2 - x^4/3 <= sin^2 x <= x^2 give

        c(theta) >= theta^2 (1 - (theta^2/12) sum_{n>=1} n^4 |beta''(n)|),

    which is positive for theta^2 < 12 / sum_{n>=1} n^4 |beta''(n)|; the grid
    argument covers the rest of (0, pi].
    """
    s = gen.integer_samples()
    pos = s.n > 0
    nn = s.n[pos].astype(float)
    theta = np.linspace(0.0, np.pi, points)
    h = theta[1] - theta[0]
    arg = np.outer(theta, nn)
    b = s.beta[s.n == 0][0] + 2.0 * (np.cos(arg) @ s.beta[pos])
    c = 4.0 * (np.sin(0.5 * arg) ** 2 @ s.ddbeta[pos])
    lip_b = 2.0 * float(np.sum(nn * np.abs(s.beta[pos])))
    lip_c = 2.0 * float(np.sum(nn * np.abs(s.ddbeta[pos])))
    moment2 = float(np.sum(nn ** 2 * s.ddbeta[pos]))
    moment4 = float(np.sum(nn ** 4 * np.abs(s.ddbeta[pos])))
    theta_small = float(np.sqrt(12.0 * moment2 / moment4)) if moment2 > 0 else 0.0
    theta_small = min(theta_small, np.pi)
    tail = theta >= 0.5 * theta_small
    out = {"grid_points": int(points),
           "b_min": float(b.min()), "b_margin_needed": 0.5 * h * lip_b,
           "b_certified": bool(b.min() > 0.5 * h * lip_b),
           "kappa2b_second_moment": moment2,
           "kappa2b_series_radius": theta_small,
           "kappa2b_min_on_tail": float(c[tail].min()),
           "kappa2b_margin_needed": 0.5 * h * lip_c,
           "kappa2b_certified": bool(moment2 > 0.0 and theta_small > 0.0
                                     and c[tail].min() > 0.5 * h * lip_c)}
    return out


# ---------------------------------------------------------------------------
# Lemma D and the zero-count theorem, checked numerically per member
# ---------------------------------------------------------------------------

def lemma_D_residual(gen: ExtendedGenerator, theta, ks=(-3, -2, -1, 1, 2, 3)) -> float:
    """Return max relative gap between phi_hat(theta + 2 pi k) and its factored form."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    worst = 0.0
    kf = gen.k_free_part(theta)
    for k in ks:
        xi = theta + 2.0 * np.pi * k
        direct = gen.fourier(xi)
        factored = kf * gen.rational_part(xi)
        scale = np.maximum(np.abs(direct), 1e-300)
        worst = max(worst, float(np.max(np.abs(direct - factored) / scale)))
    return worst


def error_sign_structure(gen: ExtendedGenerator, theta) -> dict:
    """Return the decomposition e_1 = -(k-free sign) x (positive alias sum).

    By Lemma D and the ordering A_{-k} > A_k > 0 (k >= 1) of the rational
    parts, kappa_1 - theta = k_free(theta) sum_{k>=1} 2 pi k (A_k - A_{-k})
    has the sign of -k_free(theta) and vanishes exactly where k_free does,
    that is at the imaginary frequencies. This function returns the two
    factors on a grid together with the observed zeros of e_1.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    K = gen.alias_K()
    ks = np.arange(1, K + 1, dtype=float)
    Xp = theta[:, None] + 2.0 * np.pi * ks[None, :]
    Xm = theta[:, None] - 2.0 * np.pi * ks[None, :]
    diff = (2.0 * np.pi * ks[None, :]) * (gen.rational_part(Xp) - gen.rational_part(Xm))
    alias_sum = diff.sum(axis=1)
    kf = gen.k_free_part(theta)
    b = gen.sample_symbols(theta, K=K)[0]
    e1 = kf * alias_sum / (theta * b)
    _, k1, _ = gen.sample_symbols(theta, K=K)
    e1_direct = k1 / theta - 1.0
    return {"alias_sum_max": float(alias_sum.max()),
            "alias_sum_all_negative": bool(np.all(alias_sum < 0.0)),
            "e1_factored_vs_direct": float(np.max(np.abs(e1 - e1_direct))),
            "k_free": kf, "alias_sum": alias_sum, "e1": e1_direct}


# ---------------------------------------------------------------------------
# The 60-digit decimal route
# ---------------------------------------------------------------------------

def _dec_cosh_sinh(x: Decimal) -> tuple[Decimal, Decimal]:
    e = x.exp()
    inv = 1 / e
    return (e + inv) / 2, (e - inv) / 2


def decimal_integer_samples(gen: ExtendedGenerator, K: int, prec: int = DECIMAL_PREC) -> dict:
    """Return beta, beta', beta'' in decimal from the factored alias sums.

    Every alias xi = 2 pi j/M + 2 pi k of the quadrature angle theta_j enters
    through the rational parts xi^{-m} prod (xi^2 + s_j)^{-1} only, the
    k-independent trigonometric parts being evaluated once per angle. The
    truncation index K is the double route's, so the two routes truncate the
    same tail and their difference measures arithmetic alone.
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
                cw = trig.cos(w)
                slots.append(("imaginary", w * w, cw))
            elif slot.kind == "real":
                ca, _ = _dec_cosh_sinh(a)
                slots.append(("real", a * a, ca))
            else:
                ca, sa = _dec_cosh_sinh(a)
                cw, sw = trig.cos(w), trig.sin(w)
                slots.append(("quad", (a, w), (ca * cw, sa * sw)))
        S0 = [Decimal(0)] * M
        S1 = [Decimal(0)] * M
        S2 = [Decimal(0)] * M
        for j in range(M):
            theta = two_pi * j / M
            half = pi * j / M
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
                    a, w = par
                    cc, ss = aux
                    kfree *= ((a * a + w * w) ** 2 * ((cc - cos_theta) ** 2 + ss * ss)
                              / ((cc - 1) ** 2 + ss * ss))
            s0 = s1 = s2 = Decimal(0)
            for k in range(-K, K + 1):
                xi = theta + two_pi * k
                if xi == 0:
                    if gen.m > 0:
                        # xi = 0 only at j = 0, k = 0, where phi_hat = 1 exactly
                        P = Decimal(1)
                    else:  # pragma: no cover
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
                            a, w = par
                            rat /= ((xi2 + a * a - w * w) ** 2 + 4 * a * a * w * w)
                    P = kfree * rat
                s0 += P
                s1 += xi * P
                s2 += xi * xi * P
            S0[j], S1[j], S2[j] = s0, s1, s2
        cos_table = {}
        sin_table = {}
        for residue in range(M):
            angle = two_pi * residue / M
            cos_table[residue] = trig.cos(angle)
            sin_table[residue] = trig.sin(angle)
        beta, dbeta, ddbeta = [], [], []
        for n in range(-nmax, nmax + 1):
            a0 = a1 = a2 = Decimal(0)
            for j in range(M):
                residue = (n * j) % M
                a0 += S0[j] * cos_table[residue]
                a1 += S1[j] * sin_table[residue]
                a2 += S2[j] * cos_table[residue]
            beta.append(a0 / M)
            dbeta.append(-a1 / M)
            ddbeta.append(-a2 / M)
        series_bound = float(trig.bound)
    return {"n": list(range(-nmax, nmax + 1)),
            "beta": [float(v) for v in beta],
            "dbeta": [float(v) for v in dbeta],
            "ddbeta": [float(v) for v in ddbeta],
            "beta_digits": [str(v) for v in beta],
            "K": int(K), "quadrature_points": int(M), "working_digits": int(prec),
            "series_truncation_bound": series_bound}


def verify_routes(gen: ExtendedGenerator, tol: float = DEFAULT_TOL,
                  route_tol: float = ROUTE_TOL,
                  symbol_tol: float = SYMBOL_ROUTE_TOL) -> dict:
    """Return the agreement of the double alias route, the decimal route, and the
    symbols from samples against the direct alias symbols.

    The sample routes must agree to route_tol (the contract's 1e-13). The
    symbols recomputed from the samples are compared with the direct alias
    sums at 41 probe angles relative to max(1, max |symbol|); that direct sum
    accumulates rounding over 2K + 1 terms of size up to |kappa_2| ~ 10, so
    its own tolerance is symbol_tol = 1e-12.
    """
    alias = gen.integer_samples(tol)
    dec = decimal_integer_samples(gen, alias.truncation["K"])
    dev = max(float(np.max(np.abs(np.asarray(dec[key]) - arr)))
              for key, arr in (("beta", alias.beta), ("dbeta", alias.dbeta),
                               ("ddbeta", alias.ddbeta)))
    probe = np.linspace(0.05, np.pi, 41)
    direct = gen.sample_symbols(probe, tol=tol)
    from_samples = gen.symbols(probe, tol)
    gap = max(float(np.max(np.abs(direct[q] - from_samples[q]))
                    / max(1.0, float(np.max(np.abs(from_samples[q])))))
              for q in range(3))
    g0 = gen.gate0_generator()
    gate0_dev = None
    if g0 is not None:
        s0 = g0.integer_samples(tol)
        gate0_dev = max(float(np.max(np.abs(a - b))) for a, b in
                        ((s0.beta, alias.beta), (s0.dbeta, alias.dbeta), (s0.ddbeta, alias.ddbeta)))
    return {"decimal_deviation": dev, "decimal_ok": bool(dev <= route_tol),
            "symbols_from_samples_versus_alias": gap, "symbols_ok": bool(gap <= symbol_tol),
            "gate0_family_deviation": gate0_dev,
            "gate0_ok": bool(gate0_dev is None or gate0_dev <= route_tol),
            "K": int(alias.truncation["K"]),
            "tail_bound": alias.truncation["tail_bound"],
            "decimal_series_bound": dec["series_truncation_bound"],
            "beta_digits": dec["beta_digits"]}


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

def parameter_counts(n: int) -> dict:
    """Return the dimension of the compact class and of the family at width n.

    A compact scheme with left half-bandwidth L and right half-bandwidth R has
    L + R + 1 coefficients, one of which is a scale and one is fixed by
    first-order consistency, so L + R - 1 free real parameters. A spline member
    of odd degree p = n - 1 has L = R = n/2 - 1, hence n - 3 free parameters in
    the compact class. Every slot of the extended family, whatever its kind,
    carries one real parameter per two exponents, so the family with core m
    has (n - m)/2 parameters, the same as the imaginary-pair family.
    """
    L = R = n // 2 - 1
    out = {"n": int(n), "p": int(n - 1), "L": int(L), "R": int(R),
           "compact_free_parameters": int(L + R - 1), "family": {}}
    for m in range(2, n + 1, 2):
        out["family"][str(m)] = {"parameters": int((n - m) // 2),
                                 "gap_to_compact": int(L + R - 1 - (n - m) // 2)}
    return out
