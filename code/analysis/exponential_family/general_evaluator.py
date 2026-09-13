"""Pointwise evaluator for the Gate 7 and Gate 8 generators.

The module of evaluator.py covers the Gate 0 multiset (0^m, +-i omega_j). Gate 7
widens it with real pairs +-alpha and quads +-alpha +- i omega, and Gate 8
multiplies the transform by P(xi^2) = 1 + sum_l p_l xi^{2l}, that is applies
P(-D^2) on the x side. This module carries the same localized Green-function
cell construction through both extensions and returns psi(x), psi'(x) and
psi''(x) at arbitrary real x.

Construction
------------

The exponent multiset Lambda holds m zeros, the two exponents +-i omega of an
imaginary pair, the two exponents +-alpha of a real pair, and the four
exponents +-alpha +- i omega of a quad, so n = |Lambda| is the support width
and the generator lives on [-n/2, n/2]. With

    L = prod_{lambda in Lambda} (D - lambda),   D(s) = prod (s - lambda),

and q the causal Green function of L, the centered normalized factor of
exponent lambda is b_lambda(x) = c_lambda exp(lambda x) 1_{[-1/2,1/2]}(x) with
c_lambda = lambda / (2 sinh(lambda/2)), and convolving the n factors gives

    phi(x) = nu sum_{l=0}^{n} A_l q(x + n/2 - l),
    A(z)   = prod_{lambda} (1 - exp(lambda) z),
    nu     = prod_{lambda} c_lambda,

which is the Gate 0 statement with the localization polynomial and the
normalization read off the multiset itself. A(z) and D(s) have real
coefficients because Lambda is closed under conjugation, and nu is real
because it is closed under conjugation and negation.

The partial fraction expansion of 1/D at a root rho of multiplicity mu comes
from the truncated series of u^mu / D(rho + u), so the simple and the confluent
case share one code path, and the analytic continuation

    Q(t) = sum_rho exp(rho t) sum_{a < mu_rho} kappa_{rho,a} t^a

satisfies A(S) Q = 0 for the unit right shift S. On the cell y = x + n/2 in
[J, J+1), with xi = y - J, the truncated sum over the terms with y - l >= 0
therefore equals minus the truncated sum over the terms with y - l < 0, and
whichever carries fewer terms yields the exponential polynomial

    phi(x) = sum_rho exp(rho xi) sum_b W_{rho,b}(J) xi^b.

The polynomial dial
-------------------

psi = P(-D^2) phi = sum_l (-1)^l p_l phi^{(2l)} keeps that form: differentiating
exp(rho xi) sum_b W_b xi^b sends W_b to rho W_b + (b+1) W_{b+1}, so the dial is
one linear recursion per root and per cell, applied 2 q times and accumulated
with the coefficients (-1)^l p_l. The support, the roots and the multiplicities
are unchanged, which is the x-side statement of psi_hat = P(xi^2) phi_hat.

Cancellation-safe cell form
---------------------------

Near an integer knot psi is small while the individual root terms are not, so
the Taylor jet of every root term is folded into one polynomial of degree
K - 1 with K = n, leaving the entire functions

    R_j(z) = sum_{i >= j} z^i / i!,   R_j = exp for j <= 0,

which vanish to order j at the origin:

    psi(x) = sum_{k<K} a_k(J) xi^k
           + sum_{channels} sum_b xi^b [ C_{b} Re R_{K-b}(rho xi)
                                       + S_{b} Im R_{K-b}(rho xi) ].

Because every remainder is O(xi^K), the a_k are exactly the one-sided Taylor
coefficients of psi at the cell anchor, so they are bounded by the derivatives
of psi itself and the polynomial part carries no cancellation. A channel is one
root rho = alpha + i omega with omega >= 0: a root off the real axis is paired
with its conjugate and carries C_b = 2 Re W_b, S_b = -2 Im W_b, while a real
root stands alone with C_b = W_b, S_b = 0. The zero root needs no channel, its
terms being polynomials already. Derivatives reuse the same functions through
d/dxi R_j(rho xi) = rho R_{j-1}(rho xi) and the Leibniz rule for the factor
xi^b, and R_{j-1} = R_j + z^{j-1}/(j-1)! builds the lower indices from the
higher one by addition alone.

Working precision
-----------------

The cell coefficients are assembled once per member in decimal arithmetic at
WORKING_DIGITS digits, because the sum over l cancels the polynomial growth of
Q against itself and the dial differentiates that cancellation 2 q more times.
Only the finished a_k, C_b and S_b are rounded to double, in the folded basis
above, where the evaluation itself is cancellation free. The ratio of the
largest term entering the assembly to max |psi| is reported as `conditioning`
and the ratio of the largest stored coefficient to max |psi| as
`representation_conditioning`.

Two limits are handled by a documented tolerance. A slot parameter at or below
SLOT_TOLERANCE is set to zero, so a quad with a vanishing real part is an
imaginary pair of multiplicity two, a pair with a vanishing parameter is two
extra zero exponents, and slots within SLOT_TOLERANCE of each other are merged
into one slot of higher multiplicity taken at their mean. The confluent
multiplicity is carried through the residue series, so a repeated slot is
evaluated by its own formula and not by a difference of nearly equal residues.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, localcontext
from functools import lru_cache
from math import comb, factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

WORKING_DIGITS = 50
SLOT_TOLERANCE = 1e-7
SERIES_LIMIT = 4.0
SERIES_TERMS = 48
EVAL_BLOCK_VERSION = 2


# ---------------------------------------------------------------------------
# Decimal complex arithmetic, used only by the one-off assembly
# ---------------------------------------------------------------------------

class DC:
    """A complex number carried by two decimals of the ambient context."""

    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0) -> None:
        self.re = re if isinstance(re, Decimal) else Decimal(re)
        self.im = im if isinstance(im, Decimal) else Decimal(im)

    def __add__(self, other: "DC") -> "DC":
        return DC(self.re + other.re, self.im + other.im)

    def __neg__(self) -> "DC":
        return DC(-self.re, -self.im)

    def __mul__(self, other: "DC") -> "DC":
        return DC(self.re * other.re - self.im * other.im,
                  self.re * other.im + self.im * other.re)

    def __truediv__(self, other: "DC") -> "DC":
        den = other.re * other.re + other.im * other.im
        return DC((self.re * other.re + self.im * other.im) / den,
                  (self.im * other.re - self.re * other.im) / den)

    def scale(self, factor: Decimal) -> "DC":
        return DC(self.re * factor, self.im * factor)

    def abs2(self) -> Decimal:
        return self.re * self.re + self.im * self.im

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"DC({self.re}, {self.im})"


class _Decimals:
    """Decimal exp, sin and cos at the ambient precision."""

    def __init__(self, trig) -> None:
        self.trig = trig

    def cexp(self, z: DC) -> DC:
        amplitude = z.re.exp()
        return DC(amplitude * self.trig.cos(z.im), amplitude * self.trig.sin(z.im))

    def csinh(self, z: DC) -> DC:
        e = z.re.exp()
        inv = 1 / e
        return DC((e - inv) / 2 * self.trig.cos(z.im),
                  (e + inv) / 2 * self.trig.sin(z.im))


# ---------------------------------------------------------------------------
# The entire functions R_j and their real and imaginary parts
# ---------------------------------------------------------------------------

def remainder_R(j: int, u, v):
    """Return Re R_j(u + i v) and Im R_j(u + i v) with R_j(z) = sum_{i>=j} z^i/i!.

    Below SERIES_LIMIT in modulus the sum is taken from its first term, which is
    the only branch any member of the documented parameter range reaches; above
    it the equivalent form exp(z) minus the truncated exponential series is
    used. R_j is exp for every j <= 0.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if j <= 0:
        amplitude = np.exp(u)
        return amplitude * np.cos(v), amplitude * np.sin(v)
    small = (u * u + v * v) <= SERIES_LIMIT * SERIES_LIMIT

    tre = np.ones(u.shape, dtype=float)
    tim = np.zeros(u.shape, dtype=float)
    for i in range(1, j + 1):
        tre, tim = (tre * u - tim * v) / i, (tre * v + tim * u) / i
    sre, sim = tre.copy(), tim.copy()
    for i in range(j + 1, j + 1 + SERIES_TERMS):
        tre, tim = (tre * u - tim * v) / i, (tre * v + tim * u) / i
        sre = sre + tre
        sim = sim + tim
    if bool(np.all(small)):
        return sre, sim

    pre = np.ones(u.shape, dtype=float)
    pim = np.zeros(u.shape, dtype=float)
    are = pre.copy()
    aim = pim.copy()
    for i in range(1, j):
        pre, pim = (pre * u - pim * v) / i, (pre * v + pim * u) / i
        are = are + pre
        aim = aim + pim
    amplitude = np.exp(u)
    dre = amplitude * np.cos(v) - are
    dim = amplitude * np.sin(v) - aim
    return np.where(small, sre, dre), np.where(small, sim, dim)


def remainder_block(j: int, u, v, depth: int):
    """Return R_j, R_{j-1}, ..., R_{j-depth} at u + i v, built by addition.

    The identity R_{j-1}(z) = R_j(z) + z^{j-1}/(j-1)! adds one monomial per
    step, so the lower indices never subtract nearly equal quantities.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    top = remainder_R(j, u, v)
    out = [top]
    for step in range(1, depth + 1):
        index = j - step
        if index < 0:
            out.append(out[-1])
            continue
        mre = np.ones(u.shape, dtype=float)
        mim = np.zeros(u.shape, dtype=float)
        for i in range(1, index + 1):
            mre, mim = (mre * u - mim * v) / i, (mre * v + mim * u) / i
        out.append((out[-1][0] + mre, out[-1][1] + mim))
    return out


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def slot_pair(item) -> tuple:
    """Return (alpha, omega) of a slot given as an object or as a pair."""
    if hasattr(item, "alpha") and hasattr(item, "omega"):
        return float(item.alpha), float(item.omega)
    alpha, omega = item
    return float(alpha), float(omega)


def normalize_slots(slots, tolerance: float = SLOT_TOLERANCE):
    """Return (extra zero exponents, grouped slots) with parameters snapped.

    A parameter at or below the tolerance is set to zero, a slot with both
    parameters zero becomes two extra zero exponents, and slots that agree to
    the tolerance are merged into one group of higher multiplicity taken at the
    mean of their parameters. The returned groups are sorted by (alpha, omega).
    """
    cleaned = []
    extra_zero_pairs = 0
    for item in slots:
        alpha, omega = slot_pair(item)
        alpha, omega = abs(alpha), abs(omega)
        if alpha <= tolerance:
            alpha = 0.0
        if omega <= tolerance:
            omega = 0.0
        if alpha == 0.0 and omega == 0.0:
            extra_zero_pairs += 1
            continue
        cleaned.append((alpha, omega))
    cleaned.sort()
    groups = []
    for alpha, omega in cleaned:
        if groups:
            a0, w0, k0 = groups[-1]
            if abs(alpha - a0) <= tolerance and abs(omega - w0) <= tolerance:
                groups[-1] = ((a0 * k0 + alpha) / (k0 + 1), (w0 * k0 + omega) / (k0 + 1), k0 + 1)
                continue
        groups.append((alpha, omega, 1))
    return extra_zero_pairs, tuple(groups)


def slot_kind(alpha: float, omega: float) -> str:
    """Return the kind of a normalized slot."""
    if alpha == 0.0:
        return "imaginary"
    if omega == 0.0:
        return "real"
    return "quad"


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------

class GeneralEvaluator:
    """Pointwise psi, psi' and psi'' of a Gate 7 or Gate 8 member."""

    SLOT_TOLERANCE = SLOT_TOLERANCE

    def __init__(self, m: int, slots=(), poly=(), *, digits: int = WORKING_DIGITS,
                 fold_order: int | None = None) -> None:
        m = int(m)
        if m < 1:
            raise ValueError("The polynomial core multiplicity m must be positive.")
        self.requested_m = m
        self.requested_slots = tuple(slot_pair(s) for s in slots)
        self.poly = tuple(float(p) for p in poly)
        self.q = len(self.poly)
        self.digits = int(digits)

        extra, groups = normalize_slots(self.requested_slots)
        self.m = m + 2 * extra
        self.collapsed_slots = extra
        self.slot_groups = groups
        self.n = self.m + sum((4 if slot_kind(a, w) == "quad" else 2) * k
                              for a, w, k in groups)
        self.p = self.n - 1
        self.smoothness = self.n - 2 - 2 * self.q
        self.support = (-0.5 * self.n, 0.5 * self.n)
        self.fold_order = int(self.n if fold_order is None else fold_order)
        self.poly_degree = self.fold_order - 1
        self._build()

    # ---------------- bookkeeping ----------------

    @property
    def omegas(self) -> tuple:
        """Return the imaginary pair frequencies, increasing, with multiplicity."""
        out = []
        for a, w, k in self.slot_groups:
            if slot_kind(a, w) == "imaginary":
                out += [w] * k
        return tuple(sorted(out))

    @property
    def real_pairs(self) -> tuple:
        out = []
        for a, w, k in self.slot_groups:
            if slot_kind(a, w) == "real":
                out += [a] * k
        return tuple(sorted(out))

    @property
    def quads(self) -> tuple:
        out = []
        for a, w, k in self.slot_groups:
            if slot_kind(a, w) == "quad":
                out += [(a, w)] * k
        return tuple(sorted(out))

    @property
    def r(self) -> int:
        return len(self.omegas)

    def label(self) -> str:
        pieces = []
        for a, w, k in self.slot_groups:
            kind = slot_kind(a, w)
            tag = (f"i{w:.6f}" if kind == "imaginary"
                   else f"r{a:.6f}" if kind == "real" else f"q({a:.6f},{w:.6f})")
            pieces += [tag] * k
        inner = ",".join(pieces) if pieces else ""
        core = f"X_{{{self.m}}}({inner})"
        if self.q:
            core += "xP[" + ",".join(f"{p:.6g}" for p in self.poly) + "]"
        return core

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return (f"GeneralEvaluator(m={self.requested_m}, slots={self.requested_slots}, "
                f"poly={self.poly})")

    # ---------------- construction ----------------

    def _exponent_plan(self):
        """Return the distinct roots with multiplicities and the channel layout.

        A root is stored as (alpha, omega, multiplicity); the zero root is the
        first entry with alpha = omega = 0 and multiplicity m. A channel is a
        root with omega >= 0 other than the zero root, paired with its conjugate
        when omega > 0.
        """
        roots = [(0.0, 0.0, self.m)]
        channels = []
        for a, w, k in self.slot_groups:
            kind = slot_kind(a, w)
            if kind == "imaginary":
                roots += [(0.0, w, k), (0.0, -w, k)]
                channels.append((0.0, w, k, True))
            elif kind == "real":
                roots += [(a, 0.0, k), (-a, 0.0, k)]
                channels.append((a, 0.0, k, False))
                channels.append((-a, 0.0, k, False))
            else:
                roots += [(a, w, k), (a, -w, k), (-a, w, k), (-a, -w, k)]
                channels.append((a, w, k, True))
                channels.append((-a, w, k, True))
        return roots, channels

    def _build(self) -> None:
        from gate1_champions import DecimalTrig  # noqa: E402  (heavy, loaded on demand)

        roots, channels = self._exponent_plan()
        self.roots = tuple(roots)
        self.channels = tuple(channels)
        self.term_count = sum(k for _, _, k, _ in channels)
        self._slots = []
        column = 0
        for alpha, omega, mult, _ in channels:
            self._slots.append((alpha, omega, mult, column))
            column += mult

        n = self.n
        K = self.fold_order
        prec = self.digits + 12
        with localcontext() as ctx:
            ctx.prec = prec
            trig = DecimalTrig(prec)
            dec = _Decimals(trig)
            cells, nu, worst = assemble_cells(self, dec, trig)
            self._decimal_cache = cells
            self.normalization = float(nu)

            # Fold the Taylor jet of every root term into one polynomial.
            cell_poly = np.zeros((n, K), dtype=float)
            cell_cos = np.zeros((n, self.term_count), dtype=float)
            cell_sin = np.zeros((n, self.term_count), dtype=float)
            root_index = {(a, w): i for i, (a, w, _) in enumerate(roots)}
            stored = Decimal(0)
            for J in range(n):
                acc = [Decimal(0)] * K
                for ri, (a, w, mu) in enumerate(roots):
                    rho = DC(Decimal(a), Decimal(w))
                    for b in range(mu):
                        coefficient = cells[J][ri][b]
                        power = DC(1)
                        for k in range(b, K):
                            acc[k] = acc[k] + (coefficient * power).re / Decimal(factorial(k - b))
                            power = power * rho
                for k in range(K):
                    cell_poly[J, k] = float(acc[k])
                    stored = max(stored, abs(acc[k]))
                column = 0
                for alpha, omega, mult, paired in channels:
                    ri = root_index[(alpha, omega)]
                    for b in range(mult):
                        coefficient = cells[J][ri][b]
                        if paired:
                            c_value = 2 * coefficient.re
                            s_value = -2 * coefficient.im
                        else:
                            c_value = coefficient.re
                            s_value = Decimal(0)
                        cell_cos[J, column + b] = float(c_value)
                        cell_sin[J, column + b] = float(s_value)
                        stored = max(stored, abs(c_value), abs(s_value))
                    column += mult

        self.cell_poly = cell_poly
        self.cell_cos = cell_cos
        self.cell_sin = cell_sin
        peak = float(np.max(np.abs(cell_poly[:, 0]))) or 1.0
        self.conditioning = float(worst) / peak
        self.representation_conditioning = float(stored) / peak

    # ---------------- evaluation ----------------

    def _cell_index(self, x: np.ndarray):
        y = np.asarray(x, dtype=float) + 0.5 * self.n
        inside = (y > 0.0) & (y < float(self.n))
        J = np.clip(np.floor(y), 0.0, self.n - 1).astype(np.intp)
        xi = y - J
        return inside, J, xi

    def derivative(self, x, order: int = 0) -> np.ndarray:
        """Return the order-th derivative of psi at x, zero outside the support."""
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
        out[inside] = self._cell_value(Ji, xii, order)
        return out.reshape(shape)

    def _cell_value(self, Ji: np.ndarray, xii: np.ndarray, order: int) -> np.ndarray:
        value = np.zeros(xii.shape, dtype=float)
        coeffs = self.cell_poly[Ji]
        for k in range(order, self.fold_order):
            fall = 1.0
            for t in range(order):
                fall *= (k - t)
            value = value + fall * coeffs[:, k] * xii ** (k - order)

        for alpha, omega, mult, column in self._slots:
            u = alpha * xii
            v = omega * xii
            for b in range(mult):
                Cc = self.cell_cos[Ji, column + b]
                Ss = self.cell_sin[Ji, column + b]
                block = remainder_block(self.fold_order - b, u, v, order)
                for i in range(0, min(order, b) + 1):
                    fall = 1.0
                    for t in range(i):
                        fall *= (b - t)
                    if fall == 0.0:
                        continue
                    d = order - i
                    rre, rim = block[d]
                    pre, pim = _power(alpha, omega, d)
                    gre = pre * rre - pim * rim
                    gim = pre * rim + pim * rre
                    weight = comb(order, i) * fall
                    value = value + weight * xii ** (b - i) * (Cc * gre + Ss * gim)
        return value

    def value(self, x) -> np.ndarray:
        """Return psi(x), exactly zero outside [-n/2, n/2]."""
        return self.derivative(x, 0)

    def d1(self, x) -> np.ndarray:
        """Return psi'(x), exactly zero outside [-n/2, n/2]."""
        return self.derivative(x, 1)

    def d2(self, x) -> np.ndarray:
        """Return psi''(x), exactly zero outside [-n/2, n/2]."""
        return self.derivative(x, 2)

    def __call__(self, x, derivative: int = 0) -> np.ndarray:
        return self.derivative(x, derivative)

    def one_sided(self, knot: int, order: int, side: str) -> float:
        """Return the one-sided derivative limit at an integer knot.

        The left limit is the cell to the left evaluated at xi = 1 and the right
        limit is the cell to the right evaluated at xi = 0, both from the cell
        representation itself, so no finite difference enters the comparison.
        """
        y = float(knot) + 0.5 * self.n
        cell = int(round(y)) - 1 if side == "left" else int(round(y))
        xi = 1.0 if side == "left" else 0.0
        if cell < 0 or cell >= self.n:
            return 0.0
        Ji = np.array([cell], dtype=np.intp)
        xii = np.array([xi], dtype=float)
        return float(self._cell_value(Ji, xii, int(order))[0])

    def offgrid_samples(self, xi, order: int = 0):
        """Return the lattice n and psi^{(order)}(n + xi) for scalar xi."""
        radius = (self.n + 1) // 2 + 1
        lattice = np.arange(-radius, radius + 1)
        return lattice, self.derivative(lattice.astype(float) + float(xi), order)

    # ---------------- the reference route ----------------

    def reference(self, x, order: int = 0) -> np.ndarray:
        """Return psi^{(order)}(x) from the unfolded cell form in decimal.

        This is the independent yardstick for the accuracy of the double
        evaluation: the same cell coefficients are rebuilt at the working
        precision and summed without folding, so the two routes share the
        construction and differ only in arithmetic.
        """
        from gate1_champions import DecimalTrig  # noqa: E402

        x = np.atleast_1d(np.asarray(x, dtype=float))
        inside, J, xi = self._cell_index(x)
        out = np.zeros(x.shape, dtype=float)
        prec = self.digits + 12
        with localcontext() as ctx:
            ctx.prec = prec
            trig = DecimalTrig(prec)
            dec = _Decimals(trig)
            cells = self._decimal_cells(dec, trig)
            for index in np.nonzero(inside)[0]:
                cell = int(J[index])
                t = Decimal(float(xi[index]))
                total = Decimal(0)
                for ri, (a, w, mu) in enumerate(self.roots):
                    rho = DC(Decimal(a), Decimal(w))
                    row = cells[cell][ri]
                    for _ in range(int(order)):
                        row = _cell_derivative(rho, row)
                    poly = DC(0)
                    power = DC(1)
                    for b in range(mu):
                        poly = poly + row[b] * power
                        power = power.scale(t)
                    total += (dec.cexp(rho.scale(t)) * poly).re
                out[index] = float(total)
        return out

    def _decimal_cells(self, dec, trig):
        """Return the decimal cell coefficients of psi, assembled once per member."""
        cached = getattr(self, "_decimal_cache", None)
        if cached is None:
            cached = assemble_cells(self, dec, trig)[0]
            self._decimal_cache = cached
        return cached

    # ---------------- the flat block the C++ evaluator reads ----------------

    def cell_table(self) -> dict:
        """Return the version 2 flat cell tables that the C++ evaluator reads."""
        return {
            "eval_version": int(EVAL_BLOCK_VERSION),
            "eval_support_n": int(self.n),
            "eval_cells": int(self.n),
            "eval_poly_degree": int(self.poly_degree),
            "eval_fold_order": int(self.fold_order),
            "eval_group_count": int(len(self.channels)),
            "eval_term_count": int(self.term_count),
            "eval_group_alpha": [float(a) for a, _, _, _ in self.channels],
            "eval_group_omega": [float(w) for _, w, _, _ in self.channels],
            "eval_group_multiplicity": [float(k) for _, _, k, _ in self.channels],
            "eval_normalization": float(self.normalization),
            "eval_conditioning": float(self.conditioning),
            "eval_representation_conditioning": float(self.representation_conditioning),
            "eval_poly": [float(v) for v in self.cell_poly.reshape(-1)],
            "eval_cos": [float(v) for v in self.cell_cos.reshape(-1)],
            "eval_sin": [float(v) for v in self.cell_sin.reshape(-1)],
        }


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

def _real_convolve(a, b):
    out = [Decimal(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _shift_series(den, rho: DC, mu: int):
    """Return the ascending Taylor coefficients of den(rho + u) in u from u^mu."""
    out = [DC(0) for _ in range(len(den))]
    for k, c in enumerate(den):
        if c == 0:
            continue
        power = [DC(1)]
        for _ in range(k):
            power.append(power[-1] * rho)
        for i in range(k + 1):
            out[i] = out[i] + power[k - i].scale(c * Decimal(comb(k, i)))
    return out[mu:]


def _series_inverse(coeffs, order: int):
    """Return the ascending series of 1 / sum_k coeffs[k] u^k up to u^order."""
    padded = list(coeffs) + [DC(0)] * (order + 1)
    out = [DC(0) for _ in range(order + 1)]
    out[0] = DC(1) / padded[0]
    for k in range(1, order + 1):
        acc = DC(0)
        for i in range(1, k + 1):
            acc = acc + padded[i] * out[k - i]
        out[k] = -(acc / padded[0])
    return out


def _cell_derivative(rho: DC, row):
    """Return the coefficients of d/dxi [exp(rho xi) sum_b row[b] xi^b]."""
    mu = len(row)
    out = []
    for b in range(mu):
        term = rho * row[b]
        if b + 1 < mu:
            term = term + row[b + 1].scale(Decimal(b + 1))
        out.append(term)
    return out


def assemble_cells(ev: "GeneralEvaluator", dec, trig):
    """Return the decimal cell coefficients of psi, the normalization and the scale.

    The returned cells[J][root] is the list of coefficients W_b of
    exp(rho xi) sum_b W_b xi^b on the cell J, already carrying the
    normalization and already carrying the polynomial dial P(-D^2). `scale` is
    the largest term that entered the localization sum, which is the numerator
    of the reported assembly conditioning. The ambient decimal context sets the
    working precision.
    """
    n = ev.n

    # A(z) = prod (1 - exp(lambda) z), real by conjugate closure.
    A = [Decimal(1)]
    for a, w, k in ev.slot_groups:
        kind = slot_kind(a, w)
        for _ in range(k):
            if kind == "imaginary":
                A = _real_convolve(A, [Decimal(1), -2 * trig.cos(Decimal(w)), Decimal(1)])
            elif kind == "real":
                ea = Decimal(a).exp()
                A = _real_convolve(A, [Decimal(1), -(ea + 1 / ea), Decimal(1)])
            else:
                ea = Decimal(a).exp()
                cw = trig.cos(Decimal(w))
                A = _real_convolve(A, [Decimal(1), -2 * ea * cw, ea * ea])
                A = _real_convolve(A, [Decimal(1), -2 * cw / ea, 1 / (ea * ea)])
    for _ in range(ev.m):
        A = _real_convolve(A, [Decimal(1), Decimal(-1)])

    # D(s) = prod (s - lambda), ascending, real by conjugate closure.
    D = [Decimal(1)]
    for a, w, k in ev.slot_groups:
        kind = slot_kind(a, w)
        for _ in range(k):
            if kind == "imaginary":
                D = _real_convolve(D, [Decimal(w) * Decimal(w), Decimal(0), Decimal(1)])
            elif kind == "real":
                D = _real_convolve(D, [-Decimal(a) * Decimal(a), Decimal(0), Decimal(1)])
            else:
                a2 = Decimal(a) * Decimal(a)
                w2 = Decimal(w) * Decimal(w)
                D = _real_convolve(D, [(a2 + w2) * (a2 + w2), Decimal(0),
                                       -2 * (a2 - w2), Decimal(0), Decimal(1)])
    D = [Decimal(0)] * ev.m + D

    # nu = prod c_lambda, real by conjugate and negation closure.
    nu = Decimal(1)
    for a, w, k in ev.slot_groups:
        kind = slot_kind(a, w)
        for _ in range(k):
            if kind == "imaginary":
                half = Decimal(w) / 2
                nu = nu * (half * half) / (trig.sin(half) * trig.sin(half))
            elif kind == "real":
                half = Decimal(a) / 2
                sh = (half.exp() - 1 / half.exp()) / 2
                nu = nu * (half * half) / (sh * sh)
            else:
                z = DC(Decimal(a) / 2, Decimal(w) / 2)
                ratio = z.abs2() / dec.csinh(z).abs2()
                nu = nu * ratio * ratio

    # Residue coefficients of 1/D at every distinct root.
    kappa = []
    for a, w, mu in ev.roots:
        rho = DC(Decimal(a), Decimal(w))
        inv = _series_inverse(_shift_series(D, rho, mu), mu - 1)
        kappa.append([inv[mu - 1 - b] / DC(Decimal(factorial(b))) for b in range(mu)])

    # Cell coefficients of phi from the localized causal Green function.
    cells = []
    scale = Decimal(0)
    for J in range(n):
        left = list(range(0, J + 1))
        right = list(range(J - n, 0))
        use, sign = (left, Decimal(1)) if len(left) <= len(right) else (right, Decimal(-1))
        per = []
        for ri, (a, w, mu) in enumerate(ev.roots):
            rho = DC(Decimal(a), Decimal(w))
            moments = [DC(0) for _ in range(mu)]
            for s in use:
                base = dec.cexp(rho.scale(Decimal(s))).scale(sign * A[J - s])
                power = DC(1)
                for c in range(mu):
                    moments[c] = moments[c] + base * power
                    power = power * DC(Decimal(s))
            row = []
            for b in range(mu):
                acc = DC(0)
                for c in range(b, mu):
                    term = kappa[ri][c] * moments[c - b].scale(Decimal(comb(c, b)) * nu)
                    acc = acc + term
                    scale = max(scale, abs(term.re), abs(term.im))
                row.append(acc)
            per.append(row)
        cells.append(per)

    # The polynomial dial psi = sum_l (-1)^l p_l phi^{(2l)}.
    if ev.q:
        dial = [Decimal(1)] + [Decimal(p) for p in ev.poly]
        for J in range(n):
            for ri, (a, w, mu) in enumerate(ev.roots):
                rho = DC(Decimal(a), Decimal(w))
                current = cells[J][ri]
                total = list(current)
                for level in range(1, len(dial)):
                    current = _cell_derivative(rho, _cell_derivative(rho, current))
                    sign = Decimal(1) if level % 2 == 0 else Decimal(-1)
                    total = [t + c.scale(sign * dial[level]) for t, c in zip(total, current)]
                cells[J][ri] = total
    return cells, nu, scale


def _power(alpha: float, omega: float, d: int):
    """Return the real and imaginary parts of (alpha + i omega)^d."""
    re, im = 1.0, 0.0
    for _ in range(d):
        re, im = re * alpha - im * omega, re * omega + im * alpha
    return re, im


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def general_evaluator_for(m: int, slots: tuple, poly: tuple) -> GeneralEvaluator:
    """Return a cached evaluator for the member with these slots and dial."""
    return GeneralEvaluator(m, slots, poly)


def evaluator_for_generator(generator) -> GeneralEvaluator:
    """Return the evaluator of a Gate 7 or Gate 8 generator object."""
    if isinstance(generator, GeneralEvaluator):
        return generator
    slots = tuple(slot_pair(s) for s in generator.slots)
    poly = tuple(float(p) for p in getattr(generator, "poly", ()))
    return general_evaluator_for(int(generator.m), slots, poly)


def evaluator_from_record(record: dict) -> GeneralEvaluator:
    """Return the evaluator of a member record in the Gate 7 or Gate 8 schema."""
    slots = [(0.0, float(w)) for w in record.get("omegas", [])]
    slots += [(float(a), 0.0) for a in record.get("real_pairs", [])]
    slots += [(float(a), float(w)) for a, w in record.get("quads", [])]
    poly = tuple(float(p) for p in record.get("polynomial", ()))
    return general_evaluator_for(int(record["m"]), tuple(slots), poly)


def evaluator_from_member(path) -> GeneralEvaluator:
    """Return the evaluator of the member file at `path`."""
    return evaluator_from_record(json.loads(Path(path).read_text()))
