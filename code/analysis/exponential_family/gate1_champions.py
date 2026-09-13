# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Gate 1 producer: champion per band and budget, the admissible ladder, the
Theorem A5 conditioning receipt, and the four Gate 1 figures.

The contract is results/exponential_family/gate1/contract.md. This script is
its single entry point. It

  1. rebuilds the Gate 0 comparator quantities that Gate 1 needs (the six
     conditioning budgets and the band specific reference maxima),
  2. optimizes every (cell, budget, band) triple under the Gate 1 objective
     and the Gate 1 hard constraints,
  3. verifies every champion through three independent sample routes and a
     200001 point confirmation of every reported band maximum,
  4. applies the frozen decision rule, and
  5. writes champions.json, ladder.json, ladder.md, theory_A5.json, the four
     figures and captions.md.

It exits nonzero if any verification fails. With --rebuild it rebuilds every
receipt from the stored optimizer records of ladder.json without repeating the
optimization. With --check it recomputes the
champions' metrics from champions.json and compares them with the stored
receipt, which is the fast gate path. With --figures-only it redraws the four
figures and captions.md from the stored receipts without repeating the
optimization.

Conventions that the contract leaves open and this producer fixes:

* Budgets. C_SLC2, C_q8, C_SLC1 and C_BMC11 are recomputed from the Gate 0
  comparator definitions and cross checked against the frozen Gate 0 values
  to 1e-12 relative. C_BMC7 and C_BMC9 are the condition numbers of the
  polynomial members E_{8,0} and E_{10,0}. The ladder is ordered by budget
  value, which puts C_BMC7 (18.53) below C_SLC2 (19.74).
* The band specific references E_ref1(B) and E_ref2(B) are the SLC D1 and D2
  band maxima on the same 20001 point grid that Gate 0 used, so E_ref1(B0)
  and E_ref2(B0) reproduce the frozen E_SLC1 and E_SLC2 exactly.
* The optimizer evaluates the objective and the cheap constraints on a 4001
  point theta grid with an alias tail tolerance of 1e-10, polishes the five
  best candidates on the 20001 point grid with a tail tolerance of 1e-12, and
  every reported number is recomputed at the contract's 1e-14 tail tolerance
  on the 20001 point grid with a 200001 point confirmation.
* Constraints 1, 2, 3 and 6 are cheap and enter the optimizer as the additive
  penalty PENALTY * max(1, J) * violation. The objective is a ratio against
  SLC, so it runs from about 0.03 on B0 to well above 30 on B2, and a
  penalty that does not carry that scale buys a one percent conditioning
  overshoot far too cheaply on the wide bands. Constraint 5, the coupled
  off-grid channel, costs a 20001 by 129 transfer evaluation and is therefore
  checked once per reported optimum rather than inside the inner loop; the
  receipt records its value and the optimum is demoted if it fails.
* Flags. The contract asks for champion / threshold class / infeasible. A
  triple whose optimizer found no admissible member inside the budget is
  "infeasible"; the best constraint clearing member with m >= 6 of a
  (band, budget) pair is "champion"; every other optimum is "threshold
  class" and carries the explicit reason, which is either the list of hard
  constraints it violates, constraint 4 being the polynomial core m = 4, or
  "not the band and budget optimum" when it clears every constraint but is
  beaten inside its own pair.
* Sample routes. Route (ii), the pair kernel convolution, is implemented in
  Gate 0 for r <= 2; an r = 3 member records the route as unavailable with
  that reason. Champions found by this gate all have r <= 2.
* The symbols from the samples against the symbols from the alias sum are
  compared after dividing by max(1, max |symbol|). The three symbols have very
  different scales, kappa_2 reaching pi^2, and the alias sum accumulates its
  roundoff on that scale, so an unnormalized absolute test would measure the
  scale of kappa_2 rather than the agreement of the two routes.
* Agreement of the three sample routes is measured as an absolute deviation,
  which is also the relative deviation against the unit normalization
  sum_n beta(n) = 1. The smallest samples of a member sit near the double
  precision floor of the trapezoid inverse, so a literal elementwise relative
  test would measure that floor and not the routes.
* Band maxima and the 200001 point confirmation. The reported band maxima
  are the frozen Gate 0 quantity, the largest value on the uniform grid
  inside the band, so every Gate 0 row is reproduced exactly. No band edge is
  a grid point of either the 20001 or the 200001 point grid, so a member
  whose band supremum sits at the edge has a grid maximum that is low by
  O(grid spacing) and the two grids differ by about 1e-3 relative there. The
  contract's 1e-6 confirmation is therefore applied to the closed-band
  supremum, which evaluates the edge as well and is grid independent to about
  1e-9; the pure grid maximum gap is reported beside it as
  confirmation_mask_relative_gap.
* Gate 0 reproduction. The comparison against the Gate 0 scan winners on B0
  is classified per row. A row coincides when both band maxima match to 1e-9
  relative, it is an improvement when the Gate 1 objective is lower, and it is
  a tie inside the optimizer tolerance when the objective rises by at most
  1e-4 relative. Only a larger rise is a regression and fails the gate: at a
  budget the optimum reaches exactly, the admitted 1e-9 slack on the
  conditioning constraint moves the attainable objective by parts in a
  million, which is optimizer bookkeeping rather than a worse member.
* Figure fig_conditioning_map sweeps omega_2 at the single fixed omega_1 of
  the B0 champion at C_SLC1, so that the three polynomial cores are read at
  the same first frequency.
* Figure fig_band_dependence re-optimizes at each band edge with 80 starts
  instead of 300, and skips a cell whose polynomial limit conditioning (the
  conditioning of E_{m+2r,0}, which is the infimum over admissible omega)
  already exceeds the budget.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from decimal import Decimal, localcontext
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import zeta

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import family as fam  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate1"
FIGDIR = OUT / "figures"
CONTRACT = OUT / "contract.md"
GATE0 = ROOT / "results" / "exponential_family" / "gate0"

XI_GRID = np.linspace(-0.5, 0.5, 129)
COARSE_POINTS = 4001
POLISH_POINTS = met.THETA_POINTS
CONFIRM_POINTS = 200001
FAST_TOL = 1e-10
POLISH_TOL = 1e-12
FINE_TOL = 1e-14
PENALTY = 100.0
BASE_SEED = 20260912
GRID_PER_AXIS = {1: 60, 2: 60, 3: 24}
MIN_STARTS = 300
FIGURE_STARTS = 80
POLISH_COUNT = 5
EDGE = 1e-4
COUPLED_FACTOR = 2.0
BUDGET_SLACK = 1e-9
CONFIRM_RTOL = 1e-6
GATE0_TIE_RTOL = 1e-4
ROUTE_TOL_QUAD = 1e-13
ROUTE_TOL_DECIMAL = 1e-14
SYMBOL_ROUTE_TOL = 1e-13
CONFLUENT_GAP = 1e-6
POLYNOMIAL_LIMIT_GAP = 1e-3
BANDS = ("B0", "B1", "B2")
BUDGET_NAMES = ("C_BMC7", "C_SLC2", "C_q8", "C_BMC9", "C_SLC1", "C_BMC11")
CELLS = [(m, r) for m in (4, 6, 8) for r in (1, 2, 3) if m + 2 * r - 1 <= 11]

DECIMAL_DIGITS = 60
DECIMAL_GUARD = 20
DECIMAL_PREC = DECIMAL_DIGITS + DECIMAL_GUARD


def sha256(path: Path) -> str:
    """Return the hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The Gate 3 sufficient conditions S_m
# ---------------------------------------------------------------------------

def sufficient_condition_S(m: int, omega: float) -> float:
    """Return S_m(omega), the k = 0 dominance sum of the Gate 1 constraints.

        S_m(omega) = sum_{k>=1} [ (omega/(2 pi k - omega))^m
                                + (omega/(2 pi k + omega))^m ].

    With x = omega/(2 pi) in (0, 1/2) the two families are shifted zeta sums,
    sum_{k>=1} (k -+ x)^{-m} = zeta(m, 1 -+ x), so

        S_m(omega) = x^m [ zeta(m, 1 - x) + zeta(m, 1 + x) ],

    which is exact and needs no truncation. S_m is strictly increasing in
    omega on (0, pi) and S_m < 1 is the positivity condition of the k = 0
    dominance argument: the term of the alias sum with k = 0 exceeds the sum
    of the magnitudes of all the others.
    """
    m = int(m)
    if m < 2:
        raise ValueError("The sufficient condition needs an exponent of at least two.")
    x = float(omega) / (2.0 * np.pi)
    if not 0.0 < x < 0.5:
        return float("inf")
    return float(x ** m * (zeta(m, 1.0 - x) + zeta(m, 1.0 + x)))


def sufficient_condition_S_series(m: int, omega: float, terms: int = 4096):
    """Return the truncated S_m sum and the bound on its neglected tail.

    The tail beyond k = K is bounded by twice the larger family, and
    (omega/(2 pi t - omega))^m decreases in t, so

        tail <= 2 int_K^inf (omega/(2 pi t - omega))^m dt
              = omega^m / (pi (m - 1)) * (2 pi K - omega)^{1 - m}.
    """
    m = int(m)
    w = float(omega)
    k = np.arange(1, int(terms) + 1, dtype=float)
    total = float(np.sum((w / (2.0 * np.pi * k - w)) ** m
                         + (w / (2.0 * np.pi * k + w)) ** m))
    K = float(terms)
    bound = w ** m / (np.pi * (m - 1)) * (2.0 * np.pi * K - w) ** (1 - m)
    return total, float(bound)


# ---------------------------------------------------------------------------
# The decimal route: 60 digit alias sums with series sine and cosine
# ---------------------------------------------------------------------------

def _decimal_pi(prec: int) -> Decimal:
    """Return pi to `prec` digits by the Chudnovsky-free classical series."""
    with localcontext() as ctx:
        ctx.prec = prec + 10
        lasts, t, s, n, na, d, da = 0, Decimal(3), Decimal(3), 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    with localcontext() as ctx:
        ctx.prec = prec
        return +s


class DecimalTrig:
    """Series sine and cosine in decimal with explicit argument reduction.

    The argument is reduced modulo 2 pi with pi carried to the working
    precision, so the reduced argument y satisfies |y| <= pi and the
    alternating Taylor series

        sin y = sum_{j>=0} (-1)^j y^{2j+1}/(2j+1)!,
        cos y = sum_{j>=0} (-1)^j y^{2j}/(2j)!,

    has monotonically decreasing terms from the second term onwards. The
    truncation error of an alternating series with decreasing terms is
    bounded by the first omitted term, which is what `bound` records.
    """

    def __init__(self, prec: int = DECIMAL_PREC) -> None:
        self.prec = int(prec)
        self.pi = _decimal_pi(self.prec + 5)
        self.two_pi = 2 * self.pi
        self.tiny = Decimal(10) ** (-(self.prec + 5))
        self.bound = Decimal(0)

    def _record(self, term: Decimal) -> None:
        if abs(term) > self.bound:
            self.bound = abs(term)

    def reduce(self, x: Decimal) -> Decimal:
        """Return x reduced modulo 2 pi into the interval [-pi, pi]."""
        turns = (x / self.two_pi).to_integral_value(rounding="ROUND_HALF_EVEN")
        return x - turns * self.two_pi

    def sin(self, x: Decimal) -> Decimal:
        y = self.reduce(x)
        y2 = y * y
        term = y
        total = y
        j = 1
        while True:
            term = -term * y2 / ((2 * j) * (2 * j + 1))
            if abs(term) < self.tiny:
                self._record(term)
                return total
            total += term
            j += 1

    def cos(self, x: Decimal) -> Decimal:
        y = self.reduce(x)
        y2 = y * y
        term = Decimal(1)
        total = Decimal(1)
        j = 1
        while True:
            term = -term * y2 / ((2 * j - 1) * (2 * j))
            if abs(term) < self.tiny:
                self._record(term)
                return total
            total += term
            j += 1

    def sinc(self, x: Decimal) -> Decimal:
        """Return sin(x)/x, by the series in x near zero and by the ratio away."""
        if x == 0:
            return Decimal(1)
        if abs(x) < Decimal("0.5"):
            x2 = x * x
            term = Decimal(1)
            total = Decimal(1)
            j = 1
            while True:
                term = -term * x2 / ((2 * j) * (2 * j + 1))
                if abs(term) < self.tiny:
                    self._record(term)
                    return total
                total += term
                j += 1
        return self.sin(x) / x


def decimal_fourier(trig: DecimalTrig, m: int, omegas, halves, xi: Decimal) -> Decimal:
    """Return phi_hat(xi) of E_{m,r}(omega) in decimal arithmetic."""
    value = trig.sinc(xi / 2) ** m
    for w, half in zip(omegas, halves):
        value *= trig.sinc((xi - w) / 2) * trig.sinc((xi + w) / 2) / half
    return value


def decimal_integer_samples(m: int, omegas, K: int, nmax: int, M: int,
                            prec: int = DECIMAL_PREC) -> dict:
    """Return beta, beta', beta'' from the decimal alias sums and the inverse.

    The sampled symbols b, i kappa_1 b and -kappa_2 b are trigonometric
    polynomials of degree nmax, so the trapezoidal inverse on M > 2 nmax
    equispaced angles is exact. The alias sums

        S_q(theta) = sum_{|k| <= K} (theta + 2 pi k)^q phi_hat(theta + 2 pi k)

    are formed in decimal at the working precision with the same truncation
    index K as the double route, so the two routes truncate the same tail and
    their difference measures arithmetic alone. Each frequency enters as the
    exact decimal value of its double, not as its shortest decimal spelling,
    so the two routes describe the same member to the last bit.

    Every sine of the alias sum is evaluated once per angle rather than once
    per term. The abscissae are xi = 2 pi j / M + 2 pi k, so every half
    argument is pi j / M + pi k + c with c one of 0 and -+ omega_l / 2, and

        sin(pi j / M + pi k + c) = (-1)^k sin(pi j / M + c).

    The k dependence of every numerator is therefore the single sign
    (-1)^(k (m + 2 r)), which is one for an even generator width, and the
    inner loop over k carries only divisions. This also removes the reduction
    of an argument of size 2 pi K, which would otherwise spend several of the
    working digits on cancellation for the members that need a large K.
    """
    with localcontext() as ctx:
        ctx.prec = int(prec)
        trig = DecimalTrig(int(prec))
        pi = trig.pi
        two_pi = trig.two_pi
        om = [Decimal(float(w)) for w in omegas]
        halves = [trig.sinc(w / 2) ** 2 for w in om]
        factors = m + 2 * len(om)
        S0 = [Decimal(0)] * M
        S1 = [Decimal(0)] * M
        S2 = [Decimal(0)] * M
        for j in range(M):
            theta = two_pi * j / M
            base = pi * j / M
            core_sine = trig.sin(base)
            minus_sine = [trig.sin(base - w / 2) for w in om]
            plus_sine = [trig.sin(base + w / 2) for w in om]
            s0 = s1 = s2 = Decimal(0)
            for k in range(-K, K + 1):
                xi = theta + two_pi * k
                half = base + pi * k
                if half == 0:
                    P = Decimal(1)
                else:
                    P = (core_sine / half) ** m
                for index, w in enumerate(om):
                    lo = half - w / 2
                    hi = half + w / 2
                    left = Decimal(1) if lo == 0 else minus_sine[index] / lo
                    right = Decimal(1) if hi == 0 else plus_sine[index] / hi
                    P *= left * right / halves[index]
                if factors % 2 and k % 2:
                    P = -P
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
                c = cos_table[residue]
                s = sin_table[residue]
                a0 += S0[j] * c
                a1 += S1[j] * s
                a2 += S2[j] * c
            beta.append(a0 / M)
            dbeta.append(-a1 / M)
            ddbeta.append(-a2 / M)
        series_bound = float(trig.bound)
    return {
        "n": list(range(-nmax, nmax + 1)),
        "beta": [float(v) for v in beta],
        "dbeta": [float(v) for v in dbeta],
        "ddbeta": [float(v) for v in ddbeta],
        "beta_digits": [str(v) for v in beta],
        "K": int(K),
        "quadrature_points": int(M),
        "working_digits": int(prec),
        "reported_digits": DECIMAL_DIGITS,
        "series_truncation_bound": series_bound,
    }


def decimal_samples_of(generator: fam.ExponentialGenerator,
                       tol: float = FINE_TOL, prec: int = DECIMAL_PREC) -> dict:
    """Return the decimal route samples of a member at the double route's K."""
    K = max(generator.required_K(q, tol) for q in (0, 1, 2))
    out = decimal_integer_samples(generator.m, generator.omegas, K,
                                  generator.nmax, generator._quadrature_points(),
                                  prec=prec)
    out["alias_tail_bound"] = {str(q): generator.tail_bound(q, K) for q in (0, 1, 2)}
    return out


# ---------------------------------------------------------------------------
# Budgets, band references and the comparator quantities Gate 1 quotes
# ---------------------------------------------------------------------------

def budgets_and_references() -> tuple[dict, dict, dict]:
    """Return the six budgets, the band references, and the comparator rows."""
    theta = met.theta_grid(POLISH_POINTS)
    slc1 = comp.compact_kappa1(comp.SLC_D1["alpha"], comp.SLC_D1["a"])
    slc2 = comp.compact_kappa2(comp.SLC_D2["alpha"], comp.SLC_D2["a0"],
                                comp.SLC_D2["a"])
    budgets = {
        "C_SLC2": met.symbol_cond(comp.compact_left(comp.SLC_D2["alpha"]))["cond"],
        "C_q8": met.symbol_cond(comp.compact_left(comp.Q8_D1["alpha"]))["cond"],
        "C_SLC1": met.symbol_cond(comp.compact_left(comp.SLC_D1["alpha"]))["cond"],
        "C_BMC7": fam.ExponentialGenerator(8, ()).cond()["cond"],
        "C_BMC9": fam.ExponentialGenerator(10, ()).cond()["cond"],
        "C_BMC11": fam.ExponentialGenerator(12, ()).cond()["cond"],
    }
    e1 = np.asarray(slc1(theta)) / theta - 1.0
    e2 = np.asarray(slc2(theta)) / theta ** 2 - 1.0
    references = {}
    for key, edge in met.BANDS.items():
        references[key] = {"edge": float(edge),
                           "E_ref1": met.band_max(theta, e1, edge),
                           "E_ref2": met.band_max(theta, e2, edge)}

    named = {}
    pairs = [("SLC1992", comp.SLC_D1, comp.SLC_D2, 4, 4),
             ("TunedQ8", comp.Q8_D1, comp.Q8_D2, 8, 8),
             ("CW12A", comp.CW12A_D1, comp.CW12A_D2, 12, 12),
             ("CW12B", comp.CW12B_D1, comp.CW12B_D2, 12, 12)]
    for name, d1, d2, order1, order2 in pairs:
        k1 = comp.compact_kappa1(d1["alpha"], d1["a"])
        k2 = comp.compact_kappa2(d2["alpha"], d2["a0"], d2["a"])
        named[name] = _comparator_row(name, k1, k2, comp.compact_left(d1["alpha"]),
                                      order1, order2, theta)
    for p in (7, 9, 11):
        generator = fam.ExponentialGenerator(p + 1, ())
        k1 = lambda t, g=generator: g.symbols(t, FINE_TOL)[1]
        k2 = lambda t, g=generator: g.symbols(t, FINE_TOL)[2]
        left = lambda t, g=generator: g.symbols(t, FINE_TOL)[0]
        named[f"BMC{p}"] = _comparator_row(f"BMC{p}", k1, k2, left,
                                           p + 1, p - 1, theta)
    return budgets, references, named


def _comparator_row(name, kappa1, kappa2, left, order1, order2, theta) -> dict:
    """Return the band maxima, conditioning and formal orders of a comparator."""
    e1 = np.asarray(kappa1(theta), dtype=float) / theta - 1.0
    e2 = np.asarray(kappa2(theta), dtype=float) / theta ** 2 - 1.0
    row = {"name": name, "order_d1": order1, "order_d2": order2,
           "cond": met.symbol_cond(left, POLISH_POINTS)["cond"]}
    for key, edge in met.BANDS.items():
        row[f"e1_max_{key}"] = met.band_max(theta, e1, edge)
        row[f"e2_max_{key}"] = met.band_max(theta, e2, edge)
    return row


def polynomial_limit_cond(m: int, r: int) -> float:
    """Return the infimum of the conditioning of the cell (m, r).

    Every pair factor of b(pi) is positive and decreasing in omega on (0, pi)
    and tends to its polynomial value 4/((2k+1)^2 pi^2) as omega goes to
    zero, so b(pi) is largest, and the conditioning smallest, in the
    polynomial limit where the member is E_{m+2r,0} = BMC_{m+2r-1}.
    """
    return float(fam.ExponentialGenerator(m + 2 * r, ()).cond()["cond"])


# ---------------------------------------------------------------------------
# The Gate 1 objective and its cheap constraints
# ---------------------------------------------------------------------------

def _objective(m: int, omegas, theta, mask, budget: float,
               ref1: float, ref2: float, tol: float):
    """Return the penalized objective, the raw J, and the condition number."""
    omegas = np.sort(np.asarray(omegas, dtype=float))
    if np.any(omegas <= EDGE) or np.any(omegas >= np.pi - EDGE):
        out = 1e6 + float(np.sum(np.maximum(0.0, EDGE - omegas)
                                 + np.maximum(0.0, omegas - (np.pi - EDGE))))
        return out, np.inf, np.inf
    if omegas.size > 1 and np.any(np.diff(omegas) <= 1e-9):
        return 1e6, np.inf, np.inf
    generator = fam.ExponentialGenerator(m, omegas)
    try:
        b, k1, k2 = generator.symbols(theta, tol)
    except (ValueError, FloatingPointError):
        return 1e6, np.inf, np.inf
    if not np.all(np.isfinite(b)) or float(b.min()) <= 0.0:
        return 1e6, np.inf, np.inf
    if not np.all(np.isfinite(k2)) or float(k2.min()) <= 0.0:
        return 1e6, np.inf, np.inf
    cond = float(max(b.max(), 1.0) / b.min())
    e1 = np.abs(k1[mask] / theta[mask] - 1.0).max()
    e2 = np.abs(k2[mask] / theta[mask] ** 2 - 1.0).max()
    J = float(max(e1 / ref1, e2 / ref2))
    scale = max(1.0, J)
    top = float(omegas[-1])
    violation = max(0.0, cond / budget - 1.0)
    violation += max(0.0, sufficient_condition_S(m, top) - 1.0)
    violation += max(0.0, sufficient_condition_S(max(2, m - 2), top) - 1.0)
    return J + PENALTY * scale * violation, J, cond


def _cheap_constraints_ok(m: int, omegas, budget: float, cond: float) -> bool:
    """Return whether the cheap Gate 1 constraints hold at a returned point."""
    if not np.isfinite(cond) or cond > budget * (1.0 + BUDGET_SLACK):
        return False
    top = float(np.max(omegas))
    if sufficient_condition_S(m, top) >= 1.0:
        return False
    if sufficient_condition_S(max(2, m - 2), top) >= 1.0:
        return False
    return True


def _grid_seeds(m: int, r: int, theta, mask, budget, ref1, ref2):
    """Return the grid seed sweep of one cell as points, objectives and conds."""
    axis = np.linspace(0.0, np.pi, GRID_PER_AXIS[r] + 2)[1:-1]
    points = list(combinations(axis, r)) if r > 1 else [(w,) for w in axis]
    values = np.empty(len(points))
    conds = np.empty(len(points))
    for j, point in enumerate(points):
        _, J, cond = _objective(m, point, theta, mask, np.inf, ref1, ref2, FAST_TOL)
        values[j] = J
        conds[j] = cond
    return np.array(points, dtype=float), values, conds


def optimize_triple(task: dict) -> dict:
    """Return the constrained optimum of one (cell, budget, band) triple."""
    m, r = int(task["m"]), int(task["r"])
    budget = float(task["budget"])
    ref1, ref2 = float(task["ref1"]), float(task["ref2"])
    edge = float(task["edge"])
    starts_wanted = int(task.get("starts", MIN_STARTS))
    started = time.time()

    theta = met.theta_grid(COARSE_POINTS)
    mask = met.band_mask(theta, edge)
    points, values, conds = _grid_seeds(m, r, theta, mask, budget, ref1, ref2)
    feasible = np.isfinite(values) & (conds <= budget)
    seeds: list[np.ndarray] = []
    if np.any(feasible):
        order = np.argsort(np.where(feasible, values, np.inf))
        seeds = [points[j] for j in order[:50]]
    rng = np.random.default_rng(task["seed"])
    while len(seeds) < starts_wanted:
        draw = np.sort(rng.uniform(EDGE, np.pi - EDGE, size=r))
        if r == 1 or np.all(np.diff(draw) > 1e-6):
            seeds.append(draw)

    evaluations = 0
    results = []
    for seed in seeds:
        fun = lambda x: _objective(m, x, theta, mask, budget, ref1, ref2, FAST_TOL)[0]
        res = minimize(fun, np.asarray(seed, dtype=float), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-12,
                                "maxiter": 400 * r, "maxfev": 400 * r})
        evaluations += int(res.nfev)
        x = np.sort(np.asarray(res.x, dtype=float))
        _, J, cond = _objective(m, x, theta, mask, budget, ref1, ref2, FAST_TOL)
        if np.isfinite(J) and _cheap_constraints_ok(m, x, budget, cond):
            results.append((J, x, cond))
    results.sort(key=lambda item: item[0])

    fine = met.theta_grid(POLISH_POINTS)
    fine_mask = met.band_mask(fine, edge)
    polished = []
    for J, x, cond in results[:POLISH_COUNT]:
        fun = lambda y: _objective(m, y, fine, fine_mask, budget, ref1, ref2,
                                   POLISH_TOL)[0]
        res = minimize(fun, x, method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-15,
                                "maxiter": 400 * r, "maxfev": 400 * r})
        evaluations += int(res.nfev)
        y = np.sort(np.asarray(res.x, dtype=float))
        _, J2, cond2 = _objective(m, y, fine, fine_mask, budget, ref1, ref2,
                                  POLISH_TOL)
        if np.isfinite(J2) and _cheap_constraints_ok(m, y, budget, cond2):
            polished.append((J2, y, cond2))
        _, J1, cond1 = _objective(m, x, fine, fine_mask, budget, ref1, ref2,
                                  POLISH_TOL)
        if np.isfinite(J1) and _cheap_constraints_ok(m, x, budget, cond1):
            polished.append((J1, x, cond1))
    polished.sort(key=lambda item: item[0])

    out = {
        "m": m, "r": r, "p": m + 2 * r - 1,
        "band": task["band"], "edge": edge,
        "budget_name": task["budget_name"], "budget": budget,
        "E_ref1": ref1, "E_ref2": ref2,
        "grid_points": int(points.shape[0]),
        "grid_per_axis": GRID_PER_AXIS[r],
        "grid_feasible": int(np.count_nonzero(feasible)),
        "starts": len(seeds),
        "successful_starts": len(results),
        "function_evaluations": evaluations,
        "polynomial_limit_cond": polynomial_limit_cond(m, r),
        "seconds": time.time() - started,
    }
    if not polished:
        out.update({"omegas": None, "J_polish": None, "cond_polish": None,
                    "status": "infeasible"})
        return out
    J, omegas, cond = polished[0]
    out.update({"omegas": [float(w) for w in omegas],
                "J_polish": float(J), "cond_polish": float(cond),
                "status": "ok"})
    return out


# ---------------------------------------------------------------------------
# Full metrics, constraint receipts and the verification routes
# ---------------------------------------------------------------------------

def theory_A5(m: int, omegas) -> dict:
    """Return the two-term Theorem A5 conditioning formula and its ingredients.

    Theorem A5 writes, for even m,

        b(pi) = sum_k (2/((2k+1) pi))^m
                prod_j omega_j^2 / (((2k+1)^2 pi^2 - omega_j^2) tan^2(omega_j/2)),

    with every term positive, and max b = b(0) = 1, so cond = 1/b(pi). The two
    dominant terms are k = 0 and k = -1, which share |2k+1| = 1, hence

        log cond ~ m log(pi/2) - log 2
                   - sum_j log[ omega_j^2 / ((pi^2 - omega_j^2) tan^2(omega_j/2)) ].
    """
    omegas = [float(w) for w in omegas]
    factors = [w * w / ((np.pi ** 2 - w * w) * np.tan(w / 2.0) ** 2) for w in omegas]
    log_cond = m * math.log(np.pi / 2.0) - math.log(2.0) - sum(
        math.log(f) for f in factors)
    return {"pair_factors": factors,
            "log_cond_two_term": float(log_cond),
            "cond_two_term": float(math.exp(log_cond))}


def band_suprema(generator: fam.ExponentialGenerator, points: int) -> dict:
    """Return the closed-band suprema of |e_1| and |e_2| on a grid of `points`.

    The Gate 0 band maximum is the largest value on the uniform grid inside
    the band. None of the three band edges is a grid point of either the 20001
    or the 200001 point grid, so when the supremum of |e_q| over the closed
    band sits at the edge the two grids bracket it differently and their
    maxima differ by O(grid spacing). The closed-band supremum evaluates the
    edge as well, which removes that bracketing difference and is the quantity
    the 200001 point confirmation compares.
    """
    theta = met.theta_grid(points)
    _, k1, k2 = generator.symbols(theta, FINE_TOL)
    e1 = np.abs(k1 / theta - 1.0)
    e2 = np.abs(k2 / theta ** 2 - 1.0)
    edges = np.array([met.BANDS[key] for key in BANDS], dtype=float)
    _, ke1, ke2 = generator.symbols(edges, FINE_TOL)
    edge_e1 = np.abs(ke1 / edges - 1.0)
    edge_e2 = np.abs(ke2 / edges ** 2 - 1.0)
    out = {}
    for j, key in enumerate(BANDS):
        mask = met.band_mask(theta, edges[j])
        out[f"e1_sup_{key}"] = float(max(e1[mask].max(), edge_e1[j]))
        out[f"e2_sup_{key}"] = float(max(e2[mask].max(), edge_e2[j]))
        out[f"e1_max_{key}"] = float(e1[mask].max())
        out[f"e2_max_{key}"] = float(e2[mask].max())
        out[f"e1_edge_{key}"] = float(edge_e1[j])
        out[f"e2_edge_{key}"] = float(edge_e2[j])
    return out


def _fine_confirmation(generator: fam.ExponentialGenerator) -> dict:
    """Return the 200001 point confirmation of the band maxima and the signs."""
    theta = met.theta_grid(CONFIRM_POINTS)
    b, _, k2 = generator.symbols(theta, FINE_TOL)
    with_zero = np.linspace(0.0, np.pi, CONFIRM_POINTS)
    b0 = generator.symbols(with_zero, FINE_TOL)[0]
    out = {"points": CONFIRM_POINTS,
           "min_b": float(b.min()), "max_b": float(b0.max()),
           "cond": float(b0.max() / b0.min()),
           "min_kappa2": float(k2.min())}
    out.update(band_suprema(generator, CONFIRM_POINTS))
    return out


def _discrete_checks(generator: fam.ExponentialGenerator) -> dict:
    """Return the discrete conditioning and the kappa_2 sign on the mode grids."""
    out = {}
    for N in met.DISCRETE_N:
        angles = 2.0 * np.pi * np.arange(N) / N
        b, _, k2 = generator.symbols(angles, FINE_TOL)
        out[f"cond_N{N}"] = float(b.max() / b.min())
        out[f"min_b_N{N}"] = float(b.min())
        out[f"min_kappa2_nonzero_N{N}"] = float(k2[1:].min())
        out[f"kappa2_zero_at_N{N}"] = float(abs(k2[0]))
    return out


def member_record(m: int, omegas, budget_name: str, budget: float,
                  references: dict, *, with_routes: bool = False) -> dict:
    """Return every Gate 1 metric, constraint receipt and flag of one member."""
    generator = fam.ExponentialGenerator(m, omegas)
    name = f"{generator.label()}@{budget_name}"
    scheme = comp.spline_scheme(name, generator, coupled_xi=XI_GRID, tol=FINE_TOL)
    row = met.evaluate(name, scheme.kappa1, scheme.kappa2, scheme.left,
                       analytic_gv=scheme.analytic_gv, coupled=scheme.coupled,
                       algebraic_order1=scheme.algebraic_order1,
                       algebraic_order2=scheme.algebraic_order2,
                       bandwidth_left=scheme.bandwidth_left,
                       bandwidth_d1=scheme.bandwidth_d1,
                       bandwidth_d2=scheme.bandwidth_d2, solves=scheme.solves,
                       extra=dict(scheme.extra))
    row["budget_name"] = budget_name
    row["budget"] = budget
    fine = _fine_confirmation(generator)
    row["confirmation"] = fine
    row.update(_discrete_checks(generator))

    coarse_sup = band_suprema(generator, POLISH_POINTS)
    row["band_suprema"] = coarse_sup
    confirm_gap = 0.0
    mask_gap = 0.0
    for key in BANDS:
        for tag in ("e1_sup_", "e2_sup_"):
            coarse = float(coarse_sup[tag + key])
            exact = float(fine[tag + key])
            confirm_gap = max(confirm_gap, abs(exact - coarse) / max(exact, 1e-300))
        for tag in ("e1_max_", "e2_max_"):
            coarse = float(row[tag + key])
            exact = float(fine[tag + key])
            mask_gap = max(mask_gap, abs(exact - coarse) / max(exact, 1e-300))
    row["confirmation_relative_gap"] = float(confirm_gap)
    row["confirmation_mask_relative_gap"] = float(mask_gap)
    row["confirmation_ok"] = bool(confirm_gap <= CONFIRM_RTOL)

    for key in met.BANDS:
        ref = references[key]
        row[f"J_{key}"] = float(max(row[f"e1_max_{key}"] / ref["E_ref1"],
                                    row[f"e2_max_{key}"] / ref["E_ref2"]))
        row[f"ratio_d1_{key}"] = float(row[f"e1_max_{key}"] / ref["E_ref1"])
        row[f"ratio_d2_{key}"] = float(row[f"e2_max_{key}"] / ref["E_ref2"])

    top = float(max(generator.omegas))
    s_hi = sufficient_condition_S(m, top)
    s_lo = sufficient_condition_S(max(2, m - 2), top)
    gaps = list(np.diff(np.asarray(generator.omegas, dtype=float))) if len(
        generator.omegas) > 1 else []
    admissible, admissible_detail = generator.admissible()
    constraints = {
        "c1_cond": {"cond": fine["cond"], "budget": budget,
                    "cond_N64": row["cond_N64"], "cond_N127": row["cond_N127"],
                    "cond_N128": row["cond_N128"],
                    "ok": bool(fine["cond"] <= budget * (1.0 + BUDGET_SLACK)
                               and max(row["cond_N64"], row["cond_N127"],
                                       row["cond_N128"])
                               <= budget * (1.0 + BUDGET_SLACK))},
        "c2_symbol_margin": {"min_b": fine["min_b"], "one_over_budget": 1.0 / budget,
                             "S_m": s_hi, "m": m,
                             "ok": bool(fine["min_b"] > 0.0
                                        and fine["min_b"] >= (1.0 / budget)
                                        * (1.0 - BUDGET_SLACK)
                                        and s_hi < 1.0)},
        "c3_kappa2_positive": {"min_kappa2": fine["min_kappa2"],
                               "S_m_minus_2": s_lo, "exponent": max(2, m - 2),
                               "min_kappa2_N64": row["min_kappa2_nonzero_N64"],
                               "min_kappa2_N127": row["min_kappa2_nonzero_N127"],
                               "min_kappa2_N128": row["min_kappa2_nonzero_N128"],
                               "ok": bool(fine["min_kappa2"] > 0.0 and s_lo < 1.0
                                          and min(
                                              row["min_kappa2_nonzero_N64"],
                                              row["min_kappa2_nonzero_N127"],
                                              row["min_kappa2_nonzero_N128"]) > 0.0)},
        "c4_formal_order": {"order_d1": m, "order_d2": m - 2,
                            "ok": bool(m >= 6)},
        "c6_admissible": {"omegas": list(generator.omegas),
                          "min_gap": float(min(gaps)) if gaps else None,
                          "confluent": bool(gaps and min(gaps) < CONFLUENT_GAP),
                          "near_polynomial_limit": bool(
                              min(generator.omegas) < POLYNOMIAL_LIMIT_GAP),
                          "omega_min": float(min(generator.omegas)),
                          "detail": admissible_detail,
                          "ok": bool(admissible)},
    }
    coupled = {}
    for key in met.BANDS:
        limit = COUPLED_FACTOR * max(row[f"e1_max_{key}"], row[f"e2_max_{key}"])
        coupled[key] = {"ec_max": row[f"ec_max_{key}"], "limit": limit,
                        "ok": bool(row[f"ec_max_{key}"] <= limit)}
    constraints["c5_coupled"] = coupled
    row["constraints"] = constraints
    row["theory_A5"] = theory_A5(m, generator.omegas)
    row["theory_A5"]["cond_actual"] = fine["cond"]
    row["theory_A5"]["relative_error"] = float(
        abs(row["theory_A5"]["cond_two_term"] - fine["cond"]) / fine["cond"])
    row["polynomial_limit_cond"] = polynomial_limit_cond(m, len(generator.omegas))
    if with_routes:
        row["sample_routes"] = sample_routes(generator)
    return row


def constraints_failed(row: dict, band: str) -> list[str]:
    """Return the labels of the hard constraints this member violates on a band."""
    bad = []
    c = row["constraints"]
    if not c["c1_cond"]["ok"]:
        bad.append("1 conditioning budget")
    if not c["c2_symbol_margin"]["ok"]:
        bad.append("2 sample symbol margin")
    if not c["c3_kappa2_positive"]["ok"]:
        bad.append("3 kappa_2 positivity")
    if not c["c4_formal_order"]["ok"]:
        bad.append("4 formal order")
    if not c["c5_coupled"][band]["ok"]:
        bad.append("5 coupled off-grid channel")
    if not c["c6_admissible"]["ok"]:
        bad.append("6 admissibility")
    return bad


def sample_routes(generator: fam.ExponentialGenerator) -> dict:
    """Return the three integer sample routes of a member and their agreement."""
    alias = generator.integer_samples(FINE_TOL)
    positive = alias.n >= 0
    out = {"alias": {"n": [int(v) for v in alias.n],
                     "beta": [float(v) for v in alias.beta],
                     "dbeta": [float(v) for v in alias.dbeta],
                     "ddbeta": [float(v) for v in alias.ddbeta],
                     "truncation": alias.truncation}}
    order = np.arange(0, int(alias.n.max()) + 1)
    if generator.r <= 2:
        quad = [fam.convolution_samples(generator.m, generator.omegas, order,
                                        derivative=d) for d in (0, 1, 2)]
        deviation = max(float(np.max(np.abs(quad[d] - arr[positive])))
                        for d, arr in enumerate((alias.beta, alias.dbeta,
                                                 alias.ddbeta)))
        out["pair_kernel_quadrature"] = {
            "n": [int(v) for v in order],
            "beta": [float(v) for v in quad[0]],
            "dbeta": [float(v) for v in quad[1]],
            "ddbeta": [float(v) for v in quad[2]],
            "deviation": deviation,
            "tolerance": ROUTE_TOL_QUAD,
            "ok": bool(deviation <= ROUTE_TOL_QUAD),
        }
    else:
        out["pair_kernel_quadrature"] = {
            "available": False,
            "reason": "the Gate 0 pair kernel quadrature covers r <= 2",
        }
    decimal_route = decimal_samples_of(generator)
    deviation = max(
        float(np.max(np.abs(np.asarray(decimal_route[key]) - arr)))
        for key, arr in (("beta", alias.beta), ("dbeta", alias.dbeta),
                         ("ddbeta", alias.ddbeta)))
    decimal_route["deviation"] = deviation
    decimal_route["tolerance"] = ROUTE_TOL_DECIMAL
    decimal_route["ok"] = bool(deviation <= ROUTE_TOL_DECIMAL)
    out["decimal"] = decimal_route

    probe = np.linspace(0.05, np.pi, 41)
    from_alias = generator.sample_symbols(probe, tol=FINE_TOL)
    from_samples = generator.symbols(probe, FINE_TOL)
    absolute = [float(np.max(np.abs(from_alias[q] - from_samples[q])))
                for q in range(3)]
    scales = [max(1.0, float(np.max(np.abs(from_samples[q])))) for q in range(3)]
    gap = float(max(a / s for a, s in zip(absolute, scales)))
    out["symbols_from_samples_versus_alias"] = {
        "deviation": gap, "absolute_deviation": absolute, "scales": scales,
        "tolerance": SYMBOL_ROUTE_TOL, "ok": bool(gap <= SYMBOL_ROUTE_TOL)}
    out["all_ok"] = bool(
        out["decimal"]["ok"]
        and out["symbols_from_samples_versus_alias"]["ok"]
        and out["pair_kernel_quadrature"].get("ok", True))
    return out


# ---------------------------------------------------------------------------
# The ladder, the champions and the frozen decision
# ---------------------------------------------------------------------------

def build_tasks(budgets: dict, references: dict,
                starts: int = MIN_STARTS) -> list[dict]:
    """Return the (cell, budget, band) triples in a deterministic order."""
    tasks = []
    for bi, band in enumerate(BANDS):
        for ni, budget_name in enumerate(BUDGET_NAMES):
            for m, r in CELLS:
                tasks.append({
                    "m": m, "r": r, "band": band,
                    "edge": references[band]["edge"],
                    "ref1": references[band]["E_ref1"],
                    "ref2": references[band]["E_ref2"],
                    "budget_name": budget_name,
                    "budget": float(budgets[budget_name]),
                    "starts": starts,
                    "seed": BASE_SEED + 100000 * bi + 10000 * ni + 100 * m + r,
                })
    return tasks


def _record_worker(item):
    """Return the full member record of one successful triple."""
    record, references = item
    row = member_record(record["m"], record["omegas"], record["budget_name"],
                        record["budget"], references)
    return record, row


def run_ladder(budgets: dict, references: dict, processes: int,
               starts: int = MIN_STARTS) -> tuple[list[dict], list[dict]]:
    """Return the optimizer records and the full metric rows of every triple."""
    tasks = build_tasks(budgets, references, starts)
    records = []
    with Pool(processes=processes) as pool:
        for done, record in enumerate(pool.imap_unordered(optimize_triple, tasks), 1):
            records.append(record)
            print(f"  [{done:3d}/{len(tasks)}] band={record['band']} "
                  f"budget={record['budget_name']} m={record['m']} r={record['r']} "
                  f"status={record['status']} J={record.get('J_polish')} "
                  f"cond={record.get('cond_polish')} "
                  f"nfev={record['function_evaluations']} "
                  f"{record['seconds']:.1f}s", flush=True)
    records.sort(key=lambda rec: (BANDS.index(rec["band"]),
                                  BUDGET_NAMES.index(rec["budget_name"]),
                                  rec["m"], rec["r"]))
    rows, index = ladder_rows(records, references, processes)
    return records, rows, index


def ladder_rows(records: list[dict], references: dict, processes: int):
    """Return the full metric rows of every successful triple and their index."""
    payload = [(rec, references) for rec in records if rec["status"] == "ok"]
    rows: list[dict] = []
    index: dict[tuple, dict] = {}
    with Pool(processes=processes) as pool:
        for record, row in pool.imap(_record_worker, payload):
            rows.append(row)
            index[(record["band"], record["m"], record["r"],
                   record["budget_name"])] = row
    return rows, index


def assemble_ladder(records: list[dict], index: dict) -> list[dict]:
    """Return one ladder entry per triple with the metrics and the flag.

    The index is keyed by the whole triple, because the same cell and budget
    carry a different optimum on each band.
    """
    ladder = []
    for record in records:
        entry = {"band": record["band"], "budget_name": record["budget_name"],
                 "budget": record["budget"], "m": record["m"], "r": record["r"],
                 "p": record["p"], "status": record["status"],
                 "omegas": record["omegas"],
                 "polynomial_limit_cond": record["polynomial_limit_cond"],
                 "optimizer": {k: record[k] for k in
                               ("grid_points", "grid_feasible", "starts",
                                "successful_starts", "function_evaluations",
                                "seconds", "J_polish", "cond_polish")}}
        if record["status"] != "ok":
            entry["flag"] = "infeasible"
            entry["reason"] = ("no admissible member of this cell meets the "
                               "budget; the polynomial limit conditioning "
                               f"{record['polynomial_limit_cond']:.6g} is the "
                               "infimum over the cell")
            ladder.append(entry)
            continue
        band = record["band"]
        row = index[(band, record["m"], record["r"], record["budget_name"])]
        entry["metrics"] = row
        entry["J"] = row[f"J_{band}"]
        entry["violations"] = constraints_failed(row, band)
        entry["flag"] = "threshold class"
        ladder.append(entry)
    return ladder


def select_champions(ladder: list[dict], references: dict) -> dict:
    """Return the champion and the threshold class runner-up of every pair."""
    champions = {}
    for band in BANDS:
        champions[band] = {}
        for budget_name in BUDGET_NAMES:
            pool = [e for e in ladder if e["band"] == band
                    and e["budget_name"] == budget_name and e["status"] == "ok"]
            clear = [e for e in pool if not e["violations"]]
            champion = min(clear, key=lambda e: e["J"]) if clear else None
            if champion is not None:
                champion["flag"] = "champion"
            rest = [e for e in pool if e is not champion]
            runner = min(rest, key=lambda e: e["J"]) if rest else None
            for entry in rest:
                if entry["violations"]:
                    entry["reason"] = "violates constraint " + "; ".join(
                        entry["violations"])
                else:
                    entry["reason"] = "not the band and budget optimum"
            champions[band][budget_name] = {
                "band": band, "edge": references[band]["edge"],
                "E_ref1": references[band]["E_ref1"],
                "E_ref2": references[band]["E_ref2"],
                "budget_name": budget_name,
                "budget": pool[0]["budget"] if pool else None,
                "champion": None if champion is None else {
                    "m": champion["m"], "r": champion["r"], "p": champion["p"],
                    "omegas": champion["omegas"], "J": champion["J"],
                    "metrics": champion["metrics"]},
                "threshold_runner_up": None if runner is None else {
                    "m": runner["m"], "r": runner["r"], "p": runner["p"],
                    "omegas": runner["omegas"], "J": runner["J"],
                    "reason": runner["reason"], "metrics": runner["metrics"]},
            }
    return champions


def attach_routes(champions: dict, processes: int) -> list[str]:
    """Verify every champion and runner-up through the three sample routes."""
    failures = []
    seen: dict[tuple, dict] = {}
    for band in BANDS:
        for budget_name in BUDGET_NAMES:
            cell = champions[band][budget_name]
            for role in ("champion", "threshold_runner_up"):
                member = cell[role]
                if member is None:
                    continue
                key = (member["m"], tuple(member["omegas"]))
                if key not in seen:
                    generator = fam.ExponentialGenerator(member["m"], member["omegas"])
                    seen[key] = sample_routes(generator)
                member["sample_routes"] = seen[key]
                if not seen[key]["all_ok"]:
                    failures.append(f"{band}/{budget_name}/{role} sample routes")
                if not member["metrics"]["confirmation_ok"]:
                    failures.append(f"{band}/{budget_name}/{role} 200001 point "
                                    "confirmation")
    return failures


def gate0_reproduction(ladder: list[dict]) -> dict:
    """Return the comparison of the Gate 1 B0 optima with the Gate 0 winners."""
    stored = json.loads((GATE0 / "table.json").read_text(encoding="utf-8"))
    winners = {}
    for row in stored["scan_winners"]:
        winners[(row["m"], row["r"], row["budget_name"])] = row
    out = []
    for entry in ladder:
        if entry["band"] != "B0" or entry["status"] != "ok":
            continue
        key = (entry["m"], entry["r"], entry["budget_name"])
        if key not in winners:
            continue
        old = winners[key]
        new = entry["metrics"]
        same = (abs(new["e1_max_B0"] - old["e1_max_B0"])
                <= 1e-9 * abs(old["e1_max_B0"])
                and abs(new["e2_max_B0"] - old["e2_max_B0"])
                <= 1e-9 * abs(old["e2_max_B0"]))
        frozen = stored["reference_band_maxima"]
        gate0_J = float(max(old["e1_max_B0"] / frozen["E_SLC1"],
                            old["e2_max_B0"] / frozen["E_SLC2"]))
        rise = (entry["J"] - gate0_J) / gate0_J if gate0_J > 0.0 else 0.0
        if same:
            verdict = "coincides"
        elif rise <= 0.0:
            verdict = "improved"
        elif rise <= GATE0_TIE_RTOL:
            verdict = "tie inside the optimizer tolerance"
        else:
            verdict = "regression"
        out.append({
            "m": entry["m"], "r": entry["r"], "budget_name": entry["budget_name"],
            "gate0_omegas": old["omegas"], "gate1_omegas": entry["omegas"],
            "gate0_J": gate0_J, "gate1_J": entry["J"],
            "relative_rise": float(rise),
            "coincides": bool(same),
            "improvement": float(gate0_J / entry["J"]) if entry["J"] > 0.0 else None,
            "verdict": verdict,
            "regression": bool(verdict == "regression"),
        })
    return {"rows": out,
            "tie_tolerance": GATE0_TIE_RTOL,
            "coincident": sum(1 for row in out if row["verdict"] == "coincides"),
            "improved": sum(1 for row in out if row["verdict"] == "improved"),
            "ties": sum(1 for row in out
                        if row["verdict"] == "tie inside the optimizer tolerance"),
            "regressions": [row for row in out if row["regression"]]}


def decide(champions: dict, references: dict, named: dict) -> dict:
    """Return the frozen Gate 1 decision outcome and its controlling numbers."""
    cell = champions["B0"]["C_SLC1"]
    champion = cell["champion"]
    ref = references["B0"]
    q8 = named["TunedQ8"]
    bmc11 = named["BMC11"]
    detail = {
        "E_ref1_B0": ref["E_ref1"], "E_ref2_B0": ref["E_ref2"],
        "E_q8_1_B0": q8["e1_max_B0"], "E_q8_2_B0": q8["e2_max_B0"],
        "E_BMC11_1_B0": bmc11["e1_max_B0"], "E_BMC11_2_B0": bmc11["e2_max_B0"],
    }
    if champion is None:
        outcome = "REJECT"
        tests = {}
    else:
        e1 = champion["metrics"]["e1_max_B0"]
        e2 = champion["metrics"]["e2_max_B0"]
        tests = {
            "e1_at_most_slc_over_five": bool(e1 <= ref["E_ref1"] / 5.0),
            "e2_at_most_slc_over_five": bool(e2 <= ref["E_ref2"] / 5.0),
            "e1_at_most_q8_over_two": bool(e1 <= q8["e1_max_B0"] / 2.0),
            "e2_at_most_q8_over_two": bool(e2 <= q8["e2_max_B0"] / 2.0),
            "e1_at_most_bmc11": bool(e1 <= bmc11["e1_max_B0"]),
            "e2_at_most_bmc11": bool(e2 <= bmc11["e2_max_B0"]),
        }
        beats = e1 < ref["E_ref1"] and e2 < ref["E_ref2"]
        if all(tests.values()):
            outcome = "TRIUMPH"
        elif beats:
            outcome = "PARTIAL"
        else:
            outcome = "REJECT"
        detail.update({"E1max_B0": e1, "E2max_B0": e2,
                       "E1_gain": ref["E_ref1"] / e1, "E2_gain": ref["E_ref2"] / e2,
                       "J": champion["J"], "cond": champion["metrics"]["cond"]})
    boundary = {}
    for band in ("B1", "B2"):
        entry = champions[band]["C_SLC1"]["champion"]
        beat = None
        if entry is not None:
            beat = bool(entry["metrics"][f"e1_max_{band}"]
                        < references[band]["E_ref1"]
                        and entry["metrics"][f"e2_max_{band}"]
                        < references[band]["E_ref2"])
        any_beat = False
        for budget_name in BUDGET_NAMES:
            other = champions[band][budget_name]["champion"]
            if other is None:
                continue
            if (other["metrics"][f"e1_max_{band}"] < references[band]["E_ref1"]
                    and other["metrics"][f"e2_max_{band}"]
                    < references[band]["E_ref2"]):
                any_beat = True
        boundary[band] = {"champion_at_C_SLC1_beats_slc": beat,
                          "any_champion_beats_slc": any_beat}
    return {
        "outcome": outcome,
        "rule": "TRIUMPH at (B0, C_SLC1) if the champion with m >= 6 meeting "
                "constraints 1 to 6 has E1max(B0) <= E_SLC1/5, E2max(B0) <= "
                "E_SLC2/5, E1max(B0) <= E_q8,1(B0)/2, E2max(B0) <= E_q8,2(B0)/2 "
                "and both at most the BMC11 values; PARTIAL if it beats SLC on "
                "both by less than five times; REJECT otherwise.",
        "tests": tests,
        "numbers": detail,
        "champion": None if champion is None else {
            "m": champion["m"], "r": champion["r"], "p": champion["p"],
            "omegas": champion["omegas"], "J": champion["J"]},
        "boundary_statements": boundary,
    }


# ---------------------------------------------------------------------------
# Receipts in markdown
# ---------------------------------------------------------------------------

def fmt(value, digits: int = 6) -> str:
    """Return a compact rendering with `digits` significant figures."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.{digits}g}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v, digits) for v in value) + "]"
    return str(value)


LADDER_COLUMNS = [
    ("budget", lambda e: e["budget_name"]),
    ("cond budget", lambda e: fmt(e["budget"])),
    ("m", lambda e: e["m"]),
    ("r", lambda e: e["r"]),
    ("p", lambda e: e["p"]),
    ("omegas", lambda e: fmt(e["omegas"])),
    ("flag", lambda e: e["flag"]),
    ("J", lambda e: fmt(e.get("J"))),
    ("E1max", lambda e: fmt(_metric(e, "e1_max_"))),
    ("E2max", lambda e: fmt(_metric(e, "e2_max_"))),
    ("ec_max", lambda e: fmt(_metric(e, "ec_max_"))),
    ("cond", lambda e: fmt(_field(e, "cond"))),
    ("S_m", lambda e: fmt(_constraint(e, "c2_symbol_margin", "S_m"))),
    ("S_m-2", lambda e: fmt(_constraint(e, "c3_kappa2_positive", "S_m_minus_2"))),
    ("reason", lambda e: e.get("reason", "")),
]


def _metric(entry: dict, prefix: str):
    if "metrics" not in entry:
        return None
    return entry["metrics"].get(prefix + entry["band"])


def _field(entry: dict, key: str):
    if "metrics" not in entry:
        return None
    return entry["metrics"].get(key)


def _constraint(entry: dict, block: str, key: str):
    if "metrics" not in entry:
        return None
    return entry["metrics"]["constraints"][block].get(key)


def ladder_markdown(ladder: list[dict], budgets: dict, references: dict,
                    decision: dict, champions: dict) -> str:
    """Return the human readable Gate 1 ladder, one section per band."""
    lines = ["# Gate 1 ladder", "",
             "Produced by code/analysis/exponential_family/gate1_champions.py",
             "under results/exponential_family/gate1/contract.md. Every row is",
             "the constrained optimum of one (cell, budget, band) triple. The",
             "flag is champion for the best constraint clearing member with",
             "m >= 6 of its (band, budget) pair, threshold class for every",
             "other optimum, and infeasible when no admissible member of the",
             "cell meets the budget.", "",
             f"Decision at (B0, C_SLC1): {decision['outcome']}.", "",
             "Conditioning budgets", "",
             "| budget | value |", "| --- | --- |"]
    for key in BUDGET_NAMES:
        lines.append(f"| {key} | {budgets[key]:.6f} |")
    lines += ["", "Band reference maxima (SLC D1 and D2 on the same band)", "",
              "| band | edge | E_ref1 | E_ref2 |", "| --- | --- | --- | --- |"]
    for band in BANDS:
        ref = references[band]
        lines.append(f"| {band} | {ref['edge']:.6f} | {ref['E_ref1']:.6e} | "
                     f"{ref['E_ref2']:.6e} |")
    lines.append("")
    titles = {"B0": "B0 = (0, 2 pi/3], the primary dealiased DNS band",
              "B1": "B1 = (0, 0.75 pi]", "B2": "B2 = (0, 0.8 pi]"}
    for band in BANDS:
        lines += [f"## Band {titles[band]}", ""]
        champion_line = []
        for budget_name in BUDGET_NAMES:
            cell = champions[band][budget_name]
            if cell["champion"] is None:
                champion_line.append(f"| {budget_name} | none | | | |")
                continue
            member = cell["champion"]
            label = fam.ExponentialGenerator(member["m"], member["omegas"]).label()
            champion_line.append(
                f"| {budget_name} | {label} | {member['J']:.6g} | "
                f"{member['metrics']['e1_max_' + band]:.6e} | "
                f"{member['metrics']['e2_max_' + band]:.6e} |")
        lines += ["Champion per budget", "",
                  "| budget | champion | J | E1max | E2max |",
                  "| --- | --- | --- | --- | --- |"] + champion_line + [""]
        lines.append("| " + " | ".join(name for name, _ in LADDER_COLUMNS) + " |")
        lines.append("| " + " | ".join("---" for _ in LADDER_COLUMNS) + " |")
        for entry in ladder:
            if entry["band"] != band:
                continue
            cells = [str(getter(entry)) for _, getter in LADDER_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Conventions", "",
              "* J = max(E1max(B)/E_ref1(B), E2max(B)/E_ref2(B)) with the SLC",
              "  D1 and D2 band maxima on the same band as the references.",
              "* Band maxima and the conditioning are the 20001 point contract",
              "  values; every reported band maximum is confirmed on a 200001",
              "  point grid to 1e-6 relative.",
              "* S_m and S_m-2 are the Gate 3 sufficient conditions evaluated at",
              "  the largest frequency of the member; both must be below one.",
              "* ec_max is the coupled off-grid channel maximum on the band; the",
              "  constraint is ec_max <= 2 max(E1max, E2max).",
              "* The polynomial limit conditioning of a cell, which is the",
              "  conditioning of E_{m+2r,0}, is the infimum of the conditioning",
              "  over the cell and explains every infeasible row.", ""]
    return "\n".join(lines)


def captions_markdown(champions: dict, references: dict, budgets: dict) -> str:
    """Return the captions of the four Gate 1 figures."""
    champ = champions["B0"]["C_SLC1"]["champion"]
    label = "none" if champ is None else fam.ExponentialGenerator(
        champ["m"], champ["omegas"]).label()
    omega1 = "none" if champ is None else f"{champ['omegas'][0]:.6f}"
    return "\n".join([
        "# Gate 1 figure captions", "",
        "Captions for the figures in results/exponential_family/gate1/figures.",
        "Every number quoted here is taken from champions.json and ladder.json.",
        "", "## fig_two_dial.pdf", "",
        "Band-maximum relative error on B0 against the formal first-derivative",
        "order of the member, for every constrained optimum of the admissible",
        "ladder at the two largest budgets. Left, the first derivative; right,",
        "the second derivative on the same abscissa. Filled markers clear every",
        "hard constraint; open markers are the threshold class with polynomial",
        "core m = 4. The published and tuned comparators are marked at their own",
        "formal order. The two dials are read across the panel: the abscissa is",
        "the polynomial core, the marker shape is the number of frequency pairs.",
        "", "## fig_error_curves.pdf", "",
        f"Relative symbol errors of the B0 champion at C_SLC1, {label}, against",
        "SLC 1992, the tuned q8 pair, CW12B and BMC11 on the whole resolved",
        "range. Left, the first derivative; right, the second. Vertical marks at",
        "2 pi/3, 0.75 pi and 0.8 pi are the three working bands; horizontal marks",
        "are the one percent, one per mille and one per ten thousand tolerances.",
        "", "## fig_conditioning_map.pdf", "",
        "Left, the left-side condition number of E_{m,2}(omega_1, omega_2)",
        f"against the second frequency at the fixed first frequency omega_1 =",
        f"{omega1}, for the three polynomial cores: the broad pale curve is the",
        "computed condition number and the dashed curve on top of it is the",
        "two-term formula of Theorem A5. The horizontal lines are the SLC D1",
        "and BMC11 conditioning budgets. Right, the relative error of the",
        "two-term formula on the same abscissa.",
        "", "## fig_band_dependence.pdf", "",
        "Objective J of the best member at C_SLC1 against the band edge T, for",
        "the two admissible polynomial cores with fourth-order or higher direct",
        "second derivatives. J is measured against the SLC D1 and D2 maxima on",
        "the same band, so the advantage over SLC ends where a curve crosses",
        "the horizontal line at one.", ""])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

CORE_COLOR = {4: "#E69F00", 6: "#0072B2", 8: "#009E73"}
BUDGET_COLOR = {"C_SLC1": "#0072B2", "C_BMC11": "#009E73"}
PAIR_MARKER = {1: "o", 2: "s", 3: "D"}
COMPARATOR_STYLE = {
    "SLC1992": ("#D55E00", "*", 11.0, (6.0, -1.0)),
    "TunedQ8": ("#CC79A7", "P", 7.0, (6.0, -7.0)),
    "CW12A": ("#56B4E9", "^", 6.5, (6.0, -1.0)),
    "CW12B": ("#8C510A", "v", 6.5, (6.0, 5.0)),
    "BMC7": ("#777777", "x", 6.5, (6.0, 5.0)),
    "BMC9": ("#3A3A3A", "+", 7.5, (6.0, -1.0)),
    "BMC11": ("#000000", "1", 9.0, (6.0, -7.0)),
}
TOLERANCE_GREY = "0.45"


def _style():
    """Return the figure module after applying the manuscript defaults."""
    import matplotlib.pyplot as plt
    import plotting_helpers as ph
    ph.apply_classy_defaults()
    plt.rcParams.update({"font.size": 9.0, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
                         "legend.fontsize": 8.0})
    return plt, ph


def _panel_tag(ax, text: str) -> None:
    """Place a bare panel tag, since the figures carry no title text."""
    ax.text(0.015, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.0, color="#2F4858")


def figure_two_dial(ladder: list[dict], named: dict, path: Path) -> None:
    """Write the two-dial figure: band error against the polynomial core."""
    plt, ph = _style()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True)
    for panel, (ax, tag, key) in enumerate(
            zip(axes, ("(a)", "(b)"), ("e1_max_B0", "e2_max_B0"))):
        for budget_name in ("C_SLC1", "C_BMC11"):
            for entry in ladder:
                if (entry["band"] != "B0" or entry["status"] != "ok"
                        or entry["budget_name"] != budget_name):
                    continue
                clear = not entry["violations"]
                ax.semilogy([entry["m"]], [entry["metrics"][key]],
                            marker=PAIR_MARKER[entry["r"]],
                            ms=6.0 if clear else 6.5,
                            mfc=BUDGET_COLOR[budget_name] if clear else "none",
                            mec=BUDGET_COLOR[budget_name], mew=1.0, ls="none")
        for name, (color, marker, size, offset) in COMPARATOR_STYLE.items():
            row = named[name]
            ax.semilogy([row["order_d1"]], [row[key]], marker=marker, ms=size,
                        color=color, mfc="none" if marker in "ov^s" else color,
                        mew=1.2, ls="none")
            ax.annotate(name, (row["order_d1"], row[key]),
                        textcoords="offset points", xytext=offset,
                        fontsize=7.2, color=color, va="center")
        ax.set_xlabel("formal first-derivative order of the member")
        ax.set_xticks([4, 6, 8, 10, 12])
        ax.set_xlim(3.0, 14.6)
        ax.grid(axis="y", color="0.90", lw=0.5)
        ph.quiet_axis(ax)
        _panel_tag(ax, tag)
    axes[0].set_ylabel(r"$\max_{B_0}|e_1|$")
    axes[1].set_ylabel(r"$\max_{B_0}|e_2|$")
    handles = [Line2D([], [], color=BUDGET_COLOR["C_SLC1"], marker="o", ls="none",
                      ms=6.0, label="optimum at $C_{\\mathrm{SLC1}}$"),
               Line2D([], [], color=BUDGET_COLOR["C_BMC11"], marker="o", ls="none",
                      ms=6.0, label="optimum at $C_{\\mathrm{BMC11}}$"),
               Line2D([], [], color="0.3", marker="o", ls="none", ms=6.0,
                      mfc="none", label="threshold class"),
               Line2D([], [], color="0.3", marker="o", ls="none", ms=6.0,
                      label="one pair"),
               Line2D([], [], color="0.3", marker="s", ls="none", ms=6.0,
                      label="two pairs"),
               Line2D([], [], color="0.3", marker="D", ls="none", ms=6.0,
                      label="three pairs")]
    ph.make_overhead_legend(fig, handles, [h.get_label() for h in handles],
                            ncol=3, bbox=(0.5, 1.005), rect=(0, 0, 1, 0.86),
                            fontsize=7.8, handslcngth=1.3, columnspacing=1.0,
                            handletextpad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def figure_error_curves(champion: dict, path: Path) -> None:
    """Write the error curve figure of the B0 champion against the comparators."""
    plt, ph = _style()
    theta = met.theta_grid(POLISH_POINTS)
    generator = fam.ExponentialGenerator(champion["m"], champion["omegas"])
    label = (rf"$E_{{{champion['m']},{len(champion['omegas'])}}}$ champion")
    curves = [(label, "#0072B2", 1.7,
               lambda t: generator.symbols(t, FINE_TOL)[1],
               lambda t: generator.symbols(t, FINE_TOL)[2])]
    for name, d1, d2, color in (
            ("SLC 1992", comp.SLC_D1, comp.SLC_D2, "#D55E00"),
            ("tuned q8", comp.Q8_D1, comp.Q8_D2, "#CC79A7"),
            ("CW12B", comp.CW12B_D1, comp.CW12B_D2, "#8C510A")):
        curves.append((name, color, 1.0,
                       comp.compact_kappa1(d1["alpha"], d1["a"]),
                       comp.compact_kappa2(d2["alpha"], d2["a0"], d2["a"])))
    bmc11 = fam.ExponentialGenerator(12, ())
    curves.append(("BMC11", "#000000", 1.0,
                   lambda t: bmc11.symbols(t, FINE_TOL)[1],
                   lambda t: bmc11.symbols(t, FINE_TOL)[2]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, tag, order in zip(axes, ("(a)", "(b)"), (1, 2)):
        for name, color, lw, k1, k2 in curves:
            kappa = k1 if order == 1 else k2
            power = theta if order == 1 else theta ** 2
            err = np.abs(np.asarray(kappa(theta), dtype=float) / power - 1.0)
            ax.semilogy(theta / np.pi, np.maximum(err, 1e-13), color=color,
                        lw=lw, label=name)
        for tol in (1e-2, 1e-3, 1e-4):
            ax.axhline(tol, color=TOLERANCE_GREY, lw=0.6, ls=(0, (4, 2.5)))
        for edge, text in ((2.0 / 3.0, r"$2\pi/3$"), (0.75, r"$0.75\pi$"),
                           (0.8, r"$0.8\pi$")):
            ax.axvline(edge, color="0.70", lw=0.6, ls=(0, (1, 2)))
            ax.text(edge, 1.5e-12, text, rotation=90, fontsize=7.0,
                    color="0.45", ha="right", va="bottom")
        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-13, 2.0)
        ax.grid(axis="y", color="0.92", lw=0.5)
        ph.quiet_axis(ax)
        _panel_tag(ax, tag)
    axes[0].set_ylabel(r"$|e_1(\theta)|$")
    axes[1].set_ylabel(r"$|e_2(\theta)|$")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = ph.make_overhead_legend(fig, handles, labels, ncol=5,
                                     bbox=(0.5, 1.005), rect=(0, 0, 1, 0.90),
                                     fontsize=7.8, handslcngth=1.9,
                                     columnspacing=1.1, handletextpad=0.4)
    for handle in legend.legend_handles:
        handle.set_linewidth(2.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def figure_conditioning_map(omega1: float, budgets: dict, path: Path) -> dict:
    """Write the conditioning map against the second frequency and return it."""
    plt, ph = _style()
    omega2 = np.linspace(omega1 + 0.02, np.pi - 0.02, 241)
    data = {"omega_1": float(omega1), "omega_2": [float(w) for w in omega2],
            "cores": {}}
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                  gridspec_kw={"width_ratios": [1.15, 0.95]})
    for m in (4, 6, 8):
        actual = np.empty(omega2.size)
        formula = np.empty(omega2.size)
        for j, w2 in enumerate(omega2):
            generator = fam.ExponentialGenerator(m, (omega1, float(w2)))
            actual[j] = generator.cond(POLISH_POINTS)["cond"]
            formula[j] = theory_A5(m, (omega1, float(w2)))["cond_two_term"]
        data["cores"][str(m)] = {"cond": [float(v) for v in actual],
                                 "cond_two_term": [float(v) for v in formula],
                                 "max_relative_error": float(
                                     np.max(np.abs(formula - actual) / actual))}
        ax.semilogy(omega2 / np.pi, actual, color=CORE_COLOR[m], lw=2.2,
                    label=f"$m={m}$", alpha=0.45)
        ax.semilogy(omega2 / np.pi, formula, color=CORE_COLOR[m], lw=1.0,
                    ls=(0, (3.5, 2.5)))
        axr.semilogy(omega2 / np.pi, np.abs(formula - actual) / actual,
                     color=CORE_COLOR[m], lw=1.5)
    for key, color, text, vertical in (("C_SLC1", "#D55E00", "SLC D1", "top"),
                                       ("C_BMC11", "#000000", "BMC11", "bottom")):
        ax.axhline(budgets[key], color=color, lw=0.8, ls=(0, (1, 2)))
        ax.text(0.98, budgets[key], f"{text} {budgets[key]:.1f}", fontsize=7.2,
                color=color, ha="right", va=vertical,
                transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"$\omega_2/\pi$")
    ax.set_ylabel("left-side condition number")
    ax.set_xlim(float(omega2[0] / np.pi), 1.0)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax)
    _panel_tag(ax, "(a)")
    axr.set_xlabel(r"$\omega_2/\pi$")
    axr.set_ylabel("relative error of the two-term formula")
    axr.set_xlim(float(omega2[0] / np.pi), 1.0)
    axr.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(axr)
    _panel_tag(axr, "(b)")
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color="0.3", lw=1.0, ls=(0, (3.5, 2.5)),
                          label="Theorem A5 two-term"))
    labels.append("Theorem A5 two-term")
    ph.make_overhead_legend(fig, handles, labels, ncol=4, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.90), fontsize=7.8, handslcngth=1.8,
                            columnspacing=1.1, handletextpad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return data


def band_dependence(budgets: dict, processes: int) -> dict:
    """Return the best J at C_SLC1 against the band edge for m in {6, 8}."""
    theta = met.theta_grid(POLISH_POINTS)
    slc1 = comp.compact_kappa1(comp.SLC_D1["alpha"], comp.SLC_D1["a"])
    slc2 = comp.compact_kappa2(comp.SLC_D2["alpha"], comp.SLC_D2["a0"],
                                comp.SLC_D2["a"])
    e1 = np.asarray(slc1(theta)) / theta - 1.0
    e2 = np.asarray(slc2(theta)) / theta ** 2 - 1.0
    budget = float(budgets["C_SLC1"])
    edges = np.linspace(0.5 * np.pi, 0.85 * np.pi, 15)
    tasks = []
    for ei, edge in enumerate(edges):
        ref1 = met.band_max(theta, e1, float(edge))
        ref2 = met.band_max(theta, e2, float(edge))
        for m in (6, 8):
            for r in (1, 2, 3):
                if m + 2 * r - 1 > 11:
                    continue
                if polynomial_limit_cond(m, r) > budget:
                    continue
                tasks.append({"m": m, "r": r, "band": f"T{ei}",
                              "edge": float(edge), "ref1": ref1, "ref2": ref2,
                              "budget_name": "C_SLC1", "budget": budget,
                              "starts": FIGURE_STARTS,
                              "seed": BASE_SEED + 900000 + 1000 * ei + 10 * m + r})
    results = []
    with Pool(processes=processes) as pool:
        for done, record in enumerate(pool.imap_unordered(optimize_triple, tasks), 1):
            results.append(record)
            print(f"  [band edge {done:2d}/{len(tasks)}] T/pi="
                  f"{record['edge'] / np.pi:.4f} m={record['m']} r={record['r']} "
                  f"J={record.get('J_polish')} {record['seconds']:.1f}s", flush=True)
    out = {"budget": budget, "budget_name": "C_SLC1",
           "edges": [float(v) for v in edges], "cores": {}}
    for m in (6, 8):
        best = []
        members = []
        for ei, edge in enumerate(edges):
            pick = [rec for rec in results
                    if rec["m"] == m and rec["band"] == f"T{ei}"
                    and rec["status"] == "ok"]
            if not pick:
                best.append(None)
                members.append(None)
                continue
            winner = min(pick, key=lambda rec: rec["J_polish"])
            best.append(float(winner["J_polish"]))
            members.append({"r": winner["r"], "omegas": winner["omegas"],
                            "cond": winner["cond_polish"]})
        out["cores"][str(m)] = {"J": best, "members": members}
    return out


def figure_band_dependence(data: dict, path: Path) -> None:
    """Write the band edge sweep of the objective at C_SLC1."""
    plt, ph = _style()
    edges = np.asarray(data["edges"], dtype=float) / np.pi
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in (6, 8):
        values = data["cores"][str(m)]["J"]
        good = np.array([v is not None for v in values])
        y = np.array([v if v is not None else np.nan for v in values], dtype=float)
        ax.semilogy(edges[good], y[good], color=CORE_COLOR[m], lw=1.7,
                    marker=PAIR_MARKER[1], ms=4.5, label=f"$m={m}$")
    ax.axhline(1.0, color="#D55E00", lw=0.9, ls=(0, (4, 2)))
    ax.text(edges[0], 1.15, "parity with SLC", fontsize=7.4, color="#D55E00",
            ha="left", va="bottom")
    for edge, text in ((2.0 / 3.0, r"$B_0$"), (0.75, r"$B_1$"), (0.8, r"$B_2$")):
        ax.axvline(edge, color="0.70", lw=0.6, ls=(0, (1, 2)))
        ax.text(edge, ax.get_ylim()[0], text, fontsize=7.2, color="0.45",
                ha="right", va="bottom")
    ax.set_xlabel(r"band edge $T/\pi$")
    ax.set_ylabel(r"$J = \max(E^{(1)}_{\max}/E_{\mathrm{ref}1},"
                  r" E^{(2)}_{\max}/E_{\mathrm{ref}2})$")
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    ph.make_overhead_legend(fig, handles, labels, ncol=2, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.90), fontsize=8.0, handslcngth=1.8,
                            columnspacing=1.2, handletextpad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Receipt comparison, the check path, and the entry point
# ---------------------------------------------------------------------------

def compare_receipt(stored, fresh, rtol: float = 1e-9, atol: float = 1e-12,
                    path: str = "") -> list[str]:
    """Return the list of receipt fields that moved beyond the tolerance."""
    bad: list[str] = []
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for key in sorted(set(stored) | set(fresh)):
            if key not in stored or key not in fresh:
                bad.append(f"{path}/{key} present in only one receipt")
                continue
            bad += compare_receipt(stored[key], fresh[key], rtol, atol, f"{path}/{key}")
        return bad
    if isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            return [f"{path} length {len(stored)} versus {len(fresh)}"]
        for j, (a, b) in enumerate(zip(stored, fresh)):
            bad += compare_receipt(a, b, rtol, atol, f"{path}[{j}]")
        return bad
    if isinstance(stored, bool) or isinstance(fresh, bool):
        return [] if stored == fresh else [f"{path}: {stored} versus {fresh}"]
    if isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        if math.isnan(float(stored)) and math.isnan(float(fresh)):
            return []
        if abs(float(stored) - float(fresh)) <= atol + rtol * abs(float(stored)):
            return []
        return [f"{path}: {stored!r} versus {fresh!r}"]
    return [] if stored == fresh else [f"{path}: {stored!r} versus {fresh!r}"]


def check_mode() -> int:
    """Recompute every champion from champions.json and compare the receipts."""
    started = time.time()
    stored = json.loads((OUT / "champions.json").read_text(encoding="utf-8"))
    budgets, references, named = budgets_and_references()
    bad = compare_receipt(stored["budgets"], budgets, path="/budgets")
    bad += compare_receipt(stored["reference_band_maxima"], references,
                           path="/reference_band_maxima")
    bad += compare_receipt(stored["comparators"], named, path="/comparators")
    members = 0
    routes = 0
    seen: dict[tuple, dict] = {}
    for band in BANDS:
        for budget_name in BUDGET_NAMES:
            cell = stored["champions"][band][budget_name]
            for role in ("champion", "threshold_runner_up"):
                member = cell.get(role)
                if member is None:
                    continue
                members += 1
                fresh = member_record(member["m"], member["omegas"],
                                      budget_name, float(budgets[budget_name]),
                                      references)
                bad += compare_receipt(member["metrics"], fresh,
                                       path=f"/{band}/{budget_name}/{role}")
                key = (member["m"], tuple(member["omegas"]))
                if key not in seen:
                    seen[key] = sample_routes(
                        fam.ExponentialGenerator(member["m"], member["omegas"]))
                    routes += 1
                bad += compare_receipt(member["sample_routes"], seen[key],
                                       path=f"/{band}/{budget_name}/{role}/routes")
                if not seen[key]["all_ok"]:
                    bad.append(f"/{band}/{budget_name}/{role} sample routes failed")
    outcome_ok = stored["decision"]["outcome"] in ("TRIUMPH", "PARTIAL", "REJECT")
    for item in bad[:20]:
        print(f"  receipt drift {item}")
    ok = not bad and outcome_ok
    print(f"G32 exponential family Gate 1 champions: {members} member receipts, "
          f"{routes} sample route receipts, {len(bad)} drifted fields, decision "
          f"{stored['decision']['outcome']}, {time.time() - started:.1f}s")
    return 0 if ok else 1


def draw_figures(ladder: list[dict], champions: dict, references: dict,
                 budgets: dict, named: dict, processes: int) -> float:
    """Write the four Gate 1 figures, their data sidecars and captions.md."""
    started = time.time()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure_two_dial(ladder, named, FIGDIR / "fig_two_dial.pdf")
    champion = champions["B0"]["C_SLC1"]["champion"]
    if champion is None:
        print("  no B0 champion at C_SLC1; the error curve figure uses the "
              "best threshold class member", flush=True)
        champion = champions["B0"]["C_SLC1"]["threshold_runner_up"]
    figure_error_curves(champion, FIGDIR / "fig_error_curves.pdf")
    cmap = figure_conditioning_map(float(champion["omegas"][0]), budgets,
                                   FIGDIR / "fig_conditioning_map.pdf")
    (FIGDIR / "conditioning_map.json").write_text(
        json.dumps(cmap, indent=2, default=float) + "\n", encoding="utf-8")
    sidecar = FIGDIR / "band_dependence.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        data = band_dependence(budgets, processes)
        sidecar.write_text(json.dumps(data, indent=2, default=float) + "\n",
                           encoding="utf-8")
    figure_band_dependence(data, FIGDIR / "fig_band_dependence.pdf")
    (OUT / "captions.md").write_text(
        captions_markdown(champions, references, budgets), encoding="utf-8")
    return time.time() - started


def figures_only(processes: int) -> int:
    """Redraw the figures from the stored receipts without re-optimizing."""
    budgets, references, named = budgets_and_references()
    ladder = json.loads((OUT / "ladder.json").read_text(encoding="utf-8"))["ladder"]
    champions = json.loads(
        (OUT / "champions.json").read_text(encoding="utf-8"))["champions"]
    seconds = draw_figures(ladder, champions, references, budgets, named, processes)
    print(f"Redrew the Gate 1 figures in {seconds:.1f}s", flush=True)
    return 0


def main(argv=None) -> int:
    """Build the Gate 1 receipts, or verify them in the fast check mode."""
    parser = argparse.ArgumentParser(description="Gate 1 champion producer.")
    parser.add_argument("--check", action="store_true",
                        help="Recompute the champions' metrics from "
                             "champions.json and compare with the receipt.")
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--starts", type=int, default=MIN_STARTS)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild every receipt from the stored optimizer "
                             "records in ladder.json without repeating the "
                             "optimization.")
    parser.add_argument("--figures-only", action="store_true",
                        help="Redraw the four figures and captions.md from the "
                             "stored ladder.json, champions.json and "
                             "figures/band_dependence.json without repeating "
                             "the optimization.")
    args = parser.parse_args(argv)
    if args.check:
        return check_mode()
    if args.figures_only:
        return figures_only(args.processes)

    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Gate 1 budgets and band references", flush=True)
    budgets, references, named = budgets_and_references()
    for key in BUDGET_NAMES:
        print(f"  {key} = {budgets[key]:.6f}", flush=True)
    for band in BANDS:
        ref = references[band]
        print(f"  {band}: E_ref1 = {ref['E_ref1']:.6e}, "
              f"E_ref2 = {ref['E_ref2']:.6e}", flush=True)

    stored0 = json.loads((GATE0 / "table.json").read_text(encoding="utf-8"))
    frozen = []
    for key in ("C_SLC2", "C_q8", "C_SLC1", "C_BMC11"):
        gap = abs(budgets[key] - stored0["budgets"][key]) / stored0["budgets"][key]
        frozen.append({"budget": key, "gate0": stored0["budgets"][key],
                       "gate1": budgets[key], "relative_gap": gap,
                       "ok": bool(gap <= 1e-12)})
    for key, gate0_key in (("B0", "E_SLC1"),):
        gap = abs(references[key]["E_ref1"]
                  - stored0["reference_band_maxima"][gate0_key]) \
            / stored0["reference_band_maxima"][gate0_key]
        frozen.append({"budget": "E_ref1(B0)", "gate0":
                       stored0["reference_band_maxima"]["E_SLC1"],
                       "gate1": references["B0"]["E_ref1"], "relative_gap": gap,
                       "ok": bool(gap <= 1e-12)})
    gap = abs(references["B0"]["E_ref2"]
              - stored0["reference_band_maxima"]["E_SLC2"]) \
        / stored0["reference_band_maxima"]["E_SLC2"]
    frozen.append({"budget": "E_ref2(B0)",
                   "gate0": stored0["reference_band_maxima"]["E_SLC2"],
                   "gate1": references["B0"]["E_ref2"], "relative_gap": gap,
                   "ok": bool(gap <= 1e-12)})
    for item in frozen:
        print(f"  [{'ok ' if item['ok'] else 'FAIL'}] frozen {item['budget']}: "
              f"relative gap {item['relative_gap']:.3g}", flush=True)

    print(f"Ladder over {len(CELLS)} cells, {len(BUDGET_NAMES)} budgets and "
          f"{len(BANDS)} bands", flush=True)
    ladder_started = time.time()
    if args.rebuild:
        print("  rebuilding from the stored optimizer records", flush=True)
        records = json.loads(
            (OUT / "ladder.json").read_text(encoding="utf-8"))["optimizer_records"]
        rows, index = ladder_rows(records, references, args.processes)
        previous = OUT / "champions.json"
        ladder_seconds = None
        if previous.exists():
            ladder_seconds = json.loads(
                previous.read_text(encoding="utf-8"))["decision"].get(
                    "ladder_wall_seconds")
    else:
        records, rows, index = run_ladder(budgets, references, args.processes,
                                          args.starts)
        ladder_seconds = time.time() - ladder_started
    ladder_cpu_seconds = float(sum(rec.get("seconds", 0.0) for rec in records))
    print(f"  ladder wall time {ladder_seconds}, worker cpu time "
          f"{ladder_cpu_seconds:.1f}s over {args.processes} processes", flush=True)

    ladder = assemble_ladder(records, index)
    champions = select_champions(ladder, references)
    route_failures = attach_routes(champions, args.processes)
    reproduction = gate0_reproduction(ladder)
    decision = decide(champions, references, named)
    decision["rebuilt_from_stored_records"] = bool(args.rebuild)
    decision["ladder_wall_seconds"] = ladder_seconds
    decision["ladder_cpu_seconds"] = ladder_cpu_seconds
    decision["processes"] = args.processes
    decision["route_failures"] = route_failures
    decision["gate0_reproduction"] = {
        "coincident": reproduction["coincident"],
        "improved": reproduction["improved"],
        "ties": reproduction["ties"],
        "regressions": [f"m={row['m']} r={row['r']} {row['budget_name']}"
                        for row in reproduction["regressions"]],
    }
    print(f"Decision at (B0, C_SLC1): {decision['outcome']}", flush=True)

    conventions = {
        "theta_points": POLISH_POINTS,
        "confirmation_points": CONFIRM_POINTS,
        "optimizer_theta_points": COARSE_POINTS,
        "alias_tail_tolerance": FINE_TOL,
        "optimizer_alias_tail_tolerance": FAST_TOL,
        "polish_alias_tail_tolerance": POLISH_TOL,
        "coupled_xi_points": int(XI_GRID.size),
        "bands": {k: float(v) for k, v in met.BANDS.items()},
        "budget_order": list(BUDGET_NAMES),
        "cells": [list(cell) for cell in CELLS],
        "starts": args.starts,
        "grid_per_axis": GRID_PER_AXIS,
        "penalty": PENALTY,
        "base_seed": BASE_SEED,
        "coupled_factor": COUPLED_FACTOR,
        "budget_slack": BUDGET_SLACK,
        "confirmation_relative_tolerance": CONFIRM_RTOL,
        "decimal_working_digits": DECIMAL_PREC,
        "decimal_reported_digits": DECIMAL_DIGITS,
    }
    header = {
        "producer": "code/analysis/exponential_family/gate1_champions.py",
        "contract": {"path": str(CONTRACT.relative_to(ROOT)),
                     "sha256": sha256(CONTRACT)},
        "conventions": conventions,
        "budgets": budgets,
        "reference_band_maxima": references,
        "comparators": named,
    }
    (OUT / "champions.json").write_text(json.dumps(
        dict(header, frozen_checks=frozen, champions=champions,
             decision=decision), indent=2, default=float) + "\n",
        encoding="utf-8")
    (OUT / "ladder.json").write_text(json.dumps(
        dict(header, ladder=ladder, optimizer_records=records,
             gate0_reproduction=reproduction), indent=2, default=float) + "\n",
        encoding="utf-8")
    (OUT / "ladder.md").write_text(
        ladder_markdown(ladder, budgets, references, decision, champions),
        encoding="utf-8")

    theory = []
    for entry in ladder:
        if entry["status"] != "ok":
            continue
        block = dict(entry["metrics"]["theory_A5"])
        block.update({"band": entry["band"], "budget_name": entry["budget_name"],
                      "m": entry["m"], "r": entry["r"], "omegas": entry["omegas"]})
        theory.append(block)
    errors = [block["relative_error"] for block in theory]
    (OUT / "theory_A5.json").write_text(json.dumps({
        "producer": header["producer"], "contract": header["contract"],
        "formula": "cond ~ 1 / (2 (2/pi)^m prod_j omega_j^2 / "
                   "((pi^2 - omega_j^2) tan^2(omega_j/2))), the k = 0 and "
                   "k = -1 terms of Theorem A5",
        "entries": theory,
        "summary": {"count": len(theory),
                    "min_relative_error": float(np.min(errors)) if errors else None,
                    "median_relative_error": (float(np.median(errors))
                                              if errors else None),
                    "max_relative_error": float(np.max(errors)) if errors else None,
                    "mean_relative_error": float(np.mean(errors)) if errors else None},
    }, indent=2, default=float) + "\n", encoding="utf-8")

    figure_seconds = 0.0
    if not args.skip_figures:
        print("Figures", flush=True)
        figure_seconds = draw_figures(ladder, champions, references, budgets,
                                      named, args.processes)
        print(f"  figure wall time {figure_seconds:.1f}s", flush=True)

    failures = [item for item in frozen if not item["ok"]]
    bad_confirmation = [entry for entry in ladder
                        if entry["status"] == "ok"
                        and not entry["metrics"]["confirmation_ok"]]
    for item in failures:
        print(f"  FAILED frozen check {item['budget']}", flush=True)
    for entry in bad_confirmation:
        print(f"  FAILED 200001 point confirmation for m={entry['m']} "
              f"r={entry['r']} {entry['budget_name']} {entry['band']}", flush=True)
    for item in route_failures:
        print(f"  FAILED sample route {item}", flush=True)
    for row in reproduction["regressions"]:
        print(f"  FAILED Gate 0 reproduction: m={row['m']} r={row['r']} "
              f"{row['budget_name']} J rose from {row['gate0_J']} to "
              f"{row['gate1_J']}", flush=True)
    print(f"Total wall time {time.time() - started:.1f}s", flush=True)
    return 1 if (failures or bad_confirmation or route_failures
                 or reproduction["regressions"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
