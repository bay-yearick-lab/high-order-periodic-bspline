"""Frozen Gate 0 comparator symbols.

Every coefficient is the literal value of the Gate 0 contract and of
code/solver/bm_apply.h. The symbol conventions are

    D1 symbol = (i/h) kappa_1(theta),   D2 symbol = (-1/h^2) kappa_2(theta),

so a compact first derivative with antisymmetric right side coefficients a_s
and left side 1 + 2 sum_l alpha_l cos(l theta) has

    kappa_1 = sum_s 2 a_s sin(s theta) / (1 + 2 sum_l alpha_l cos(l theta)),

and a compact second derivative with symmetric right side coefficients
(a_0, a_s) has

    kappa_2 = -(a_0 + sum_s 2 a_s cos(s theta)) / (1 + 2 sum_l alpha_l cos(l theta)).

Bandwidth is the count of nonzero diagonals of the circulant operator. The
solve count is the number of cyclic band solves needed to produce both D1 u
and D2 u: a spline member needs one, because D1 = A_1 A_0^{-1} and
D2 = A_2 A_0^{-1} share the coefficient vector A_0^{-1} u, a compact pair with
two different left sides needs two, and an explicit scheme needs none.

The algebraic order carried in the table is the published formal order of the
scheme, as the contract requires. A Taylor residual scan of the scheme's own
order conditions is reported beside it as a detected order, using the absolute
threshold ORDER_RESIDUAL_FLOOR. That scan is informative only when the
published coefficients carry enough digits: coefficients quoted to seven or
eight digits leave residuals near 1e-7 in the conditions they are meant to
satisfy, so a scheme whose genuine leading residual is smaller than that
reports no detected order.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import spline_analysis as sa  # noqa: E402
from family import ExponentialGenerator  # noqa: E402

ORDER_RESIDUAL_FLOOR = 1e-6

SLC_D1 = {"alpha": (0.5771439, 0.0896406),
           "a": (1.3025166 / 2.0, 0.9935500 / 4.0, 0.03750245 / 6.0)}
SLC_D2 = {"alpha": (0.50209266, 0.05569169),
           "a": (0.21564935, 1.7233220 / 4.0, 0.17659730 / 9.0),
           "a0": -2.0 * (0.21564935 + 1.7233220 / 4.0 + 0.17659730 / 9.0)}
Q8_D1 = {"alpha": (0.4742169929451113,),
         "a": (0.7598825426879071, 0.13489507072322338, -0.023260830361993556,
               0.004069991868971751, -0.00039063351582985587)}
Q8_D2 = {"alpha": (0.4320698009820234,), "a0": -1.555224078876143,
         "a": (0.35890831739343104, 0.45913001629633177, -0.04560561305199756,
               0.00559124812826596, -0.0004119293279594219)}
CW12A_D1 = {"alpha": (0.40937878082640167, -0.06462037441519262),
            "a": (0.8359351614953894, 0.06465601447530303, -0.05254555071859491,
                  0.011976865033900547, -0.0025753384161538,
                  0.00038591694604337513, -2.8343087156344804e-05)}
CW12A_D2 = {"alpha": (0.1601749806496769, -0.21726126868097),
            "a0": -2.4227916839619907,
            "a": (0.9043854478128212, 0.5970618140260426, -0.3252810784893023,
                  0.040779787773292164, -0.006239267401751708,
                  0.0007337128953475037, -4.4574635453931686e-05)}
CW12B_D1 = {"alpha": (0.5796083747693036, 0.09310954748722111),
            "a": (0.6474244112956682, 0.25095267402010113, 0.00827175994799237,
                  -0.0003731748662601286, 1.3116508343528986e-05)}
CW12B_D2 = {"alpha": (0.5273008328764947, 0.06895526765898032),
            "a0": -1.2490664485294838,
            "a": (0.15392188750639246, 0.43804625186979695, 0.03355551188077371,
                  -0.001018485995992867, 2.8059003771746482e-05)}
DRP_D1 = {"alpha": (), "a": (0.79926643, -0.18941314, 0.02651995)}


def compact_left(alpha):
    """Return the left side symbol 1 + 2 sum_l alpha_l cos(l theta)."""
    alpha = tuple(float(v) for v in alpha)

    def left(theta):
        theta = np.asarray(theta, dtype=float)
        value = np.ones(theta.shape)
        for l, al in enumerate(alpha, start=1):
            value = value + 2.0 * al * np.cos(l * theta)
        return value

    return left


def compact_kappa1(alpha, a):
    """Return kappa_1 of a compact antisymmetric first derivative."""
    left = compact_left(alpha)
    a = tuple(float(v) for v in a)

    def kappa1(theta):
        theta = np.asarray(theta, dtype=float)
        num = np.zeros(theta.shape)
        for s, coeff in enumerate(a, start=1):
            num = num + 2.0 * coeff * np.sin(s * theta)
        return num / left(theta)

    return kappa1


def compact_kappa2(alpha, a0, a):
    """Return kappa_2 of a compact symmetric second derivative."""
    left = compact_left(alpha)
    a = tuple(float(v) for v in a)
    a0 = float(a0)

    def kappa2(theta):
        theta = np.asarray(theta, dtype=float)
        num = np.full(theta.shape, a0)
        for s, coeff in enumerate(a, start=1):
            num = num + 2.0 * coeff * np.cos(s * theta)
        return -num / left(theta)

    return kappa2


def detect_order_d1(alpha, a, terms: int = 14) -> int:
    """Return the formal order of a compact first derivative from its residual."""
    for j in range(terms):
        c = 2.0 * sum(a[s - 1] * (-1.0) ** j * s ** (2 * j + 1) / factorial(2 * j + 1)
                      for s in range(1, len(a) + 1))
        c -= 2.0 * sum(alpha[l - 1] * (-1.0) ** j * l ** (2 * j) / factorial(2 * j)
                       for l in range(1, len(alpha) + 1))
        if j == 0:
            c -= 1.0
        if abs(c) > ORDER_RESIDUAL_FLOOR:
            return 2 * j
    return None


def detect_order_d2(alpha, a0, a, terms: int = 14) -> int:
    """Return the formal order of a compact second derivative from its residual."""
    for j in range(terms):
        c = 2.0 * sum(a[s - 1] * (-1.0) ** j * s ** (2 * j) / factorial(2 * j)
                      for s in range(1, len(a) + 1))
        if j == 0:
            c += a0
        if j >= 1:
            d = 2.0 * sum(alpha[l - 1] * (-1.0) ** (j - 1) * l ** (2 * j - 2)
                          / factorial(2 * j - 2) for l in range(1, len(alpha) + 1))
            if j == 1:
                d += 1.0
            c += d
        if abs(c) > ORDER_RESIDUAL_FLOOR:
            return max(0, 2 * j - 2)
    return None


@dataclass
class Scheme:
    """One scheme of the Gate 0 table with everything the metrics need."""

    name: str
    kappa1: object
    kappa2: object
    left: object
    bandwidth_left: object
    bandwidth_d1: object
    bandwidth_d2: object
    solves: object
    algebraic_order1: object
    algebraic_order2: object
    published_order1: object = None
    published_order2: object = None
    analytic_gv: object = None
    coupled: object = None
    extra: dict = field(default_factory=dict)


def spline_scheme(name: str, generator: ExponentialGenerator, *, coupled_xi=None,
                  tol: float = 1e-14) -> Scheme:
    """Return the Scheme of a spline or exponential spline collocation member."""
    samples = generator.integer_samples(tol)
    nz = np.abs(samples.beta) > 1e-14
    nz1 = np.abs(samples.dbeta) > 1e-14
    nz2 = np.abs(samples.ddbeta) > 1e-14
    coupled = None
    if coupled_xi is not None:
        coupled = lambda theta: generator.coupled_error(theta, coupled_xi, tol)
    return Scheme(
        name=name,
        kappa1=lambda theta: generator.symbols(theta, tol)[1],
        kappa2=lambda theta: generator.symbols(theta, tol)[2],
        left=lambda theta: generator.symbols(theta, tol)[0],
        bandwidth_left=int(np.count_nonzero(nz)),
        bandwidth_d1=int(np.count_nonzero(nz1)),
        bandwidth_d2=int(np.count_nonzero(nz2)),
        solves=1,
        algebraic_order1=generator.m,
        algebraic_order2=generator.m - 2,
        published_order1=generator.m,
        published_order2=generator.m - 2,
        analytic_gv=lambda theta: generator.symbol_derivative(theta, tol),
        coupled=coupled,
        extra={"m": generator.m, "r": generator.r, "p": generator.p,
               "omegas": list(generator.omegas),
               "truncation": samples.truncation},
    )


def _compact_pair(name, d1, d2, published1, published2, solves=2) -> Scheme:
    left1 = compact_left(d1["alpha"])
    k1 = compact_kappa1(d1["alpha"], d1["a"])
    if d2 is None:
        return Scheme(name=name, kappa1=k1, kappa2=None, left=left1,
                      bandwidth_left=2 * len(d1["alpha"]) + 1,
                      bandwidth_d1=2 * len(d1["a"]),
                      bandwidth_d2="none", solves=0 if not d1["alpha"] else 1,
                      algebraic_order1=published1,
                      algebraic_order2="none",
                      published_order1=published1, published_order2="none",
                      extra={"d2": "none",
                             "detected_order_d1": detect_order_d1(d1["alpha"], d1["a"])})
    k2 = compact_kappa2(d2["alpha"], d2["a0"], d2["a"])
    return Scheme(name=name, kappa1=k1, kappa2=k2, left=left1,
                  bandwidth_left=2 * len(d1["alpha"]) + 1,
                  bandwidth_d1=2 * len(d1["a"]),
                  bandwidth_d2=2 * len(d2["a"]) + 1,
                  solves=solves,
                  algebraic_order1=published1,
                  algebraic_order2=published2,
                  published_order1=published1, published_order2=published2,
                  extra={"left_d2_alpha": list(d2["alpha"]),
                         "bandwidth_left_d2": 2 * len(d2["alpha"]) + 1,
                         "detected_order_d1": detect_order_d1(d1["alpha"], d1["a"]),
                         "detected_order_d2": detect_order_d2(d2["alpha"], d2["a0"],
                                                              d2["a"]),
                         "cond_left_d2": None})


def fourier_scheme() -> Scheme:
    """Return the global Fourier boundary scheme kappa_1 = theta, kappa_2 = theta^2."""
    return Scheme(
        name="Fourier",
        kappa1=lambda theta: np.asarray(theta, dtype=float),
        kappa2=lambda theta: np.asarray(theta, dtype=float) ** 2,
        left=lambda theta: np.ones(np.asarray(theta, dtype=float).shape),
        bandwidth_left="global", bandwidth_d1="global", bandwidth_d2="global",
        solves=0, algebraic_order1="exact", algebraic_order2="exact",
        published_order1="exact", published_order2="exact",
        analytic_gv=lambda theta: np.ones(np.asarray(theta, dtype=float).shape),
    )


def comparator_schemes(coupled_xi=None) -> list[Scheme]:
    """Return every comparator the Gate 0 contract requires in each table."""
    schemes = [
        _compact_pair("SLC1992", SLC_D1, SLC_D2, 4, 4),
        _compact_pair("TunedQ8", Q8_D1, Q8_D2, 8, 8),
        _compact_pair("CW12A", CW12A_D1, CW12A_D2, 12, 12),
        _compact_pair("CW12B", CW12B_D1, CW12B_D2, 12, 12),
        _compact_pair("DRP-TamWebb", DRP_D1, None, 4, "none"),
    ]
    for p in (5, 7, 9, 11):
        generator = ExponentialGenerator(p + 1, ())
        schemes.append(spline_scheme(f"BMC{p}", generator, coupled_xi=coupled_xi))
    schemes.append(fourier_scheme())
    return schemes


def bmc_symbol_check(p: int, theta) -> tuple[float, float]:
    """Return the agreement of the E_{p+1,0} symbols with the collocation symbols."""
    generator = ExponentialGenerator(p + 1, ())
    _, k1, k2 = generator.symbols(theta)
    ref1 = sa.kprime_collocation(theta, p)
    ref2 = sa.ksecond_collocation(theta, p)
    return (float(np.max(np.abs(k1 - ref1))),
            float(np.max(np.abs(k2 - ref2)) / max(1.0, float(np.max(np.abs(ref2))))))
