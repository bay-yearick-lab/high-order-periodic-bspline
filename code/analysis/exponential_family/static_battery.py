"""Static yardstick battery for one generator specification, or many at once.

A specification names a member of the exponential cardinal B-spline family in
the widest form the frozen code supports: an even or odd polynomial core
multiplicity m, imaginary pair frequencies omega_j in (0, pi), real pairs
alpha_j > 0, quads (alpha_j, omega_j), and an even polynomial factor

    psi_hat(xi) = P(xi^2) phi_hat(xi),    P(t) = 1 + p_1 t + ... + p_q t^q,

with P(0) = 1 so that psi_hat(0) = 1 is preserved. The generator transform of
the pair and quad part is the one of gate7_family,

    phi_hat(xi) = sinc(xi/2)^m prod_slots g(s_j, xi),
    g(s, xi)    = s (cosh sqrt(s) - cos xi) / ((cosh sqrt(s) - 1) (xi^2 + s)),
    s           = lambda^2,  lambda = alpha + i omega,

which for alpha = 0 is the Gate 0 pair factor
sinc((xi - omega)/2) sinc((xi + omega)/2) / sinc(omega/2)^2.

Three frozen back ends carry those three cases and this module only dispatches
between them, so that every number it prints is produced by the same code that
produced the frozen receipts:

    imaginary pairs only, P = 1   family.ExponentialGenerator      (Gate 0/1),
    any slot kinds, P = 1         gate7_family.ExtendedGenerator   (Gate 7),
    any slot kinds, P non trivial gate8_family.PolynomialGenerator (Gate 8).

Every yardstick is the Gate 0 metric of metrics.py on its 20,001-point grid
theta_j = j pi / (P + 1), and every reach is the first contiguous crossing of
the tolerance with linear interpolation between the bracketing grid points,
refined by Brent's method on the same bracket. The battery reports, per
specification:

* the band maxima and root mean squares of e_1 = kappa_1/theta - 1 and
  e_2 = kappa_2/theta^2 - 1 on B0 = (0, 2 pi/3], B1 = (0, 3 pi/4],
  B2 = (0, 4 pi/5];
* the reaches of |e_1| and |e_2| at 1e-2, 1e-3, 1e-4 and the group velocity
  reach of |d kappa_1 / d theta - 1| at 1e-2, in multiples of pi;
* the conditioning max b / min b of the symmetric positive definite left side
  b, continuously on (0, pi) and on the mode grids of N = 64, 127, 128, with
  max b, min b and the location of the minimum;
* the positivity certificates of b and of kappa_2 b: both are trigonometric
  polynomials of degree nmax whose derivatives are bounded by
  2 sum_n n |coefficient|, so a grid minimum exceeding half the grid spacing
  times that bound is a proof of positivity between the grid points
  (gate7_family.certify_positivity), with kappa_2 b handled near theta = 0 by
  its series lower bound theta^2 (1 - (theta^2/12) sum n^4 |beta''(n)|);
* the sufficient conditions S_m(omega) < 1 and S_{m-2}(omega) < 1 of the k = 0
  dominance argument where they apply, with S_m(omega) = sum_{k >= 1}
  [(omega/(2 pi k - omega))^m + (omega/(2 pi k + omega))^m] evaluated both by
  the exact shifted zeta form x^m [zeta(m, 1 - x) + zeta(m, 1 + x)],
  x = omega/(2 pi), and by the truncated series the later gates use;
* the formal orders, fitted by the log-log slope of the error on the metrics
  window and algebraic (m for D1, m - 2 for D2);
* the bandwidth of the left side and of the two right sides, counted as the
  nonzero lattice samples, and the solve count, which is one because
  D1 = A_1 A_0^{-1} and D2 = A_2 A_0^{-1} share the coefficient vector;
* the coupled off-grid channel e_c(theta) = max_xi max_q |T_q(theta, xi) - 1|
  on a 129-point xi grid of [-1/2, 1/2], its band maxima and its reaches;
* on request, the 60-digit decimal recomputation of the integer samples, the
  Gate 1 decimal route, and its deviation from the double precision route;
* a comparison against SLC's fourth-order pentadiagonal pair, the tuned
  eighth-order pair, CW12A, CW12B, BMC9 and BMC11, with per-yardstick ratios,
  the signed margins (value - target)/target for the yardsticks where more is
  better and 1 - value/target for the band maxima, and the dominance verdict,
  which is the sign of the smallest margin over the Gate 7 yardstick set
  R1(1e-2), R1(1e-3), R1(1e-4), G(1e-2), E1max(B0), E2max(B0). That verdict
  says nothing about conditioning, which the gates carry as a separate
  constraint, so the conditioning ratio against the comparator's own left side
  and the verdict with that budget imposed are reported beside it.

Specifications are evaluated one per process by a multiprocessing pool with a
chunked map, so a list of candidates costs the time of the slowest chunk. The
sweep mode builds the specifications of a parameter grid (one axis per slot
parameter and per polynomial coefficient, linear or logarithmic) and writes
one comma separated row per grid point.

The reproduction mode checks this battery against the frozen receipts: every
Gate 1 champion of results/exponential_family/gate1/champions.json to 1e-9 in
relative deviation on every metric, and every reported member of
results/exponential_family/gate7/dominance.json to 1e-6.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import family as fam  # noqa: E402
import gate1_champions as g1  # noqa: E402
import gate7_family as g7  # noqa: E402
import metrics as met  # noqa: E402

try:  # the polynomial factor is optional
    import gate8_family as g8  # noqa: E402
except Exception:  # pragma: no cover - exercised only without gate8
    g8 = None

PI = float(np.pi)
XI_POINTS = 129
FINE_TOL = 1e-14
POSITIVITY_POINTS = 200001
ROUTE_TOL = 1e-13
SAMPLE_FLOOR = 1e-14
COMPARATORS = ("SLC1992", "TunedQ8", "CW12A", "CW12B", "BMC9", "BMC11")
DOMINANCE_YARDSTICKS = ("reach_d1_0.01", "reach_d1_0.001", "reach_d1_0.0001",
                        "gv_reach_1e-2", "e1_max_B0", "e2_max_B0")
STRICT_YARDSTICKS = DOMINANCE_YARDSTICKS + (
    "reach_d2_0.01", "reach_d2_0.001", "reach_d2_0.0001",
    "e1_max_B1", "e2_max_B1", "e1_max_B2", "e2_max_B2")
RATIO_YARDSTICKS = STRICT_YARDSTICKS + ("e1_rms_B0", "e2_rms_B0", "cond")
HIGHER_IS_BETTER = {key: key.startswith("reach") or key.startswith("gv_")
                    for key in RATIO_YARDSTICKS}
GATE1 = ROOT / "results" / "exponential_family" / "gate1" / "champions.json"
GATE7 = ROOT / "results" / "exponential_family" / "gate7" / "dominance.json"
GATE1_RTOL = 1e-9
GATE7_RTOL = 1e-6
CHECK_ATOL = 1e-12

CSV_COLUMNS = ("label", "backend", "m", "r", "n", "p", "parameters", "feasible",
               "reason", "cond", "cond_N64", "cond_N127", "cond_N128",
               "min_b", "max_b", "argmin_b_over_pi", "kappa2_min",
               "e1_max_B0", "e2_max_B0", "e1_rms_B0", "e2_rms_B0",
               "e1_max_B1", "e2_max_B1", "e1_max_B2", "e2_max_B2",
               "reach_d1_0.01", "reach_d1_0.001", "reach_d1_0.0001",
               "reach_d2_0.01", "reach_d2_0.001", "reach_d2_0.0001",
               "gv_reach_1e-2", "rho1", "rho2",
               "order_d1_fitted", "order_d2_fitted",
               "ec_max_B0", "reach_ec_0.01",
               "b_certified", "kappa2b_certified", "S_m", "S_m_minus_2",
               "margin_SLC1992", "binding_SLC1992", "dominates_SLC1992",
               "cond_ratio_SLC1992", "dominates_at_budget_SLC1992",
               "margin_TunedQ8", "binding_TunedQ8", "dominates_TunedQ8",
               "cond_ratio_TunedQ8", "dominates_at_budget_TunedQ8",
               "seconds")
BUDGET_SLACK = 1e-9


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

def normalise_spec(spec: dict) -> dict:
    """Return the canonical form of one generator specification.

    Accepted keys: m, omegas (or imaginary), real_pairs (or reals), quads,
    polynomial (or poly) as [p_1, ..., p_q] of P(t) = 1 + sum_l p_l t^l, or
    polynomial_scaled as c_l = p_l pi^{2l}, label, backend. A slots list of
    [alpha, omega] pairs is an alternative to the three kind lists; it fixes
    the order in which the slot factors multiply, which matters only in the
    last bits of the floating point product.
    """
    if not isinstance(spec, dict):
        raise ValueError("a specification must be a mapping")
    m = int(spec["m"])
    if "slots" in spec:
        slots = [(float(a), float(w)) for a, w in spec["slots"]]
        omegas = [w for a, w in slots if a == 0.0]
        reals = [a for a, w in slots if w == 0.0]
        quads = [[a, w] for a, w in slots if a != 0.0 and w != 0.0]
    else:
        slots = None
        omegas = [float(w) for w in spec.get("omegas", spec.get("imaginary", ()))]
        reals = [float(a) for a in spec.get("real_pairs", spec.get("reals", ()))]
        quads = [[float(a), float(w)] for a, w in spec.get("quads", ())]
    poly = [float(p) for p in spec.get("polynomial", spec.get("poly", ()))]
    if "polynomial_scaled" in spec:
        poly = [float(c) / PI ** (2 * (l + 1))
                for l, c in enumerate(spec["polynomial_scaled"])]
    out = {"m": m, "omegas": omegas, "real_pairs": reals, "quads": quads,
           "polynomial": poly, "backend": str(spec.get("backend", "auto"))}
    if slots is not None:
        out["slots"] = [[a, w] for a, w in slots]
    out["label"] = str(spec.get("label") or default_label(out))
    return out


def default_label(spec: dict) -> str:
    """Return a readable label built from the exponent multiset."""
    parts = [f"i{w:.6f}" for w in spec["omegas"]]
    parts += [f"r{a:.6f}" for a in spec["real_pairs"]]
    parts += [f"q({a:.6f},{w:.6f})" for a, w in spec["quads"]]
    head = f"E_{{{spec['m']}}}(" + ",".join(parts) + ")"
    if spec["polynomial"]:
        head += "xP[" + ",".join(f"{p:.6g}" for p in spec["polynomial"]) + "]"
    return head


def backend_of(spec: dict) -> str:
    """Return the frozen back end that carries this specification."""
    choice = spec.get("backend", "auto")
    if choice != "auto":
        return choice
    if spec["polynomial"]:
        return "gate8"
    if spec["real_pairs"] or spec["quads"]:
        return "gate7"
    return "gate0"


def make_generator(spec: dict):
    """Return the generator object of a specification and the back end name."""
    backend = backend_of(spec)
    if "slots" in spec:
        slots = [(float(a), float(w)) for a, w in spec["slots"]]
    else:
        slots = [(0.0, w) for w in spec["omegas"]]
        slots += [(a, 0.0) for a in spec["real_pairs"]]
        slots += [(a, w) for a, w in spec["quads"]]
    if backend == "gate8":
        if g8 is None:
            raise ValueError("gate8_family is not available for a polynomial factor")
        return g8.PolynomialGenerator(spec["m"], slots, spec["polynomial"]), backend
    if spec["polynomial"]:
        raise ValueError("a polynomial factor needs the gate8 back end")
    if backend == "gate7":
        return g7.ExtendedGenerator(spec["m"], slots), backend
    if spec["real_pairs"] or spec["quads"]:
        raise ValueError("real pairs and quads need the gate7 or gate8 back end")
    return fam.ExponentialGenerator(spec["m"], spec["omegas"]), backend


# ---------------------------------------------------------------------------
# The pieces of the battery
# ---------------------------------------------------------------------------

def stencil_block(gen) -> dict:
    """Return the bandwidths and the solve count of the collocation member.

    The bandwidth is the number of nonzero lattice samples, that is the number
    of nonzero diagonals of the circulant operator, and one cyclic band solve
    produces both derivatives because D1 = A_1 A_0^{-1} and D2 = A_2 A_0^{-1}
    share the coefficient vector A_0^{-1} u.
    """
    samples = gen.integer_samples(FINE_TOL) if _takes_tol(gen) else gen.integer_samples()
    return {"bandwidth_left": int(np.count_nonzero(np.abs(samples.beta) > SAMPLE_FLOOR)),
            "bandwidth_d1": int(np.count_nonzero(np.abs(samples.dbeta) > SAMPLE_FLOOR)),
            "bandwidth_d2": int(np.count_nonzero(np.abs(samples.ddbeta) > SAMPLE_FLOOR)),
            "nominal_bandwidth_left": int(2 * gen.nmax - 1),
            "radius": int(gen.nmax), "solves": 1,
            "truncation": samples.truncation}


def _takes_tol(gen) -> bool:
    """Return whether the back end exposes the Gate 0 tolerance argument."""
    return isinstance(gen, fam.ExponentialGenerator)


def symbol_callables(gen, xi: np.ndarray):
    """Return kappa_1, kappa_2, b, the analytic group velocity, and e_c."""
    if _takes_tol(gen):
        k1 = lambda t: gen.symbols(t, FINE_TOL)[1]
        k2 = lambda t: gen.symbols(t, FINE_TOL)[2]
        left = lambda t: gen.symbols(t, FINE_TOL)[0]
        gv = lambda t: gen.symbol_derivative(t, FINE_TOL)
        coupled = lambda t: gen.coupled_error(t, xi, FINE_TOL)
    else:
        k1 = lambda t: gen.symbols(t)[1]
        k2 = lambda t: gen.symbols(t)[2]
        left = lambda t: gen.symbols(t)[0]
        gv = lambda t: gen.symbol_derivative(t)
        coupled = lambda t: gen.coupled_error(t, xi)
    return k1, k2, left, gv, coupled


def conditioning_block(gen, theta_points: int = met.THETA_POINTS,
                       discrete=met.DISCRETE_N) -> dict:
    """Return the conditioning of the left side with the extreme locations.

    The continuous grid is linspace(0, pi, theta_points), the grid of
    metrics.symbol_cond, so the reported cond is the frozen one; the mode grid
    of N points is theta_l = 2 pi l / N, on which the discrete operator of a
    periodic problem with N modes actually acts.
    """
    if _takes_tol(gen):
        symbols = lambda t: gen.symbols(t, FINE_TOL)
    else:
        symbols = gen.symbols
    theta = np.linspace(0.0, PI, theta_points)
    b, _, k2 = symbols(theta)
    lo, hi = int(np.argmin(b)), int(np.argmax(b))
    out = {"points": int(theta_points),
           "cond": float(b[hi] / b[lo]),
           "min_b": float(b[lo]), "max_b": float(b[hi]),
           "argmin_b_over_pi": float(theta[lo] / PI),
           "argmax_b_over_pi": float(theta[hi] / PI),
           "positive": bool(np.all(np.isfinite(b)) and b[lo] > 0.0),
           "kappa2_min": float(k2[1:].min()),
           "min_left": float(b[lo]), "max_left": float(b[hi])}
    for N in discrete:
        angles = 2.0 * PI * np.arange(N) / N
        bn, _, k2n = symbols(angles)
        j = int(np.argmin(bn))
        out[f"cond_N{N}"] = float(bn.max() / bn[j])
        out[f"min_b_N{N}"] = float(bn[j])
        out[f"max_b_N{N}"] = float(bn.max())
        out[f"argmin_b_N{N}"] = int(j)
        out[f"argmin_b_over_pi_N{N}"] = float(angles[j] / PI)
        out[f"min_kappa2_nonzero_N{N}"] = float(k2n[1:].min())
        out[f"kappa2_zero_at_N{N}"] = float(abs(k2n[0]))
    return out


def sufficient_block(gen, spec: dict, backend: str) -> dict:
    """Return the S_m conditions of the member, by both routes where defined.

    S_m(omega) < 1 makes the k = 0 term of the alias sum for b exceed the sum
    of the magnitudes of all the others, hence b > 0; S_{m-2}(omega) < 1 does
    the same for the second moment sum, hence kappa_2 > 0. Both are conditions
    on the largest imaginary frequency alone, so they are vacuous for a member
    with no imaginary pair, and they are sufficient, never necessary: the
    positivity certificate decides the member.
    """
    if backend == "gate8":
        out = dict(g8.sufficient_conditions(gen))
        out["route"] = "gate8_family.sufficient_conditions"
        out["S_m"] = out.get("S_m_P")
        out["S_m_minus_2"] = out.get("S_m_minus_2_P")
    elif backend == "gate7":
        out = dict(g7.sufficient_conditions(gen))
        out["route"] = "gate7_family.sufficient_conditions"
    else:
        omegas = list(spec["omegas"])
        out = {"route": "gate1_champions.sufficient_condition_S",
               "has_imaginary": bool(omegas),
               "omega_below_pi": bool(all(0.0 < w < PI for w in omegas)),
               "re_s_above_minus_pi2": True}
        if omegas:
            top = max(omegas)
            out["S_m"] = g1.sufficient_condition_S(spec["m"], top)
            out["S_m_minus_2"] = g1.sufficient_condition_S(max(2, spec["m"] - 2), top)
        else:
            out["S_m"] = 0.0
            out["S_m_minus_2"] = 0.0
        out["theorem_P"] = bool(out["omega_below_pi"] and out["S_m"] < 1.0)
        out["theorem_N"] = bool(spec["m"] >= 4 and out["omega_below_pi"]
                                and out["S_m_minus_2"] < 1.0)
    omegas = list(spec["omegas"]) + [w for _, w in spec["quads"]]
    if omegas:
        top = max(omegas)
        out["omega_max"] = float(top)
        out["S_m_zeta"] = g1.sufficient_condition_S(spec["m"], top)
        out["S_m_minus_2_zeta"] = g1.sufficient_condition_S(max(2, spec["m"] - 2), top)
    else:
        out["omega_max"] = None
        out["S_m_zeta"] = 0.0
        out["S_m_minus_2_zeta"] = 0.0
    out["applies"] = bool(omegas)
    return out


def decimal_block(gen, backend: str) -> dict:
    """Return the 60-digit decimal recomputation of the integer samples.

    The decimal route sums the same truncated alias series in 60 working
    digits with series sine and cosine, so its deviation from the double
    precision samples measures arithmetic alone, the truncation being shared.
    """
    if backend == "gate8":
        routes = g8.verify_routes(gen)
        routes["route"] = "gate8_family.verify_routes"
        return routes
    if backend == "gate7":
        routes = g7.verify_routes(gen)
        routes["route"] = "gate7_family.verify_routes"
        return routes
    alias = gen.integer_samples(FINE_TOL)
    dec = g1.decimal_samples_of(gen)
    deviation = max(float(np.max(np.abs(np.asarray(dec[key]) - arr)))
                    for key, arr in (("beta", alias.beta), ("dbeta", alias.dbeta),
                                     ("ddbeta", alias.ddbeta)))
    probe = np.linspace(0.05, PI, 41)
    direct = gen.sample_symbols(probe, tol=FINE_TOL)
    from_samples = gen.symbols(probe, FINE_TOL)
    gap = max(float(np.max(np.abs(direct[q] - from_samples[q]))
                    / max(1.0, float(np.max(np.abs(from_samples[q])))))
              for q in range(3))
    return {"route": "gate1_champions.decimal_samples_of",
            "decimal_deviation": deviation,
            "decimal_ok": bool(deviation <= ROUTE_TOL),
            "symbols_from_samples_versus_alias": gap,
            "symbols_ok": bool(gap <= 1e-12),
            "K": int(dec["K"]), "quadrature_points": int(dec["quadrature_points"]),
            "working_digits": int(dec["working_digits"]),
            "series_truncation_bound": dec["series_truncation_bound"],
            "alias_tail_bound": dec["alias_tail_bound"],
            "beta_digits": dec["beta_digits"]}


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

def comparator_targets(names=COMPARATORS) -> dict:
    """Return the frozen Gate 0 metric row of every comparator of the table."""
    out = {}
    for scheme in comp.comparator_schemes():
        if scheme.name not in names:
            continue
        row = met.evaluate(scheme.name, scheme.kappa1, scheme.kappa2, scheme.left,
                           analytic_gv=scheme.analytic_gv,
                           algebraic_order1=scheme.algebraic_order1,
                           algebraic_order2=scheme.algebraic_order2)
        out[scheme.name] = {key: _value(row, key) for key in RATIO_YARDSTICKS}
    return out


def _value(row: dict, key: str):
    """Return the yardstick of a metric row, the refined value for a reach."""
    if key.startswith("reach"):
        return row.get(key + "_refined", row.get(key))
    return row.get(key)


def margins_against(row: dict, target: dict, keys) -> dict:
    """Return the signed relative margins of a member against one comparator.

    A yardstick where more is better has margin (value - target)/target; a band
    maximum, where less is better, has margin 1 - value/target. Both are
    positive exactly when the member is at least as good as the comparator.
    """
    out = {}
    for key in keys:
        value, goal = _value(row, key), target.get(key)
        if value is None or goal is None or goal == 0.0:
            continue
        out[key] = float((value - goal) / goal if HIGHER_IS_BETTER[key]
                         else 1.0 - value / goal)
    return out


def comparison_block(row: dict, targets: dict, cond: float | None = None) -> dict:
    """Return ratios, margins and the dominance verdict against every comparator.

    The verdict named dominates is the sign of the smallest margin over the six
    yardsticks of the Gate 7 dominance contract and says nothing about
    conditioning, which the gates carry as a separate constraint: the member
    must fit a budget, which for a comparator is the condition number of its own
    left side. The verdict named dominates_at_budget adds that constraint, so a
    member with a hugely ill conditioned left side cannot buy resolution with it.
    """
    out = {}
    for name, target in targets.items():
        ratios = {}
        for key in RATIO_YARDSTICKS:
            value, goal = _value(row, key), target.get(key)
            if value is None or goal is None or goal == 0.0:
                continue
            ratios[key] = float(value / goal)
        margins = margins_against(row, target, RATIO_YARDSTICKS)
        core = {k: margins[k] for k in DOMINANCE_YARDSTICKS if k in margins}
        strict = {k: margins[k] for k in STRICT_YARDSTICKS if k in margins}
        entry = {"ratios": ratios, "margins": margins}
        if core:
            binding = min(core, key=core.get)
            entry.update({"smallest_margin": float(core[binding]),
                          "binding_yardstick": binding,
                          "dominates": bool(core[binding] >= 0.0)})
        else:  # pragma: no cover - a comparator always carries the core set
            entry.update({"smallest_margin": None, "binding_yardstick": None,
                          "dominates": False})
        budget = target.get("cond")
        value = row.get("cond") if cond is None else cond
        if budget and value is not None:
            entry["cond"] = float(value)
            entry["cond_budget"] = float(budget)
            entry["cond_ratio"] = float(value / budget)
            entry["cond_ok"] = bool(value <= budget * (1.0 + BUDGET_SLACK))
            entry["dominates_at_budget"] = bool(entry.get("dominates")
                                                and entry["cond_ok"])
        if strict:
            binding = min(strict, key=strict.get)
            entry.update({"smallest_margin_strict": float(strict[binding]),
                          "binding_yardstick_strict": binding,
                          "dominates_strict": bool(strict[binding] >= 0.0)})
            entry["dominates_strict_at_budget"] = bool(
                entry["dominates_strict"] and entry.get("cond_ok", True))
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# One specification
# ---------------------------------------------------------------------------

def default_options(**kwargs) -> dict:
    """Return the option dictionary of the battery with the requested overrides."""
    options = {"theta_points": met.THETA_POINTS, "xi_points": XI_POINTS,
               "coupled": True, "positivity": True, "decimal": False,
               "comparison": True, "positivity_points": POSITIVITY_POINTS,
               "samples": False}
    options.update({k: v for k, v in kwargs.items() if v is not None})
    return options


def evaluate_spec(spec: dict, options: dict | None = None,
                  targets: dict | None = None) -> dict:
    """Return the full battery of one specification, errors captured in place.

    An imaginary-pair member with an even core whose Gate 0 alias route cannot
    reach the tolerance (the second moment sum of a member with few exponents
    converges too slowly for a double precision tail bound) is retried on the
    Gate 7 back end, whose slot factorization sums the same series with the
    tighter factorwise bound.
    """
    options = default_options() if options is None else options
    started = time.time()
    spec = normalise_spec(spec)
    result = _evaluate_with_backend(spec, backend_of(spec), options, targets)
    if result.get("error") and spec["backend"] == "auto" and not spec["polynomial"] \
            and backend_of(spec) == "gate0" and spec["m"] % 2 == 0:
        retry = _evaluate_with_backend(spec, "gate7", options, targets)
        if not retry.get("error"):
            retry["backend_fallback"] = {"from": "gate0", "reason": result["error"]}
            result = retry
    result["seconds"] = time.time() - started
    return result


def _evaluate_with_backend(spec: dict, backend: str, options: dict,
                           targets: dict | None) -> dict:
    """Return the battery of a specification on one named back end."""
    started = time.time()
    result = {"label": spec["label"], "spec": spec}
    try:
        gen, backend = make_generator(dict(spec, backend=backend))
    except Exception as exc:
        result.update({"backend": backend, "feasible": False,
                       "error": f"{type(exc).__name__}: {exc}",
                       "seconds": time.time() - started})
        return result
    result["backend"] = backend
    xi = np.linspace(-0.5, 0.5, int(options["xi_points"]))
    try:
        k1, k2, left, gv, coupled = symbol_callables(gen, xi)
        stencil = stencil_block(gen)
        unavailable = None
        if options["coupled"]:
            try:  # the off-grid alias sum has its own tail bound, which a member
                coupled(np.array([1.0]))  # of four exponents cannot satisfy
            except Exception as exc:
                unavailable = f"{type(exc).__name__}: {exc}"
                coupled = None
        result["coupled_unavailable"] = unavailable
        row = met.evaluate(spec["label"], k1, k2, left,
                           analytic_gv=gv,
                           coupled=coupled if options["coupled"] else None,
                           algebraic_order1=spec["m"],
                           algebraic_order2=spec["m"] - 2,
                           bandwidth_left=stencil["bandwidth_left"],
                           bandwidth_d1=stencil["bandwidth_d1"],
                           bandwidth_d2=stencil["bandwidth_d2"],
                           solves=stencil["solves"],
                           theta_points=int(options["theta_points"]))
        cond = conditioning_block(gen, theta_points=int(options["theta_points"]))
        admissible, detail = gen.admissible()
        result.update({
            "family": family_block(gen, spec, backend),
            "admissible": {"ok": bool(admissible), "detail": detail},
            "metrics": row,
            "conditioning": cond,
            "stencil": {k: v for k, v in stencil.items() if k != "truncation"},
            "truncation": stencil["truncation"],
            "sufficient_conditions": sufficient_block(gen, spec, backend),
            "feasible": bool(admissible and cond["positive"]
                             and cond["kappa2_min"] > 0.0),
            "error": None,
        })
        result["reason"] = "admissible" if result["feasible"] else detail
        if options["positivity"]:
            result["positivity"] = g7.certify_positivity(
                gen, points=int(options["positivity_points"]))
        if options["decimal"]:
            result["decimal"] = decimal_block(gen, backend)
        if options["samples"]:
            samples = (gen.integer_samples(FINE_TOL) if _takes_tol(gen)
                       else gen.integer_samples())
            result["samples"] = {"n": [int(v) for v in samples.n],
                                 "beta": [float(v) for v in samples.beta],
                                 "dbeta": [float(v) for v in samples.dbeta],
                                 "ddbeta": [float(v) for v in samples.ddbeta]}
        if options["comparison"]:
            targets = comparator_targets() if targets is None else targets
            result["comparison"] = comparison_block(row, targets, cond["cond"])
            for tag, name in (("slc", "SLC1992"), ("q8", "TunedQ8")):
                entry = result["comparison"].get(name, {})
                result[f"dominates_{tag}"] = bool(entry.get("dominates", False))
                result[f"dominates_{tag}_at_budget"] = bool(
                    entry.get("dominates_at_budget", False))
    except Exception as exc:
        result.update({"feasible": False, "error": f"{type(exc).__name__}: {exc}"})
    result["seconds"] = time.time() - started
    return result


def family_block(gen, spec: dict, backend: str) -> dict:
    """Return the bookkeeping of the exponent multiset of a member."""
    out = {"m": int(spec["m"]), "omegas": list(spec["omegas"]),
           "real_pairs": list(spec["real_pairs"]),
           "quads": [list(q) for q in spec["quads"]],
           "polynomial": list(spec["polynomial"]),
           "r": len(spec["omegas"]), "p": int(gen.p), "nmax": int(gen.nmax),
           "backend": backend}
    out["n"] = int(getattr(gen, "n", spec["m"] + 2 * len(spec["omegas"])))
    out["exponents"] = out["n"]
    out["parameters"] = int(getattr(gen, "parameters",
                                    len(spec["omegas"]) + len(spec["real_pairs"])
                                    + 2 * len(spec["quads"]) + len(spec["polynomial"])))
    if backend == "gate8":
        out["polynomial_degree"] = int(gen.q)
        out["smoothness"] = int(gen.smoothness)
        out["polynomial_scaled"] = [float(p * PI ** (2 * (l + 1)))
                                    for l, p in enumerate(spec["polynomial"])]
    return out


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(options: dict, targets: dict | None) -> None:
    _WORKER["options"] = options
    _WORKER["targets"] = targets


def _worker(spec: dict) -> dict:
    return evaluate_spec(spec, _WORKER["options"], _WORKER["targets"])


def chunk_size(count: int, processes: int) -> int:
    """Return a chunk that keeps every process fed without starving the tail."""
    return max(1, min(8, count // max(1, 4 * processes) or 1))


def evaluate_specs(specs, options: dict | None = None, processes: int = 1,
                   targets: dict | None = None, stream=None) -> list:
    """Return the battery of every specification, one process per generator."""
    specs = [normalise_spec(s) for s in specs]
    options = default_options() if options is None else options
    if targets is None and options["comparison"]:
        targets = comparator_targets()
    if processes <= 1 or len(specs) == 1:
        _init_worker(options, targets)
        out = []
        for spec in specs:
            row = _worker(spec)
            out.append(row)
            if stream is not None:
                stream(row)
        return out
    with Pool(processes, initializer=_init_worker,
              initargs=(options, targets)) as pool:
        out = []
        for row in pool.imap(_worker, specs,
                             chunksize=chunk_size(len(specs), processes)):
            out.append(row)
            if stream is not None:
                stream(row)
    return out


# ---------------------------------------------------------------------------
# The parameter sweep
# ---------------------------------------------------------------------------

def axis_values(axis: dict) -> np.ndarray:
    """Return the sample points of one sweep axis, linear or logarithmic."""
    lo, hi = (float(v) for v in axis["range"])
    count = int(axis.get("count", 1))
    if count == 1:
        return np.array([lo])
    if str(axis.get("scale", "linear")) == "log":
        if lo <= 0.0:
            raise ValueError("a logarithmic axis needs a positive lower end")
        return np.exp(np.linspace(np.log(lo), np.log(hi), count))
    return np.linspace(lo, hi, count)


def sweep_specs(grid: dict) -> list:
    """Return the specifications of a parameter grid.

    The grid names m (an integer or a list) and a list of axes. An axis of
    kind "imaginary" or "real" carries one parameter, an axis of kind "quad"
    carries the two sub axes alpha and omega, and an axis of kind "poly"
    carries one coefficient of P. With "ordered" true the points whose
    frequencies of equal kind are not strictly increasing are dropped, which
    removes the permutations of one exponent multiset.
    """
    ms = grid["m"] if isinstance(grid.get("m"), (list, tuple)) else [grid["m"]]
    axes = list(grid.get("axes", ()))
    values, kinds = [], []
    for axis in axes:
        kind = str(axis["kind"])
        if kind == "quad":
            values.append(axis_values(axis["alpha"]))
            kinds.append("quad_alpha")
            values.append(axis_values(axis["omega"]))
            kinds.append("quad_omega")
        else:
            values.append(axis_values(axis))
            kinds.append(kind)
    ordered = bool(grid.get("ordered", True))
    label = str(grid.get("label", "sweep"))
    backend = grid.get("backend", "auto")
    specs = []
    points = list(itertools.product(*values)) if values else [()]
    for m in ms:
        for point in points:
            omegas, reals, quads, poly = [], [], [], []
            pending = None
            for kind, value in zip(kinds, point):
                if kind == "imaginary":
                    omegas.append(float(value))
                elif kind == "real":
                    reals.append(float(value))
                elif kind == "quad_alpha":
                    pending = float(value)
                elif kind == "quad_omega":
                    quads.append([pending, float(value)])
                elif kind == "poly":
                    poly.append(float(value))
                else:
                    raise ValueError(f"unknown sweep axis kind {kind}")
            if ordered and (any(np.diff(omegas) <= 0.0) or any(np.diff(reals) <= 0.0)):
                continue
            specs.append({"m": int(m), "omegas": omegas, "real_pairs": reals,
                          "quads": quads, "polynomial": poly, "backend": backend,
                          "label": f"{label}|m{m}|" + "_".join(f"{v:.6g}" for v in point)})
    return specs


def sweep_options(grid: dict, base: dict) -> dict:
    """Return the options of a sweep.

    They follow the battery options of the command line unless the grid names
    its own; the decimal route and the sample dump are always off, being far
    too slow for a grid and of no use in a map of a family.
    """
    options = dict(base)
    options["coupled"] = bool(grid.get("coupled", base.get("coupled", True)))
    options["decimal"] = bool(grid.get("decimal", False))
    options["positivity"] = bool(grid.get("positivity", base.get("positivity", True)))
    options["samples"] = False
    return options


def csv_row(result: dict) -> dict:
    """Return the flat comma separated view of one battery result."""
    row = {key: "" for key in CSV_COLUMNS}
    row["label"] = result.get("label", "")
    row["backend"] = result.get("backend", "")
    row["feasible"] = result.get("feasible", False)
    row["reason"] = result.get("error") or result.get("reason", "")
    row["seconds"] = result.get("seconds")
    fam_block = result.get("family", {})
    for key in ("m", "r", "n", "p", "parameters"):
        row[key] = fam_block.get(key, "")
    metrics = result.get("metrics", {})
    cond = result.get("conditioning", {})
    positivity = result.get("positivity", {})
    suff = result.get("sufficient_conditions", {})
    for key in ("cond", "cond_N64", "cond_N127", "cond_N128", "min_b", "max_b",
                "argmin_b_over_pi", "kappa2_min"):
        row[key] = cond.get(key, "")
    for key in ("e1_max_B0", "e2_max_B0", "e1_rms_B0", "e2_rms_B0", "e1_max_B1",
                "e2_max_B1", "e1_max_B2", "e2_max_B2", "gv_reach_1e-2", "rho1",
                "rho2", "order_d1_fitted", "order_d2_fitted", "ec_max_B0"):
        row[key] = metrics.get(key, "")
    for key in ("reach_d1_0.01", "reach_d1_0.001", "reach_d1_0.0001",
                "reach_d2_0.01", "reach_d2_0.001", "reach_d2_0.0001",
                "reach_ec_0.01"):
        row[key] = metrics.get(key + "_refined", metrics.get(key, ""))
    row["b_certified"] = positivity.get("b_certified", "")
    row["kappa2b_certified"] = positivity.get("kappa2b_certified", "")
    row["S_m"] = suff.get("S_m", "")
    row["S_m_minus_2"] = suff.get("S_m_minus_2", "")
    for name in ("SLC1992", "TunedQ8"):
        entry = result.get("comparison", {}).get(name, {})
        row[f"margin_{name}"] = entry.get("smallest_margin", "")
        row[f"binding_{name}"] = entry.get("binding_yardstick", "")
        row[f"dominates_{name}"] = entry.get("dominates", "")
        row[f"cond_ratio_{name}"] = entry.get("cond_ratio", "")
        row[f"dominates_at_budget_{name}"] = entry.get("dominates_at_budget", "")
    return row


def run_sweep(grid: dict, csv_path: Path, options: dict, processes: int) -> dict:
    """Evaluate a parameter grid in parallel and write one row per point."""
    specs = sweep_specs(grid)
    options = sweep_options(grid, options)
    started = time.time()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle = csv_path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    counter = {"n": 0, "feasible": 0, "dominating": 0,
               "dominating_at_budget": 0}

    def sink(result: dict) -> None:
        writer.writerow(csv_row(result))
        counter["n"] += 1
        counter["feasible"] += int(bool(result.get("feasible")))
        counter["dominating"] += int(bool(result.get("dominates_slc")))
        counter["dominating_at_budget"] += int(
            bool(result.get("dominates_slc_at_budget")))
        if counter["n"] % 200 == 0:
            handle.flush()

    try:
        evaluate_specs(specs, options, processes, stream=sink)
    finally:
        handle.close()
    return {"points": len(specs), "written": counter["n"],
            "feasible": counter["feasible"], "dominating_slc": counter["dominating"],
            "dominating_slc_at_budget": counter["dominating_at_budget"],
            "csv": str(csv_path), "seconds": time.time() - started,
            "processes": int(processes)}


# ---------------------------------------------------------------------------
# Reproduction of the frozen receipts
# ---------------------------------------------------------------------------

def _compare_floats(stored: dict, fresh: dict, rtol: float,
                    atol: float = CHECK_ATOL) -> dict:
    """Return the worst relative deviation over the float keys of two rows."""
    worst, key_worst, compared, failures = 0.0, None, 0, []
    for key, value in stored.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        other = fresh.get(key)
        if not isinstance(other, (int, float)) or isinstance(other, bool):
            continue
        compared += 1
        scale = max(abs(float(value)), atol)
        deviation = abs(float(value) - float(other)) / scale
        if deviation > worst:
            worst, key_worst = deviation, key
        if deviation > rtol:
            failures.append({"key": key, "stored": float(value),
                             "fresh": float(other), "relative": deviation})
    return {"compared": compared, "worst": worst, "worst_key": key_worst,
            "failures": failures, "ok": not failures}


def gate1_cases() -> list:
    """Return the specification and stored metrics of every Gate 1 champion."""
    data = json.loads(GATE1.read_text())
    cases = []
    for band, budgets in data["champions"].items():
        for budget_name, entry in budgets.items():
            champion = entry.get("champion")
            if not champion:
                continue
            cases.append({"key": f"{band}|{budget_name}",
                          "spec": {"m": champion["m"], "omegas": champion["omegas"],
                                   "label": champion["metrics"]["name"]},
                          "metrics": champion["metrics"]})
    return cases


def gate7_cases() -> list:
    """Return the specification and stored metrics of every Gate 7 member."""
    data = json.loads(GATE7.read_text())
    cases = []
    for key, record in data["summary"].items():
        verification = data["verification"].get(key)
        if verification is None:
            continue
        cases.append({"key": key,
                      "spec": {"m": record["m"], "slots": record["slots"],
                               "backend": "gate7",
                               "label": key.replace("|", "_")},
                      "metrics": verification["metrics"],
                      "positivity": verification.get("positivity")})
    return cases


def run_check(processes: int, gate1: bool = True, gate7: bool = True) -> dict:
    """Return the reproduction report against the frozen receipts."""
    report = {"gate1_rtol": GATE1_RTOL, "gate7_rtol": GATE7_RTOL, "gates": {}}
    targets = comparator_targets()
    for name, enabled, cases, rtol in (("gate1", gate1, gate1_cases, GATE1_RTOL),
                                       ("gate7", gate7, gate7_cases, GATE7_RTOL)):
        if not enabled:
            continue
        items = cases()
        options = default_options(comparison=False, positivity=True, decimal=False)
        started = time.time()
        results = evaluate_specs([item["spec"] for item in items], options,
                                 processes, targets=targets)
        block = {"members": [], "ok": True, "worst": 0.0, "worst_key": None,
                 "worst_member": None, "seconds": time.time() - started,
                 "count": len(items)}
        for item, result in zip(items, results):
            entry = {"key": item["key"], "label": result["label"],
                     "seconds": result["seconds"], "error": result.get("error")}
            if result.get("error"):
                entry["ok"] = False
                block["ok"] = False
                block["members"].append(entry)
                continue
            row = dict(result["metrics"])
            row.update(result["conditioning"])
            row.update(result["stencil"])
            comparison = _compare_floats(item["metrics"], row, rtol)
            entry.update(comparison)
            if item.get("positivity") and result.get("positivity"):
                pos = _compare_floats(item["positivity"], result["positivity"], rtol)
                entry["positivity"] = pos
                entry["ok"] = bool(comparison["ok"] and pos["ok"])
                entry["worst"] = max(comparison["worst"], pos["worst"])
            else:
                entry["ok"] = comparison["ok"]
            if entry["worst"] > block["worst"]:
                block["worst"] = entry["worst"]
                block["worst_key"] = entry.get("worst_key")
                block["worst_member"] = item["key"]
            block["ok"] = bool(block["ok"] and entry["ok"])
            block["members"].append(entry)
        block["max_seconds_per_member"] = max((m["seconds"] for m in block["members"]),
                                              default=0.0)
        report["gates"][name] = block
    report["ok"] = all(block["ok"] for block in report["gates"].values())
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown(results: list) -> str:
    """Return the compact table of a battery run."""
    header = ("| label | back end | ok | m | n | cond | cond/SLC | E1max B0 | "
              "E2max B0 | R1(1e-2) | R1(1e-3) | R1(1e-4) | R2(1e-3) | G(1e-2) | "
              "ec max B0 | b cert | k2 cert | margin SLC | binding | "
              "margin q8 | seconds |")
    lines = ["# Static battery", "",
             "Reaches in multiples of pi, refined by Brent's method on the grid",
             "bracket; margins are the smallest signed margin over R1(1e-2),",
             "R1(1e-3), R1(1e-4), G(1e-2), E1max(B0), E2max(B0); ok is the",
             "admissibility of the member, and an inadmissible member still",
             "carries its numbers so that the failure can be read.", "",
             header, "| " + " | ".join(["---"] * 21) + " |"]
    for result in results:
        if result.get("error"):
            lines.append(f"| {result['label']} | {result.get('backend', '')} | no | "
                         + " | ".join(["-"] * 17) + f" | {result['error']} |")
            continue
        metrics = result["metrics"]
        fam_block = result["family"]
        positivity = result.get("positivity", {})
        slc = result.get("comparison", {}).get("SLC1992", {})
        q8 = result.get("comparison", {}).get("TunedQ8", {})
        lines.append("| " + " | ".join([
            result["label"], result["backend"], _fmt(result.get("feasible")),
            str(fam_block["m"]), str(fam_block["n"]),
            _fmt(result["conditioning"]["cond"], 2),
            _fmt(slc.get("cond_ratio"), 3),
            _fmt(metrics["e1_max_B0"]), _fmt(metrics["e2_max_B0"]),
            _fmt(metrics["reach_d1_0.01_refined"]),
            _fmt(metrics["reach_d1_0.001_refined"]),
            _fmt(metrics["reach_d1_0.0001_refined"]),
            _fmt(metrics.get("reach_d2_0.001_refined")),
            _fmt(metrics["gv_reach_1e-2"]),
            _fmt(metrics.get("ec_max_B0")),
            _fmt(positivity.get("b_certified")),
            _fmt(positivity.get("kappa2b_certified")),
            _fmt(slc.get("smallest_margin")), str(slc.get("binding_yardstick", "-")),
            _fmt(q8.get("smallest_margin")), f"{result['seconds']:.2f}"]) + " |")
    return "\n".join(lines) + "\n"


def check_markdown(report: dict) -> str:
    """Return the compact table of a reproduction run."""
    lines = ["# Static battery reproduction of the frozen receipts", "",
             "| gate | members | tolerance | worst relative deviation | worst key | "
             "worst member | verdict | seconds | slowest member |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, block in report["gates"].items():
        tol = report[f"{name}_rtol"]
        lines.append(f"| {name} | {block['count']} | {tol:g} | {block['worst']:.3e} | "
                     f"{block['worst_key']} | {block['worst_member']} | "
                     f"{'PASS' if block['ok'] else 'FAIL'} | {block['seconds']:.1f} | "
                     f"{block['max_seconds_per_member']:.2f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def load_specs(args) -> list:
    """Return the specifications named on the command line."""
    specs = []
    if args.specs:
        text = Path(args.specs).read_text()
        data = json.loads(text)
        specs += data["generators"] if isinstance(data, dict) and "generators" in data \
            else (data if isinstance(data, list) else [data])
    for item in args.spec or ():
        data = json.loads(item)
        specs += data if isinstance(data, list) else [data]
    return specs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--specs", type=str, default=None,
                        help="path to a JSON list of generator specifications, or "
                             "an object with a generators field")
    parser.add_argument("--spec", action="append", default=None,
                        help="one inline JSON specification, repeatable")
    parser.add_argument("--sweep", type=str, default=None,
                        help="path to a JSON parameter grid, evaluated in parallel")
    parser.add_argument("--csv", type=str, default=None,
                        help="output path of the sweep table (default sweep.csv)")
    parser.add_argument("--check", action="store_true",
                        help="reproduce the frozen Gate 1 and Gate 7 receipts")
    parser.add_argument("--gate", type=str, default="gate1,gate7",
                        help="which receipts the check reproduces")
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    parser.add_argument("--md", action="store_true", help="print a markdown table")
    parser.add_argument("--md-out", type=str, default=None,
                        help="write the markdown table to a path")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--decimal", action="store_true",
                        help="add the 60-digit decimal check of the integer samples")
    parser.add_argument("--no-coupled", action="store_true",
                        help="skip the coupled off-grid channel")
    parser.add_argument("--no-positivity", action="store_true",
                        help="skip the positivity certificates")
    parser.add_argument("--no-comparison", action="store_true",
                        help="skip the comparator block")
    parser.add_argument("--samples", action="store_true",
                        help="include the integer samples in the output")
    parser.add_argument("--xi-points", type=int, default=XI_POINTS)
    parser.add_argument("--theta-points", type=int, default=met.THETA_POINTS)
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the JSON to standard output")
    args = parser.parse_args(argv)

    options = default_options(theta_points=args.theta_points,
                              xi_points=args.xi_points,
                              coupled=not args.no_coupled,
                              positivity=not args.no_positivity,
                              comparison=not args.no_comparison,
                              decimal=args.decimal, samples=args.samples)

    if args.check:
        gates = args.gate.split(",")
        report = run_check(args.processes, gate1="gate1" in gates,
                           gate7="gate7" in gates)
        text = check_markdown(report)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1, default=float) + "\n")
        if args.md_out:
            Path(args.md_out).write_text(text)
        if args.md or not args.out:
            print(text)
        if not args.quiet and args.out is None and not args.md:
            print(json.dumps(report, indent=1, default=float))
        return 0 if report["ok"] else 1

    if args.sweep:
        grid = json.loads(Path(args.sweep).read_text())
        csv_path = Path(args.csv) if args.csv else Path("sweep.csv")
        summary = run_sweep(grid, csv_path, options, args.processes)
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=1, default=float) + "\n")
        if not args.quiet:
            print(json.dumps(summary, indent=1, default=float))
        return 0

    specs = load_specs(args)
    if not specs:
        parser.error("give --specs, --spec, --sweep or --check")
    results = evaluate_specs(specs, options, args.processes)
    payload = {"producer": "code/analysis/exponential_family/static_battery.py",
               "options": options, "count": len(results),
               "seconds": sum(r.get("seconds", 0.0) for r in results),
               "results": results}
    text = markdown(results)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, default=float) + "\n")
    if args.md_out:
        Path(args.md_out).write_text(text)
    if args.md:
        print(text)
    if not args.quiet and not args.out:
        print(json.dumps(payload, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
