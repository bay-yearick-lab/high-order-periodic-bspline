"""Gate 8, Question 3: dominance search over the augmented family E_{m,r} x P_q.

The search space at width n (n = 8 or 10 exponents, bandwidth n - 1) is the
union of cells (m, slot kinds, q): an even core m >= 4, (n - m)/2 pair slots
of the Gate 7 kinds (imaginary pair I with omega in (0, pi), real pair R with
alpha > 0, quad Q with both), and a real polynomial P(xi^2) of degree q with
P(0) = 1 multiplying the transform. The polynomial is carried in the scaled
coefficients c_l = p_l pi^{2 l}, so that P(theta^2) = 1 + sum_l c_l (theta/pi)^{2l},
and is admissible when P > 0 on [0, pi^2] (a real zero of psi_hat inside the
band makes the recovery singular, theory.md Proposition 3). The degree is at
most (n - 4)/2 (psi in C^2, needed for the literal second derivative and
the off-grid channels); the cells with q = (n - 6)/2 are the C^4
restriction and are reported separately. Every cell has (n - m)/2 + q real
parameters; at q = (n - 4)/2 this equals the dimension of the compact class
of type (n/2 - 1, n/2 - 1) and formal order m.

Yardsticks, constraints, margins and the threshold objective are those of
gate7_dominance.py (Gate 0 metrics code; grid-independent reaches at the
tolerance times (1 - REACH_SAFETY) inside the search; the smallest margin
against a comparator maximized by Nelder-Mead from the best seeds), with two
changes. The coupled channel enters the inner loop through the residue
closed form of gate8_family.py (a coarse xi grid inside the search, 129
points in the verification block), because the polynomial factor can lower
the nodal error by cancellation between aliases while the off-grid transfers
see the alias mass without cancellation. The seeds are a Sobol sequence over
the cell's box together with structured seeds built from polynomial roots
(soft contacts in (pi, 2 pi), complex pairs, negative roots), because the
grids of Gate 7 do not scale to six parameters.

Every reported member is re-evaluated with metrics.evaluate on the contract
grid, its band maxima and reaches confirmed on 200,001 angles, its samples
verified by the alias route with the exact zeta tail against the residue
route and the 60-digit decimal route to 1e-13, and its positivity certified
by the grid-plus-Lipschitz argument. The conditioning is max b / min b with
both extrema computed (max b need not be one for an augmented member).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import gate7_dominance as gd  # noqa: E402
import gate8_family as g8  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate8"
MEMBERS = OUT / "members"
GATE7 = ROOT / "results" / "exponential_family" / "gate7"

B0 = 2.0 * np.pi / 3.0
PI = np.pi
XI_SEARCH = np.linspace(-0.5, 0.5, 9)
XI_FINE = np.linspace(-0.5, 0.5, 17)
XI_FULL = np.linspace(-0.5, 0.5, 129)
C_BOX = {1: (-5.0, 3.0), 2: (-3.0, 6.0), 3: (-3.0, 3.0), 4: (-2.0, 2.0)}
SLOT_GRID_POINTS = {1: 40, 2: 14, 3: 9, 4: 6}
SOBOL_POWER = {1: 10, 2: 11, 3: 12, 4: 12, 5: 13, 6: 13, 7: 13}
SEEDS_PER_CELL = 10
NM_MAXITER = 1200
NM_RESTARTS = 2
POLISH_MAXITER = 250
POLISH_SCALES = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
POLISH_SAMPLES = 120
SEARCH_THETA = 4001
COUPLED_STRIDE = 4
YARDSTICKS = gd.YARDSTICKS
HIGHER_IS_BETTER = gd.HIGHER_IS_BETTER
COUPLED_FACTOR = gd.COUPLED_FACTOR
REACH_SAFETY = gd.REACH_SAFETY
INFEASIBLE = gd.INFEASIBLE
BUDGET_SLACK = gd.BUDGET_SLACK

CELLS = {
    8: [(4, "II", 1), (4, "II", 2), (4, "IR", 1), (4, "IR", 2), (4, "RR", 1), (4, "RR", 2),
        (4, "Q", 1), (4, "Q", 2), (6, "I", 1), (6, "I", 2), (6, "R", 1), (6, "R", 2),
        (8, "", 1), (8, "", 2)],
    10: [(4, "III", 2), (4, "III", 3), (4, "IIR", 2), (4, "IIR", 3), (4, "IRR", 2), (4, "IRR", 3),
         (4, "RRR", 2), (4, "RRR", 3), (4, "QI", 2), (4, "QI", 3), (4, "QR", 2), (4, "QR", 3),
         (6, "II", 2), (6, "II", 3), (6, "IR", 2), (6, "IR", 3), (6, "RR", 2), (6, "RR", 3),
         (6, "Q", 2), (6, "Q", 3), (8, "I", 2), (8, "I", 3), (8, "R", 2), (8, "R", 3),
         (10, "", 2), (10, "", 3)],
}
# polynomial seed configurations in x = t / pi^2: real roots (1, 4) are soft
# contacts, negative roots keep P increasing, complex pairs (u, v) = 1/x
POLY_ROOT_SEEDS = {
    1: [[1.15], [1.3], [1.6], [2.2], [3.5], [-1.0], [-4.0]],
    2: [[1.15, 1.6], [1.2, 2.5], [1.4, 3.5], [1.1, 1.3], [1.6, 2.2], [-2.0, 1.4], [-3.0, 2.5],
        ("c", 0.6, 0.4), ("c", 0.4, 0.7), ("c", 0.8, 0.2), ("c", 0.2, 1.0), ("c", -0.3, 0.6)],
    3: [[1.15, 1.5, 2.5], [1.1, 1.3, 1.8], [1.2, 2.0, 4.0], [-2.0, 1.2, 1.8], [-3.0, 1.4, 3.0],
        (("c", 0.6, 0.4), 1.3), (("c", 0.3, 0.8), 1.6), (("c", 0.8, 0.3), 2.5), (("c", 0.4, 0.5), -2.0)],
}


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------

def slot_dimension(kinds: str) -> int:
    return gd.cell_dimension(kinds)


def cell_dimension(kinds: str, q: int) -> int:
    return slot_dimension(kinds) + q


def poly_from_scaled(c) -> list:
    """Return [p_1, ..., p_q] from the scaled coefficients c_l = p_l pi^{2l}."""
    return [float(cl) / PI ** (2 * (l + 1)) for l, cl in enumerate(c)]


def scaled_from_roots(spec) -> list:
    """Return the scaled coefficients of prod (1 - x / x_i) from a root specification.

    A real entry x_i is a real root in x = t/pi^2; ("c", u, v) is the complex
    pair with 1/x = u +- i v, that is the factor 1 - 2 u x + (u^2 + v^2) x^2.
    """
    poly = np.array([1.0])
    items = spec if isinstance(spec, list) or (isinstance(spec, tuple) and spec and spec[0] != "c") else [spec]
    for item in items:
        if isinstance(item, tuple) and item[0] == "c":
            _, u, v = item
            factor = np.array([1.0, -2.0 * u, u * u + v * v])
        else:
            factor = np.array([1.0, -1.0 / float(item)])
        poly = np.convolve(poly, factor)
    return [float(v) for v in poly[1:]]


def unpack(kinds: str, q: int, x):
    x = list(x)
    ds = slot_dimension(kinds)
    slots = gd.unpack(kinds, x[:ds])
    poly = poly_from_scaled(x[ds:ds + q])
    return slots, poly


def in_box(kinds: str, q: int, x) -> bool:
    ds = slot_dimension(kinds)
    if not gd.in_box(kinds, x[:ds]):
        return False
    for l in range(q):
        lo, hi = C_BOX[l + 1]
        if not lo <= x[ds + l] <= hi:
            return False
    return True


def seeds_of(kinds: str, q: int, seed: int = 20260913) -> np.ndarray:
    """Return the seed set of a cell: Sobol points of the box plus structured seeds."""
    ds = slot_dimension(kinds)
    d = ds + q
    lows, highs = [], []
    la, lb = np.log(gd.ALPHA_RANGE[0]), np.log(gd.ALPHA_RANGE[1])
    for k in kinds:
        if k == "I":
            lows.append(gd.OMEGA_RANGE[0]); highs.append(gd.OMEGA_RANGE[1])
        elif k == "R":
            lows.append(la); highs.append(lb)
        else:
            lows += [la, gd.OMEGA_RANGE[0]]; highs += [lb, gd.OMEGA_RANGE[1]]
    for l in range(q):
        lo, hi = C_BOX[l + 1]
        lows.append(lo); highs.append(hi)
    sampler = qmc.Sobol(d, scramble=True, seed=seed)
    sob = qmc.scale(sampler.random_base2(SOBOL_POWER[d]), lows, highs)
    # structured seeds: coarse slot grid times polynomial root configurations
    axes = []
    for k in kinds:
        pts = SLOT_GRID_POINTS[ds]
        if k == "I":
            axes.append(np.linspace(gd.OMEGA_RANGE[0], gd.OMEGA_RANGE[1], pts))
        elif k == "R":
            axes.append(np.linspace(la, lb, pts))
        else:
            axes.append(np.linspace(la, lb, pts))
            axes.append(np.linspace(gd.OMEGA_RANGE[0], gd.OMEGA_RANGE[1], pts))
    slot_grid = np.array(list(itertools.product(*axes))) if axes else np.zeros((1, 0))
    if ds > 0:
        keep = np.ones(slot_grid.shape[0], dtype=bool)
        i = 0
        for k, group in itertools.groupby(kinds):
            count = len(list(group))
            if k in ("I", "R") and count > 1:
                block = slot_grid[:, i:i + count]
                keep &= np.all(np.diff(block, axis=1) > 0.0, axis=1)
            i += count if k in ("I", "R") else 2 * count
        slot_grid = slot_grid[keep]
    polys = [scaled_from_roots(spec) for spec in POLY_ROOT_SEEDS[q]] + [[0.0] * q]
    structured = np.array([list(s) + list(p) for s in slot_grid for p in polys])
    structured = structured[np.array([in_box(kinds, q, x) for x in structured])]
    return np.vstack([sob, structured])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class Evaluator:
    """Nodal and coupled yardsticks of one cell, with the constraints."""

    def __init__(self, m: int, kinds: str, q: int, C: float, theta_points: int = SEARCH_THETA,
                 coupled: bool = True, xi=XI_SEARCH, coupled_stride: int = COUPLED_STRIDE,
                 safety: float = REACH_SAFETY):
        self.m, self.kinds, self.q, self.C = m, kinds, q, C
        self.safety = safety
        self.theta = met.theta_grid(theta_points)
        self.cond_theta = np.linspace(0.0, PI, theta_points)
        self.coupled = coupled
        self.xi = np.asarray(xi, dtype=float)
        self.mask_B0 = met.band_mask(self.theta, B0)
        self.theta_c = self.theta[self.mask_B0][::coupled_stride]

    def generator(self, x):
        slots, poly = unpack(self.kinds, self.q, x)
        return g8.PolynomialGenerator(self.m, slots, poly)

    def __call__(self, x) -> dict:
        if not in_box(self.kinds, self.q, x):
            return {"feasible": False, "reason": "outside box"}
        slots, poly = unpack(self.kinds, self.q, x)
        if g8.poly_min_on_band(poly) <= 0.0:
            return {"feasible": False, "reason": "P not positive on the band"}
        try:
            gen = g8.PolynomialGenerator(self.m, slots, poly)
            b, k1, k2 = gen.symbols(self.theta)
            bc = gen.symbols(self.cond_theta)[0]
        except (ValueError, FloatingPointError) as exc:
            return {"feasible": False, "reason": str(exc)}
        row = {"feasible": True}
        if not np.all(np.isfinite(bc)) or bc.min() <= 0.0:
            row.update({"feasible": False, "reason": "b not positive", "cond": np.inf})
            return row
        row["cond"] = float(bc.max() / bc.min())
        row["max_b"] = float(bc.max())
        row["min_b"] = float(bc.min())
        row["kappa2_min"] = float(k2.min())
        if not np.all(np.isfinite(k2)) or k2.min() <= 0.0:
            row.update({"feasible": False, "reason": "kappa_2 not positive"})
            return row
        th = self.theta
        e1 = k1 / th - 1.0
        e2 = k2 / th ** 2 - 1.0
        e1_fn = lambda t: gen.symbols(t)[1] / t - 1.0
        shrink = 1.0 - self.safety
        for tol in met.REACH_TOLS:
            row[f"reach_d1_{tol:g}"] = gd.robust_reach(th, e1, tol * shrink, e1_fn) / PI
        gv = gen.symbol_derivative(th) - 1.0
        gv_fn = lambda t: gen.symbol_derivative(t) - 1.0
        row["gv_reach_1e-2"] = gd.robust_reach(th, gv, met.GV_TOL * shrink, gv_fn) / PI
        row["e1_max_B0"] = met.band_max(th, e1, B0)
        row["e2_max_B0"] = met.band_max(th, e2, B0)
        if self.coupled:
            try:
                ec = gen.coupled_error(self.theta_c, self.xi)
                row["ec_max_B0"] = float(np.max(ec))
            except ValueError as exc:
                row["ec_max_B0"] = np.inf
                row["coupled_reason"] = str(exc)
        row["cond_ok"] = bool(row["cond"] <= self.C * (1.0 + BUDGET_SLACK))
        return row


def objective_value(row: dict, target, yardstick, C: float) -> float:
    if not row.get("feasible"):
        return INFEASIBLE
    if row["cond"] > C * (1.0 + BUDGET_SLACK):
        return INFEASIBLE - (row["cond"] / C - 1.0)
    if yardstick is None:
        mg = gd.margins(row, target)
        return float(min(mg.values()))
    v = row[yardstick]
    return float(v) if HIGHER_IS_BETTER[yardstick] else float(-np.log10(max(v, 1e-300)))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def stochastic_polish(ev: Evaluator, x0, target, yardstick, C: float, rng=None,
                      scales=POLISH_SCALES, samples: int = POLISH_SAMPLES):
    """Return a hill-climb of the threshold objective by shrinking random steps.

    The threshold objective is a minimax of yardsticks that jump when the
    crossing lobe of a reach switches, so its local optimum is a vertex where
    several margins tie and where finite-difference gradients are useless
    (the slope of the group-velocity reach scales like 1/eps at such a point).
    Random perturbations at decreasing scales, accepted only when they
    improve the objective, converge to that vertex without derivatives.
    """
    rng = np.random.default_rng(20260913) if rng is None else rng
    x = np.asarray(x0, dtype=float).copy()
    d = x.size
    value = objective_value(ev(x), target, yardstick, C)
    for scale in scales:
        for _ in range(samples):
            trial = x + scale * rng.normal(size=d)
            v = objective_value(ev(trial), target, yardstick, C)
            if v > value:
                x, value = trial, v
    return x, value


def realization_seeds(kinds: str, q: int) -> list:
    """Return the SLC realization embedded in a cell (real pairs at the removal limit)."""
    seeds = []
    if q < 2:
        return seeds
    for spec in REALIZATION_SEEDS:
        slots = list(spec["slots"])
        c = list(spec["c"]) + [0.0] * (q - len(spec["c"]))
        x = []
        # kinds must contain the imaginary pairs of the seed; extra R slots go to the box edge
        needed = [w for _, w in slots]
        extra_R = kinds.count("R")
        if kinds.count("I") != len(needed) or kinds.count("Q") or extra_R != len(kinds) - len(needed):
            continue
        it = iter(sorted(needed))
        for k in kinds:
            x.append(next(it) if k == "I" else np.log(gd.ALPHA_RANGE[1]))
        x += c[:q]
        if in_box(kinds, q, x):
            seeds.append(np.array(x, dtype=float))
    return seeds


REALIZATION_SEEDS = [
    # SLC 1992 D1 as E_{4,2}(0.6366 pi, 0.7321 pi) x P_2, gate8_realization.py
    {"slots": [(0.0, 1.9999625621433885), (0.0, 2.29989575197614)],
     "c": [-0.1994520556764855, -0.2976533590238645]},
]


def _search_task(item: dict) -> dict:
    n, m, kinds, q, C = item["n"], item["m"], item["kinds"], item["q"], item["C"]
    target, yardstick = item["target"], item["yardstick"]
    t0 = time.time()
    coarse = Evaluator(m, kinds, q, C, theta_points=2001, coupled=(yardstick is None),
                       xi=XI_SEARCH[::2], coupled_stride=8)
    seeds = seeds_of(kinds, q)
    extra = realization_seeds(kinds, q) if m == 4 else []
    if extra:
        seeds = np.vstack([np.array(extra), seeds])
    values = np.array([objective_value(coarse(x), target, yardstick, C) for x in seeds])
    order = np.argsort(values)[::-1]
    top = seeds[order[:SEEDS_PER_CELL]]
    fine = Evaluator(m, kinds, q, C, theta_points=SEARCH_THETA, coupled=(yardstick is None))
    f = lambda x: -objective_value(fine(x), target, yardstick, C)
    best = None
    for seed in top:
        res = minimize(f, seed, method="Nelder-Mead",
                       options={"maxiter": NM_MAXITER, "xatol": 1e-7, "fatol": 1e-10})
        for _ in range(NM_RESTARTS):
            again = minimize(f, res.x, method="Nelder-Mead",
                             options={"maxiter": NM_MAXITER // 2, "xatol": 1e-8, "fatol": 1e-11})
            if again.fun < res.fun:
                res = again
        if best is None or res.fun < best.fun:
            best = res
    # polish on the contract grid: Nelder-Mead, then the stochastic hill-climb
    polish = Evaluator(m, kinds, q, C, theta_points=met.THETA_POINTS, coupled=(yardstick is None),
                       xi=XI_FINE, coupled_stride=2)
    fp = lambda x: -objective_value(polish(x), target, yardstick, C)
    res = minimize(fp, best.x, method="Nelder-Mead",
                   options={"maxiter": POLISH_MAXITER, "xatol": 1e-9, "fatol": 1e-12})
    x = res.x if res.fun <= best.fun + 1e-12 else best.x
    x, _ = stochastic_polish(polish, x, target, yardstick, C)
    row = Evaluator(m, kinds, q, C, theta_points=met.THETA_POINTS, coupled=True, xi=XI_FINE,
                    coupled_stride=1, safety=0.0)(x)
    value = objective_value(row, target, yardstick, C)
    row_safe = Evaluator(m, kinds, q, C, theta_points=met.THETA_POINTS, coupled=True, xi=XI_FINE,
                         coupled_stride=2)(x)
    value_safe = objective_value(row_safe, target, yardstick, C)
    slots, poly = unpack(kinds, q, x)
    out = {"n": n, "m": m, "kinds": kinds, "q": q, "smoothness": n - 2 - 2 * q, "C": C,
           "yardstick": yardstick, "target_name": item.get("target_name"),
           "x": [float(v) for v in x],
           "slots": [[float(a), float(w)] for a, w in slots], "polynomial": [float(p) for p in poly],
           "polynomial_scaled": [float(v) for v in x[slot_dimension(kinds):]],
           "polynomial_roots": g8.poly_roots_xi(poly),
           "value": float(value), "value_at_safety": float(value_safe), "row": row,
           "seeds": int(seeds.shape[0]), "seed_best": float(values[order[0]]),
           "seconds": time.time() - t0}
    if target is not None and row.get("feasible"):
        out["margins"] = gd.margins(row, target)
    return out


def full_metrics(gen: g8.PolynomialGenerator, name: str) -> dict:
    row = met.evaluate(name, lambda t: gen.symbols(t)[1], lambda t: gen.symbols(t)[2],
                       lambda t: gen.symbols(t)[0], analytic_gv=lambda t: gen.symbol_derivative(t),
                       coupled=lambda t: gen.coupled_error(t, XI_FULL),
                       algebraic_order1=gen.m, algebraic_order2=gen.m - 2,
                       bandwidth_left=2 * gen.nmax - 1, bandwidth_d1=2 * gen.nmax - 2,
                       bandwidth_d2=2 * gen.nmax - 1, solves=1)
    fine = met.theta_grid(200001)
    b, k1, k2 = gen.symbols(fine)
    e1, e2 = k1 / fine - 1.0, k2 / fine ** 2 - 1.0
    row["fine_e1_max_B0"] = met.band_max(fine, e1, B0)
    row["fine_e2_max_B0"] = met.band_max(fine, e2, B0)
    for tol in met.REACH_TOLS:
        row[f"fine_reach_d1_{tol:g}"] = met.reach(fine, e1, tol, err_fn=lambda t: gen.symbols(t)[1] / t - 1.0)["refined"] / PI
    gv = gen.symbol_derivative(fine) - 1.0
    row["fine_gv_reach_1e-2"] = met.reach(fine, gv, met.GV_TOL)["linear"] / PI
    row["fine_cond"] = float(b.max() / b.min())
    row["fine_max_b"] = float(b.max())
    row["fine_min_b"] = float(b.min())
    zeros = fine[np.flatnonzero(np.sign(e1[:-1]) * np.sign(e1[1:]) < 0)]
    row["e1_zeros_over_pi"] = [float(z / PI) for z in zeros]
    row["kappa2_min_over_grid"] = float(k2[1:].min())
    return row


def verify_member(gen: g8.PolynomialGenerator, name: str) -> dict:
    return {"label": gen.label(), "metrics": full_metrics(gen, name),
            "routes": g8.verify_routes(gen), "positivity": g8.certify_positivity(gen),
            "sufficient_conditions": g8.sufficient_conditions(gen),
            "lemma_D_residual": g7_lemma_residual(gen),
            "record": gen.member_record(name)}


def g7_lemma_residual(gen: g8.PolynomialGenerator) -> float:
    import gate7_family as g7
    return g7.lemma_D_residual(gen.base, np.linspace(0.05, PI, 200))


def _verify_task(item):
    m, slots, poly, name = item
    return verify_member(g8.PolynomialGenerator(m, slots, poly), name)


def dominates(rec: dict, ver: dict) -> bool:
    vm = ver["metrics"]
    ec_ratio = vm["ec_max_B0"] / (COUPLED_FACTOR * max(vm["e1_max_B0"], vm["e2_max_B0"]))
    pos = ver["positivity"]
    return bool(rec["value"] >= 0.0 and ec_ratio <= 1.0 and pos["b_certified"] and pos["kappa2b_certified"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(processes: int = 10, quick: bool = False, widths=None, budget_names=None,
          cells_filter=None) -> dict:
    targets = gd.comparator_targets()
    slc_C = targets["SLC1992"]["cond"]
    q8_C = targets["TunedQ8"]["cond"]
    bmc11_C = comp.ExponentialGenerator(12, ()).cond()["cond"]
    budgets = {"C_q8": q8_C, "C_SLC1": slc_C, "C_BMC11": bmc11_C}
    if budget_names is None:
        budget_names = ["C_SLC1", "C_BMC11"]
    budgets = {k: v for k, v in budgets.items() if k in budget_names}
    widths = tuple(widths) if widths else ((8,) if quick else (10, 8))
    tasks = []
    for n in widths:
        cells = CELLS[n][:2] if quick else CELLS[n]
        if cells_filter:
            cells = [c for c in cells if f"{c[0]}{c[1]}q{c[2]}" in cells_filter]
        for m, kinds, q in cells:
            for bname, C in budgets.items():
                for tname in ("SLC1992", "TunedQ8"):
                    tasks.append({"n": n, "m": m, "kinds": kinds, "q": q, "C": C, "budget": bname,
                                  "target": targets[tname], "target_name": tname, "yardstick": None})
                for y in YARDSTICKS:
                    tasks.append({"n": n, "m": m, "kinds": kinds, "q": q, "C": C, "budget": bname,
                                  "target": None, "target_name": None, "yardstick": y})
    with Pool(processes) as pool:
        results = pool.map(_search_task, tasks, chunksize=1)
    for task, res in zip(tasks, results):
        res["budget"] = task["budget"]
    summary = {}
    for n in widths:
        for bname in budgets:
            for restrict in ("", "|C4"):
                def pick(cands):
                    if restrict:
                        cands = [r for r in cands if r["smoothness"] >= 4]
                    return max(cands, key=lambda r: r["value"]) if cands else None
                for tname in ("SLC1992", "TunedQ8"):
                    cand = [r for r in results if r["n"] == n and r["budget"] == bname
                            and r["yardstick"] is None and r["target_name"] == tname]
                    best = pick(cand)
                    if best is not None:
                        summary[f"{n}|{bname}|{tname}{restrict}"] = best
                for y in YARDSTICKS:
                    cand = [r for r in results if r["n"] == n and r["budget"] == bname and r["yardstick"] == y]
                    best = pick(cand)
                    if best is not None:
                        summary[f"{n}|{bname}|{y}{restrict}"] = best
    items, keys = [], []
    for key, rec in summary.items():
        items.append((rec["m"], [tuple(s) for s in rec["slots"]], rec["polynomial"],
                      key.replace("|", "_")))
        keys.append(key)
    with Pool(processes) as pool:
        verified = pool.map(_verify_task, items, chunksize=1)
    verification = dict(zip(keys, verified))
    return {"targets": targets, "budgets": budgets, "widths": list(widths),
            "results": results, "summary": summary, "verification": verification,
            "settings": {"OMEGA_RANGE": list(gd.OMEGA_RANGE), "ALPHA_RANGE": list(gd.ALPHA_RANGE),
                         "C_BOX": {str(k): list(v) for k, v in C_BOX.items()},
                         "SOBOL_POWER": SOBOL_POWER, "SLOT_GRID_POINTS": SLOT_GRID_POINTS,
                         "SEEDS_PER_CELL": SEEDS_PER_CELL, "NM_MAXITER": NM_MAXITER,
                         "NM_RESTARTS": NM_RESTARTS, "POLISH_MAXITER": POLISH_MAXITER,
                         "POLISH_SCALES": list(POLISH_SCALES), "POLISH_SAMPLES": POLISH_SAMPLES,
                         "SEARCH_THETA": SEARCH_THETA, "COUPLED_FACTOR": COUPLED_FACTOR,
                         "REACH_SAFETY": REACH_SAFETY}}


def merge(parts: list) -> dict:
    data = {"targets": parts[0]["targets"], "budgets": {}, "widths": [], "results": [],
            "summary": {}, "verification": {}, "settings": parts[0]["settings"], "merged_from": len(parts)}
    for part in parts:
        data["budgets"].update(part["budgets"])
        data["widths"] += [w for w in part["widths"] if w not in data["widths"]]
        data["results"] += part["results"]
        data["summary"].update(part["summary"])
        data["verification"].update(part["verification"])
    data["widths"] = sorted(data["widths"], reverse=True)
    order = ["C_q8", "C_SLC1", "C_BMC11"]
    data["budgets"] = {k: data["budgets"][k] for k in order if k in data["budgets"]}
    return data


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(v, digits=4):
    return gd._fmt(v, digits)


def _member_label(rec) -> str:
    parts = []
    for a, w in rec["slots"]:
        if a == 0.0:
            parts.append(f"i {w / PI:.4f}pi")
        elif w == 0.0:
            parts.append(f"r {a:.4f}")
        else:
            parts.append(f"q({a:.4f}, {w / PI:.4f}pi)")
    c = ", ".join(f"{v:.4f}" for v in rec["polynomial_scaled"])
    return f"m={rec['m']} " + ", ".join(parts) + f"; c=[{c}]"


def _roots_label(rec) -> str:
    roots = rec["polynomial_roots"]
    parts = [f"soft {t / PI:.3f}pi" for t in roots["soft_contacts_theta"]]
    parts += [f"real xi {x / PI:.2f}pi" for x in roots["real_positive_xi"] if x >= 2.0 * PI]
    parts += [f"neg t {t:.2f}" for t in roots["negative_t"]]
    parts += [f"cplx {re:.2f}{im:+.2f}i" for re, im in roots["complex_t"]]
    return "; ".join(parts) if parts else "P = 1"


def _gate7_summary():
    path = GATE7 / "dominance.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def markdown(data: dict, ceiling: dict | None, gate7: dict | None) -> str:
    lines = ["# Gate 8 dominance search over the augmented family E_{m,r} x P_q", "",
             "Produced by code/analysis/exponential_family/gate8_dominance.py under",
             "results/exponential_family/gate8/contract.md. Reaches in multiples of pi;",
             "every number from the Gate 0 metrics code; verification blocks in dominance.json.",
             "c = [c_1, ...] are the scaled polynomial coefficients, P(theta^2) = 1 + sum_l c_l (theta/pi)^{2l};",
             "the roots column lists the soft contacts theta_0 = 2 pi - xi_0 of real roots xi_0 in (pi, 2 pi),",
             "real roots beyond 2 pi, negative roots in t, and complex roots in t = xi^2.",
             "A dominating member must have smallest margin >= 0, ec/2max <= 1, and certified b and kappa_2.", ""]
    t = data["targets"]
    lines += ["Targets", "", "| comparator | cond | R1(1e-2) | R1(1e-3) | R1(1e-4) | G(1e-2) | E1max B0 | E2max B0 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in t.items():
        lines.append(f"| {name} | {row['cond']:.3f} | " + " | ".join(_fmt(row[k]) for k in YARDSTICKS) + " |")
    lines.append("")
    for n in data["widths"]:
        lines += [f"## Bandwidth {n - 1} (n = {n} exponents, degree {n - 1}, q <= {(n - 4) // 2} for C^2)", ""]
        for bname, C in data["budgets"].items():
            if f"{n}|{bname}|SLC1992" not in data["summary"]:
                continue
            lines += [f"### Budget {bname} = {C:.3f}", ""]
            for restrict, title in (("", "all cells (C^2 and smoother)"), ("|C4", "C^4 restriction (q <= (n - 6)/2)")):
                lines += [f"Dominance candidates, {title} (largest smallest-margin member)", "",
                          "| against | member | roots | C^k | cond | smallest margin | binding | Gate 7 margin | R1(1e-2) | R1(1e-3) | R1(1e-4) | G | E1max | E2max | ec/2max | b, k2 certified | verdict |",
                          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
                for tname in ("SLC1992", "TunedQ8"):
                    key = f"{n}|{bname}|{tname}{restrict}"
                    if key not in data["summary"]:
                        continue
                    rec = data["summary"][key]
                    ver = data["verification"][key]
                    vm = ver["metrics"]
                    mg = rec.get("margins", {})
                    binding = min(mg, key=mg.get) if mg else "infeasible"
                    ec_ratio = vm["ec_max_B0"] / (COUPLED_FACTOR * max(vm["e1_max_B0"], vm["e2_max_B0"]))
                    pos = ver["positivity"]
                    cert = f"{pos['b_certified']}, {pos['kappa2b_certified']}"
                    verdict = "DOMINATES" if dominates(rec, ver) else "does not dominate"
                    g7m = "n/a"
                    if gate7 is not None and f"{n}|{bname}|{tname}" in gate7["summary"]:
                        g7m = f"{gate7['summary'][f'{n}|{bname}|{tname}']['value']:+.4f}"
                    lines.append(f"| {tname} | {_member_label(rec)} | {_roots_label(rec)} | {rec['smoothness']} | {vm['cond']:.2f} | {rec['value']:+.4f} | {binding} | {g7m} | "
                                 + " | ".join(_fmt(vm[k + '_refined'] if k.startswith('reach') else vm[k]) for k in YARDSTICKS)
                                 + f" | {ec_ratio:.3f} | {cert} | {verdict} |")
                lines.append("")
            lines += ["Per-yardstick family ceiling (each yardstick maximized alone, same constraints, all cells)", "",
                      "| yardstick | family best | C^4 best | Gate 7 best | member | roots | cond | SLC | q8 | compact ceiling at (L, R) = (n/2 - 1, n/2 - 1) |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for y in YARDSTICKS:
                rec = data["summary"][f"{n}|{bname}|{y}"]
                ver = data["verification"][f"{n}|{bname}|{y}"]["metrics"]
                key = y + "_refined" if y.startswith("reach") else y
                c4 = "n/a"
                if f"{n}|{bname}|{y}|C4" in data["verification"]:
                    c4 = _fmt(data["verification"][f"{n}|{bname}|{y}|C4"]["metrics"][key])
                g7v = "n/a"
                if gate7 is not None and f"{n}|{bname}|{y}" in gate7["verification"]:
                    g7v = _fmt(gate7["verification"][f"{n}|{bname}|{y}"]["metrics"][key])
                ceil = "n/a"
                if ceiling is not None:
                    L = R = n // 2 - 1
                    if y.startswith("reach"):
                        tol = float(y.split("_")[-1])
                        for c in ceiling["records"]:
                            if c["kind"] == "reach" and c["L"] == L and c["R"] == R and abs(c["C"] - C) < 1e-6 and c["tol"] == tol:
                                ceil = f"[{c['reach_lower']:.4f}, {c['reach_upper']:.4f}]"
                    elif y in ("e1_max_B0", "e2_max_B0"):
                        d = 1 if y == "e1_max_B0" else 2
                        for c in ceiling["records"]:
                            if c["kind"] == "minimax" and c["L"] == L and c["R"] == R and abs(c["C"] - C) < 1e-6 and c["derivative"] == d:
                                lo = "none" if c["lower_bound"] is None else f"{c['lower_bound']:.3e}"
                                ceil = f"[{lo}, {c['achieved']:.3e}]"
                lines.append(f"| {y} | {_fmt(ver[key])} | {c4} | {g7v} | {_member_label(rec)} | {_roots_label(rec)} | {ver['cond']:.2f} | "
                             f"{_fmt(t['SLC1992'][y])} | {_fmt(t['TunedQ8'][y])} | {ceil} |")
            lines.append("")
    lines += ["## Verification of every reported member", "",
              "| key | member | decimal dev | residue dev | symbols vs alias | endpoint samples | b certified | kappa2 certified | Theorem P'' | Theorem N'' | max b | fine E1max | fine R1(1e-3) | fine G | fine cond |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, ver in data["verification"].items():
        m = ver["metrics"]
        r = ver["routes"]
        p = ver["positivity"]
        s = ver["sufficient_conditions"]
        lines.append(f"| {key} | {ver['label']} | {r['decimal_deviation']:.2e} | {r['residue_deviation']:.2e} | {r['symbols_from_samples_versus_alias']:.2e} | {r['endpoint_samples']:.1e} | "
                     f"{p['b_certified']} | {p['kappa2b_certified']} | {s['theorem_P']} | {s['theorem_N']} | {m['fine_max_b']:.4f} | "
                     f"{m['fine_e1_max_B0']:.3e} | {m['fine_reach_d1_0.001']:.4f} | {m['fine_gv_reach_1e-2']:.4f} | {m['fine_cond']:.3f} |")
    return "\n".join(lines) + "\n"


def write_members(data: dict) -> list:
    dominating = []
    for key, rec in data["summary"].items():
        if rec["yardstick"] is None and dominates(rec, data["verification"][key]):
            dominating.append(key)
            MEMBERS.mkdir(parents=True, exist_ok=True)
            name = "X8_" + key.replace("|", "_")
            record = dict(data["verification"][key]["record"])
            record["name"] = name
            record["dominates"] = rec["target_name"]
            record["budget"] = rec["budget"]
            record["margins"] = rec.get("margins")
            record["verification"] = {"routes": {k: v for k, v in data["verification"][key]["routes"].items() if k != "beta_digits"},
                                      "positivity": data["verification"][key]["positivity"]}
            (MEMBERS / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return dominating


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--widths", type=str, default=None)
    parser.add_argument("--budgets", type=str, default=None)
    parser.add_argument("--cells", type=str, default=None, help="comma separated cell ids like 4IIq2")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--merge", type=str, default=None)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ceiling = json.loads((GATE7 / "ceiling.json").read_text()) if (GATE7 / "ceiling.json").exists() else None
    gate7 = _gate7_summary()
    if args.merge is not None:
        data = merge([json.loads((OUT / f"dominance{sfx}.json").read_text()) for sfx in args.merge.split(",")])
        (OUT / "dominance.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
        text = markdown(data, ceiling, gate7)
        (OUT / "dominance.md").write_text(text)
        print(text)
        print("dominating members:", write_members(data))
        return 0
    widths = [int(w) for w in args.widths.split(",")] if args.widths else None
    budget_names = args.budgets.split(",") if args.budgets else None
    cells = args.cells.split(",") if args.cells else None
    data = build(args.processes, quick=args.quick, widths=widths, budget_names=budget_names, cells_filter=cells)
    suffix = args.suffix if args.suffix is not None else ("_quick" if args.quick else "")
    (OUT / f"dominance{suffix}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    text = markdown(data, ceiling, gate7)
    (OUT / f"dominance{suffix}.md").write_text(text)
    print(text)
    if suffix == "":
        print("dominating members:", write_members(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
