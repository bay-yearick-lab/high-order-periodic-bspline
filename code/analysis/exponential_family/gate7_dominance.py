"""Gate 7, Question 3: dominance search over the family with real parts.

The search space at bandwidth n (n = 8, 10 or 12 exponents, degree p = n - 1) is
the union of cells (m, slot kinds): even polynomial core m >= 4 (m = 2 leaves
the direct second derivative inconsistent, Proposition O), and q = (n - m)/2
pair slots each of which is an imaginary pair (parameter omega in (0, pi)), a
real pair (parameter alpha > 0), or half of a quad (two slots, parameters
alpha > 0 and omega in (0, pi)). Every cell has (n - m)/2 continuous real
parameters. Constraints: left symbol positive with max/min <= C (Lemma F gives
max b = 1 for every member searched, omega < pi), kappa_2 > 0 on (0, pi].

Yardsticks (Gate 0 metrics code, metrics.py, on its 20,001-point grid; the
reaches inside the search refine every interior lobe of the error exactly so
that the threshold objective cannot exploit the grid, see robust_reach):

    R1(1e-2), R1(1e-3), R1(1e-4)   first-derivative reaches (refined crossing),
    G(1e-2)                        group-velocity reach,
    E1max(B0), E2max(B0)           band maxima on (0, 2 pi/3],
    ec_max(B0) <= 2 max(E1max, E2max)   coupled off-grid channel.

Dominance of a comparator means every reach and G at least the comparator's,
every band maximum at most the comparator's, and the coupled condition. The
signed margin of a yardstick is the relative excess, (value - target)/target
for reaches and 1 - value/target for maxima, and the search maximizes the
smallest margin (a threshold objective) by Nelder-Mead from the best seeds
of a dense grid over each cell; a member with smallest margin >= 0 dominates.
Inside the search every reach is taken at the tolerance times (1 - REACH_SAFETY)
so that the returned member keeps every interior lobe of the error strictly
below the tolerance; the reach is a discontinuous function of the parameters
whose supremum is approached by members with a lobe touching the tolerance,
and the reported value is the reach at the true tolerance of a member with
that margin. The coupled channel is evaluated on the polished member (33 xi points in the
search record, 129 in the verification block) and enters the verdict, not the
inner Nelder-Mead loop, because its margin is never the binding one.
Independently, each yardstick is maximized alone under the same constraints
to give the family's per-yardstick ceiling, which is placed against the
compact ceiling of Question 1 and against the comparators.

Every reported member is re-evaluated with metrics.evaluate on the contract
grid, its band maxima and reaches confirmed on 200,001 angles, its samples
verified by the 60-digit decimal route to 1e-13, and its positivity certified
(gate7_family.certify_positivity).
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
from scipy.optimize import brentq, minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import gate7_family as g7  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate7"
MEMBERS = OUT / "members"

B0 = 2.0 * np.pi / 3.0
XI_COARSE = np.linspace(-0.5, 0.5, 33)
XI_FULL = np.linspace(-0.5, 0.5, 129)
OMEGA_RANGE = (0.05 * np.pi, 0.985 * np.pi)
ALPHA_RANGE = (0.2, 12.0)
GRID_POINTS = {1: 400, 2: 60, 3: 30, 4: 14}
SEEDS_PER_CELL = 12
NM_MAXITER = 600
INFEASIBLE = -10.0
BUDGET_SLACK = 1e-9
YARDSTICKS = ("reach_d1_0.01", "reach_d1_0.001", "reach_d1_0.0001", "gv_reach_1e-2",
              "e1_max_B0", "e2_max_B0")
HIGHER_IS_BETTER = {"reach_d1_0.01": True, "reach_d1_0.001": True, "reach_d1_0.0001": True,
                    "gv_reach_1e-2": True, "e1_max_B0": False, "e2_max_B0": False}
COUPLED_FACTOR = 2.0
LOBE_FRACTION = 0.8
REACH_SAFETY = 1e-3

CELLS = {
    12: [(4, "IIII"), (4, "IIIR"), (4, "IIRR"), (4, "IRRR"), (4, "RRRR"),
         (4, "QII"), (4, "QIR"), (4, "QRR"), (4, "QQ"),
         (6, "III"), (6, "IIR"), (6, "IRR"), (6, "RRR"), (6, "QI"), (6, "QR"),
         (8, "II"), (8, "IR"), (8, "RR"), (8, "Q"), (10, "I"), (10, "R")],
    10: [(4, "III"), (4, "IIR"), (4, "IRR"), (4, "RRR"), (4, "QI"), (4, "QR"),
         (6, "II"), (6, "IR"), (6, "RR"), (6, "Q"), (8, "I"), (8, "R")],
    8: [(4, "II"), (4, "IR"), (4, "RR"), (4, "Q"), (6, "I"), (6, "R")],
}


# ---------------------------------------------------------------------------
# Parametrization of a cell
# ---------------------------------------------------------------------------

def cell_dimension(kinds: str) -> int:
    return sum(2 if k == "Q" else 1 for k in kinds)


def unpack(kinds: str, x) -> list:
    """Return the slots of a parameter vector: omegas for I, alphas for R,
    (alpha, omega) for Q. Alphas are carried as log(alpha)."""
    x = list(x)
    slots = []
    i = 0
    for k in kinds:
        if k == "I":
            slots.append((0.0, x[i]))
            i += 1
        elif k == "R":
            slots.append((np.exp(x[i]), 0.0))
            i += 1
        else:
            slots.append((np.exp(x[i]), x[i + 1]))
            i += 2
    return slots


def in_box(kinds: str, x) -> bool:
    i = 0
    la, lb = np.log(ALPHA_RANGE[0]), np.log(ALPHA_RANGE[1])
    for k in kinds:
        if k == "I":
            if not OMEGA_RANGE[0] <= x[i] <= OMEGA_RANGE[1]:
                return False
            i += 1
        elif k == "R":
            if not la <= x[i] <= lb:
                return False
            i += 1
        else:
            if not (la <= x[i] <= lb and OMEGA_RANGE[0] <= x[i + 1] <= OMEGA_RANGE[1]):
                return False
            i += 2
    return True


def grid_of(kinds: str) -> np.ndarray:
    """Return the seed grid of a cell, with ordered duplicates removed."""
    d = cell_dimension(kinds)
    pts = GRID_POINTS[d]
    axes = []
    for k in kinds:
        if k == "I":
            axes.append(np.linspace(OMEGA_RANGE[0], OMEGA_RANGE[1], pts))
        elif k == "R":
            axes.append(np.linspace(np.log(ALPHA_RANGE[0]), np.log(ALPHA_RANGE[1]), pts))
        else:
            axes.append(np.linspace(np.log(ALPHA_RANGE[0]), np.log(ALPHA_RANGE[1]), pts))
            axes.append(np.linspace(OMEGA_RANGE[0], OMEGA_RANGE[1], pts))
    grid = np.array(list(itertools.product(*axes)))
    # remove permutations of identical kinds
    keep = np.ones(grid.shape[0], dtype=bool)
    i = 0
    for k, group in itertools.groupby(kinds):
        count = len(list(group))
        if k in ("I", "R") and count > 1:
            block = grid[:, i:i + count]
            keep &= np.all(np.diff(block, axis=1) > 0.0, axis=1)
        i += count if k in ("I", "R") else 2 * count
    return grid[keep]


# ---------------------------------------------------------------------------
# Grid-independent reach
# ---------------------------------------------------------------------------

def robust_reach(theta: np.ndarray, err: np.ndarray, tol: float, err_fn) -> float:
    """Return the first crossing of |e| = tol with every interior lobe refined.

    The grid value of metrics.reach is the first grid crossing; an optimizer
    driving a threshold objective parks interior lobes of |e| exactly at the
    tolerance on the grid, where the continuous lobe maximum lies above it
    between the grid points. Every local maximum of |e| on the grid before the
    grid crossing whose value exceeds LOBE_FRACTION tol is therefore refined
    by a bounded scalar search on its bracketing interval; if the refined
    maximum exceeds tol, the reach is the root of |e| - tol on the rising side
    of that lobe. Otherwise the grid bracket is refined by Brent's method as
    in metrics.reach.
    """
    mag = np.abs(err)
    f = lambda t: abs(float(np.atleast_1d(err_fn(np.array([t])))[0]))
    bad = np.flatnonzero(mag > tol)
    j = int(bad[0]) if bad.size else mag.size
    if j == 0:
        return 0.0
    interior = np.flatnonzero((mag[1:j - 1] >= mag[:j - 2]) & (mag[1:j - 1] >= mag[2:j])
                              & (mag[1:j - 1] > LOBE_FRACTION * tol)) + 1
    for i in interior:
        lo, hi = float(theta[i - 1]), float(theta[i + 1])
        res = minimize_scalar(lambda t: -f(t), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-12})
        if -res.fun > tol:
            left = i - 1
            while left > 0 and mag[left] >= tol:
                left -= 1
            return float(brentq(lambda t: f(t) - tol, float(theta[left]), float(res.x),
                                xtol=1e-14, maxiter=200))
    if j >= mag.size:
        return float(theta[-1])
    t0, t1 = float(theta[j - 1]), float(theta[j])
    if f(t0) < tol < f(t1):
        return float(brentq(lambda t: f(t) - tol, t0, t1, xtol=1e-14, maxiter=200))
    e0, e1 = float(mag[j - 1]), float(mag[j])
    return t0 + (tol - e0) * (t1 - t0) / (e1 - e0)


# ---------------------------------------------------------------------------
# Fast evaluation of the nodal yardsticks
# ---------------------------------------------------------------------------

class Evaluator:
    """Nodal yardsticks of one cell on a theta grid, with the constraints."""

    def __init__(self, m: int, kinds: str, C: float, theta_points: int = met.THETA_POINTS,
                 coupled: bool = False, safety: float = REACH_SAFETY):
        self.m, self.kinds, self.C = m, kinds, C
        self.safety = safety
        self.theta = met.theta_grid(theta_points)
        self.cond_theta = np.linspace(0.0, np.pi, theta_points)
        self.coupled = coupled
        self.mask_B0 = met.band_mask(self.theta, B0)

    def generator(self, x):
        return g7.ExtendedGenerator(self.m, unpack(self.kinds, x))

    def __call__(self, x, refine: bool = False) -> dict:
        if not in_box(self.kinds, x):
            return {"feasible": False, "reason": "outside box"}
        gen = self.generator(x)
        try:
            b, k1, k2 = gen.symbols(self.theta)
            bc = gen.symbols(self.cond_theta)[0]
        except ValueError as exc:
            return {"feasible": False, "reason": str(exc)}
        row = {"feasible": True}
        if not np.all(np.isfinite(bc)) or bc.min() <= 0.0:
            row.update({"feasible": False, "reason": "b not positive", "cond": np.inf})
            return row
        row["cond"] = float(bc.max() / bc.min())
        row["max_b"] = float(bc.max())
        row["kappa2_min"] = float(k2.min())
        if k2.min() <= 0.0:
            row.update({"feasible": False, "reason": "kappa_2 not positive"})
            return row
        th = self.theta
        e1 = k1 / th - 1.0
        e2 = k2 / th ** 2 - 1.0
        e1_fn = lambda t: gen.symbols(t)[1] / t - 1.0
        shrink = 1.0 - self.safety
        for tol in met.REACH_TOLS:
            row[f"reach_d1_{tol:g}"] = robust_reach(th, e1, tol * shrink, e1_fn) / np.pi
        gv = gen.symbol_derivative(th) - 1.0
        gv_fn = lambda t: gen.symbol_derivative(t) - 1.0
        row["gv_reach_1e-2"] = robust_reach(th, gv, met.GV_TOL * shrink, gv_fn) / np.pi
        row["e1_max_B0"] = met.band_max(th, e1, B0)
        row["e2_max_B0"] = met.band_max(th, e2, B0)
        if self.coupled:
            ec = gen.coupled_error(th[self.mask_B0], XI_COARSE)
            row["ec_max_B0"] = float(ec.max())
        row["cond_ok"] = bool(row["cond"] <= self.C * (1.0 + BUDGET_SLACK))
        return row


def margins(row: dict, target: dict) -> dict:
    out = {}
    for key in YARDSTICKS:
        v, t = row[key], target[key]
        out[key] = (v - t) / t if HIGHER_IS_BETTER[key] else 1.0 - v / t
    if "ec_max_B0" in row:
        out["coupled"] = 1.0 - row["ec_max_B0"] / (COUPLED_FACTOR * max(row["e1_max_B0"], row["e2_max_B0"]))
    return out


def objective_value(row: dict, target: dict | None, yardstick: str | None, C: float) -> float:
    """Return the scalar to maximize: the smallest margin, or one yardstick."""
    if not row.get("feasible"):
        return INFEASIBLE
    if row["cond"] > C * (1.0 + BUDGET_SLACK):
        return INFEASIBLE - (row["cond"] / C - 1.0)
    if yardstick is None:
        return float(min(margins(row, target).values()))
    v = row[yardstick]
    return float(v) if HIGHER_IS_BETTER[yardstick] else float(-np.log10(max(v, 1e-300)))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search_task(item: dict) -> dict:
    """Grid scan then Nelder-Mead polish for one (n, cell, budget, objective)."""
    n, m, kinds, C, target, yardstick = (item["n"], item["m"], item["kinds"], item["C"],
                                          item["target"], item["yardstick"])
    t0 = time.time()
    coarse = Evaluator(m, kinds, C, theta_points=4001, coupled=False)
    grid = grid_of(kinds)
    values = np.array([objective_value(coarse(x), target, yardstick, C) for x in grid])
    order = np.argsort(values)[::-1]
    seeds = grid[order[:SEEDS_PER_CELL]]
    fine = Evaluator(m, kinds, C, coupled=False)
    best = None
    d = cell_dimension(kinds)
    for seed in seeds:
        f = lambda x: -objective_value(fine(x), target, yardstick, C)
        res = minimize(f, seed, method="Nelder-Mead",
                       options={"maxiter": NM_MAXITER, "xatol": 1e-7, "fatol": 1e-10,
                                "initial_simplex": None})
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    row = Evaluator(m, kinds, C, coupled=True, safety=0.0)(x, refine=True)
    value = objective_value(row, target, yardstick, C)
    row_safe = Evaluator(m, kinds, C, coupled=False)(x)
    value_safe = objective_value(row_safe, target, yardstick, C)
    out = {"n": n, "m": m, "kinds": kinds, "C": C, "yardstick": yardstick,
           "target_name": item.get("target_name"), "x": [float(v) for v in x],
           "slots": [[float(a), float(w)] for a, w in unpack(kinds, x)],
           "value": float(value), "value_at_safety": float(value_safe), "row": row,
           "grid_points": int(grid.shape[0]),
           "grid_best": float(values[order[0]]), "seconds": time.time() - t0}
    if target is not None and row.get("feasible"):
        out["margins"] = margins(row, target)
    return out


def comparator_targets() -> dict:
    """Return the yardsticks of SLC and tuned q8 from the frozen metrics code."""
    out = {}
    for scheme in comp.comparator_schemes():
        if scheme.name not in ("SLC1992", "TunedQ8"):
            continue
        row = met.evaluate(scheme.name, scheme.kappa1, scheme.kappa2, scheme.left,
                           analytic_gv=scheme.analytic_gv)
        out[scheme.name] = {key: float(row[key if not key.startswith("reach") else key + "_refined"])
                            for key in YARDSTICKS}
        out[scheme.name]["cond"] = float(row["cond"])
    return out


def full_metrics(gen: g7.ExtendedGenerator, name: str) -> dict:
    """Return the complete Gate 0 metric row of a member (129 xi, analytic gv)."""
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
        row[f"fine_reach_d1_{tol:g}"] = met.reach(fine, e1, tol, err_fn=lambda t: gen.symbols(t)[1] / t - 1.0)["refined"] / np.pi
    row["fine_cond"] = float(b.max() / b.min())
    zeros = fine[np.flatnonzero(np.sign(e1[:-1]) * np.sign(e1[1:]) < 0)]
    row["e1_zeros_over_pi"] = [float(z / np.pi) for z in zeros]
    return row


def verify_member(gen: g7.ExtendedGenerator, name: str) -> dict:
    """Return the complete verification block of a member."""
    return {"label": gen.label(), "metrics": full_metrics(gen, name),
            "routes": g7.verify_routes(gen), "positivity": g7.certify_positivity(gen),
            "sufficient_conditions": g7.sufficient_conditions(gen),
            "lemma_D_residual": g7.lemma_D_residual(gen, np.linspace(0.05, np.pi, 200)),
            "record": gen.member_record(name)}


def _verify_task(item):
    m, slots, name = item
    return verify_member(g7.ExtendedGenerator(m, slots), name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(processes: int = 12, quick: bool = False, widths=None, budget_names=None) -> dict:
    targets = comparator_targets()
    slc_C = targets["SLC1992"]["cond"]
    q8_C = targets["TunedQ8"]["cond"]
    bmc11_C = comp.ExponentialGenerator(12, ()).cond()["cond"]
    budgets = {"C_q8": q8_C, "C_SLC1": slc_C, "C_BMC11": bmc11_C}
    if budget_names:
        budgets = {k: v for k, v in budgets.items() if k in budget_names}
    tasks = []
    if widths is None:
        widths = (10,) if quick else (10, 8)
    widths = tuple(widths)
    for n in widths:
        cells = CELLS[n][:2] if quick else CELLS[n]
        for m, kinds in cells:
            for bname, C in budgets.items():
                for tname in ("SLC1992", "TunedQ8"):
                    tasks.append({"n": n, "m": m, "kinds": kinds, "C": C, "budget": bname,
                                  "target": targets[tname], "target_name": tname, "yardstick": None})
                for y in YARDSTICKS:
                    tasks.append({"n": n, "m": m, "kinds": kinds, "C": C, "budget": bname,
                                  "target": None, "target_name": None, "yardstick": y})
    with Pool(processes) as pool:
        results = pool.map(_search_task, tasks, chunksize=1)
    for task, res in zip(tasks, results):
        res["budget"] = task["budget"]
    # best dominance candidate per (n, budget, comparator) and per-yardstick family ceilings
    summary = {}
    for n in widths:
        for bname in budgets:
            for tname in ("SLC1992", "TunedQ8"):
                cand = [r for r in results if r["n"] == n and r["budget"] == bname
                        and r["yardstick"] is None and r["target_name"] == tname]
                best = max(cand, key=lambda r: r["value"])
                summary[f"{n}|{bname}|{tname}"] = best
            for y in YARDSTICKS:
                cand = [r for r in results if r["n"] == n and r["budget"] == bname and r["yardstick"] == y]
                best = max(cand, key=lambda r: r["value"])
                summary[f"{n}|{bname}|{y}"] = best
    # verification of every summary member
    items, keys = [], []
    for key, rec in summary.items():
        items.append((rec["m"], [tuple(s) for s in rec["slots"]], key.replace("|", "_")))
        keys.append(key)
    with Pool(processes) as pool:
        verified = pool.map(_verify_task, items, chunksize=1)
    verification = dict(zip(keys, verified))
    return {"targets": targets, "budgets": budgets, "widths": list(widths),
            "results": results, "summary": summary, "verification": verification,
            "settings": {"OMEGA_RANGE": list(OMEGA_RANGE), "ALPHA_RANGE": list(ALPHA_RANGE),
                         "GRID_POINTS": GRID_POINTS, "SEEDS_PER_CELL": SEEDS_PER_CELL,
                         "NM_MAXITER": NM_MAXITER, "COUPLED_FACTOR": COUPLED_FACTOR}}


def merge(parts: list) -> dict:
    """Return the union of several dominance receipts (widths sorted descending)."""
    data = {"targets": parts[0]["targets"], "budgets": {}, "widths": [], "results": [],
            "summary": {}, "verification": {}, "settings": parts[0]["settings"], "merged_from": len(parts)}
    for part in parts:
        data.setdefault("notes", []).extend(part.get("notes", []))
        data["budgets"].update(part["budgets"])
        data["widths"] += [w for w in part["widths"] if w not in data["widths"]]
        data["results"] += part["results"]
        data["summary"].update(part["summary"])
        data["verification"].update(part["verification"])
    data["widths"] = sorted(data["widths"], reverse=True)
    order = ["C_q8", "C_SLC1", "C_BMC11"]
    data["budgets"] = {k: data["budgets"][k] for k in order if k in data["budgets"]}
    return data


def _fmt(v, digits=4):
    if v is None:
        return "none"
    if isinstance(v, float) and (abs(v) < 1e-3 or abs(v) >= 1e4):
        return f"{v:.3e}"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _slots_label(rec):
    parts = []
    for a, w in rec["slots"]:
        if a == 0.0:
            parts.append(f"i {w / np.pi:.4f}pi")
        elif w == 0.0:
            parts.append(f"r {a:.4f}")
        else:
            parts.append(f"q({a:.4f}, {w / np.pi:.4f}pi)")
    return f"m={rec['m']} " + ", ".join(parts)


def markdown(data: dict, ceiling: dict | None) -> str:
    lines = ["# Gate 7 dominance search over the family with real parts", "",
             "Produced by code/analysis/exponential_family/gate7_dominance.py under",
             "results/exponential_family/gate7/contract.md. Reaches in multiples of pi;",
             "every number from the Gate 0 metrics code; verification blocks in dominance.json.", ""]
    for note in data.get("notes", []):
        lines += [note, ""]
    t = data["targets"]
    lines += ["Targets", "", "| comparator | cond | R1(1e-2) | R1(1e-3) | R1(1e-4) | G(1e-2) | E1max B0 | E2max B0 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in t.items():
        lines.append(f"| {name} | {row['cond']:.3f} | " + " | ".join(_fmt(row[k]) for k in YARDSTICKS) + " |")
    lines.append("")
    for n in data["widths"]:
        lines += [f"## Bandwidth {n - 1} (n = {n} exponents, degree {n - 1})", ""]
        for bname, C in data["budgets"].items():
            if f"{n}|{bname}|SLC1992" not in data["summary"]:
                continue
            lines += [f"### Budget {bname} = {C:.3f}", "",
                      "Dominance candidates (largest smallest-margin member)", "",
                      "| against | member | cond | smallest margin | binding yardstick | R1(1e-2) | R1(1e-3) | R1(1e-4) | G | E1max | E2max | ec/2max | verdict |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for tname in ("SLC1992", "TunedQ8"):
                rec = data["summary"][f"{n}|{bname}|{tname}"]
                ver = data["verification"][f"{n}|{bname}|{tname}"]["metrics"]
                mg = rec.get("margins", {})
                binding = min(mg, key=mg.get) if mg else "infeasible"
                ec_ratio = ver["ec_max_B0"] / (COUPLED_FACTOR * max(ver["e1_max_B0"], ver["e2_max_B0"]))
                verdict = "DOMINATES" if (rec["value"] >= 0.0 and ec_ratio <= 1.0) else "does not dominate"
                lines.append(f"| {tname} | {_slots_label(rec)} | {ver['cond']:.2f} | {rec['value']:+.4f} | {binding} | "
                             + " | ".join(_fmt(ver[k + '_refined'] if k.startswith('reach') else ver[k]) for k in YARDSTICKS)
                             + f" | {ec_ratio:.3f} | {verdict} |")
            lines += ["", "Per-yardstick family ceiling (each yardstick maximized alone, same constraints)", "",
                      "| yardstick | family best | member | cond | SLC | q8 | compact ceiling at (L, R) = (n/2 - 1, n/2 - 1) |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for y in YARDSTICKS:
                rec = data["summary"][f"{n}|{bname}|{y}"]
                ver = data["verification"][f"{n}|{bname}|{y}"]["metrics"]
                key = y + "_refined" if y.startswith("reach") else y
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
                lines.append(f"| {y} | {_fmt(ver[key])} | {_slots_label(rec)} | {ver['cond']:.2f} | "
                             f"{_fmt(t['SLC1992'][y])} | {_fmt(t['TunedQ8'][y])} | {ceil} |")
            lines.append("")
    lines += ["## Verification of every reported member", "",
              "| key | member | decimal deviation | symbols vs alias | b certified | kappa2 certified | Theorem P | Theorem N | Lemma D residual | fine E1max | fine R1(1e-3) | fine cond |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, ver in data["verification"].items():
        m = ver["metrics"]
        r = ver["routes"]
        p = ver["positivity"]
        s = ver["sufficient_conditions"]
        lines.append(f"| {key} | {ver['label']} | {r['decimal_deviation']:.2e} | {r['symbols_from_samples_versus_alias']:.2e} | "
                     f"{p['b_certified']} | {p['kappa2b_certified']} | {s['theorem_P']} | {s['theorem_N']} | "
                     f"{ver['lemma_D_residual']:.1e} | {m['fine_e1_max_B0']:.3e} | {m['fine_reach_d1_0.001']:.4f} | {m['fine_cond']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--widths", type=str, default=None,
                        help="comma separated exponent counts, e.g. 12 (default 10,8)")
    parser.add_argument("--budgets", type=str, default=None,
                        help="comma separated budget names among C_q8,C_SLC1,C_BMC11")
    parser.add_argument("--suffix", type=str, default=None, help="output file suffix")
    parser.add_argument("--merge", type=str, default=None,
                        help="comma separated suffixes of dominance<suffix>.json files to merge "
                             "into dominance.json and dominance.md, no search")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ceiling = None
    if (OUT / "ceiling.json").exists():
        ceiling = json.loads((OUT / "ceiling.json").read_text())
    if args.merge is not None:
        data = merge([json.loads((OUT / f"dominance{sfx}.json").read_text())
                      for sfx in args.merge.split(",")])
        (OUT / "dominance.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
        text = markdown(data, ceiling)
        (OUT / "dominance.md").write_text(text)
        print(text)
        return 0
    widths = [int(w) for w in args.widths.split(",")] if args.widths else None
    budget_names = args.budgets.split(",") if args.budgets else None
    data = build(args.processes, quick=args.quick, widths=widths, budget_names=budget_names)
    suffix = args.suffix if args.suffix is not None else ("_quick" if args.quick else "")
    (OUT / f"dominance{suffix}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    text = markdown(data, ceiling)
    (OUT / f"dominance{suffix}.md").write_text(text)
    dominating = []
    for key, rec in data["summary"].items():
        if rec["yardstick"] is None and rec["value"] >= 0.0:
            ver = data["verification"][key]["metrics"]
            ec_ratio = ver["ec_max_B0"] / (COUPLED_FACTOR * max(ver["e1_max_B0"], ver["e2_max_B0"]))
            if ec_ratio <= 1.0:
                dominating.append(key)
                MEMBERS.mkdir(parents=True, exist_ok=True)
                name = "X7_" + key.replace("|", "_")
                record = data["verification"][key]["record"]
                record["name"] = name
                (MEMBERS / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(text)
    print("dominating members:", dominating)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
