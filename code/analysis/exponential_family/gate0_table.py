# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Gate 0 producer: comparator table, refined scan, and the frozen decision.

The contract is results/exponential_family/gate0/contract.md. This script is
its single entry point. It

  1. builds the comparator table with every frozen metric,
  2. runs the refined scan over the cells (m, r) with m in {4, 6, 8},
     r in {1, 2, 3}, p = m + 2 r - 1 <= 11 and the four conditioning budgets,
  3. applies the frozen decision rule, and
  4. writes table.json, table.md, scan.json and decision.json.

It exits nonzero if any reproduction check fails. With --check it recomputes
only the comparator table and the E_{4,2}(0.1, 2.36) reproduction and compares
them with the stored receipt, which is the fast gate path.

Scan conventions that the contract leaves open and this producer fixes:

* The objective and the constraint are evaluated on a 4001 point theta grid
  and with an alias tail tolerance of 1e-10 inside the optimizer; every
  reported number for a winner is recomputed on the 20001 point grid with the
  1e-14 tail tolerance of the contract.
* The optimizer works with the sorted frequency vector, which costs nothing
  because phi_hat is symmetric under permutations of the frequencies.
* Constraint violation enters Nelder-Mead as the additive penalty
  PENALTY * max(0, cond/C - 1); a returned point is accepted only if its
  recomputed condition number satisfies the budget.
* The decision rule reads "at C = C_SLC1", and every winner found under a
  smaller budget is also feasible there, so the rule is applied to all winners
  whose recomputed condition number is at most C_SLC1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for path in (str(CODE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import comparators as comp  # noqa: E402
import family as fam  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate0"
CONTRACT = OUT / "contract.md"

XI_GRID = np.linspace(-0.5, 0.5, 129)
COARSE_POINTS = 4001
FAST_TOL = 1e-10
FINE_TOL = 1e-14
PENALTY = 100.0
BASE_SEED = 20260912
GRID_PER_AXIS = {1: 60, 2: 60, 3: 24}
MIN_STARTS = 200
POLISH_COUNT = 5
EDGE = 1e-4

FROZEN_ANCHOR = {
    "m": 4,
    "omegas": [0.1, 2.36],
    "beta": [0.45570408509669036, 0.24238702597207298,
             0.029461339406865419, 0.00029959207271641975],
    "cond": 34.18391129180412,
    "reach_d1_1pct_over_pi": 0.783922599452434,
    "reach_coupled_1pct_over_pi": 0.7776723532534208,
}


def sha256(path: Path) -> str:
    """Return the hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheme_row(scheme) -> dict:
    """Return the full metric row of one scheme."""
    return met.evaluate(
        scheme.name, scheme.kappa1, scheme.kappa2, scheme.left,
        analytic_gv=scheme.analytic_gv, coupled=scheme.coupled,
        algebraic_order1=scheme.algebraic_order1,
        algebraic_order2=scheme.algebraic_order2,
        bandwidth_left=scheme.bandwidth_left, bandwidth_d1=scheme.bandwidth_d1,
        bandwidth_d2=scheme.bandwidth_d2, solves=scheme.solves,
        extra=dict(scheme.extra, published_order_d1=scheme.published_order1,
                   published_order_d2=scheme.published_order2))


def comparator_table() -> tuple[list[dict], dict]:
    """Return the comparator rows and the four conditioning budgets."""
    rows = [scheme_row(s) for s in comp.comparator_schemes(coupled_xi=XI_GRID)]
    index = {row["name"]: row for row in rows}
    slc_d2_cond = met.symbol_cond(comp.compact_left(comp.SLC_D2["alpha"]))
    index["SLC1992"]["cond_left_d2"] = slc_d2_cond["cond"]
    q8_d2_cond = met.symbol_cond(comp.compact_left(comp.Q8_D2["alpha"]))
    index["TunedQ8"]["cond_left_d2"] = q8_d2_cond["cond"]
    for name, key in (("CW12A", comp.CW12A_D2), ("CW12B", comp.CW12B_D2)):
        index[name]["cond_left_d2"] = met.symbol_cond(
            comp.compact_left(key["alpha"]))["cond"]
    budgets = {
        "C_SLC2": slc_d2_cond["cond"],
        "C_q8": index["TunedQ8"]["cond"],
        "C_SLC1": index["SLC1992"]["cond"],
        "C_BMC11": index["BMC11"]["cond"],
    }
    return rows, budgets


def reproduction_checks() -> tuple[list[dict], dict]:
    """Return the reproduction checks of the frozen receipts and sanity cases."""
    checks: list[dict] = []
    values: dict = {}

    def record(name: str, got, want, tol, kind="relative"):
        if want is None:
            ok = True
            detail = "recorded"
        elif kind == "relative":
            ok = abs(got - want) <= tol * max(1.0, abs(want))
            detail = f"|{got!r} - {want!r}| <= {tol:g} rel"
        else:
            ok = abs(got - want) <= tol
            detail = f"|{got!r} - {want!r}| <= {tol:g} abs"
        checks.append({"check": name, "value": got, "reference": want,
                       "tolerance": tol, "kind": kind, "ok": bool(ok),
                       "detail": detail})
        values[name] = got
        return ok

    generator = fam.anchor_generator()
    samples = generator.integer_samples(FINE_TOL)
    beta_pos = samples.beta[samples.n >= 0]
    for j, want in enumerate(FROZEN_ANCHOR["beta"]):
        record(f"anchor_beta_{j}", float(beta_pos[j]), want, 1e-12)
    cond = generator.cond()
    record("anchor_cond", cond["cond"], FROZEN_ANCHOR["cond"], 1e-12)

    theta = met.theta_grid()
    e1_fn = lambda t: generator.symbols(t, FINE_TOL)[1] / np.asarray(t, float) - 1.0
    res = met.reach(theta, e1_fn(theta), 1e-2, err_fn=e1_fn)
    record("anchor_reach_d1_1pct_over_pi", res["refined"] / np.pi,
           FROZEN_ANCHOR["reach_d1_1pct_over_pi"], 1e-9)
    record("anchor_reach_d1_1pct_over_pi_linear", res["linear"] / np.pi, None, 0.0)
    ec_fn = lambda t: generator.coupled_error(t, XI_GRID, FINE_TOL)
    res = met.reach(theta, ec_fn(theta), 1e-2, err_fn=ec_fn)
    record("anchor_reach_coupled_1pct_over_pi", res["refined"] / np.pi,
           FROZEN_ANCHOR["reach_coupled_1pct_over_pi"], 1e-9)
    record("anchor_reach_coupled_1pct_over_pi_linear", res["linear"] / np.pi, None, 0.0)

    order = np.arange(0, samples.n.max() + 1)
    quad = fam.convolution_samples(4, (0.1, 2.36), order)
    quad1 = fam.convolution_samples(4, (0.1, 2.36), order, derivative=1)
    quad2 = fam.convolution_samples(4, (0.1, 2.36), order, derivative=2)
    record("anchor_quadrature_beta_deviation",
           float(np.max(np.abs(quad - samples.beta[samples.n >= 0]))), 0.0, 1e-12, "absolute")
    record("anchor_quadrature_dbeta_deviation",
           float(np.max(np.abs(quad1 - samples.dbeta[samples.n >= 0]))), 0.0, 1e-12, "absolute")
    record("anchor_quadrature_ddbeta_deviation",
           float(np.max(np.abs(quad2 - samples.ddbeta[samples.n >= 0]))), 0.0, 1e-12, "absolute")
    for w in (0.1, 2.36):
        record(f"pair_kernel_mass_{w}", fam.pair_kernel_mass(w), 1.0, 1e-12, "absolute")

    probe = np.linspace(1e-6, np.pi, 4001)
    cubic = fam.ExponentialGenerator(4, ())
    k1 = cubic.symbols(probe, FINE_TOL)[1]
    record("cubic_kappa1_deviation",
           float(np.max(np.abs(k1 - 3.0 * np.sin(probe) / (2.0 + np.cos(probe))))),
           0.0, 1e-13, "absolute")
    for p in (3, 5, 7, 9, 11):
        d1, d2 = comp.bmc_symbol_check(p, probe)
        record(f"bmc{p}_kappa1_deviation", d1, 0.0, 1e-13, "absolute")
        record(f"bmc{p}_kappa2_relative_deviation", d2, 0.0, 1e-13, "absolute")

    coarse = np.linspace(0.05, np.pi, 21)
    alias = generator.sample_symbols(coarse, tol=FINE_TOL)
    trig = generator.symbols(coarse, FINE_TOL)
    record("anchor_alias_versus_sample_symbols",
           float(max(np.max(np.abs(alias[q] - trig[q])) for q in range(3))),
           0.0, 1e-12, "absolute")

    return checks, values


def cells() -> list[tuple[int, int]]:
    """Return the scan cells (m, r) with degree at most eleven."""
    return [(m, r) for m in (4, 6, 8) for r in (1, 2, 3) if m + 2 * r - 1 <= 11]


def _objective(m: int, omegas, theta, edges, budget: float,
               scale1: float, scale2: float, tol: float):
    """Return the penalized objective, the raw ratio, and the condition number."""
    omegas = np.sort(np.asarray(omegas, dtype=float))
    if np.any(omegas <= EDGE) or np.any(omegas >= np.pi - EDGE):
        out = 1e6 + float(np.sum(np.maximum(0.0, EDGE - omegas)
                                 + np.maximum(0.0, omegas - (np.pi - EDGE))))
        return out, np.inf, np.inf
    if np.any(np.diff(omegas) <= 1e-9):
        return 1e6, np.inf, np.inf
    generator = fam.ExponentialGenerator(m, omegas)
    try:
        b, k1, k2 = generator.symbols(theta, tol)
    except (ValueError, FloatingPointError):
        return 1e6, np.inf, np.inf
    if not np.all(np.isfinite(b)) or float(b.min()) <= 0.0:
        return 1e6, np.inf, np.inf
    cond = float(max(b.max(), 1.0) / b.min())
    e1 = np.abs(k1[edges] / theta[edges] - 1.0).max()
    e2 = np.abs(k2[edges] / theta[edges] ** 2 - 1.0).max()
    ratio = float(max(e1 / scale1, e2 / scale2))
    return ratio + PENALTY * max(0.0, cond / budget - 1.0), ratio, cond


def _grid_seeds(m: int, r: int, theta, edges, scale1, scale2):
    """Return the grid seed sweep for one cell as arrays of points and values."""
    axis = np.linspace(0.0, np.pi, GRID_PER_AXIS[r] + 2)[1:-1]
    points = list(combinations(axis, r)) if r > 1 else [(w,) for w in axis]
    ratios = np.empty(len(points))
    conds = np.empty(len(points))
    for j, point in enumerate(points):
        _, ratio, cond = _objective(m, point, theta, edges, np.inf,
                                    scale1, scale2, FAST_TOL)
        ratios[j] = ratio
        conds[j] = cond
    return np.array(points, dtype=float), ratios, conds


def scan_cell(task: dict) -> dict:
    """Return the best member of one (m, r, budget) cell."""
    m, r = task["m"], task["r"]
    budget = task["budget"]
    scale1, scale2 = task["scale1"], task["scale2"]
    theta = met.theta_grid(COARSE_POINTS)
    edges = met.band_mask(theta, met.BANDS["B0"])
    started = time.time()

    points, ratios, conds = _grid_seeds(m, r, theta, edges, scale1, scale2)
    feasible = np.isfinite(ratios) & (conds <= budget)
    seeds: list[np.ndarray] = []
    if np.any(feasible):
        order = np.argsort(np.where(feasible, ratios, np.inf))
        seeds = [points[j] for j in order[:50]]
    rng = np.random.default_rng(task["seed"])
    while len(seeds) < MIN_STARTS:
        draw = np.sort(rng.uniform(EDGE, np.pi - EDGE, size=r))
        if r == 1 or np.all(np.diff(draw) > 1e-6):
            seeds.append(draw)

    evaluations = 0
    results = []
    for seed in seeds:
        fun = lambda x: _objective(m, x, theta, edges, budget,
                                   scale1, scale2, FAST_TOL)[0]
        res = minimize(fun, np.asarray(seed, dtype=float), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 400 * r,
                                "maxfev": 400 * r})
        evaluations += int(res.nfev)
        _, ratio, cond = _objective(m, res.x, theta, edges, budget,
                                    scale1, scale2, FAST_TOL)
        if np.isfinite(ratio) and cond <= budget:
            results.append((ratio, np.sort(np.asarray(res.x, dtype=float)), cond))
    results.sort(key=lambda item: item[0])

    polished = []
    for ratio, x, cond in results[:POLISH_COUNT]:
        fun = lambda y: _objective(m, y, theta, edges, budget,
                                   scale1, scale2, FAST_TOL)[0]
        res = minimize(fun, x, method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-15, "maxiter": 800 * r,
                                "maxfev": 800 * r})
        evaluations += int(res.nfev)
        _, ratio2, cond2 = _objective(m, res.x, theta, edges, budget,
                                      scale1, scale2, FAST_TOL)
        if np.isfinite(ratio2) and cond2 <= budget:
            polished.append((ratio2, np.sort(np.asarray(res.x, dtype=float)), cond2))
        polished.append((ratio, x, cond))
    polished.sort(key=lambda item: item[0])

    out = {
        "m": m, "r": r, "p": m + 2 * r - 1,
        "budget_name": task["budget_name"], "budget": budget,
        "grid_points": int(points.shape[0]),
        "grid_per_axis": GRID_PER_AXIS[r],
        "grid_feasible": int(np.count_nonzero(feasible)),
        "grid_best_ratio": float(np.min(ratios[feasible])) if np.any(feasible) else None,
        "starts": len(seeds),
        "successful_starts": len(results),
        "function_evaluations": evaluations,
        "seconds": time.time() - started,
    }
    if not polished:
        out.update({"omegas": None, "ratio": None, "cond": None,
                    "status": "no feasible member"})
        return out
    ratio, omegas, cond = polished[0]
    out.update({"omegas": [float(w) for w in omegas],
                "ratio_coarse": float(ratio), "cond_coarse": float(cond),
                "status": "ok"})
    return out


def winner_row(record: dict, scale1: float, scale2: float) -> dict:
    """Return the full contract metrics of one scan winner."""
    generator = fam.ExponentialGenerator(record["m"], record["omegas"])
    name = f"{generator.label()}@{record['budget_name']}"
    scheme = comp.spline_scheme(name, generator, coupled_xi=XI_GRID, tol=FINE_TOL)
    row = scheme_row(scheme)
    row["budget_name"] = record["budget_name"]
    row["budget"] = record["budget"]
    row["ratio_to_slc"] = float(max(row["e1_max_B0"] / scale1,
                                     row["e2_max_B0"] / scale2))
    row["ratio_d1"] = float(row["e1_max_B0"] / scale1)
    row["ratio_d2"] = float(row["e2_max_B0"] / scale2)
    row["admissible"] = list(generator.admissible())
    row["feasible"] = bool(row["cond"] <= record["budget"] * (1.0 + 1e-9))
    row["omega_min"] = float(min(record["omegas"]))
    row["near_polynomial_limit"] = bool(row["omega_min"] < 1e-3)
    return row


def run_scan(budgets: dict, scale1: float, scale2: float, processes: int):
    """Return the scan records and the winner metric rows."""
    tasks = []
    for index, (budget_name, budget) in enumerate(sorted(budgets.items())):
        for m, r in cells():
            tasks.append({"m": m, "r": r, "budget": float(budget),
                          "budget_name": budget_name, "scale1": scale1,
                          "scale2": scale2,
                          "seed": BASE_SEED + 1000 * m + 100 * r + index})
    records = []
    with Pool(processes=processes) as pool:
        for record in pool.imap_unordered(scan_cell, tasks):
            records.append(record)
            print(f"  cell m={record['m']} r={record['r']} "
                  f"budget={record['budget_name']} "
                  f"ratio={record.get('ratio_coarse')} "
                  f"cond={record.get('cond_coarse')} "
                  f"starts={record['starts']} "
                  f"nfev={record['function_evaluations']} "
                  f"{record['seconds']:.1f}s", flush=True)
    records.sort(key=lambda rec: (rec["budget_name"], rec["m"], rec["r"]))
    rows = [winner_row(rec, scale1, scale2) for rec in records if rec["status"] == "ok"]
    return records, rows


def decide(rows: list[dict], budgets: dict, scale1: float, scale2: float) -> dict:
    """Return the frozen decision rule outcome and its controlling numbers."""
    limit = budgets["C_SLC1"]
    pool = [row for row in rows if row["cond"] <= limit * (1.0 + 1e-9)]
    triumph = [row for row in pool
               if row["e1_max_B0"] <= scale1 / 3.0 and row["e2_max_B0"] <= scale2 / 3.0]
    beats = [row for row in pool
             if row["e1_max_B0"] < scale1 and row["e2_max_B0"] < scale2]
    best = min(pool, key=lambda row: row["ratio_to_slc"]) if pool else None
    if triumph:
        outcome = "TRIUMPH"
        controlling = min(triumph, key=lambda row: row["ratio_to_slc"])
    elif beats:
        outcome = "PARTIAL"
        controlling = min(beats, key=lambda row: row["ratio_to_slc"])
    else:
        outcome = "REJECT"
        controlling = best
    return {
        "outcome": outcome,
        "rule": "TRIUMPH if some member with p <= 11 at cond <= C_SLC1 has "
                "E1max(B0) <= E_SLC1/3 and E2max(B0) <= E_SLC2/3; PARTIAL if "
                "the best member beats SLC on both by less than three times; "
                "REJECT otherwise.",
        "E_SLC1": scale1,
        "E_SLC2": scale2,
        "C_SLC1": limit,
        "members_at_C_SLC1": len(pool),
        "members_beating_slc_on_both": len(beats),
        "members_meeting_threefold": len(triumph),
        "controlling_member": None if controlling is None else {
            "name": controlling["name"],
            "m": controlling["m"], "r": controlling["r"], "p": controlling["p"],
            "omegas": controlling["omegas"],
            "E1max_B0": controlling["e1_max_B0"],
            "E2max_B0": controlling["e2_max_B0"],
            "E1_gain": scale1 / controlling["e1_max_B0"],
            "E2_gain": scale2 / controlling["e2_max_B0"],
            "cond": controlling["cond"],
            "budget_name": controlling["budget_name"],
        },
        "best_member": None if best is None else {
            "name": best["name"], "omegas": best["omegas"],
            "ratio_to_slc": best["ratio_to_slc"], "cond": best["cond"],
        },
    }


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


GROUPS = [
    ("Band errors on B0 = (0, 2pi/3], B1 = (0, 0.75pi], B2 = (0, 0.8pi]",
     ["e1_max_B0", "e1_rms_B0", "e1_max_B1", "e1_rms_B1", "e1_max_B2", "e1_rms_B2",
      "e2_max_B0", "e2_rms_B0", "e2_max_B1", "e2_rms_B1", "e2_max_B2", "e2_rms_B2"]),
    ("Reach in multiples of pi, first contiguous crossing, linear interpolation",
     ["reach_d1_0.01", "reach_d1_0.001", "reach_d1_0.0001",
      "reach_d2_0.01", "reach_d2_0.001", "reach_d2_0.0001", "gv_reach_1e-2",
      "gv_error"]),
    ("Reach in multiples of pi, Brent refined crossing",
     ["reach_d1_0.01_refined", "reach_d1_0.001_refined", "reach_d1_0.0001_refined",
      "reach_d2_0.01_refined", "reach_d2_0.001_refined", "reach_d2_0.0001_refined"]),
    ("Conditioning of the left side and the discrete values",
     ["cond", "min_left", "max_left", "cond_N64", "cond_N127", "cond_N128",
      "cond_left_d2"]),
    ("Spectral radii, RK4 limits, and the kappa_2 sign check",
     ["rho1", "rk4_hyperbolic", "rho2", "rk4_parabolic",
      "sign_N64", "sign_N127", "sign_N128"]),
    ("Coupled off-grid channel on 129 xi points",
     ["coupled", "ec_max_B0", "ec_max_B1", "ec_max_B2",
      "reach_ec_0.01", "reach_ec_0.001", "reach_ec_0.0001",
      "reach_ec_0.01_refined"]),
    ("Formal orders",
     ["order_d1_fitted", "order_d1_algebraic", "order_d1_window", "order_d1_points",
      "order_d1_plateau", "order_d2_fitted", "order_d2_algebraic",
      "order_d2_window", "order_d2_points", "order_d2_plateau"]),
    ("Bandwidth and solve count",
     ["bandwidth_left", "bandwidth_d1", "bandwidth_d2", "solves"]),
    ("Member data",
     ["m", "r", "p", "omegas", "omega_min", "near_polynomial_limit",
      "budget_name", "budget", "ratio_d1", "ratio_d2", "ratio_to_slc",
      "feasible"]),
]


def markdown_table(rows: list[dict], budgets: dict, decision: dict,
                   checks: list[dict], scale1: float, scale2: float) -> str:
    """Return the human readable Gate 0 table."""
    lines = ["# Gate 0 table", "",
             "Produced by code/analysis/exponential_family/gate0_table.py under",
             "results/exponential_family/gate0/contract.md. Every row is one",
             "scheme. The metric columns are split into groups only for width;",
             "each group carries the same row order.", "",
             f"Reference band maxima on B0: E_SLC1 = {scale1:.6e}, "
             f"E_SLC2 = {scale2:.6e}.", "",
             "Conditioning budgets", ""]
    lines.append("| budget | value |")
    lines.append("| --- | --- |")
    for key in ("C_SLC2", "C_q8", "C_SLC1", "C_BMC11"):
        lines.append(f"| {key} | {budgets[key]:.6f} |")
    lines += ["", f"Decision: {decision['outcome']}.", ""]
    for title, keys in GROUPS:
        present = [key for key in keys if any(key in row for row in rows)]
        if not present:
            continue
        lines += [f"## {title}", ""]
        lines.append("| scheme | " + " | ".join(present) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in present) + " |")
        for row in rows:
            cells_out = [fmt(row.get(key)) for key in present]
            lines.append(f"| {row['name']} | " + " | ".join(cells_out) + " |")
        lines.append("")
    lines += ["## Reproduction checks", "",
              "| check | value | reference | tolerance | kind | ok |",
              "| --- | --- | --- | --- | --- | --- |"]
    for check in checks:
        lines.append("| {check} | {value} | {reference} | {tol} | {kind} | {ok} |".format(
            check=check["check"], value=fmt(check["value"], 12),
            reference=fmt(check["reference"], 12), tol=fmt(check["tolerance"]),
            kind=check["kind"], ok="yes" if check["ok"] else "NO"))
    lines.append("")
    lines += ["## Conventions", "",
              "* Theta grid: 20001 uniform points on (0, pi], ending at pi.",
              "* Reach: first contiguous crossing with linear interpolation; the",
              "  Brent refined value of the same bracket is reported beside it.",
              "* Group velocity: analytic derivative for the spline members,",
              "  centered difference with step 1e-5 for the compact symbols; the",
              "  reported gv_error is the measured analytic versus difference gap",
              "  for spline members and the truncation estimate otherwise.",
              "* Solve count: cyclic band solves needed to produce D1 u and D2 u",
              "  together; a spline member needs one because both share A_0.",
              "* Bandwidth: nonzero diagonals of the circulant left and right sides.",
              "* Algebraic order: the contract rule gives nodal D1 order m and",
              "  direct D2 order m - 2 for a spline member with even polynomial",
              "  core m. A member flagged near_polynomial_limit sits at the",
              "  coalescing boundary where a conjugate pair collapses onto two",
              "  further zero exponents, which raises the effective core from m to",
              "  m + 2; for such a member the fitted order is the one to read.",
              ""]
    return "\n".join(lines)


def compare_receipt(stored: dict, fresh: dict, rtol: float = 1e-9,
                    atol: float = 1e-12, path: str = "") -> list[str]:
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
    """Recompute the comparator table and the anchor receipt and compare them."""
    started = time.time()
    stored = json.loads((OUT / "table.json").read_text(encoding="utf-8"))
    rows, budgets = comparator_table()
    checks, _ = reproduction_checks()
    fresh = {"budgets": budgets, "comparators": rows,
             "reproduction": {c["check"]: c["value"] for c in checks}}
    kept = {"budgets": stored["budgets"], "comparators": stored["comparators"],
            "reproduction": stored["reproduction"]}
    bad = compare_receipt(kept, fresh)
    failed = [c for c in checks if not c["ok"]]
    for item in bad[:20]:
        print(f"  receipt drift {item}")
    for item in failed:
        print(f"  reproduction failure {item['check']}: {item['detail']}")
    ok = not bad and not failed
    print(f"G31 exponential family Gate 0 receipt: "
          f"{len(rows)} comparator rows, {len(checks)} reproduction checks, "
          f"{len(bad)} drifted fields, {len(failed)} failed checks, "
          f"{time.time() - started:.1f}s")
    return 0 if ok else 1


def main(argv=None) -> int:
    """Build the Gate 0 receipts, or verify them in the fast check mode."""
    parser = argparse.ArgumentParser(description="Gate 0 table producer.")
    parser.add_argument("--check", action="store_true",
                        help="Recompute only the comparator table and the anchor "
                             "reproduction and compare with the stored receipt.")
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the receipts from the stored scan.json "
                             "without repeating the optimization.")
    args = parser.parse_args(argv)
    if args.check:
        return check_mode()

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Gate 0 comparator table", flush=True)
    rows, budgets = comparator_table()
    index = {row["name"]: row for row in rows}
    scale1 = index["SLC1992"]["e1_max_B0"]
    scale2 = index["SLC1992"]["e2_max_B0"]
    for key in ("C_SLC2", "C_q8", "C_SLC1", "C_BMC11"):
        print(f"  {key} = {budgets[key]:.6f}", flush=True)
    print(f"  E_SLC1 = {scale1:.6e}   E_SLC2 = {scale2:.6e}", flush=True)

    print("Reproduction checks", flush=True)
    checks, _ = reproduction_checks()
    failed = [c for c in checks if not c["ok"]]
    for check in checks:
        print(f"  [{'ok ' if check['ok'] else 'FAIL'}] {check['check']}: "
              f"{check['value']!r}", flush=True)

    scan_seconds = 0.0
    records: list[dict] = []
    winners: list[dict] = []
    if args.rebuild:
        print("Rebuilding winners from the stored scan", flush=True)
        records = json.loads((OUT / "scan.json").read_text(encoding="utf-8"))["cells"]
        winners = [winner_row(rec, scale1, scale2)
                   for rec in records if rec["status"] == "ok"]
        scan_seconds = float(sum(rec.get("seconds", 0.0) for rec in records))
    elif not args.skip_scan:
        print("Refined scan", flush=True)
        scan_started = time.time()
        records, winners = run_scan(budgets, scale1, scale2, args.processes)
        scan_seconds = time.time() - scan_started
        print(f"  scan wall time {scan_seconds:.1f}s", flush=True)

    decision = decide(winners, budgets, scale1, scale2)
    decision["scan_seconds"] = scan_seconds
    decision["reproduction_failures"] = [c["check"] for c in failed]
    print(f"Decision: {decision['outcome']}", flush=True)

    table = {
        "producer": "code/analysis/exponential_family/gate0_table.py",
        "contract": {"path": str(CONTRACT.relative_to(ROOT)),
                     "sha256": sha256(CONTRACT)},
        "conventions": {
            "theta_points": met.THETA_POINTS,
            "bands": {k: v for k, v in met.BANDS.items()},
            "reach_tolerances": list(met.REACH_TOLS),
            "coupled_xi_points": int(XI_GRID.size),
            "alias_tail_tolerance": FINE_TOL,
            "optimizer_alias_tail_tolerance": FAST_TOL,
            "optimizer_theta_points": COARSE_POINTS,
        },
        "budgets": budgets,
        "reference_band_maxima": {"E_SLC1": scale1, "E_SLC2": scale2},
        "comparators": rows,
        "scan_winners": winners,
        "reproduction": {c["check"]: c["value"] for c in checks},
        "reproduction_checks": checks,
    }
    (OUT / "table.json").write_text(json.dumps(table, indent=2, default=float) + "\n",
                                    encoding="utf-8")
    (OUT / "table.md").write_text(
        markdown_table(rows + winners, budgets, decision, checks, scale1, scale2),
        encoding="utf-8")
    if not args.rebuild:
        (OUT / "scan.json").write_text(json.dumps(
            {"cells": records, "budgets": budgets,
             "reference_band_maxima": {"E_SLC1": scale1, "E_SLC2": scale2},
             "settings": {"grid_per_axis": GRID_PER_AXIS, "min_starts": MIN_STARTS,
                          "polish_count": POLISH_COUNT, "penalty": PENALTY,
                          "base_seed": BASE_SEED, "coarse_points": COARSE_POINTS,
                          "fast_tol": FAST_TOL}},
            indent=2, default=float) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float)
                                       + "\n", encoding="utf-8")
    print(f"Total wall time {time.time() - started:.1f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
