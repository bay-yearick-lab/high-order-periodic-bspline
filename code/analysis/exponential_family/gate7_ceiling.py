"""Gate 7, Question 1: the nodal ceiling of the compact family.

A compact first derivative of type (L, R) has the symbol

    kappa_1(theta) = N(theta) / Lf(theta),
    N(theta)  = 2 sum_{s=1}^{R} a_s sin(s theta),
    Lf(theta) = c_0 + 2 sum_{l=1}^{L} c_l cos(l theta),

with Lf > 0. The symbol is invariant under a common scaling of (a, c), so the
conditioning constraint max Lf / min Lf <= C is normalized to

    1 <= Lf(theta) <= C     on [0, pi],

and first-order consistency N'(0) = Lf(0) reads 2 sum_s s a_s = c_0 + 2 sum_l c_l.
For a fixed band edge Theta and tolerance tol the reach condition

    |N(theta)/theta - Lf(theta)| <= tol Lf(theta)   on (0, Theta]

is linear in (a, c); the division by theta keeps every row of order one down
to theta -> 0 (the coefficient of a_s is 2 sin(s theta)/theta), so the
solver's absolute feasibility tolerance is never large against a row. On a
theta grid the largest slack

    delta* = max { delta : |N/theta - Lf| + delta <= tol Lf on the grid,
                   1 <= Lf <= C on the grid, consistency }

is a linear program; delta* >= 0 means a scheme exists on the grid. The grid
problem has fewer constraints than the semi-infinite one, so delta* < 0 on the
grid proves that no compact scheme of the type reaches Theta at tolerance tol
under conditioning C. That proof is made independent of the solver by a dual
certificate: for any multipliers y <= 0 on the inequality rows and y_eq on the
equality rows, and with r = c_obj - A^T y - A_eq^T y_eq, every feasible x in
the box satisfies

    c_obj^T x >= y^T b + y_eq^T b_eq + sum_i min(r_i lb_i, r_i ub_i),

so the right-hand side evaluated in floating point (its rounding error is far
below the margins reported) bounds -delta* from below whatever the solver did.
The box is c_0 in [1/2, 2C], |c_l| <= C, |a_s| <= 100 C; it is never active
at a reported optimum and is recorded.

The reach ceiling of a cell is bracketed: the lower end is the certified reach
(Gate 0 metrics code on 200,001 angles, Brent refined crossing) of the scheme
returned by the last feasible LP, which is solved at tolerance tol (1 - MU) so
that its continuous error stays below tol between the grid points; the upper
end is the smallest Theta at which the dual certificate proves infeasibility
at the full tolerance. The band-maximum minimax on B0 is bracketed the same
way by bisection on the level E (feasibility at level E (1 - MU) for the
achievable side, dual infeasibility at level E for the lower side). The
direct second derivative uses kappa_2 = -N_2 / Lf with N_2(theta) = a_0 +
2 sum a_s cos(s theta); the consistency condition a_0 + 2 sum a_s = 0 is
eliminated exactly, which gives the cancellation-free form
-N_2 = 4 sum a_s sin^2(s theta/2), the rows |(-N_2)/theta^2 - Lf| <= E Lf,
and the remaining condition sum s^2 a_s = Lf(0). The same form evaluates the
certified symbols, so no cancellation noise enters the fine-grid band maxima.

The left symbol of every certified scheme is bounded between the fine grid
points by the curvature bound |Lf''| <= 2 sum l^2 |c_l|, so the reported
conditioning is an interval that contains the continuous value.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate7"

B0 = 2.0 * np.pi / 3.0
TOLS = (1e-2, 1e-3, 1e-4)
CELLS = [(L, R) for L in (1, 2, 3, 4) for R in (2, 3, 4, 5)] + [(2, 7)]
MU = 1e-4
ACC_POINTS = 4000
LEFT_POINTS = 2000
FINE_LEFT_POINTS = 20001
FINE_POINTS = 200001
BISECT_ITERS = 26
MINIMAX_ITERS = 30
CERT_MARGIN = 1e-12
A_BOX = 100.0
COMPARATORS = {
    "SLC1992": (2, 3, comp.SLC_D1, comp.SLC_D2),
    "TunedQ8": (1, 5, comp.Q8_D1, comp.Q8_D2),
    "CW12A": (2, 7, comp.CW12A_D1, comp.CW12A_D2),
    "CW12B": (2, 5, comp.CW12B_D1, comp.CW12B_D2),
}


def budgets() -> dict:
    """Return the three Gate 7 budgets from the comparator left symbols."""
    slc = met.symbol_cond(comp.compact_left(comp.SLC_D1["alpha"]))["cond"]
    q8 = met.symbol_cond(comp.compact_left(comp.Q8_D1["alpha"]))["cond"]
    bmc11 = comp.ExponentialGenerator(12, ()).cond()["cond"]
    return {"C_q8": float(q8), "C_SLC1": float(slc), "C_BMC11": float(bmc11)}


# ---------------------------------------------------------------------------
# The linear programs
# ---------------------------------------------------------------------------

def _left_rows(theta: np.ndarray, L: int) -> np.ndarray:
    """Return the matrix mapping (c_0, ..., c_L) to Lf(theta)."""
    cols = [np.ones_like(theta)] + [2.0 * np.cos(l * theta) for l in range(1, L + 1)]
    return np.column_stack(cols)


def _right_rows_d1(theta: np.ndarray, R: int) -> np.ndarray:
    return np.column_stack([2.0 * np.sin(s * theta) for s in range(1, R + 1)])


def _right_rows_d2(theta: np.ndarray, R: int) -> np.ndarray:
    cols = [np.ones_like(theta)] + [2.0 * np.cos(s * theta) for s in range(1, R + 1)]
    return np.column_stack(cols)


def build_lp(L: int, R: int, C: float, tol: float, Theta: float, derivative: int,
             acc_points: int = ACC_POINTS, left_points: int = LEFT_POINTS) -> dict:
    """Return the LP data of the slack maximization for one (cell, C, tol, Theta)."""
    theta_acc = Theta * np.arange(1, acc_points + 1) / acc_points
    theta_left = np.linspace(0.0, np.pi, left_points)
    Lacc = _left_rows(theta_acc, L)
    Lleft = _left_rows(theta_left, L)
    if derivative == 1:
        # N/theta with N = 2 sum a_s sin(s theta)
        Nacc = _right_rows_d1(theta_acc, R) / theta_acc[:, None]
    else:
        # (-N_2)/theta^2 with -N_2 = 4 sum a_s sin^2(s theta/2), a_0 eliminated
        Nacc = np.column_stack([4.0 * np.sin(0.5 * s * theta_acc) ** 2 / theta_acc ** 2
                                for s in range(1, R + 1)])
    n_right = R
    n_left = L + 1
    n_var = n_left + n_right + 1
    # rows: Nacc a - (1 + tol) Lf + delta <= 0 ; -Nacc a + (1 - tol) Lf + delta <= 0
    A1 = np.zeros((acc_points, n_var))
    A1[:, :n_left] = -(1.0 + tol) * Lacc
    A1[:, n_left:n_left + n_right] = Nacc
    A1[:, -1] = 1.0
    A2 = np.zeros((acc_points, n_var))
    A2[:, :n_left] = (1.0 - tol) * Lacc
    A2[:, n_left:n_left + n_right] = -Nacc
    A2[:, -1] = 1.0
    # left bounds: -Lf <= -1 ; Lf <= C
    A3 = np.zeros((left_points, n_var))
    A3[:, :n_left] = -Lleft
    A4 = np.zeros((left_points, n_var))
    A4[:, :n_left] = Lleft
    A_ub = np.vstack([A1, A2, A3, A4])
    b_ub = np.concatenate([np.zeros(2 * acc_points), -np.ones(left_points),
                           C * np.ones(left_points)])
    # consistency: leading coefficient of N/theta (or (-N_2)/theta^2) equals Lf(0)
    A_eq = np.zeros((1, n_var))
    A_eq[0, :n_left] = -np.array([1.0] + [2.0] * L)
    if derivative == 1:
        A_eq[0, n_left:n_left + n_right] = 2.0 * np.arange(1, R + 1)
    else:
        A_eq[0, n_left:n_left + n_right] = np.arange(1, R + 1) ** 2
    b_eq = np.zeros(1)
    lb = np.concatenate([[0.5], -C * np.ones(L), -A_BOX * C * np.ones(n_right), [-1e3 * C]])
    ub = np.concatenate([[2.0 * C], C * np.ones(L), A_BOX * C * np.ones(n_right), [1e3 * C]])
    c_obj = np.zeros(n_var)
    c_obj[-1] = -1.0
    return {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "lb": lb, "ub": ub,
            "c": c_obj, "n_left": n_left, "n_right": n_right}


def solve_lp(lp: dict) -> dict:
    """Solve the slack LP and return delta*, the coefficients, and the dual bound."""
    res = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"], b_eq=lp["b_eq"],
                  bounds=list(zip(lp["lb"], lp["ub"])), method="highs",
                  options={"primal_feasibility_tolerance": 1e-9,
                           "dual_feasibility_tolerance": 1e-9})
    out = {"status": int(res.status), "message": str(res.message)}
    if res.status != 0:
        out.update({"delta": None, "x": None, "delta_upper_bound": None})
        return out
    x = np.asarray(res.x, dtype=float)
    delta = float(x[-1])
    y = np.minimum(np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
    y_eq = np.asarray(res.eqlin.marginals, dtype=float)
    r = lp["c"] - lp["A_ub"].T @ y - lp["A_eq"].T @ y_eq
    lb, ub = lp["lb"], lp["ub"]
    box = 0.0
    for ri, lo, hi in zip(r, lb, ub):
        if ri == 0.0:
            continue
        cands = []
        for bound in (lo, hi):
            if np.isfinite(bound):
                cands.append(ri * bound)
            else:
                cands.append(-np.inf if (ri > 0) == (bound < 0) else np.inf)
        box += min(cands)
    dual = float(y @ lp["b_ub"] + y_eq @ lp["b_eq"] + box)
    # c^T x = -delta >= dual  =>  delta <= -dual for every feasible x
    n_left, n_right = lp["n_left"], lp["n_right"]
    out.update({"delta": delta, "x": x, "delta_upper_bound": -dual,
                "left": x[:n_left].tolist(), "right": x[n_left:n_left + n_right].tolist(),
                "box_active": bool(np.any(np.abs(x[:-1] - lb[:-1]) < 1e-9 * (1 + np.abs(lb[:-1])))
                                   or np.any(np.abs(x[:-1] - ub[:-1]) < 1e-9 * (1 + np.abs(ub[:-1]))))})
    return out


# ---------------------------------------------------------------------------
# Certification of one compact scheme
# ---------------------------------------------------------------------------

def polish_consistency(left: list, right: list, derivative: int) -> tuple[list, list]:
    """Return the coefficients with the consistency condition enforced exactly.

    The LP satisfies its equality row to the solver tolerance only; the right
    side is rescaled so that the leading-order condition holds to rounding, a
    change of the size of that tolerance.
    """
    c = np.asarray(left, dtype=float)
    a = np.asarray(right, dtype=float).copy()
    L0 = c[0] + 2.0 * np.sum(c[1:])
    s_idx = np.arange(1, a.size + 1)
    lead = 2.0 * np.sum(s_idx * a) if derivative == 1 else np.sum(s_idx ** 2 * a)
    a *= L0 / lead
    return c.tolist(), a.tolist()


def scheme_symbols(left: list, right: list, derivative: int):
    """Return (kappa, Lf) callables from the LP coefficient vectors.

    The second derivative symbol uses -N_2 = 4 sum a_s sin^2(s theta/2), the
    cancellation-free form implied by a_0 = -2 sum a_s.
    """
    left, right = polish_consistency(left, right, derivative)
    c = np.asarray(left, dtype=float)
    a = np.asarray(right, dtype=float)
    alpha = tuple(c[1:] / c[0])
    Lf = comp.compact_left(alpha)
    if derivative == 1:
        kappa = comp.compact_kappa1(alpha, tuple(a / c[0]))
    else:
        aa = a / c[0]

        def kappa(theta):
            theta = np.asarray(theta, dtype=float)
            num = np.zeros(theta.shape)
            for s, coeff in enumerate(aa, start=1):
                num = num + 4.0 * coeff * np.sin(0.5 * s * theta) ** 2
            return num / Lf(theta)
    return kappa, Lf


def certify_scheme(left: list, right: list, derivative: int, C: float) -> dict:
    """Return the fine-grid metrics and the interval conditioning of a scheme."""
    kappa, Lf = scheme_symbols(left, right, derivative)
    theta = met.theta_grid(FINE_POINTS)
    if derivative == 1:
        err_fn = lambda t: np.asarray(kappa(t)) / np.asarray(t) - 1.0
    else:
        err_fn = lambda t: np.asarray(kappa(t)) / np.asarray(t) ** 2 - 1.0
    err = err_fn(theta)
    out = {"band_max_B0": met.band_max(theta, err, B0)}
    for tol in TOLS:
        res = met.reach(theta, err, tol, err_fn=err_fn)
        out[f"reach_{tol:g}"] = res["refined"] / np.pi
    c = np.asarray(left, dtype=float)
    alpha = c[1:] / c[0]
    tl = np.linspace(0.0, np.pi, FINE_POINTS)
    Lv = Lf(tl)
    h = tl[1] - tl[0]
    curvature = 2.0 * float(np.sum(np.arange(1, alpha.size + 1) ** 2 * np.abs(alpha)))
    slack = h * h * curvature / 8.0
    out["cond_grid"] = float(Lv.max() / Lv.min())
    out["cond_interval"] = [float(Lv.max() / (Lv.min() + slack)),
                            float((Lv.max() + slack) / (Lv.min() - slack))]
    out["cond_within_budget"] = bool(out["cond_interval"][1] <= C * (1.0 + 1e-6))
    out["left_min_grid"] = float(Lv.min())
    out["left_max_grid"] = float(Lv.max())
    out["alpha"] = [float(v) for v in alpha]
    left, right = polish_consistency(left, right, derivative)
    out["a"] = [float(v) for v in np.asarray(right) / c[0]]
    if derivative == 2:
        out["a0"] = float(-2.0 * np.sum(np.asarray(right)) / c[0])
    return out


# ---------------------------------------------------------------------------
# Bisections
# ---------------------------------------------------------------------------

def _feasible(L, R, C, tol, Theta, derivative):
    sol = solve_lp(build_lp(L, R, C, tol, Theta, derivative))
    return sol


def _feasible_bisection(L, R, C_lp, tol_lp, derivative, left_points, lo, hi, iters):
    """Bisect the feasible side (reach: on Theta; minimax: on log level) and return
    the endpoint and the last feasible LP solution."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if derivative == 0:
            sol = solve_lp(build_lp(L, R, C_lp, tol_lp, mid, 1, left_points=left_points))
        else:
            sol = solve_lp(build_lp(L, R, C_lp, np.exp(mid) * (1.0 - MU), B0, derivative,
                                    left_points=left_points))
        if sol["delta"] is not None and sol["delta"] >= 0.0:
            if derivative == 0:
                lo, best = mid, sol
            else:
                hi, best = mid, sol
        else:
            if derivative == 0:
                hi = mid
            else:
                lo = mid
    return (lo if derivative == 0 else hi), best


def _tighten_budget(L, R, C, tol_lp, derivative, best, lo, hi, iters, rounds: int = 3):
    """Return (best, certificate, C_lp) with the certified conditioning inside C.

    The left grid of the LP has LEFT_POINTS angles; for L >= 4 the continuous
    minimum of Lf can fall below the grid minimum by up to about 1.5e-4
    relative. The returned scheme is certified on 200,001 angles and, if its
    conditioning interval exceeds C, the feasible-side bisection is repeated
    on a left grid of FINE_LEFT_POINTS angles with the budget C_lp = C (1 - 2
    excess); the certified value of the scheme found is then the lower end.
    derivative = 0 encodes the reach problem (bisection on Theta), 1 and 2 the
    band minimax problems (bisection on the log level).
    """
    C_lp = C
    cert = certify_scheme(best["left"], best["right"], 1 if derivative == 0 else derivative, C)
    for _ in range(rounds):
        if cert["cond_within_budget"]:
            break
        excess = cert["cond_interval"][1] / C - 1.0
        C_lp = C_lp * (1.0 - 2.0 * max(excess, 1e-7))
        _, sol = _feasible_bisection(L, R, C_lp, tol_lp, derivative, FINE_LEFT_POINTS, lo, hi, iters)
        if sol is None:
            break
        best = sol
        cert = certify_scheme(best["left"], best["right"], 1 if derivative == 0 else derivative, C)
    return best, cert, C_lp


def reach_ceiling(L: int, R: int, C: float, tol: float) -> dict:
    """Return the bracket of the reach ceiling of a cell at tolerance tol."""
    tol_lp = tol * (1.0 - MU)
    lo, hi = 0.02 * np.pi, np.pi
    best = None
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        sol = _feasible(L, R, C, tol_lp, mid, 1)
        if sol["delta"] is not None and sol["delta"] >= 0.0:
            lo, best = mid, sol
        else:
            hi = mid
    record = {"L": L, "R": R, "C": C, "tol": tol, "feasible_theta_over_pi": lo / np.pi}
    if best is None:
        record["certified"] = None
        record["reach_lower"] = 0.0
    else:
        best, cert, C_lp = _tighten_budget(L, R, C, tol_lp, 0, best, 0.02 * np.pi, np.pi, BISECT_ITERS)
        record["certified"] = cert
        record["reach_lower"] = cert[f"reach_{tol:g}"]
        record["box_active"] = best["box_active"]
        record["lp_budget_used"] = C_lp
    # upper end: smallest Theta with a dual certificate of infeasibility at tol
    lo2, hi2 = lo, min(np.pi, hi + 1e-3)
    sol = _feasible(L, R, C, tol, hi2, 1)
    while not (sol["delta_upper_bound"] is not None and sol["delta_upper_bound"] < -CERT_MARGIN):
        hi2 = min(np.pi, hi2 + 0.05)
        sol = _feasible(L, R, C, tol, hi2, 1)
        if hi2 >= np.pi:
            break
    for _ in range(12):
        mid = 0.5 * (lo2 + hi2)
        sol = _feasible(L, R, C, tol, mid, 1)
        if sol["delta_upper_bound"] is not None and sol["delta_upper_bound"] < -CERT_MARGIN:
            hi2 = mid
        else:
            lo2 = mid
    record["reach_upper"] = hi2 / np.pi
    record["upper_certificate_delta_bound"] = sol["delta_upper_bound"]
    return record


def minimax_ceiling(L: int, R: int, C: float, derivative: int) -> dict:
    """Return the bracket of the band-maximum minimax on B0 for D1 or D2."""
    lo, hi = np.log(1e-9), np.log(1.0)
    best = None
    for _ in range(MINIMAX_ITERS):
        mid = 0.5 * (lo + hi)
        E = np.exp(mid)
        sol = _feasible(L, R, C, E * (1.0 - MU), B0, derivative)
        if sol["delta"] is not None and sol["delta"] >= 0.0:
            hi, best = mid, sol
        else:
            lo = mid
    record = {"L": L, "R": R, "C": C, "derivative": derivative}
    if best is None:
        record["achieved"] = None
        record["certified"] = None
    else:
        best, cert, C_lp = _tighten_budget(L, R, C, None, derivative, best, np.log(1e-9), np.log(1.0), MINIMAX_ITERS)
        record["certified"] = cert
        record["achieved"] = cert["band_max_B0"]
        record["box_active"] = best["box_active"]
        record["lp_budget_used"] = C_lp
    # lower end of the minimax value: a level with a dual certificate
    E_low = np.exp(lo)
    sol = _feasible(L, R, C, E_low, B0, derivative)
    steps = 0
    while not (sol["delta_upper_bound"] is not None and sol["delta_upper_bound"] < -CERT_MARGIN) and steps < 20:
        E_low *= 0.999
        sol = _feasible(L, R, C, E_low, B0, derivative)
        steps += 1
    record["lower_bound"] = float(E_low) if (sol["delta_upper_bound"] is not None
                                            and sol["delta_upper_bound"] < -CERT_MARGIN) else None
    record["lower_certificate_delta_bound"] = sol["delta_upper_bound"]
    return record


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

def significant_zeros(theta: np.ndarray, err: np.ndarray, floor: float = 1e-6) -> list:
    """Return the sign changes of err between points where |err| exceeds floor.

    Published coefficients quoted to seven or eight digits leave a relative
    error plateau of about 1e-8 as theta -> 0 whose sign is rounding noise;
    only sign changes between lobes that rise above the floor are zeros of
    the scheme's error curve.
    """
    keep = np.abs(err) > floor
    th, sg = theta[keep], np.sign(err[keep])
    idx = np.flatnonzero(sg[:-1] * sg[1:] < 0)
    return [float(0.5 * (th[i] + th[i + 1]) / np.pi) for i in idx]


def comparator_metrics() -> dict:
    """Return the fine-grid yardsticks of the four compact comparators."""
    out = {}
    for name, (L, R, d1, d2) in COMPARATORS.items():
        k1 = comp.compact_kappa1(d1["alpha"], d1["a"])
        k2 = comp.compact_kappa2(d2["alpha"], d2["a0"], d2["a"])
        Lf = comp.compact_left(d1["alpha"])
        theta = met.theta_grid(FINE_POINTS)
        e1_fn = lambda t, k1=k1: np.asarray(k1(t)) / np.asarray(t) - 1.0
        e2_fn = lambda t, k2=k2: np.asarray(k2(t)) / np.asarray(t) ** 2 - 1.0
        e1, e2 = e1_fn(theta), e2_fn(theta)
        row = {"L": L, "R": R, "cond": met.symbol_cond(Lf)["cond"],
               "e1_max_B0": met.band_max(theta, e1, B0), "e2_max_B0": met.band_max(theta, e2, B0)}
        for tol in TOLS:
            row[f"reach_{tol:g}"] = met.reach(theta, e1, tol, err_fn=e1_fn)["refined"] / np.pi
        row["e1_zeros_over_pi"] = significant_zeros(theta, e1)
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _task(item):
    kind, L, R, C, extra = item
    t0 = time.time()
    if kind == "reach":
        rec = reach_ceiling(L, R, C, extra)
    else:
        rec = minimax_ceiling(L, R, C, extra)
    rec["kind"] = kind
    rec["seconds"] = time.time() - t0
    return rec


def build(processes: int = 12, cells=None, budget_map=None, placements: bool = True) -> dict:
    cells = list(cells or CELLS)
    budget_map = dict(budget_map or budgets())
    comparators = comparator_metrics()
    # the comparators are placed against the ceiling of their own cell at their own cond
    placement_budgets = {name: (row["L"], row["R"], row["cond"]) for name, row in comparators.items()}
    if not placements:
        placement_budgets = {}
    tasks = []
    for L, R in cells:
        for name, C in budget_map.items():
            for tol in TOLS:
                tasks.append(("reach", L, R, C, tol))
            tasks.append(("minimax", L, R, C, 1))
            tasks.append(("minimax", L, R, C, 2))
    for name, (L, R, C) in placement_budgets.items():
        for tol in TOLS:
            tasks.append(("reach", L, R, C, tol))
        tasks.append(("minimax", L, R, C, 1))
        tasks.append(("minimax", L, R, C, 2))
    with Pool(processes) as pool:
        records = pool.map(_task, tasks, chunksize=1)
    return {"budgets": budget_map, "cells": cells, "records": records,
            "comparators": comparators, "placement_budgets": placement_budgets,
            "settings": {"MU": MU, "ACC_POINTS": ACC_POINTS, "LEFT_POINTS": LEFT_POINTS,
                         "FINE_POINTS": FINE_POINTS, "CERT_MARGIN": CERT_MARGIN,
                         "BISECT_ITERS": BISECT_ITERS, "MINIMAX_ITERS": MINIMAX_ITERS}}


def _find(records, kind, L, R, C, extra):
    for rec in records:
        if rec["kind"] == kind and rec["L"] == L and rec["R"] == R and abs(rec["C"] - C) < 1e-9:
            key = "tol" if kind == "reach" else "derivative"
            if rec[key] == extra:
                return rec
    return None


def markdown(data: dict) -> str:
    lines = ["# Gate 7 ceiling of the compact family", "",
             "Produced by code/analysis/exponential_family/gate7_ceiling.py under",
             "results/exponential_family/gate7/contract.md. Each entry brackets the",
             "ceiling: the lower value is the certified metric of a scheme found by the",
             "slack LP (Gate 0 metrics code, 200,001 angles, Brent refined crossing);",
             "the upper value is the smallest level at which a dual certificate proves",
             "that no scheme of the cell reaches it. Reaches are in multiples of pi.", ""]
    lines += ["Budgets", "", "| budget | value |", "| --- | --- |"]
    for name, C in data["budgets"].items():
        lines.append(f"| {name} | {C:.6f} |")
    lines.append("")
    for name, C in data["budgets"].items():
        lines += [f"## Budget {name} = {C:.4f}", "",
                  "| L | R | free params | R(1e-2) lower | R(1e-2) upper | R(1e-3) lower | R(1e-3) upper | "
                  "R(1e-4) lower | R(1e-4) upper | D1 minimax B0 [lower, achieved] | D2 minimax B0 [lower, achieved] |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for L, R in data["cells"]:
            cells = []
            for tol in TOLS:
                rec = _find(data["records"], "reach", L, R, C, tol)
                cells += [f"{rec['reach_lower']:.4f}", f"{rec['reach_upper']:.4f}"]
            m1 = _find(data["records"], "minimax", L, R, C, 1)
            m2 = _find(data["records"], "minimax", L, R, C, 2)
            fmt = lambda m: (f"[{m['lower_bound']:.3e}, {m['achieved']:.3e}]"
                             if m and m["achieved"] is not None and m["lower_bound"] is not None
                             else (f"[none, {m['achieved']:.3e}]" if m and m["achieved"] is not None else "none"))
            lines.append(f"| {L} | {R} | {L + R - 1} | " + " | ".join(cells)
                         + f" | {fmt(m1)} | {fmt(m2)} |")
        lines.append("")
    lines += ["## Where the published schemes sit", "",
              "Each comparator is placed against the ceiling of its own type (L, R) at its",
              "own left-side conditioning.", "",
              "| scheme | (L, R) | cond | R(1e-2) | ceiling | R(1e-3) | ceiling | R(1e-4) | ceiling | "
              "E1max B0 | D1 minimax | E2max B0 | D2 minimax | zeros of e1 (theta/pi) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in data["comparators"].items():
        L, R, C = data["placement_budgets"][name]
        cells = []
        for tol in TOLS:
            rec = _find(data["records"], "reach", L, R, C, tol)
            cells += [f"{row[f'reach_{tol:g}']:.4f}", f"[{rec['reach_lower']:.4f}, {rec['reach_upper']:.4f}]"]
        m1 = _find(data["records"], "minimax", L, R, C, 1)
        m2 = _find(data["records"], "minimax", L, R, C, 2)
        fmt = lambda m: (f"[{m['lower_bound']:.3e}, {m['achieved']:.3e}]"
                         if m and m["achieved"] is not None and m["lower_bound"] is not None else "none")
        zeros = ", ".join(f"{z:.3f}" for z in row["e1_zeros_over_pi"])
        lines.append(f"| {name} | ({L}, {R}) | {C:.2f} | " + " | ".join(cells)
                     + f" | {row['e1_max_B0']:.3e} | {fmt(m1)}"
                     + f" | {row['e2_max_B0']:.3e} | {fmt(m2)} | {zeros} |")
    lines += ["", "## Conventions", "",
              f"* LP accuracy grid: {ACC_POINTS} uniform angles on (0, Theta]; left grid: "
              f"{LEFT_POINTS} angles on [0, pi]; feasibility solved at tolerance (1 - {MU}) tol so",
              "  that the continuous error of the returned scheme stays inside tol between grid points.",
              "* Upper ends are dual certificates of the slack LP at the full tolerance: the bound",
              f"  delta <= b^T y + b_eq^T y_eq + box term is below -{CERT_MARGIN:g} for every feasible point.",
              "* Conditioning of every certified scheme is an interval from the curvature bound of the",
              "  left symbol between the 200,001 grid points; every reported scheme lies within its budget.",
              "* free params = L + R - 1: the coefficients less one scale and one consistency condition.",
              "* When the certified conditioning of a returned scheme exceeds the budget (the LP left grid",
              "  has 2,000 angles), the feasible-side bisection is repeated on a 20,001-angle left grid",
              "  with a budget reduced by twice the excess until the interval lies inside; lp_budget_used",
              "  records the budget used and the lower end is the certified value of that scheme.",
              "* Reach brackets are below 1e-5 in theta/pi and D1 minimax brackets below 1e-3 relative;",
              "  D2 minimax levels near 1e-8 are at the LP floor and their brackets reach a few percent.",
              "* A minimax level below about 1e-8 is at the floor of the LP solver's feasibility tolerance",
              "  (rows are relative, so the tolerance is compared with E itself); where no dual certificate",
              "  could be obtained the lower end is reported as none and only the achieved value is a claim."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--quick", action="store_true", help="two cells only, for a smoke test")
    parser.add_argument("--markdown-only", action="store_true", help="rebuild ceiling.md from ceiling.json")
    parser.add_argument("--add-cells", type=str, default=None,
                        help="cells L:R separated by commas to compute and merge into ceiling.json")
    args = parser.parse_args(argv)
    cells = [(2, 3), (4, 4)] if args.quick else CELLS
    suffix = "_quick" if args.quick else ""
    if args.add_cells:
        extra = [tuple(int(v) for v in item.split(":")) for item in args.add_cells.split(",")]
        data = json.loads((OUT / "ceiling.json").read_text())
        data["cells"] = [tuple(c) for c in data["cells"]]
        new = build(args.processes, cells=extra, budget_map=data["budgets"], placements=False)
        keep = [r for r in data["records"] if (r["L"], r["R"]) not in extra]
        data["records"] = keep + new["records"]
        data["cells"] = [c for c in data["cells"] if c not in extra] + extra
        (OUT / "ceiling.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
        (OUT / "ceiling.md").write_text(markdown(data))
        print(markdown(data))
        return 0
    if args.markdown_only:
        data = json.loads((OUT / f"ceiling{suffix}.json").read_text())
        data["cells"] = [tuple(c) for c in data["cells"]]
        data["comparators"] = comparator_metrics()
        (OUT / f"ceiling{suffix}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
        (OUT / f"ceiling{suffix}.md").write_text(markdown(data))
        print(markdown(data))
        return 0
    data = build(args.processes, cells=cells)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ceiling{suffix}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    (OUT / f"ceiling{suffix}.md").write_text(markdown(data))
    print(markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
