"""Gate 8: the ceiling of the compact class of formal order at least m.

The Gate 7 ceiling LP (gate7_ceiling.py) imposes first-order consistency
only, so its optima are generically not fourth order and lie outside the
image of the augmented family, every member of which has formal order at
least its core m (theory.md, Proposition O' and Theorem 6). The attainable
target of a one-field member with core m is therefore the ceiling of the
compact class of type (L, R) and formal order at least m, which is the
same semi-infinite program with the additional linear equality rows

    sum_s s^{2j+1} a_s / (2j+1)! = sum_l l^{2j} c_l / (2j)! (with c_0 for l = 0),   j = 1, ..., (m - 2)/2,

the vanishing of the Taylor coefficients of N(theta) - theta Lf(theta) at
theta^{2j+1} (N = 2 sum a_s sin(s theta), Lf = c_0 + 2 sum c_l cos(l theta)).
The dual certificate of Lemma 9 applies unchanged (the rows are equalities
with free multipliers). This module adds the rows to gate7_ceiling.build_lp
and reports, for each (L, R) of bandwidth 7 and 9 and each budget, the
brackets of the reaches R(1e-2), R(1e-3), R(1e-4) and the D1 band minimax
of the class of order >= 4 beside the Gate 7 (order >= 2) values. Every
certified scheme is checked to have a detected order of at least m by the
Taylor residuals of comparators.detect_order_d1.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from math import factorial
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
import gate7_ceiling as gc  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate8"
GATE7 = ROOT / "results" / "exponential_family" / "gate7"
CELLS = [(3, 3), (4, 4), (2, 3)]
ORDERS = (4, 6)


def order_rows(L: int, R: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the equality rows of formal order >= order for the D1 LP variables."""
    rows = []
    n_left = L + 1
    n_var = n_left + R + 1
    for j in range(1, (order - 2) // 2 + 1):
        row = np.zeros(n_var)
        # coefficient of theta^{2j+1} in 2 sum a_s sin(s theta): 2 a_s (-1)^j s^{2j+1}/(2j+1)!
        # coefficient of theta^{2j+1} in theta Lf: (-1)^j [c_0 delta_{j0} + 2 sum c_l l^{2j}/(2j)!]
        for s in range(1, R + 1):
            row[n_left + s - 1] = 2.0 * s ** (2 * j + 1) / factorial(2 * j + 1)
        for l in range(1, L + 1):
            row[l] = -2.0 * l ** (2 * j) / factorial(2 * j)
        rows.append(row)
    return np.array(rows), np.zeros(len(rows))


def build_lp(L, R, C, tol, Theta, derivative, order):
    lp = gc.build_lp(L, R, C, tol, Theta, derivative)
    if derivative == 1 and order > 2:
        A, b = order_rows(L, R, order)
        lp["A_eq"] = np.vstack([lp["A_eq"], A])
        lp["b_eq"] = np.concatenate([lp["b_eq"], b])
    return lp


def reach_ceiling(L, R, C, tol, order) -> dict:
    tol_lp = tol * (1.0 - gc.MU)
    lo, hi = 0.02 * np.pi, np.pi
    best = None
    for _ in range(gc.BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        sol = gc.solve_lp(build_lp(L, R, C, tol_lp, mid, 1, order))
        if sol["delta"] is not None and sol["delta"] >= 0.0:
            lo, best = mid, sol
        else:
            hi = mid
    rec = {"L": L, "R": R, "C": C, "tol": tol, "order": order, "kind": "reach"}
    if best is None:
        rec["reach_lower"] = 0.0
        rec["certified"] = None
    else:
        cert = gc.certify_scheme(best["left"], best["right"], 1, C)
        rec["certified"] = cert
        rec["reach_lower"] = cert[f"reach_{tol:g}"]
        rec["detected_order"] = comp.detect_order_d1(cert["alpha"], cert["a"])
    lo2, hi2 = lo, min(np.pi, hi + 1e-3)
    sol = gc.solve_lp(build_lp(L, R, C, tol, hi2, 1, order))
    while not (sol["delta_upper_bound"] is not None and sol["delta_upper_bound"] < -gc.CERT_MARGIN):
        hi2 = min(np.pi, hi2 + 0.05)
        sol = gc.solve_lp(build_lp(L, R, C, tol, hi2, 1, order))
        if hi2 >= np.pi:
            break
    for _ in range(12):
        mid = 0.5 * (lo2 + hi2)
        sol = gc.solve_lp(build_lp(L, R, C, tol, mid, 1, order))
        if sol["delta_upper_bound"] is not None and sol["delta_upper_bound"] < -gc.CERT_MARGIN:
            hi2 = mid
        else:
            lo2 = mid
    rec["reach_upper"] = hi2 / np.pi
    return rec


def minimax_ceiling(L, R, C, order) -> dict:
    lo, hi = np.log(1e-9), np.log(1.0)
    best = None
    for _ in range(gc.MINIMAX_ITERS):
        mid = 0.5 * (lo + hi)
        E = np.exp(mid)
        sol = gc.solve_lp(build_lp(L, R, C, E * (1.0 - gc.MU), gc.B0, 1, order))
        if sol["delta"] is not None and sol["delta"] >= 0.0:
            hi, best = mid, sol
        else:
            lo = mid
    rec = {"L": L, "R": R, "C": C, "order": order, "kind": "minimax", "derivative": 1}
    if best is None:
        rec["achieved"] = None
    else:
        cert = gc.certify_scheme(best["left"], best["right"], 1, C)
        rec["certified"] = cert
        rec["achieved"] = cert["band_max_B0"]
        rec["detected_order"] = comp.detect_order_d1(cert["alpha"], cert["a"])
    E_low = np.exp(lo)
    sol = gc.solve_lp(build_lp(L, R, C, E_low, gc.B0, 1, order))
    steps = 0
    while not (sol["delta_upper_bound"] is not None and sol["delta_upper_bound"] < -gc.CERT_MARGIN) and steps < 20:
        E_low *= 0.999
        sol = gc.solve_lp(build_lp(L, R, C, E_low, gc.B0, 1, order))
        steps += 1
    rec["lower_bound"] = float(E_low) if (sol["delta_upper_bound"] is not None
                                        and sol["delta_upper_bound"] < -gc.CERT_MARGIN) else None
    return rec


def _task(item):
    kind, L, R, C, order, extra = item
    if kind == "reach":
        return reach_ceiling(L, R, C, extra, order)
    return minimax_ceiling(L, R, C, order)


def build(processes: int = 6) -> dict:
    budgets = gc.budgets()
    budgets = {k: v for k, v in budgets.items() if k in ("C_SLC1", "C_BMC11")}
    tasks = []
    for L, R in CELLS:
        for name, C in budgets.items():
            for order in ORDERS:
                for tol in gc.TOLS:
                    tasks.append(("reach", L, R, C, order, tol))
                tasks.append(("minimax", L, R, C, order, None))
    with Pool(processes) as pool:
        records = pool.map(_task, tasks, chunksize=1)
    gate7 = json.loads((GATE7 / "ceiling.json").read_text()) if (GATE7 / "ceiling.json").exists() else None
    return {"budgets": budgets, "cells": CELLS, "orders": list(ORDERS), "records": records,
            "gate7_records": gate7["records"] if gate7 else None}


def markdown(data: dict) -> str:
    lines = ["# Gate 8 ceiling of the compact class with formal order (Question 2, dimension)", "",
             "Produced by code/analysis/exponential_family/gate8_ceiling.py. Same LP and dual",
             "certificate as Gate 7 (ceiling.md) with the order conditions added as equality rows;",
             "order 2 is the Gate 7 value (consistency only). Reaches in multiples of pi; brackets",
             "[certified scheme, dual certificate]. The augmented family with core m lies inside the",
             "class of order >= m, so the order-4 column is the ceiling a one-field member with m = 4",
             "can attain at most, and the order-2 column is what it cannot reach in general.", ""]
    for name, C in data["budgets"].items():
        lines += [f"## Budget {name} = {C:.4f}", "",
                  "| L | R | order | free params | R(1e-2) | R(1e-3) | R(1e-4) | D1 minimax B0 [lower, achieved] | detected order of certified schemes |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for L, R in data["cells"]:
            # order 2 from Gate 7
            if data["gate7_records"]:
                cells = []
                for tol in gc.TOLS:
                    rec = gc._find(data["gate7_records"], "reach", L, R, C, tol)
                    cells.append(f"[{rec['reach_lower']:.4f}, {rec['reach_upper']:.4f}]" if rec else "n/a")
                m1 = gc._find(data["gate7_records"], "minimax", L, R, C, 1)
                mm = (f"[{m1['lower_bound']:.3e}, {m1['achieved']:.3e}]" if m1 and m1.get("lower_bound") is not None
                      else "n/a")
                lines.append(f"| {L} | {R} | 2 (Gate 7) | {L + R - 1} | " + " | ".join(cells) + f" | {mm} | n/a |")
            for order in data["orders"]:
                cells, det = [], []
                for tol in gc.TOLS:
                    rec = [r for r in data["records"] if r["kind"] == "reach" and r["L"] == L and r["R"] == R
                           and abs(r["C"] - C) < 1e-9 and r["order"] == order and r["tol"] == tol][0]
                    cells.append(f"[{rec['reach_lower']:.4f}, {rec['reach_upper']:.4f}]")
                    det.append(str(rec.get("detected_order")))
                m1 = [r for r in data["records"] if r["kind"] == "minimax" and r["L"] == L and r["R"] == R
                      and abs(r["C"] - C) < 1e-9 and r["order"] == order][0]
                lo = "none" if m1.get("lower_bound") is None else f"{m1['lower_bound']:.3e}"
                det.append(str(m1.get("detected_order")))
                lines.append(f"| {L} | {R} | {order} | {L + R - 1 - (order - 2) // 2} | " + " | ".join(cells)
                             + f" | [{lo}, {m1['achieved']:.3e}] | {', '.join(det)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    data = build(args.processes)
    (OUT / "ceiling_order.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    text = markdown(data)
    (OUT / "ceiling_order.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
