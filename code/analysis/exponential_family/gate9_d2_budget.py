"""Best direct second derivative the family reaches at the comparator's own D2 budget.

The equal-conditioning rule of the article holds the comparator's
first-derivative left side fixed, at condition number 93.37. The comparator's
separately tuned second derivative carries a second left side of its own, at
condition number 19.74. This probe answers the question that follows: how
accurate is the best family member whose single left side is held to that
cheaper budget instead?

The scan is over the pentadiagonal cells of the same half-bandwidth as the
comparator's second-derivative left side, width six, that is E_{4,1}(omega)
with one linear transform polynomial and the polynomial rung E_{6,0} with the
same factor, on the frozen static battery of this repository. Every symbol is
the Gate 0 metric on its 20,001-point grid.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import metrics as met
import static_battery as sb

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "exponential_family" / "gate9" / "d2_budget_probe.json"
SLC_D2_COND = 19.735139796373170
XI = np.linspace(-0.5, 0.5, 9)


def evaluate(m: int, omega: float, p1: float) -> dict | None:
    spec = {"m": m,
            "omegas": [omega] if omega > 0 else [],
            "polynomial": [p1] if p1 != 0.0 else []}
    try:
        generator, _ = sb.make_generator(sb.normalise_spec(spec))
        kappa1, kappa2, left, _, _ = sb.symbol_callables(generator, XI)
        values = np.asarray(left(np.linspace(0.0, np.pi, 4001)))
        if float(np.min(values)) <= 0.0:
            return None
        theta = met.theta_grid()
        band = met.BANDS["B0"]
        e1 = np.asarray(kappa1(theta)) / theta - 1.0
        e2 = np.asarray(kappa2(theta)) / theta ** 2 - 1.0
        return {
            "m": m, "omega": float(omega), "p1": float(p1),
            "cond": float(values.max() / values.min()),
            "e1_max_B0": float(met.band_max(theta, e1, band)),
            "e2_max_B0": float(met.band_max(theta, e2, band)),
            "kappa2_min": float(np.min(np.asarray(kappa2(theta))[theta < np.pi])),
        }
    except Exception:
        return None


def main() -> int:
    rows = []
    for omega in np.linspace(0.30, 0.95, 27) * np.pi:
        for p1 in np.concatenate([[0.0], np.linspace(-0.08, 0.08, 33)]):
            row = evaluate(4, float(omega), float(p1))
            if row is not None:
                rows.append(row)
    for p1 in np.linspace(-0.10, 0.10, 81):
        row = evaluate(6, 0.0, float(p1))
        if row is not None:
            rows.append(row)
    feasible = [r for r in rows
                if r["cond"] <= SLC_D2_COND and r["kappa2_min"] > 0.0]
    feasible.sort(key=lambda r: r["e2_max_B0"])
    payload = {
        "producer": "code/analysis/exponential_family/gate9_d2_budget.py",
        "budget_cond": SLC_D2_COND,
        "width": 6,
        "cells": ["E_{4,1}(omega) x P_1", "E_{6,0} x P_1"],
        "evaluated": len(rows),
        "feasible": len(feasible),
        "best": feasible[0] if feasible else None,
        "best_ten": feasible[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"d2_budget_probe: {len(rows)} evaluated, {len(feasible)} inside the budget")
    if feasible:
        best = feasible[0]
        print(f"  best e2_max_B0 = {best['e2_max_B0']:.6g} at cond {best['cond']:.4f} "
              f"(m={best['m']}, omega={best['omega']:.6f}, p1={best['p1']:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
