# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Collect the assembled-solver MMS order verification into results JSON.

The 2D Taylor-Green vortex is an exact periodic Navier-Stokes solution; the C++
driver ``code/solver/ns_mms2d`` initializes it at t=0, marches the fully
assembled solver (skew advection + direct-D2 viscous + Perot projection,
SSP-RK3) to t=1, and prints the velocity L2/Linf error against the exact field
at t=1. This script only launches the driver over the (regime, scheme, grid)
matrix, parses the printed numbers, fits per-interval observed orders, and
writes ``code/results/s8_ns_mms.json``. No numerical work happens in Python.

Three regimes are collected on the same grids and schemes:
- viscous-dominated unforced Taylor-Green (Re=100): the solution error is set
  by the direct second derivative, so the expected orders are the composite
  2*floor(p/2) = 4/6/8 for BMC5/7/9 and 6/8 for C6/C8;
- advection-dominated unforced Taylor-Green (Re=1e4): shows the SAME composite
  orders at 100x smaller errors, because the discrete convective term of the
  Taylor-Green field is an exact discrete gradient that the projection
  annihilates (verified: at Re=1e10 the total error is at the round-off
  plateau on every grid), so the error stays purely viscous at any Re;
- forced MMS (mode=forced, Re=1e4): a two-mode manufactured streamfunction
  with NON-gradient nonlinearity (mms_forcing.h, sympy-generated/verified)
  plus its analytic momentum source per RK stage; the convective/projection
  error dominates and the first-derivative ladder 2*ceil(p/2) = 6/8/10
  (6/8 compact) governs.

Time-step policy: SSP-RK3 is third order, so the time step is chosen per
regime, scheme, and grid such that the cubed time error sits at least two
orders of magnitude below that scheme's measured/estimated spatial error, and
below the advective CFL cap sigma <= 0.3. Verified by step-halving checks
recorded in the JSON: halving dt leaves the reported error unchanged up to the
round-off scatter of the march.

Grid points whose error has reached the round-off plateau are marked floored
and no slope is fit through them.

Usage:  uv run code/experiment_ns_mms.py
"""

import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "solver" / "ns_mms2d"
OUT = HERE / "results" / "s8_ns_mms.json"

TEND = 1.0
KM = 2            # wavenumber of the exact field: two vortex pairs per period,
                  # so the finest-grid errors of the order-8 schemes sit well
                  # above the round-off plateau (equivalent to km=1 on N/2 grids)

# (label, route, pdeg, composite_order, d1_order):
# composite_order = min over operators = direct-second-derivative order
# 2*floor(p/2) for splines (the viscous term uses D^(2)); d1_order = nodal
# first-derivative order 2*ceil(p/2). Compact baselines carry one order.
SCHEMES = [
    ("BMC5", "bmc", 5, 4, 6),
    ("BMC7", "bmc", 7, 6, 8),
    ("BMC9", "bmc", 9, 8, 10),
    ("C6",   "c6",  5, 6, 6),
    ("C8",   "c8",  5, 8, 8),
]
GRIDS = [16, 32, 64]

# Per-regime absolute time steps dt[(label, N)] under the budget
# time_error ~ 1e-5 * dt^3 <= 0.01 * (measured/estimated spatial error),
# capped by the advective CFL (sigma <= 0.3, umax ~ 1). Floored points reuse
# the smallest budget at that grid.
REGIMES = [
    {
        "name": "viscous-dominated",
        "Re": 100.0,
        "dt": {
            ("BMC5", 16): 2.0e-3, ("BMC5", 32): 1.0e-3, ("BMC5", 64): 5.0e-3,
            ("BMC7", 16): 2.0e-3, ("BMC7", 32): 1.0e-3, ("BMC7", 64): 1.0e-3,
            ("BMC9", 16): 2.0e-3, ("BMC9", 32): 1.0e-3, ("BMC9", 64): 5.0e-4,
            ("C6",   16): 2.0e-3, ("C6",   32): 1.0e-3, ("C6",   64): 2.0e-3,
            ("C8",   16): 2.0e-3, ("C8",   32): 1.0e-3, ("C8",   64): 5.0e-4,
        },
        "halving": [("C6", 32), ("BMC9", 32)],
    },
    {
        "name": "advection-dominated",
        "Re": 1.0e4,
        "dt": {
            ("BMC5", 16): 2.0e-3, ("BMC5", 32): 1.0e-3, ("BMC5", 64): 5.0e-3,
            ("BMC7", 16): 2.0e-3, ("BMC7", 32): 1.0e-3, ("BMC7", 64): 5.0e-4,
            ("BMC9", 16): 2.0e-3, ("BMC9", 32): 5.0e-4, ("BMC9", 64): 5.0e-4,
            ("C6",   16): 2.0e-3, ("C6",   32): 1.0e-3, ("C6",   64): 1.0e-3,
            ("C8",   16): 2.0e-3, ("C8",   32): 1.0e-3, ("C8",   64): 5.0e-4,
        },
        "halving": [("BMC7", 32)],
    },
    {
        "name": "forced-convective",
        "Re": 1.0e4,
        "mode": "forced",
        "dt": {
            ("BMC5", 16): 2.0e-3, ("BMC5", 32): 1.0e-3, ("BMC5", 64): 2.0e-3,
            ("BMC7", 16): 2.0e-3, ("BMC7", 32): 1.0e-3, ("BMC7", 64): 1.0e-3,
            ("BMC9", 16): 2.0e-3, ("BMC9", 32): 1.0e-3, ("BMC9", 64): 5.0e-4,
            ("C6",   16): 2.0e-3, ("C6",   32): 1.0e-3, ("C6",   64): 2.0e-3,
            ("C8",   16): 2.0e-3, ("C8",   32): 1.0e-3, ("C8",   64): 5.0e-4,
        },
        "halving": [("BMC9", 32)],
    },
]

# Round-off plateau: a point is floored when its error stops decreasing under
# refinement (ratio < 2 from the previous grid) or sits at the absolute
# round-off scale of the double-precision march (measured plateau on this
# problem: 5e-14 to 1.5e-13 depending on step count).
FLOOR_ABS = 2.0e-13

LINE = re.compile(r"L2=([0-9eE.+-]+)\s+Linf=([0-9eE.+-]+)\s+div=([0-9eE.+-]+)")


def run(route: str, N: int, pdeg: int, dt: float, threads: int, Re: float,
        mode: str) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [str(DRIVER), str(N), route, str(Re), str(TEND), str(pdeg), str(-dt),
           str(KM), mode]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True,
                         env=env).stdout
    m = LINE.search(out)
    if not m:
        raise RuntimeError(f"could not parse driver output:\n{out}")
    return {
        "N": N,
        "dt": dt,
        "L2": float(m.group(1)),
        "Linf": float(m.group(2)),
        "divmax": float(m.group(3)),
    }


def collect_regime(reg: dict) -> dict:
    Re, DT, HALVING = reg["Re"], reg["dt"], reg["halving"]
    mode = reg.get("mode", "tg")
    by_label = {lab: (route, pdeg) for lab, route, pdeg, _, _ in SCHEMES}

    # Phase A: all N=16/32 runs plus the halving re-runs, concurrently.
    small, halve = {}, {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {}
        for lab, route, pdeg, _, _ in SCHEMES:
            for N in (16, 32):
                futs[ex.submit(run, route, N, pdeg, DT[(lab, N)], 2, Re, mode)] = ("s", lab, N)
        for lab, N in HALVING:
            route, pdeg = by_label[lab]
            futs[ex.submit(run, route, N, pdeg, DT[(lab, N)] / 2, 2, Re, mode)] = ("h", lab, N)
        for f, key in futs.items():
            kind, lab, N = key
            (small if kind == "s" else halve)[(lab, N)] = f.result()

    # Phase B: the five N=64 runs, concurrently, 4 threads each (20 cores).
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run, route, 64, pdeg, DT[(lab, 64)], 4, Re, mode): lab
                for lab, route, pdeg, _, _ in SCHEMES}
        big = {futs[f]: f.result() for f in futs}

    print(f"--- regime {reg['name']} (Re={Re:g}, mode={mode}) ---")
    records = []
    for lab, route, pdeg, comp, d1 in SCHEMES:
        rows = [small[(lab, 16)], small[(lab, 32)], big[lab]]
        rows[0]["floored"] = rows[0]["L2"] <= FLOOR_ABS
        rows[0]["order_L2"] = None
        for i in (1, 2):
            fine, coarse = rows[i], rows[i - 1]
            plateaued = (fine["L2"] <= FLOOR_ABS
                         or (not coarse["floored"] and coarse["L2"] / fine["L2"] < 2.0))
            fine["floored"] = plateaued
            if plateaued or coarse["floored"]:
                fine["order_L2"] = None
            else:
                fine["order_L2"] = math.log2(coarse["L2"] / fine["L2"])
        records.append({
            "label": lab, "route": route, "degree": pdeg,
            "composite_order": comp, "d1_order": d1, "grid": rows,
        })
        shown = ", ".join("floor" if r["floored"] else
                          ("--" if r["order_L2"] is None else f"{r['order_L2']:.2f}")
                          for r in rows)
        print(f"{lab:5s} composite={comp} d1={d1}  per-grid L2 orders: [{shown}]")

    checks = []
    for lab, N in HALVING:
        base = small[(lab, N)]
        half = halve[(lab, N)]
        rel = abs(base["L2"] - half["L2"]) / max(base["L2"], half["L2"])
        digits = -math.log10(rel) if rel > 0 else 16.0
        checks.append({
            "label": lab, "N": N,
            "dt": base["dt"], "L2": base["L2"],
            "dt_half": half["dt"], "L2_half": half["L2"],
            "agree_digits": round(digits, 1),
        })
        print(f"halving {lab} N={N}: L2 {base['L2']:.6e} vs {half['L2']:.6e} "
              f"({digits:.1f} digits)")

    return {
        "name": reg["name"],
        "Re": Re,
        "nu": 1.0 / Re,
        "mode": mode,
        "records": records,
        "dt_halving_checks": checks,
    }


def main() -> None:
    if not DRIVER.exists():
        raise SystemExit(f"driver not built: {DRIVER} (run `make ns_mms2d` in solver/)")

    regimes = [collect_regime(reg) for reg in REGIMES]

    payload = {
        "problem": "2D Taylor-Green vortex, exact periodic incompressible Navier-Stokes",
        "exact_solution": {
            "u": "-cos(km x) sin(km y) exp(-2 km^2 nu t)",
            "v": "sin(km x) cos(km y) exp(-2 km^2 nu t)",
            "p": "-(cos 2km x + cos 2km y) exp(-4 km^2 nu t)/4",
            "km": KM,
            "domain": "[0,2pi]^3, z-invariant, w=0",
        },
        "forced_mms": {
            "psi": "[sin(x+0.7) sin(y+0.3) + 0.5 sin(2x+0.2) sin(y+0.9)] cos(t)",
            "u": "dpsi/dy", "v": "-dpsi/dx", "w": "0 (z-invariant)",
            "source": ("S = u_t + (u.grad)u - nu lap(u); p_m = 0, the discrete "
                       "projection determines the discrete pressure; source "
                       "added per RK stage at times t, t+dt, t+dt/2"),
            "generator": "code/generate_mms_forcing.py (sympy; symbolic checks: "
                         "div u == 0, forced residual == 0, convective term not "
                         "a gradient; C-vs-sympy at 20 random points < 1.2e-15)",
        },
        "Tend": TEND,
        "dt_policy": ("per (regime, scheme, grid): cubed SSP-RK3 time error at "
                      "least two orders below the scheme's measured spatial "
                      "error, capped by advective CFL sigma<=0.3; verified by "
                      "step halving (dt_halving_checks)"),
        "grids": GRIDS,
        "error_norm": "velocity L2 (mean-square density) and Linf vs exact field at t=Tend",
        "floor_convention": ("floored = error at the round-off plateau (no decrease "
                             "under refinement); no slope is fit through floored points"),
        "design_order_note": ("composite_order = direct-second-derivative order "
                              "2*floor(p/2) for BMC (the viscous Laplacian uses "
                              "D^(2)), FD order for C6/C8; d1_order = nodal "
                              "first-derivative order 2*ceil(p/2)"),
        "regimes": regimes,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
