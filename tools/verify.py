"""One-command verification of the headline results.

Part A recomputes the discrete identities and convergence orders live from the
operator library in code/spline_analysis.py. Part B re-derives the evolved-flow,
transport and immersed-wall numbers from the machine-readable receipts in
results/. Exits non-zero if any check fails.

    make verify            (or: uv run python tools/verify.py)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import spline_analysis as sa  # noqa: E402

EF = ROOT / "results" / "exponential_family"
FAILURES: list[str] = []


def check(ok: bool, label: str) -> str:
    if not ok:
        FAILURES.append(label)
    return "ok" if ok else "FAIL"


def identities() -> None:
    print("A1. Discrete identities on a periodic grid, N = 64")
    n, h = 64, 1.0 / 64
    collapse = 0.0
    for p in range(1, 6):
        collapse = max(collapse,
                       np.abs(sa.collocation_derivative_matrix(2 * p + 1, n) - sa.iga_derivative_matrix(p, n)).max(),
                       np.abs(sa.collocation_second_derivative_matrix(2 * p + 1, n)
                              - sa.iga_second_derivative_matrix(p, n)).max())
    sbp = sym = 0.0
    top = -np.inf
    rng = np.random.default_rng(0)
    ops = [(sa.collocation_derivative_matrix(p, n, h), sa.collocation_second_derivative_matrix(p, n, h))
           for p in range(2, 12)]
    ops += [(sa.iga_derivative_matrix(p, n, h), sa.iga_second_derivative_matrix(p, n, h)) for p in range(1, 6)]
    H = sa.periodic_sbp_norm(n, h)
    for D, D2 in ops:
        sbp = max(sbp, np.abs(H @ D + D.T @ H).max() / np.abs(D).max())
        sym = max(sym, np.abs(H @ D2 - (H @ D2).T).max() / np.abs(D2).max())
        for _ in range(4):
            u = rng.standard_normal(n)
            u -= u.mean()
            top = max(top, float(u @ H @ D2 @ u))
    print(f"   BMC_(2p+1) = BMG_p, first and second derivative, p = 1..5   max |diff| = {collapse:.1e}  "
          f"{check(collapse < 1e-12, 'collapse identity')}")
    print(f"   SBP  H D + D^T H = 0, BMC2..BMC11 and BMG1..BMG5           rel. residual = {sbp:.1e}  "
          f"{check(sbp < 1e-13, 'SBP identity')}")
    print(f"   H D2 symmetric, 15 operators                                rel. residual = {sym:.1e}  "
          f"{check(sym < 1e-13, 'D2 symmetry')}")
    print(f"   <D2 u, u>_H <= 0 on zero-mean fields                        max = {top:.2e}  "
          f"{check(top < 0, 'D2 sign')}")


def convergence() -> None:
    print("\nA2. First-derivative convergence, u = sin 2x + cos(3x)/2, max norm, N = 48 -> 64")
    print("    degree p   formal order 2*ceil(p/2)   observed")
    for p in (2, 3, 4, 5, 6, 7, 8, 9):
        errs = []
        for n in (48, 64):
            h = 2 * np.pi / n
            x = np.arange(n) * h
            u = np.sin(2 * x) + 0.5 * np.cos(3 * x)
            du = 2 * np.cos(2 * x) - 1.5 * np.sin(3 * x)
            errs.append(np.abs(sa.collocation_derivative_matrix(p, n, h) @ u - du).max())
        q = 2 * ((p + 1) // 2)
        obs = np.log(errs[0] / errs[1]) / np.log(64 / 48)
        print(f"    BMC{p:<2d}      {q:>2d}                         {obs:5.2f}   {check(abs(obs - q) < 0.25, f'order BMC{p}')}")


def taylor_green() -> None:
    print("\nB1. Dealiased Taylor-Green, Re = 1600, t* = 9.0026: relative RMS velocity error against the")
    print("    same-grid dealiased Fourier pseudospectral solution (spatial operator error alone)")
    d6 = json.loads((EF / "gate9" / "e2_decision_core6.json").read_text())["e_full"]
    d5 = json.loads((EF / "gate9" / "e2_decision.json").read_text())["e_full"]
    print("    route                                   64^3        128^3      gain over spectral-like compact")
    names = [("spectral-like compact (pentadiagonal)", "sl"), ("enriched member X6 x P2", "core6"),
             ("polynomial rung BMC11", "bmc_p11"), ("band member E(6,2)", "exp")]
    for label, key in names:
        e64, e128 = d6["N64"][key], d6["N128"][key]
        g64, g128 = d6["N64"]["sl"] / e64, d6["N128"]["sl"] / e128
        print(f"    {label:<40s}{e64:.3e}   {e128:.3e}   {g64:5.2f}x, {g128:5.2f}x")
    g = [d6[f"N{n}"]["sl"] / d6[f"N{n}"]["core6"] for n in (64, 128)]
    b = [d5[f"N{n}"]["sl"] / d5[f"N{n}"]["exp"] for n in (64, 128)]
    check(4.3 < min(g) and max(g) < 4.6, "enriched member TGV gain")
    check(37 < min(b) and max(b) < 43, "band member TGV gain")


def transport() -> None:
    print("\nB2. Long-horizon dispersive transport, N = 128: crossings of the periodic box before the")
    print("    packet error reaches 1 percent (closed form, checked against SSP-RK3 to 6e-11)")
    d = json.loads((EF / "gate9" / "e3_transport_core6.json").read_text())["cells_1d"]
    print("    carrier theta_c     spectral-like compact   enriched member   band member E(6,2)")
    for key, label in (("1.256637", "0.4 pi"), ("1.570796", "0.5 pi"), ("1.884956", "0.6 pi")):
        c = d[key]
        print(f"    {label:<19s} {c['slc_q4']['horizon_at_1_percent']:8.3f}             "
              f"{c['core6']['horizon_at_1_percent']:7.3f} ({c['core6']['crossing_ratio_over_slc_q4']:4.2f}x)   "
              f"{c['champion']['horizon_at_1_percent']:6.3f} ({c['champion']['crossing_ratio_over_slc_q4']:5.1f}x)")
    check(abs(d["1.570796"]["champion"]["crossing_ratio_over_slc_q4"] - 37.0) < 0.1, "transport horizon")


def wall() -> None:
    print("\nB3. Immersed smooth wall, periodic Helmholtz with Schur-complement marker coupling, N = 96..192,")
    print("    four marker shifts: BMC9 against the strongest compact route (C10 + 12-point evaluation layer)")
    s = json.loads((ROOT / "results" / "ibm_cert" / "summary.json").read_text())
    a = s["adoption_answer"]
    tr, ex = a["accuracy_factors"]["wall_trace"], a["accuracy_factors"]["exterior_relative_l2"]
    ttr, tex = a["time_premium_ratios"]["wall_trace_blocker"], a["time_premium_ratios"]["exterior_blocker"]
    rows = list(csv.DictReader(open(ROOT / "results" / "ibm_cert" / "full_results.csv")))
    inv = sum(r["invariants_pass"] == "True" for r in rows)
    print(f"    worst-shift error reduction: wall trace {tr:.2f}x, exterior field {ex:.2f}x")
    print(f"    complete one-solve time ratio: {ttr:.2f}x and {tex:.2f}x")
    print(f"    runs with every discrete invariant passing: {inv} / {len(rows)}")
    check(tr > 62.7 and ex > 32.1 and inv == len(rows), "wall factors")


def main() -> int:
    t0 = time.perf_counter()
    identities()
    convergence()
    taylor_green()
    transport()
    wall()
    print(f"\n{'PASS' if not FAILURES else 'FAIL: ' + ', '.join(FAILURES)}  ({time.perf_counter() - t0:.2f} s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
