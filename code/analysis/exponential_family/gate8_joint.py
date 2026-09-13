"""Gate 8: joint dominance in the order-4 compact class, then inversion of the family map.

Step (a). In the compact class of type (L, R) with formal order >= 4 and left
conditioning <= C, find the largest common margin delta such that a scheme
satisfies simultaneously, against SLC's yardsticks,

    R(1e-2) >= (1 + delta) R_L(1e-2),   R(1e-3) >= (1 + delta) R_L(1e-3),
    R(1e-4) >= (1 + delta) R_L(1e-4),   G(1e-2) >= (1 + delta) G_L,
    E1max(B0) <= (1 - delta) E1_L.

For a fixed delta the reach and band-maximum conditions are the semi-infinite
linear rows of gate7_ceiling (|N/theta - Lf| <= tol Lf on a theta grid), the
conditioning is 1 <= Lf <= C, consistency and the fourth-order condition are
equality rows (gate8_ceiling.order_rows). The group-velocity condition
|d kappa_1/d theta - 1| <= 1e-2 on (0, Theta_G] is not linear
(d kappa_1/d theta = (N' Lf - N Lf')/Lf^2); it is imposed by sequential
linearization around the previous iterate,

    d kappa_1/d theta - 1 = (eps + theta eps')/Lf - theta eps Lf'/Lf^2,   eps = N/theta - Lf,

with Lf and Lf' frozen from the previous scheme and a trust region on the
coefficients, and every iterate is certified by the Gate 0 metrics code on
the true (nonlinear) yardsticks: a delta is feasible only when a certified
scheme has every true margin at least delta minus a tolerance. Bisection on
delta gives the joint D1 ceiling delta* of the class (the D2 of a compact
pair is a separate operator and is not constrained here; the realized
member carries its own literal D2, step (c)).

Step (b). The certified scheme at delta* (and on the segment toward SLC's
scheme if needed) is handed to the realization solver of gate8_realization
(core m = 4, (n - 4)/2 slots, P of degree (n - 4)/2, the C^2 limit at width
n = 2 (L + 1)); an exact realization (residual at the 1e-8 level or better)
is a one-field member with the LP scheme as its nodal D1.

Step (c). The realized member is verified with gate8_dominance.verify_member:
every Gate 0 yardstick including the literal second derivative, the
positivity certificates of b and kappa_2 b, the coupled channel (129 xi),
the three sample routes; a member that dominates SLC on every yardstick
(margins >= 0, coupled within 2x, certified) is written to members/.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
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
import gate7_dominance as gd  # noqa: E402
import gate8_ceiling as g8c  # noqa: E402
import gate8_dominance as gdom  # noqa: E402
import gate8_family as g8  # noqa: E402
import gate8_realization as gr  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate8"
MEMBERS = OUT / "members"
PI = np.pi
B0 = 2.0 * PI / 3.0
MU = 1e-4
ACC_POINTS = 3000
LEFT_POINTS = 2000
G_ITER = 14
DELTA_CHECK = 2e-4
TRUST = 0.05
COND_MARGIN = 1e-4
COND_ACCEPT = 1e-9
ORDER = 4
CELLS = {8: (3, 3), 10: (4, 4)}


# ---------------------------------------------------------------------------
# compact symbols and true margins
# ---------------------------------------------------------------------------

def compact_functions(alpha, a):
    Lf = comp.compact_left(alpha)
    k1 = comp.compact_kappa1(alpha, a)
    alpha = np.asarray(alpha, dtype=float)
    a = np.asarray(a, dtype=float)

    def dk1(theta):
        theta = np.asarray(theta, dtype=float)
        N = np.zeros(theta.shape)
        dN = np.zeros(theta.shape)
        for s, coeff in enumerate(a, start=1):
            N += 2.0 * coeff * np.sin(s * theta)
            dN += 2.0 * coeff * s * np.cos(s * theta)
        L = Lf(theta)
        dL = np.zeros(theta.shape)
        for l, al in enumerate(alpha, start=1):
            dL += -2.0 * al * l * np.sin(l * theta)
        return (dN * L - N * dL) / (L * L)

    return Lf, k1, dk1


def true_yardsticks(alpha, a, theta_points: int = met.THETA_POINTS) -> dict:
    """Return SLC's D1 yardsticks of a compact scheme by the Gate 0 code."""
    Lf, k1, dk1 = compact_functions(alpha, a)
    theta = met.theta_grid(theta_points)
    e1_fn = lambda t: k1(t) / t - 1.0
    e1 = e1_fn(theta)
    row = {}
    for tol in met.REACH_TOLS:
        row[f"reach_d1_{tol:g}"] = met.reach(theta, e1, tol, err_fn=e1_fn)["refined"] / PI
    gv = dk1(theta) - 1.0
    row["gv_reach_1e-2"] = met.reach(theta, gv, met.GV_TOL)["linear"] / PI
    row["e1_max_B0"] = met.band_max(theta, e1, B0)
    row.update(met.symbol_cond(Lf, theta_points))
    return row


def margins_d1(row: dict, target: dict) -> dict:
    keys = ("reach_d1_0.01", "reach_d1_0.001", "reach_d1_0.0001", "gv_reach_1e-2", "e1_max_B0")
    return {k: ((row[k] - target[k]) / target[k] if gd.HIGHER_IS_BETTER[k] else 1.0 - row[k] / target[k]) for k in keys}


# ---------------------------------------------------------------------------
# the joint LP
# ---------------------------------------------------------------------------

def build_joint_lp(L, R, C, delta, target, G_state=None, x_prev=None, trust=None) -> dict:
    n_left = L + 1
    n_right = R
    n_var = n_left + n_right + 1
    A_blocks, b_blocks = [], []

    def acc_block(tol, Theta, pts):
        th = Theta * np.arange(1, pts + 1) / pts
        Lacc = gc._left_rows(th, L)
        Nacc = gc._right_rows_d1(th, R) / th[:, None]
        A1 = np.zeros((pts, n_var)); A2 = np.zeros((pts, n_var))
        A1[:, :n_left] = -(1.0 + tol) * Lacc; A1[:, n_left:n_left + n_right] = Nacc; A1[:, -1] = 1.0
        A2[:, :n_left] = (1.0 - tol) * Lacc; A2[:, n_left:n_left + n_right] = -Nacc; A2[:, -1] = 1.0
        A_blocks.extend([A1, A2]); b_blocks.extend([np.zeros(pts), np.zeros(pts)])

    for tol, key in ((1e-2, "reach_d1_0.01"), (1e-3, "reach_d1_0.001"), (1e-4, "reach_d1_0.0001")):
        Theta = min(PI, target[key] * PI * (1.0 + delta))
        acc_block(tol * (1.0 - MU), Theta, ACC_POINTS)
    acc_block(target["e1_max_B0"] * (1.0 - delta) * (1.0 - MU), B0, ACC_POINTS)
    if G_state is not None:
        th, Lf0, dLf0 = G_state
        Lacc = gc._left_rows(th, L)
        dLacc = np.column_stack([np.zeros_like(th)] + [-2.0 * l * np.sin(l * th) for l in range(1, L + 1)])
        Nacc = gc._right_rows_d1(th, R) / th[:, None]
        dNacc = np.column_stack([2.0 * (s * np.cos(s * th) * th - np.sin(s * th)) / th ** 2 for s in range(1, R + 1)])
        # eps = Nacc a - Lacc c ; eps' = dNacc a - dLacc c
        # g = (eps + th eps')/Lf0 - th eps dLf0/Lf0^2
        w1 = (1.0 / Lf0)[:, None]
        w2 = (th * dLf0 / Lf0 ** 2)[:, None]
        Ga = (Nacc + th[:, None] * dNacc) * w1 - Nacc * w2
        Gc = (-Lacc - th[:, None] * dLacc) * w1 + Lacc * w2
        Grow = np.zeros((th.size, n_var))
        Grow[:, :n_left] = Gc; Grow[:, n_left:n_left + n_right] = Ga
        tolg = met.GV_TOL * (1.0 - MU)
        A1 = Grow.copy(); A1[:, -1] = 1.0
        A2 = -Grow.copy(); A2[:, -1] = 1.0
        A_blocks.extend([A1, A2]); b_blocks.extend([tolg * np.ones(th.size), tolg * np.ones(th.size)])
    theta_left = np.linspace(0.0, PI, LEFT_POINTS)
    Lleft = gc._left_rows(theta_left, L)
    A3 = np.zeros((LEFT_POINTS, n_var)); A3[:, :n_left] = -Lleft
    A4 = np.zeros((LEFT_POINTS, n_var)); A4[:, :n_left] = Lleft
    A_blocks.extend([A3, A4]); b_blocks.extend([-np.ones(LEFT_POINTS), C * (1.0 - COND_MARGIN) * np.ones(LEFT_POINTS)])
    A_ub = np.vstack(A_blocks); b_ub = np.concatenate(b_blocks)
    A_eq = np.zeros((1, n_var)); A_eq[0, :n_left] = -np.array([1.0] + [2.0] * L)
    A_eq[0, n_left:n_left + n_right] = 2.0 * np.arange(1, R + 1)
    b_eq = np.zeros(1)
    Aord, bord = g8c.order_rows(L, R, ORDER)
    A_eq = np.vstack([A_eq, Aord]); b_eq = np.concatenate([b_eq, bord])
    lb = np.concatenate([[0.5], -C * np.ones(L), -gc.A_BOX * C * np.ones(n_right), [-1e3 * C]])
    ub = np.concatenate([[2.0 * C], C * np.ones(L), gc.A_BOX * C * np.ones(n_right), [1e3 * C]])
    if x_prev is not None and trust is not None:
        xp = np.asarray(x_prev, dtype=float)
        width = trust * np.maximum(1.0, np.abs(xp[:-1]))
        lb[:-1] = np.maximum(lb[:-1], xp[:-1] - width)
        ub[:-1] = np.minimum(ub[:-1], xp[:-1] + width)
    c_obj = np.zeros(n_var); c_obj[-1] = -1.0
    return {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "lb": lb, "ub": ub,
            "c": c_obj, "n_left": n_left, "n_right": n_right}


def scheme_of(sol) -> tuple[list, list]:
    left, right = gc.polish_consistency(sol["left"], sol["right"], 1)
    c = np.asarray(left); a = np.asarray(right)
    return [float(v) for v in c[1:] / c[0]], [float(v) for v in a / c[0]]


def slc_vector(L: int, R: int) -> np.ndarray:
    """Return SLC's D1 as an LP vector (c_0, c_l, a_s, slack) scaled to min Lf = 1."""
    alpha = list(comp.SLC_D1["alpha"]) + [0.0] * (L - 2)
    a = list(comp.SLC_D1["a"]) + [0.0] * (R - 3)
    Lf = comp.compact_left(alpha)
    scale = 1.0 / float(np.min(Lf(np.linspace(0.0, PI, LEFT_POINTS))))
    return np.array([scale] + [scale * v for v in alpha] + [scale * v for v in a] + [0.0])


def g_state(alpha, a, delta, target, scale: float = 1.0):
    """Return the frozen (theta, Lf, Lf') of the G linearization in the LP's scaling.

    The LP variables carry c_0 = scale (1 <= Lf <= C), so the frozen left
    symbol must be scaled the same way as the linear error eps = N/theta - Lf.
    """
    Lf, _, _ = compact_functions(alpha, a)
    Theta_G = min(PI, target["gv_reach_1e-2"] * PI * (1.0 + delta))
    th = Theta_G * np.arange(1, ACC_POINTS + 1) / ACC_POINTS
    dL = np.zeros(th.shape)
    for l, al in enumerate(alpha, start=1):
        dL += -2.0 * al * l * np.sin(l * th)
    return (th, scale * Lf(th), scale * dL)


def feasible(L, R, C, delta, target, verbose=False, x_start=None):
    """Return (alpha, a, row, margins) with every true margin >= delta - DELTA_CHECK, else None.

    Sequential linear programming from SLC's own scheme (feasible at
    delta = 0): the G rows are linearized at the current iterate, a trust
    region bounds the step, a step is accepted when the true smallest margin
    improves, and the trust region shrinks otherwise.
    """
    x_prev = slc_vector(L, R) if x_start is None else np.asarray(x_start, dtype=float)
    n_left = L + 1
    alpha0 = list(x_prev[1:n_left] / x_prev[0]); a0 = list(x_prev[n_left:n_left + R] / x_prev[0])
    row0 = true_yardsticks(alpha0, a0)
    mg0 = margins_d1(row0, target)
    best = (alpha0, a0, row0, min(mg0.values()), mg0)
    state = g_state(alpha0, a0, delta, target, scale=float(x_prev[0]))
    trust = 0.2
    for it in range(G_ITER):
        lp = build_joint_lp(L, R, C, delta, target, G_state=state, x_prev=x_prev, trust=trust)
        sol = gc.solve_lp(lp)
        if sol["delta"] is None or sol["delta"] < 0.0:
            trust *= 0.5
            if trust < 2e-3:
                break
            continue
        alpha, a = scheme_of(sol)
        row = true_yardsticks(alpha, a)
        mg = margins_d1(row, target)
        mmin = min(mg.values())
        if verbose:
            print(f"    delta {delta:.5f} it {it} trust {trust:.3f} min margin {mmin:+.5f} cond {row['cond']:.6f} "
                  f"G {row['gv_reach_1e-2']:.4f} lp slack {sol['delta']:.2e}", flush=True)
        if mmin >= delta - DELTA_CHECK and row["cond"] <= C * (1.0 + COND_ACCEPT):
            return alpha, a, row, mg
        if mmin > best[3]:
            best = (alpha, a, row, mmin, mg)
            x_prev = np.asarray(sol["x"])
            state = g_state(alpha, a, delta, target, scale=float(x_prev[0]))
            trust = min(0.4, 1.5 * trust)
        else:
            trust *= 0.5
            if trust < 2e-3:
                break
    if best[3] >= delta - DELTA_CHECK and best[2]["cond"] <= C * (1.0 + COND_ACCEPT):
        return best[0], best[1], best[2], best[4]
    return None


def joint_ceiling(L, R, C, target, verbose=True) -> dict:
    lo, hi = -0.02, 0.25
    best = None
    res0 = feasible(L, R, C, 0.0, target, verbose)
    if res0 is None:
        return {"L": L, "R": R, "C": C, "delta_star": None, "note": "infeasible at delta = 0 with the G linearization"}
    best = (0.0,) + res0
    lo = 0.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        res = feasible(L, R, C, mid, target, verbose)
        if res is not None:
            lo, best = mid, (mid,) + res
        else:
            hi = mid
    delta, alpha, a, row, mg = best
    return {"L": L, "R": R, "C": C, "delta_star": delta, "delta_upper": hi, "alpha": alpha, "a": a,
            "row": row, "margins": mg, "detected_order": comp.detect_order_d1(alpha, a)}


# ---------------------------------------------------------------------------
# realization and verification
# ---------------------------------------------------------------------------

def realize(alpha, a, n: int, m: int, q: int, processes: int, seeds: int, name: str) -> dict:
    spec = {"n": n, "d1": {"alpha": tuple(alpha), "a": tuple(a)}, "d2": None, "cores": (m,), "degrees": (q,),
            "published_order": ORDER}
    gr.TARGETS[name] = spec
    cell = gr.fit_cell(name, m, q, processes, seeds=seeds)
    return cell


def verify_and_record(cell: dict, target: dict, name: str, C: float) -> dict:
    gen = gr.build_generator(cell["m"], cell["x"], cell["slots_count"], cell["q"])
    ver = gdom.verify_member(gen, name)
    vm = ver["metrics"]
    row = {k: (vm[k + "_refined"] if k.startswith("reach") else vm[k]) for k in gd.YARDSTICKS}
    row["ec_max_B0"] = vm["ec_max_B0"]
    mg = gd.margins(row, target)
    ec_ratio = vm["ec_max_B0"] / (gd.COUPLED_FACTOR * max(vm["e1_max_B0"], vm["e2_max_B0"]))
    pos = ver["positivity"]
    dominates = bool(min(mg.values()) >= 0.0 and ec_ratio <= 1.0 and pos["b_certified"] and pos["kappa2b_certified"]
                     and vm["cond"] <= C * (1.0 + COND_ACCEPT))
    out = {"label": gen.label(), "margins": mg, "smallest_margin": float(min(mg.values())), "ec_ratio": float(ec_ratio),
           "cond": vm["cond"], "max_b": vm.get("max_left"), "dominates": dominates,
           "positivity": {k: pos[k] for k in ("b_certified", "kappa2b_certified", "b_min", "kappa2b_min_on_tail")},
           "routes": {k: v for k, v in ver["routes"].items() if k != "beta_digits"},
           "sufficient_conditions": {k: v for k, v in ver["sufficient_conditions"].items() if k != "roots"},
           "metrics": vm, "record": ver["record"]}
    if dominates:
        MEMBERS.mkdir(parents=True, exist_ok=True)
        rec = dict(ver["record"]); rec["name"] = name; rec["dominates"] = "SLC1992"; rec["margins"] = mg
        rec["verification"] = {"routes": out["routes"], "positivity": pos}
        (MEMBERS / f"{name}.json").write_text(json.dumps(rec, indent=2) + "\n")
    return out


def run(n: int, C: float, budget_name: str, target: dict, processes: int, seeds: int, segment: bool = True) -> dict:
    L, R = CELLS[n]
    m, q = 4, (n - 4) // 2
    t0 = time.time()
    print(f"[{budget_name}, n={n}] joint LP in the order-4 ({L}, {R}) class", flush=True)
    ceil = joint_ceiling(L, R, C, target)
    print(f"  delta* = {ceil.get('delta_star')} (upper {ceil.get('delta_upper')}), margins {ceil.get('margins')}", flush=True)
    out = {"n": n, "C": C, "budget": budget_name, "ceiling": ceil, "seconds_lp": time.time() - t0, "realizations": []}
    if ceil.get("delta_star") is None:
        return out
    slc_alpha = list(comp.SLC_D1["alpha"]) + [0.0] * (L - 2)
    slc_a = list(comp.SLC_D1["a"]) + [0.0] * (R - 3)
    lambdas = [1.0] + ([0.5, 0.25, 0.1] if segment else [])
    for lam in lambdas:
        alpha = [(1 - lam) * x + lam * y for x, y in zip(slc_alpha, ceil["alpha"])]
        a = [(1 - lam) * x + lam * y for x, y in zip(slc_a, ceil["a"])]
        row = true_yardsticks(alpha, a)
        mg = margins_d1(row, target)
        name = f"X8_joint_{budget_name}_n{n}_lam{lam:g}"
        t1 = time.time()
        cell = realize(alpha, a, n, m, q, processes, seeds, name)
        block = {"lambda": lam, "alpha": alpha, "a": a, "d1_margins": mg, "d1_cond": row["cond"],
                 "residual_max": cell["residual_max"], "exact": cell["exact_realization"],
                 "generator": cell.get("generator"), "newton": cell.get("newton"), "seconds": time.time() - t1}
        print(f"  lambda {lam}: D1 min margin {min(mg.values()):+.5f}, realization residual {cell['residual_max']:.2e} "
              f"({time.time() - t1:.0f} s)", flush=True)
        if cell["exact_realization"] or cell["residual_max"] < 1e-6:
            block["verification"] = verify_and_record(cell, target, name, C)
            v = block["verification"]
            print(f"    member {v['label']}: smallest margin {v['smallest_margin']:+.5f}, ec ratio {v['ec_ratio']:.3f}, "
                  f"cond {v['cond']:.3f}, certified {v['positivity']['b_certified']}/{v['positivity']['kappa2b_certified']}, "
                  f"dominates {v['dominates']}; margins {v['margins']}", flush=True)
            out["realizations"].append(block)
            if v["dominates"]:
                break
        else:
            out["realizations"].append(block)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=9)
    parser.add_argument("--widths", type=str, default="10,8")
    parser.add_argument("--budgets", type=str, default="C_SLC1,C_BMC11")
    parser.add_argument("--seeds", type=int, default=900)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    targets = gd.comparator_targets()
    budgets = gc.budgets()
    data = {"targets": targets, "runs": []}
    for n in [int(w) for w in args.widths.split(",")]:
        for bname in args.budgets.split(","):
            res = run(n, budgets[bname], bname, targets["SLC1992"], args.processes, args.seeds)
            data["runs"].append(res)
            (OUT / f"joint{args.suffix}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
