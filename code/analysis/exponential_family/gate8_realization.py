"""Gate 8, Question 2: are SLC's spectral-like D1 and tuned q8 one-field collocation schemes?

A compact first derivative of type (L, R) with left side 1 + 2 sum_l alpha_l cos(l theta)
and right side sum_s 2 a_s sin(s theta) is the collocation derivative
[psi'(j - k)] [psi(j - k)]^{-1} of a generator psi exactly when the integer
samples satisfy

    psi(l) / psi(0) = alpha_l   (l = 1..nmax - 1, alpha_l = 0 beyond L),
    psi'(s) / psi(0) = -a_s     (s = 1..nmax - 1, a_s = 0 beyond R),

because b(theta) = psi(0) + 2 sum psi(n) cos(n theta) and
kappa_1 b = -2 sum psi'(n) sin(n theta). The generator psi = P(-D^2) phi has
width n = 2 nmax, its samples at |n| = nmax vanish (C^2 endpoint), so the
scheme must have L, R <= nmax - 1: SLC's (2, 3) needs n = 8 with psi(3) = 0
imposed, tuned q8's (1, 5) needs n = 12 with psi(2..5) = 0 imposed.

Unknowns. The core m, the real monic polynomial Q(s) = prod_j (s - s_j) of
degree (n - m)/2 whose roots are the slot parameters (a negative root is an
imaginary pair, a positive one a real pair, a conjugate pair a quad), and
P(t) = 1 + sum p_l t^l of degree q <= (n - 4)/2. The map (Q, P) -> samples
is real-analytic; the residual vector is fitted by least squares from many
seeds, polished, and then a square Newton solve on the most independent
equations (QR with pivoting on the Jacobian) is reported with the residual
of the equations it leaves out.

Counting. The sample ratios number 2 (nmax - 1); the generator satisfies
first-order consistency and formal order m automatically (e_1 = O(theta^m)),
which the target satisfies to its printed precision, so 1 + (m - 2)/2 of
the equations are implied. For SLC (n = 8, exactly fourth order, so m = 4)
this leaves 4 equations for the 4 unknowns (2 slots, q = 2); for tuned q8
(n = 12, exactly eighth order, so m = 8) 6 equations for 6 unknowns (2
slots, q = 4). Both systems are square; whether they have a real admissible
solution is the numerical question answered here. A residual at the level
of the printed coefficients (about 1e-8 for SLC, whose printed values miss
their own consistency condition by 2e-8) counts as an exact realization.

For a realized generator the literal second derivative kappa_2 = -B_2/B_0 of
the same field is computed with its sign certificate, and its off-grid
transfers T_q (residue route) are compared with the comparator's own D2 and
with the Gate 5a Lagrange augmentation routes (nodal operators then a
centered Lagrange layer of 8 or 12 nodes, and the differentiated Lagrange
interpolant).
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
from scipy.linalg import qr
from scipy.optimize import least_squares, root

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import gate7_family as g7  # noqa: E402
import gate8_family as g8  # noqa: E402
import lagrange_layers as ll  # noqa: E402
import metrics as met  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate8"
PI = np.pi
B0 = 2.0 * PI / 3.0
XI_FULL = np.linspace(-0.5, 0.5, 129)
SEEDS = {0: 60, 1: 400, 2: 1200, 3: 2000, 4: 2500}
EXACT_LEVEL = 1e-8
ALPHA_CAP = 30.0

TARGETS = {
    "SLC1992": {"n": 8, "d1": comp.SLC_D1, "d2": comp.SLC_D2, "cores": (4, 6, 8), "degrees": (1, 2),
                 "published_order": 4},
    "TunedQ8": {"n": 12, "d1": comp.Q8_D1, "d2": comp.Q8_D2, "cores": (4, 6, 8, 10, 12), "degrees": (3, 4),
                "published_order": 8},
}


# ---------------------------------------------------------------------------
# Targets and parametrization
# ---------------------------------------------------------------------------

def target_vector(spec: dict) -> np.ndarray:
    n = spec["n"]
    nmax = n // 2
    alpha = list(spec["d1"]["alpha"]) + [0.0] * (nmax - 1)
    a = list(spec["d1"]["a"]) + [0.0] * (nmax - 1)
    return np.array(alpha[:nmax - 1] + [-v for v in a[:nmax - 1]], dtype=float)


def slots_from_Q(qcoeffs) -> list:
    """Return the Gate 7 slots of the monic real polynomial s^L + q_{L-1} s^{L-1} + ... + q_0."""
    qcoeffs = [float(v) for v in qcoeffs]
    if not qcoeffs:
        return []
    roots = np.roots([1.0] + qcoeffs)
    slots = []
    used = np.zeros(roots.size, dtype=bool)
    for i, s in enumerate(roots):
        if used[i]:
            continue
        if abs(s.imag) <= 1e-9 * (1.0 + abs(s)):
            sr = float(s.real)
            if abs(sr) < 1e-8:
                raise ValueError("a slot parameter vanished (degenerate with the core)")
            if sr < 0.0:
                slots.append(g7.Slot(0.0, float(np.sqrt(-sr))))
            else:
                a = float(np.sqrt(sr))
                if a > ALPHA_CAP:
                    raise ValueError("real pair beyond the cap")
                slots.append(g7.Slot(a, 0.0))
            used[i] = True
        else:
            lam = np.sqrt(s + 0j)
            a, w = abs(lam.real), abs(lam.imag)
            if a > ALPHA_CAP:
                raise ValueError("quad real part beyond the cap")
            slots.append(g7.Slot(a, w))
            used[i] = True
            # mark the conjugate
            j = int(np.argmin(np.abs(roots - s.conjugate()) + used * 1e9))
            used[j] = True
    return slots


def Q_from_slots(slots) -> list:
    svals = []
    for slot in slots:
        s = slot.s
        svals += [s, s.conjugate()] if slot.kind == "quad" else [s]
    poly = np.array([1.0 + 0j])
    for s in svals:
        poly = np.convolve(poly, np.array([1.0, -s]))
    return [float(v.real) for v in poly[1:]]


def build_generator(m: int, x, Ls: int, q: int) -> g8.PolynomialGenerator:
    x = list(x)
    slots = slots_from_Q(x[:Ls])
    poly = [float(c) / PI ** (2 * (l + 1)) for l, c in enumerate(x[Ls:Ls + q])]
    return g8.PolynomialGenerator(m, slots, poly)


def sample_ratios(gen: g8.PolynomialGenerator) -> np.ndarray:
    s = gen.integer_samples()
    c = int(np.searchsorted(s.n, 0))
    psi0 = s.beta[c]
    left = s.beta[c + 1:c + gen.nmax] / psi0
    right = s.dbeta[c + 1:c + gen.nmax] / psi0
    return np.concatenate([left, right])


def residual(x, m: int, Ls: int, q: int, target: np.ndarray) -> np.ndarray:
    try:
        gen = build_generator(m, x, Ls, q)
        return sample_ratios(gen) - target
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.full(target.size, 1e3)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def random_seed(rng, Ls: int, q: int) -> np.ndarray:
    slots = []
    for _ in range(Ls):
        kind = rng.choice(["I", "R", "Q"], p=[0.5, 0.3, 0.2]) if Ls >= 1 else "I"
        if kind == "I":
            slots.append(g7.Slot(0.0, float(rng.uniform(0.15 * PI, 0.97 * PI))))
        elif kind == "R":
            slots.append(g7.Slot(float(np.exp(rng.uniform(np.log(0.3), np.log(12.0)))), 0.0))
        else:
            if len(slots) + 2 > Ls:
                slots.append(g7.Slot(0.0, float(rng.uniform(0.15 * PI, 0.97 * PI))))
            else:
                slots.append(g7.Slot(float(np.exp(rng.uniform(np.log(0.3), np.log(8.0)))),
                                     float(rng.uniform(0.15 * PI, 0.97 * PI))))
    # a quad counts two slots; trim or pad to exactly Ls
    count = sum(2 if s.kind == "quad" else 1 for s in slots)
    while count > Ls:
        slots.pop()
        count = sum(2 if s.kind == "quad" else 1 for s in slots)
    while count < Ls:
        slots.append(g7.Slot(0.0, float(rng.uniform(0.15 * PI, 0.97 * PI))))
        count += 1
    qc = Q_from_slots(slots) if Ls else []
    # polynomial: roots in x = t/pi^2
    factor = np.array([1.0])
    remaining = q
    while remaining > 0:
        choice = rng.random()
        if remaining >= 2 and choice < 0.35:
            u, v = rng.uniform(-0.6, 0.9), rng.uniform(0.1, 1.2)
            factor = np.convolve(factor, [1.0, -2.0 * u, u * u + v * v])
            remaining -= 2
        elif choice < 0.8:
            xr = rng.uniform(1.03, 4.5)
            factor = np.convolve(factor, [1.0, -1.0 / xr])
            remaining -= 1
        else:
            xr = -np.exp(rng.uniform(np.log(0.5), np.log(8.0)))
            factor = np.convolve(factor, [1.0, -1.0 / xr])
            remaining -= 1
    return np.array(qc + [float(v) for v in factor[1:]])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _fit_task(item) -> dict:
    m, Ls, q, target, seed = item
    x0 = np.asarray(seed, dtype=float)
    try:
        res = least_squares(residual, x0, args=(m, Ls, q, target), method="trf",
                            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
        x, cost = res.x, float(np.max(np.abs(res.fun)))
    except Exception as exc:  # pragma: no cover - solver failure on a wild seed
        return {"x": [float(v) for v in x0], "residual_max": 1e3, "error": str(exc)}
    return {"x": [float(v) for v in x], "residual_max": cost}


def square_newton(x, m: int, Ls: int, q: int, target: np.ndarray) -> dict:
    """Solve the most independent d equations exactly and report the rest."""
    d = Ls + q
    f = lambda y: residual(y, m, Ls, q, target)
    J = np.empty((target.size, d))
    h = 1e-7
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        J[:, i] = (f(x + e) - f(x - e)) / (2.0 * h)
    _, _, piv = qr(J.T, pivoting=True)
    chosen = sorted(int(v) for v in piv[:d])
    others = [i for i in range(target.size) if i not in chosen]
    sol = root(lambda y: f(y)[chosen], x, method="hybr", tol=1e-15)
    full = f(sol.x)
    return {"chosen_equations": chosen, "left_out_equations": others,
            "x": [float(v) for v in sol.x], "residual_chosen_max": float(np.max(np.abs(full[chosen]))),
            "residual_left_out": [float(full[i]) for i in others],
            "residual_max": float(np.max(np.abs(full))), "success": bool(sol.success),
            "jacobian_singular_values": [float(v) for v in np.linalg.svd(J, compute_uv=False)]}


def fit_cell(name: str, m: int, q: int, processes: int, seeds: int | None = None) -> dict:
    spec = TARGETS[name]
    n = spec["n"]
    Ls = (n - m) // 2
    target = target_vector(spec)
    rng = np.random.default_rng(20260913 + 7 * m + q)
    count = seeds if seeds is not None else SEEDS[min(Ls + (q > 2), 4)]
    seed_list = [random_seed(rng, Ls, q) for _ in range(count)]
    if Ls == 0 and q == 1:
        seed_list = [np.array([c]) for c in np.linspace(-4.0, 2.0, 61)]
    t0 = time.time()
    with Pool(processes) as pool:
        fits = pool.map(_fit_task, [(m, Ls, q, target, s) for s in seed_list], chunksize=4)
    fits.sort(key=lambda r: r["residual_max"])
    best = fits[0]
    # polish with Levenberg-Marquardt from the best point
    polished = least_squares(residual, np.asarray(best["x"]), args=(m, Ls, q, target), method="lm",
                             xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    x = polished.x
    res_vec = residual(x, m, Ls, q, target)
    out = {"target": name, "n": n, "m": m, "slots_count": Ls, "q": q, "smoothness": n - 2 - 2 * q,
           "unknowns": Ls + q, "equations": int(target.size),
           "implied_equations": 1 + (m - 2) // 2,
           "seeds": int(count), "seconds": time.time() - t0,
           "x": [float(v) for v in x], "residual": [float(v) for v in res_vec],
           "residual_max": float(np.max(np.abs(res_vec))), "residual_norm": float(np.linalg.norm(res_vec)),
           "distinct_local_minima": sorted({round(r["residual_max"], 10) for r in fits[:50]})[:8],
           "exact_realization": bool(np.max(np.abs(res_vec)) <= EXACT_LEVEL)}
    try:
        gen = build_generator(m, x, Ls, q)
        out["generator"] = describe(gen)
        out["newton"] = square_newton(x, m, Ls, q, target)
    except ValueError as exc:
        out["generator"] = {"error": str(exc)}
    return out


def describe(gen: g8.PolynomialGenerator) -> dict:
    cond = gen.cond()
    adm = gen.admissible()
    return {"label": gen.label(), "m": gen.m, "n": gen.n, "q": gen.q, "smoothness": gen.smoothness,
            "slots": [[float(s.alpha), float(s.omega)] for s in gen.slots],
            "slot_kinds": [s.kind for s in gen.slots],
            "omegas_over_pi": [float(w / PI) for w in gen.omegas],
            "real_pairs": [float(a) for a in gen.real_pairs],
            "quads": [[float(a), float(w / PI)] for a, w in gen.quads],
            "Q_coefficients": Q_from_slots(gen.slots),
            "polynomial": [float(p) for p in gen.poly],
            "polynomial_scaled": [float(p * PI ** (2 * (l + 1))) for l, p in enumerate(gen.poly)],
            "polynomial_roots": g8.poly_roots_xi(gen.poly),
            "cond": cond, "admissible": adm}


# ---------------------------------------------------------------------------
# Analysis of a realized generator
# ---------------------------------------------------------------------------

def lagrange_route_ec(kappa1, kappa2, n_nodes: int, faithful: str, theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Return e_c(theta) of a compact pair with a Lagrange layer (Gate 5a conventions)."""
    hat0 = ll.transfer(n_nodes, theta, xi, 0)
    if faithful == "operator":
        k1 = (np.asarray(kappa1(theta), dtype=float) / theta)[:, None]
        k2 = (np.asarray(kappa2(theta), dtype=float) / theta ** 2)[:, None]
        hat1, hat2 = k1 * hat0, k2 * hat0
    else:
        hat1 = ll.transfer(n_nodes, theta, xi, 1) / (1j * theta[:, None])
        hat2 = -ll.transfer(n_nodes, theta, xi, 2) / (theta[:, None] ** 2)
    carrier = np.exp(-1j * np.outer(theta, xi))
    worst = None
    for hat in (hat0, hat1, hat2):
        err = np.abs(carrier * hat - 1.0)
        worst = err if worst is None else np.maximum(worst, err)
    return worst.max(axis=1)


def analyze_realization(name: str, gen: g8.PolynomialGenerator) -> dict:
    """Return the nodal, second-derivative and off-grid receipts of a realized generator."""
    spec = TARGETS[name]
    d1, d2 = spec["d1"], spec["d2"]
    kappa1_c = comp.compact_kappa1(d1["alpha"], d1["a"])
    kappa2_c = comp.compact_kappa2(d2["alpha"], d2["a0"], d2["a"])
    left_c = comp.compact_left(d1["alpha"])
    theta = met.theta_grid()
    b, k1, k2 = gen.symbols(theta)
    out = {"symbol_agreement": {
        "kappa1_max_abs_dev": float(np.max(np.abs(k1 - kappa1_c(theta)))),
        "left_max_abs_dev": float(np.max(np.abs(b / b[0] * left_c(theta)[0] - left_c(theta)))),
        "kappa1_relative_dev_max": float(np.max(np.abs(k1 - kappa1_c(theta)) / np.maximum(np.abs(kappa1_c(theta)), 1e-300)))}}
    row = met.evaluate(f"{name}_realized", lambda t: gen.symbols(t)[1], lambda t: gen.symbols(t)[2],
                       lambda t: gen.symbols(t)[0], analytic_gv=lambda t: gen.symbol_derivative(t),
                       coupled=lambda t: gen.coupled_error(t, XI_FULL),
                       algebraic_order1=gen.m, algebraic_order2=gen.m - 2,
                       bandwidth_left=2 * gen.nmax - 1, bandwidth_d1=2 * gen.nmax - 2,
                       bandwidth_d2=2 * gen.nmax - 1, solves=1)
    own = met.evaluate(name, kappa1_c, kappa2_c, left_c, algebraic_order1=spec["published_order"],
                       algebraic_order2=spec["published_order"])
    own["cond_left_d2"] = met.symbol_cond(comp.compact_left(d2["alpha"]))["cond"]
    out["metrics_realized"] = row
    out["metrics_comparator"] = own
    out["second_derivative"] = {
        "kappa2_min_on_grid": float(k2[1:].min()), "kappa2_positive_on_grid": bool(k2[1:].min() > 0.0),
        "positivity": g8.certify_positivity(gen), "sufficient_conditions": g8.sufficient_conditions(gen),
        "e2_max_B0_realized": row["e2_max_B0"], "e2_max_B0_comparator": own["e2_max_B0"],
        "reach_d2_1e-2_realized": row["reach_d2_0.01_refined"], "reach_d2_1e-2_comparator": own["reach_d2_0.01_refined"],
        "reach_d2_1e-3_realized": row["reach_d2_0.001_refined"], "reach_d2_1e-3_comparator": own["reach_d2_0.001_refined"],
        "order_d2_fitted": row["order_d2_fitted"], "cond_shared_left": row["cond"],
        "cond_comparator_left_d2": own["cond_left_d2"]}
    # off-grid: realized field against the Lagrange augmentations of the comparator's own pair
    ec_real = gen.coupled_error(theta, XI_FULL)
    offgrid = {"realized": {"ec_max_B0": met.band_max(theta, ec_real, B0),
                            "reach_ec_1e-2": met.reach(theta, ec_real, 1e-2)["linear"] / PI,
                            "reach_ec_1e-3": met.reach(theta, ec_real, 1e-3)["linear"] / PI}}
    for n_nodes in (8, 12):
        for faithful in ("operator", "field"):
            ec = np.empty(theta.size)
            for start in range(0, theta.size, 1000):
                block = theta[start:start + 1000]
                ec[start:start + block.size] = lagrange_route_ec(kappa1_c, kappa2_c, n_nodes, faithful, block, XI_FULL)
            offgrid[f"{faithful}_L{n_nodes}"] = {"ec_max_B0": met.band_max(theta, ec, B0),
                                                 "reach_ec_1e-2": met.reach(theta, ec, 1e-2)["linear"] / PI,
                                                 "reach_ec_1e-3": met.reach(theta, ec, 1e-3)["linear"] / PI}
    champion = g7.ExtendedGenerator(6, [(0.0, 1.7701927977475826), (0.0, 2.0606855355641316)])
    ec_ch = champion.coupled_error(theta, XI_FULL)
    offgrid["champion_E62"] = {"ec_max_B0": met.band_max(theta, ec_ch, B0),
                               "reach_ec_1e-2": met.reach(theta, ec_ch, 1e-2)["linear"] / PI,
                               "reach_ec_1e-3": met.reach(theta, ec_ch, 1e-3)["linear"] / PI}
    out["offgrid"] = offgrid
    out["routes"] = {k: v for k, v in g8.verify_routes(gen).items() if k != "beta_digits"}
    out["record"] = gen.member_record(f"{name}_realized")
    return out


# ---------------------------------------------------------------------------
# Driver and markdown
# ---------------------------------------------------------------------------

def anchor_check() -> dict:
    """Return the exact reproduction of the eighth-order pentadiagonal scheme by E_{6,0} x (1 - t/28)."""
    gen = g8.PolynomialGenerator(6, [], [-1.0 / 28.0])
    s = gen.integer_samples()
    c = int(np.searchsorted(s.n, 0))
    psi0 = s.beta[c]
    ratios = {"psi1_over_psi0": float(s.beta[c + 1] / psi0), "psi2_over_psi0": float(s.beta[c + 2] / psi0),
              "psi3_over_psi0": float(s.beta[c + 3] / psi0),
              "a": float(-2.0 * s.dbeta[c + 1] / psi0), "b": float(-4.0 * s.dbeta[c + 2] / psi0),
              "c": float(-6.0 * s.dbeta[c + 3] / psi0)}
    exact = {"alpha": 4.0 / 9.0, "beta": 1.0 / 36.0, "a": 40.0 / 27.0, "b": 25.0 / 54.0, "c": 0.0}
    theta = met.theta_grid(4001)
    k1 = gen.symbols(theta)[1]
    ref = comp.compact_kappa1((exact["alpha"], exact["beta"]), (exact["a"] / 2.0, exact["b"] / 4.0))(theta)
    return {"generator": describe(gen), "ratios": ratios, "exact": exact,
            "max_dev": max(abs(ratios["psi1_over_psi0"] - exact["alpha"]), abs(ratios["psi2_over_psi0"] - exact["beta"]),
                           abs(ratios["psi3_over_psi0"]), abs(ratios["a"] - exact["a"]), abs(ratios["b"] - exact["b"]),
                           abs(ratios["c"])),
            "kappa1_max_dev": float(np.max(np.abs(k1 - ref)))}


def build(processes: int = 10, quick: bool = False) -> dict:
    data = {"anchor": anchor_check(), "targets": {}, "cells": []}
    for name, spec in TARGETS.items():
        data["targets"][name] = {"n": spec["n"], "target_vector": [float(v) for v in target_vector(spec)],
                                 "detected_order": comp.detect_order_d1(spec["d1"]["alpha"], spec["d1"]["a"]),
                                 "published_order": spec["published_order"]}
        cores = spec["cores"][:1] if quick else spec["cores"]
        for m in cores:
            for q in spec["degrees"]:
                if spec["n"] - 2 - 2 * q < 2:
                    continue
                # the square cell (m = published order, q at the C^2 limit) gets the full
                # seed budget; the other cells are overdetermined or lack the order and
                # only need a reliable best residual
                square = (m == spec["published_order"] and q == (spec["n"] - 4) // 2)
                seeds = 40 if quick else (None if square else 500)
                cell = fit_cell(name, m, q, processes, seeds=seeds)
                print(f"{name} m={m} q={q}: residual_max {cell['residual_max']:.3e} ({cell['seconds']:.0f} s)", flush=True)
                data["cells"].append(cell)
    # analyze the best exact realization per target (or the best fit)
    data["realizations"] = {}
    for name in TARGETS:
        cells = [c for c in data["cells"] if c["target"] == name and "error" not in c["generator"]]
        if not cells:
            continue
        best = min(cells, key=lambda c: c["residual_max"])
        gen = build_generator(best["m"], best["x"], best["slots_count"], best["q"])
        block = {"cell": {k: best[k] for k in ("m", "q", "slots_count", "residual_max", "exact_realization")},
                 "analysis": analyze_realization(name, gen)}
        data["realizations"][name] = block
    return data


def _fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.3e}" if (abs(v) < 1e-3 or abs(v) >= 1e4) and v != 0.0 else f"{v:.{digits}f}"
    return str(v)


def markdown(data: dict) -> str:
    L = ["# Gate 8 realization of published compact schemes (Question 2)", "",
         "Produced by code/analysis/exponential_family/gate8_realization.py under",
         "results/exponential_family/gate8/contract.md. The sample-matching problem of the",
         "module docstring is solved by least squares over the real coefficients of Q(s)",
         "and P(t) from random seeds, polished by Levenberg-Marquardt, then a square Newton",
         "solve of the most independent equations. Residuals are absolute, on the ratios",
         "psi(l)/psi(0) and psi'(s)/psi(0); a residual at the printed-coefficient level",
         f"({EXACT_LEVEL:g}) counts as an exact realization.", ""]
    a = data["anchor"]
    L += ["## Anchor: the rational realization, made exact", "",
          f"E_{{6,0}} x (1 - t/28): psi(1)/psi(0) = {a['ratios']['psi1_over_psi0']:.15f} (4/9), "
          f"psi(2)/psi(0) = {a['ratios']['psi2_over_psi0']:.15f} (1/36), psi(3)/psi(0) = {a['ratios']['psi3_over_psi0']:.1e}, "
          f"a = {a['ratios']['a']:.15f} (40/27), b = {a['ratios']['b']:.15f} (25/54), c = {a['ratios']['c']:.1e}; "
          f"max deviation from the rationals {a['max_dev']:.1e}, symbol deviation {a['kappa1_max_dev']:.1e}.",
          "This is the eighth-order pentadiagonal five-point scheme (the outer derivative sample vanishes), so",
          "the polynomial factor raises the formal order of E_{6,0} from six to eight: the coefficient of",
          "theta^6 in e_1 is a linear function of p_1 with the root p_1 = -1/28.", ""]
    L += ["## Cells", "",
          "| target | n | m | slots | q | C^k | unknowns | equations | implied | seeds | best residual max | residual norm | exact? | Newton residual (chosen / left out) | generator |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in data["cells"]:
        g = c["generator"]
        if "error" in g:
            label = "error: " + g["error"]
        else:
            slots = ", ".join(f"{k[0]}({a:.4f},{w / PI:.4f}pi)" for (a, w), k in zip(g["slots"], g["slot_kinds"]))
            label = f"m={g['m']} {slots}; c={[round(v, 4) for v in g['polynomial_scaled']]}; cond {g['cond']['cond']:.2f}; {g['admissible'][1]}"
        newton = "n/a"
        if "newton" in c:
            nw = c["newton"]
            newton = f"{nw['residual_chosen_max']:.1e} / " + ", ".join(f"{v:.1e}" for v in nw["residual_left_out"])
        L.append(f"| {c['target']} | {c['n']} | {c['m']} | {c['slots_count']} | {c['q']} | {c['smoothness']} | {c['unknowns']} | {c['equations']} | {c['implied_equations']} | {c['seeds']} | "
                 f"{c['residual_max']:.3e} | {c['residual_norm']:.3e} | {c['exact_realization']} | {newton} | {label} |")
    L.append("")
    for name, block in data.get("realizations", {}).items():
        an = block["analysis"]
        cell = block["cell"]
        L += [f"## {name}: best realization (m = {cell['m']}, q = {cell['q']}, residual {cell['residual_max']:.2e}, exact: {cell['exact_realization']})", ""]
        L += [f"Symbol agreement with the printed scheme on the contract grid: max |kappa_1 - kappa_1^SLC| = {an['symbol_agreement']['kappa1_max_abs_dev']:.2e}, "
              f"relative {an['symbol_agreement']['kappa1_relative_dev_max']:.2e}.", ""]
        r, o = an["metrics_realized"], an["metrics_comparator"]
        L += ["| quantity | realized generator (one field) | comparator (its own D1 and D2) |", "| --- | --- | --- |"]
        for key, lab in (("cond", "cond (D1 left side)"), ("reach_d1_0.01_refined", "R1(1e-2)"), ("reach_d1_0.001_refined", "R1(1e-3)"),
                         ("reach_d1_0.0001_refined", "R1(1e-4)"), ("gv_reach_1e-2", "G(1e-2)"), ("e1_max_B0", "E1max(B0)"),
                         ("e2_max_B0", "E2max(B0)"), ("reach_d2_0.01_refined", "R2(1e-2)"), ("reach_d2_0.001_refined", "R2(1e-3)"),
                         ("order_d1_fitted", "fitted order D1"), ("order_d2_fitted", "fitted order D2")):
            L.append(f"| {lab} | {_fmt(r[key])} | {_fmt(o[key])} |")
        sd = an["second_derivative"]
        L.append(f"| D2 left-side cond | {sd['cond_shared_left']:.2f} (shared with D1) | {sd['cond_comparator_left_d2']:.2f} (separate solve) |")
        L.append(f"| kappa_2 > 0 on (0, pi] | min {sd['kappa2_min_on_grid']:.3e}; certified b: {sd['positivity']['b_certified']}, kappa_2 b: {sd['positivity']['kappa2b_certified']} | by construction (published) |")
        L.append("")
        L += ["Off-grid channel e_c (129 xi, Gate 0 contract):", "",
              "| route | ec_max(B0) | reach_ec(1e-2) | reach_ec(1e-3) |", "| --- | --- | --- | --- |"]
        for rname, rv in an["offgrid"].items():
            L.append(f"| {rname} | {rv['ec_max_B0']:.3e} | {rv['reach_ec_1e-2']:.4f} | {rv['reach_ec_1e-3']:.4f} |")
        L.append("")
        rt = an["routes"]
        L.append(f"Routes: decimal deviation {rt['decimal_deviation']:.1e}, residue deviation {rt['residue_deviation']:.1e}, endpoint samples {rt['endpoint_samples']:.1e}.")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    data = build(args.processes, quick=args.quick)
    suffix = "_quick" if args.quick else ""
    (OUT / f"realization{suffix}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    text = markdown(data)
    (OUT / f"realization{suffix}.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
