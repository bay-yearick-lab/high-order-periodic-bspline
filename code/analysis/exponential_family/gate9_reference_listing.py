"""Run the supplement's thirty-line reference listing and check it against this repository.

The supplement prints a self-contained numpy implementation of one member: the
banded left side, the single cyclic solve, both nodal derivatives, one off-grid
read from the deposited cell tables, and one divergence-free projection in the
operator's own metric. This script runs exactly that code and compares it with
the frozen routines, so that the printed listing is verified rather than
asserted.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import gate8_family as g8
import general_evaluator as ge

MEMBER = (ROOT / "results" / "exponential_family" / "gate8" / "members"
          / "X9_m6_C_SLC1_SLC1992_6IIq2.json")
OUT = ROOT / "results" / "exponential_family" / "gate9" / "reference_listing_check.json"


def listing(path: Path) -> dict:
    """The supplement listing, verbatim in substance, returning what it computed."""
    M = json.load(open(path))
    R = M["radius"]
    b, db, ddb = (np.array(M[k]) for k in ("beta", "dbeta", "ddbeta"))
    N = 64
    h = 2 * np.pi / N
    x = h * np.arange(N)
    u = np.sin(x) + 0.3 * np.cos(3 * x)
    th = 2 * np.pi * np.fft.fftfreq(N)
    n = np.arange(1, R + 1)
    bsym = b[0] + 2 * (np.cos(np.outer(th, n)) @ b[1:])
    c = np.fft.ifft(np.fft.fft(u) / bsym).real

    def band(w, odd):
        out = np.zeros(N) if odd else w[0] * c
        for q in n:
            out = out + w[q] * (np.roll(c, q) + (-1 if odd else 1) * np.roll(c, -q))
        return out

    d1 = band(db, True) / h
    d2 = band(ddb, False) / h ** 2
    K = M["eval_fold_order"]
    cells = M["eval_cells"]
    poly = np.array(M["eval_poly"]).reshape(cells, M["eval_poly_degree"] + 1)
    cs = np.array(M["eval_cos"]).reshape(cells, -1)
    sn = np.array(M["eval_sin"]).reshape(cells, -1)
    rho = np.array(M["eval_group_alpha"]) + 1j * np.array(M["eval_group_omega"])
    mult = np.array(M["eval_group_multiplicity"], dtype=int)

    def Rj(j, z, terms=60):
        i = np.arange(j, j + terms)
        return np.sum(z ** i / [math.factorial(int(k)) for k in i])

    def psi(t):
        y = t + M["eval_support_n"] / 2
        if not 0 <= y < M["eval_support_n"]:
            return 0.0
        J = int(np.floor(y))
        xi = y - J
        out = np.polyval(poly[J][::-1], xi)
        col = 0
        for g, r in enumerate(rho):
            for bb in range(mult[g]):
                Rv = Rj(K - bb, r * xi)
                out += xi ** bb * (cs[J][col] * Rv.real + sn[J][col] * Rv.imag)
                col += 1
        return out

    t0 = 0.37 / h
    j0 = int(np.floor(t0))
    value = sum(c[(j0 + q) % N] * psi(t0 - (j0 + q)) for q in range(-R, R + 1))
    kap = np.where(th == 0, 1.0, -2 * (np.sin(np.outer(th, n)) @ db[1:]) / bsym)
    rng = np.random.default_rng(7)
    U = np.fft.fft2(rng.standard_normal((N, N)))
    V = np.fft.fft2(rng.standard_normal((N, N)))
    KX, KY = np.meshgrid(kap, kap, indexing="ij")
    den = np.where(KX ** 2 + KY ** 2 == 0, 1.0, KX ** 2 + KY ** 2)
    S = (KX * U + KY * V) / den
    Up, Vp = U - KX * S, V - KY * S
    divergence = (np.max(np.abs(KX * Up + KY * Vp))
                  / np.max(np.abs(KX * U + KY * V)))
    return {"M": M, "R": R, "b": b, "db": db, "ddb": ddb, "N": N, "h": h,
            "d1": d1, "d2": d2, "x": x, "psi": psi, "value": float(value),
            "divergence": float(divergence)}


def main() -> int:
    out = listing(MEMBER)
    M, R, N = out["M"], out["R"], out["N"]
    b, db, ddb = out["b"], out["db"], out["ddb"]
    n = np.arange(1, R + 1)
    theta = 2 * np.pi * np.fft.fftfreq(N)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    theta = theta[theta > 1e-12]
    generator = g8.from_record(M)
    b_ref, k1_ref, k2_ref = generator.symbols(theta)
    b_list = b[0] + 2 * (np.cos(np.outer(theta, n)) @ b[1:])
    k1_list = -2 * (np.sin(np.outer(theta, n)) @ db[1:]) / b_list
    k2_list = -(ddb[0] + 2 * (np.cos(np.outer(theta, n)) @ ddb[1:])) / b_list
    evaluator = ge.evaluator_from_record(M)
    points = np.linspace(-5.0, 5.0, 401)
    psi_list = np.array([out["psi"](t) for t in points])
    psi_ref = np.asarray(evaluator.value(points))
    x = out["x"]
    payload = {
        "producer": "code/analysis/exponential_family/gate9_reference_listing.py",
        "member": str(MEMBER.relative_to(ROOT)),
        "grid": N,
        "b_max_deviation": float(np.max(np.abs(b_list - b_ref))),
        "kappa1_max_deviation": float(np.max(np.abs(k1_list - k1_ref))),
        "kappa2_max_deviation": float(np.max(np.abs(k2_list - k2_ref))),
        "generator_points": int(points.size),
        "generator_max_deviation": float(np.max(np.abs(psi_list - psi_ref))),
        "projection_relative_divergence": out["divergence"],
        "offgrid_value_at_0p37": out["value"],
        "d1_max_error_against_exact": float(
            np.max(np.abs(out["d1"] - (np.cos(x) - 0.9 * np.sin(3 * x))))),
        "d2_max_error_against_exact": float(
            np.max(np.abs(out["d2"] - (-np.sin(x) - 2.7 * np.cos(3 * x))))),
        "tolerance": 1e-12,
    }
    payload["pass"] = bool(
        max(payload["b_max_deviation"], payload["kappa1_max_deviation"],
            payload["kappa2_max_deviation"], payload["generator_max_deviation"],
            payload["projection_relative_divergence"]) <= payload["tolerance"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    for key in ("b_max_deviation", "kappa1_max_deviation", "kappa2_max_deviation",
                "generator_max_deviation", "projection_relative_divergence"):
        print(f"  {key} = {payload[key]:.3e}")
    print(f"reference_listing_check: pass={payload['pass']} -> {OUT}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
