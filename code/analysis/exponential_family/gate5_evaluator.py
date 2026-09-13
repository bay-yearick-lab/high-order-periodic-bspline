"""Gate 5 section 5.0: the six verification checks of the generator evaluator.

Single entry point. It writes

  results/exponential_family/gate5/evaluator_checks.json   the six checks,
  results/exponential_family/gate5/exp_eval_table.csv       10,000 rows
                                                            (x, phi, phi', phi''),

extends results/exponential_family/members/E62_champion.json with the flat
"eval_*" block that code/solver/exp_eval.h reads (existing fields are never
touched), and then builds and runs code/solver/exp_eval_check to close check
(vi). Exits nonzero if any check fails.

The checks follow the contract literally:

  (i)   integer samples equal the alias-route samples of family.py to 1e-14;
  (ii)  the off-grid transfers T_0, T_1, T_2 assembled from phi agree with the
        alias-sum transfers of Gate 0 to 1e-12, at 200 angles and 2,000 random
        positions;
  (iii) phi, phi' and phi'' are C^{p-1} across every knot, one-sided limits to
        1e-13;
  (iv)  the translates partition unity to 1e-14 at 1,000 points;
  (v)   the polynomial limit omega -> 0 reproduces the cardinal B-spline to
        1e-13;
  (vi)  the C++ evaluator agrees with the Python one to 1e-14.

Two conventions the contract leaves open are resolved here and recorded in the
receipt.

* Check (ii) assembles the transfers from the shifted lattice samples in the
  cancellation-free form of the Gate 0 coupled channel, that is with
  sum_n phi'(n + xi) = 0 and sum_n phi''(n + xi) = sum_n n phi''(n + xi) = 0
  removed analytically. Both sides of the comparison then carry the same
  conditioning in theta, and the residual is the difference of the two routes
  to phi rather than the difference of two cancellation patterns.
* Check (iii) takes the one-sided limits from the cell representation itself,
  the left limit being the left cell at xi = 1 and the right limit the right
  cell at xi = 0, so no finite difference or offset enters. Orders 0 through
  p - 1 are compared, and the jump of the order-p derivative is reported
  beside them as the witness that the smoothness class is exactly C^{p-1}.
* Check (v) gates on the two exact statements of the limit, namely the members
  with r = 0 and a member whose frequencies sit below the evaluator's
  documented zero-frequency tolerance, both against
  spline_analysis.centered_cardinal_bspline. The approach to the limit at
  frequencies the evaluator treats as genuine is recorded as a convergence
  table beside the gate, because the deviation there is the O(omega^2) of the
  member itself and not an error of the evaluator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for path in (str(CODE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import spline_analysis as sa  # noqa: E402
from evaluator import ExponentialEvaluator, evaluator_for  # noqa: E402
from family import ExponentialGenerator  # noqa: E402

GATE5 = ROOT / "results" / "exponential_family" / "gate5"
MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
TABLE = GATE5 / "exp_eval_table.csv"
SOLVER = CODE / "solver"

CHAMPION = (6, (1.7701927977475826, 2.0606855355641316))
MEMBERS = [
    ("champion E_{6,2}", CHAMPION[0], CHAMPION[1]),
    ("BMC9 E_{10,0}", 10, ()),
    ("BMC11 E_{12,0}", 12, ()),
    ("anchor E_{4,2}(0.1, 2.36)", 4, (0.1, 2.36)),
    ("confluent E_{4,2}(0.73, 0.73)", 4, (0.73, 0.73)),
]

TOL_SAMPLES = 1e-14
TOL_TRANSFERS = 1e-12
TOL_SMOOTH = 1e-13
TOL_PARTITION = 1e-14
TOL_POLYNOMIAL = 1e-13
TOL_CPP = 1e-14

SEED = 20260912
TABLE_ROWS = 10000


def check_integer_samples() -> dict:
    """Return check (i): lattice samples against the alias route of family.py."""
    detail = []
    worst = 0.0
    for name, m, omegas in MEMBERS:
        ev = ExponentialEvaluator(m, omegas)
        gen = ExponentialGenerator(ev.m, ev.omegas)
        samples = gen.integer_samples()
        lattice = samples.n.astype(float)
        d0 = float(np.max(np.abs(ev.value(lattice) - samples.beta)))
        d1 = float(np.max(np.abs(ev.d1(lattice) - samples.dbeta)))
        d2 = float(np.max(np.abs(ev.d2(lattice) - samples.ddbeta)))
        worst = max(worst, d0, d1, d2)
        detail.append({"member": name, "n": int(ev.n), "p": int(ev.p),
                       "conditioning": ev.conditioning,
                       "max_dev_phi": d0, "max_dev_d1": d1, "max_dev_d2": d2})
    return {"achieved": worst, "tolerance": TOL_SAMPLES, "pass": worst <= TOL_SAMPLES,
            "members": detail}


def assemble_transfers(ev: ExponentialEvaluator, gen: ExponentialGenerator,
                       theta: np.ndarray, xi: np.ndarray):
    """Return T_0, T_1, T_2 built from the evaluator's shifted lattice samples."""
    radius = (ev.n + 1) // 2 + 2
    lattice = np.arange(-radius, radius + 1).astype(float)
    s = [np.empty((lattice.size, xi.size)) for _ in range(3)]
    for i, node in enumerate(lattice):
        for q in range(3):
            s[q][i] = ev.derivative(node + xi, q)
    arg = np.outer(theta, lattice)
    expo = np.exp(-1j * arg)
    b = gen.symbols(theta)[0]
    shift = np.exp(-1j * np.outer(theta, xi))
    T0 = shift * (expo @ s[0]) / b[:, None]
    T1 = shift * (-1j) * ((expo - 1.0) @ s[1]) / (theta[:, None] * b[:, None])
    T2 = shift * (-1.0) * ((expo - 1.0 + 1j * arg) @ s[2]) / (theta[:, None] ** 2 * b[:, None])
    return T0, T1, T2


def check_transfers() -> dict:
    """Return check (ii): assembled transfers against the Gate 0 alias sums."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-6.0, 6.0, 2000)
    xi = x - np.round(x)
    theta = np.pi * np.arange(1, 201) / 200.0
    detail = []
    worst = 0.0
    for name, m, omegas in MEMBERS[:2] + MEMBERS[3:4]:
        ev = ExponentialEvaluator(m, omegas)
        gen = ExponentialGenerator(ev.m, ev.omegas)
        T = assemble_transfers(ev, gen, theta, xi)
        R = gen.offgrid_transfers(theta, xi)
        dev = [float(np.max(np.abs(T[q] - R[q]))) for q in range(3)]
        worst = max(worst, *dev)
        detail.append({"member": name, "max_dev_T0": dev[0], "max_dev_T1": dev[1],
                       "max_dev_T2": dev[2]})
    return {"achieved": worst, "tolerance": TOL_TRANSFERS, "pass": worst <= TOL_TRANSFERS,
            "angles": 200, "positions": int(xi.size), "members": detail}


def check_smoothness() -> dict:
    """Return check (iii): one-sided limits of every derivative through p - 1."""
    detail = []
    worst = 0.0
    for name, m, omegas in MEMBERS:
        ev = ExponentialEvaluator(m, omegas)
        knots = range(-(ev.n // 2), ev.n // 2 + 1)
        per_order = []
        member_worst = 0.0
        for order in range(0, ev.p):
            jump = max(abs(ev.one_sided(k, order, "left") - ev.one_sided(k, order, "right"))
                       for k in knots)
            per_order.append(float(jump))
            member_worst = max(member_worst, float(jump))
        top = max(abs(ev.one_sided(k, ev.p, "left") - ev.one_sided(k, ev.p, "right"))
                  for k in knots)
        worst = max(worst, member_worst)
        detail.append({"member": name, "p": int(ev.p),
                       "max_jump_orders_0_to_p_minus_1": member_worst,
                       "max_jump_order_0": per_order[0],
                       "max_jump_order_1": per_order[1],
                       "max_jump_order_2": per_order[2],
                       "max_jump_order_p": float(top)})
    return {"achieved": worst, "tolerance": TOL_SMOOTH, "pass": worst <= TOL_SMOOTH,
            "members": detail}


def check_partition() -> dict:
    """Return check (iv): the translates partition unity at 1,000 points."""
    rng = np.random.default_rng(SEED + 1)
    detail = []
    worst = 0.0
    for name, m, omegas in MEMBERS:
        ev = ExponentialEvaluator(m, omegas)
        x = rng.uniform(-3.0, 3.0, 1000)
        reach = ev.n // 2 + 4
        total = np.zeros_like(x)
        first = np.zeros_like(x)
        second = np.zeros_like(x)
        for k in range(-reach, reach + 1):
            total += ev.value(x - k)
            first += ev.d1(x - k)
            second += ev.d2(x - k)
        d0 = float(np.max(np.abs(total - 1.0)))
        d1 = float(np.max(np.abs(first)))
        d2 = float(np.max(np.abs(second)))
        worst = max(worst, d0)
        detail.append({"member": name, "max_dev_sum_phi": d0,
                       "max_sum_d1": d1, "max_sum_d2": d2})
    return {"achieved": worst, "tolerance": TOL_PARTITION, "pass": worst <= TOL_PARTITION,
            "points": 1000, "members": detail}


def check_polynomial_limit() -> dict:
    """Return check (v): the omega -> 0 limit against the cardinal B-spline."""
    gated = []
    worst = 0.0
    for degree in (3, 5, 7, 9, 11):
        ev = ExponentialEvaluator(degree + 1, ())
        x = np.linspace(-0.5 * (degree + 1) - 0.5, 0.5 * (degree + 1) + 0.5, 3001)
        d0 = float(np.max(np.abs(ev.value(x) - sa.centered_cardinal_bspline(degree, x))))
        d1 = float(np.max(np.abs(ev.d1(x) - sa.centered_cardinal_bspline_derivative(degree, x))))
        d2 = float(np.max(np.abs(ev.d2(x)
                                 - sa.centered_cardinal_bspline_second_derivative(degree, x))))
        worst = max(worst, d0, d1, d2)
        gated.append({"case": f"r = 0, degree {degree}", "max_dev_phi": d0,
                      "max_dev_d1": d1, "max_dev_d2": d2})
    tiny = 0.1 * ExponentialEvaluator.ZERO_FREQUENCY_TOLERANCE
    ev = ExponentialEvaluator(6, (tiny, tiny))
    x = np.linspace(-5.5, 5.5, 3001)
    d0 = float(np.max(np.abs(ev.value(x) - sa.centered_cardinal_bspline(9, x))))
    d1 = float(np.max(np.abs(ev.d1(x) - sa.centered_cardinal_bspline_derivative(9, x))))
    d2 = float(np.max(np.abs(ev.d2(x) - sa.centered_cardinal_bspline_second_derivative(9, x))))
    worst = max(worst, d0, d1, d2)
    gated.append({"case": f"E_{{6,2}}(omega = {tiny:.1e}) collapsed by the zero tolerance",
                  "max_dev_phi": d0, "max_dev_d1": d1, "max_dev_d2": d2})
    approach = []
    for w in (0.8, 0.4, 0.2, 0.1, 0.05):
        evw = ExponentialEvaluator(6, (w, w))
        dev = float(np.max(np.abs(evw.value(x) - sa.centered_cardinal_bspline(9, x))))
        approach.append({"omega": w, "max_dev_phi": dev,
                         "conditioning": evw.conditioning})
    return {"achieved": worst, "tolerance": TOL_POLYNOMIAL, "pass": worst <= TOL_POLYNOMIAL,
            "gated_cases": gated, "approach_to_the_limit": approach,
            "zero_frequency_tolerance": ExponentialEvaluator.ZERO_FREQUENCY_TOLERANCE}


def extend_member_file() -> dict:
    """Add the flat eval_* block to the champion member file, keeping the rest."""
    original = json.loads(MEMBER.read_text())
    ev = evaluator_for(CHAMPION[0], CHAMPION[1])
    block = ev.cell_table()
    merged = dict(original)
    added = []
    for key, value in block.items():
        if key in original:
            if original[key] != value:
                raise RuntimeError(f"refusing to change the existing member field {key}")
        else:
            added.append(key)
        merged[key] = value
    for key, value in original.items():
        if merged[key] != value:
            raise RuntimeError(f"refusing to change the existing member field {key}")
    MEMBER.write_text(json.dumps(merged, indent=2) + "\n")
    return {"file": str(MEMBER.relative_to(ROOT)),
            "evaluator_fields": list(block.keys()),
            "fields_written_for_the_first_time": added,
            "pre_existing_fields_unchanged": [k for k in original if k not in block]}


def write_table() -> dict:
    """Write the 10,000-row (x, phi, phi', phi'') table the C++ check reads."""
    ev = evaluator_for(CHAMPION[0], CHAMPION[1])
    half = 0.5 * ev.n
    knots = np.concatenate([np.arange(-ev.n, ev.n + 1) * 0.5,
                            np.array([-half, half, -half - 0.25, half + 0.25])])
    rng = np.random.default_rng(SEED + 2)
    rest = rng.uniform(-half - 0.5, half + 0.5, TABLE_ROWS - knots.size)
    x = np.concatenate([knots, rest])
    v0 = ev.value(x)
    v1 = ev.d1(x)
    v2 = ev.d2(x)
    GATE5.mkdir(parents=True, exist_ok=True)
    lines = ["# Gate 5 section 5.0 check (vi): the Python evaluator of the champion",
             "# member E_{6,2}(1.7701927977475826, 2.0606855355641316) at 10,000 points.",
             "# The first rows are the integer and half integer knots and the support ends.",
             "x,phi,dphi,ddphi"]
    for i in range(x.size):
        lines.append(f"{float(x[i]):.17g},{float(v0[i]):.17g},"
                     f"{float(v1[i]):.17g},{float(v2[i]):.17g}")
    TABLE.write_text("\n".join(lines) + "\n")
    return {"file": str(TABLE.relative_to(ROOT)), "rows": int(x.size),
            "range": [float(x.min()), float(x.max())]}


def check_cpp() -> dict:
    """Return check (vi): build and run the C++ evaluator check."""
    build = subprocess.run(["make", "exp_eval_check"], cwd=SOLVER,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return {"achieved": None, "tolerance": TOL_CPP, "pass": False,
                "error": "build failed", "stderr": build.stderr[-2000:]}
    run = subprocess.run(["./exp_eval_check", "--exp-member", str(MEMBER),
                          "--table", str(TABLE)], cwd=SOLVER,
                         capture_output=True, text=True)
    achieved = None
    per_channel = {}
    for line in run.stdout.splitlines():
        if "max deviation" in line:
            achieved = float(line.split("max deviation")[1].split("(")[0].strip())
        if "max|C++ - Python| d" in line:
            tag = line.split("max|C++ - Python| d")[1]
            per_channel[f"d{tag[0]}"] = float(tag.split()[1])
    return {"achieved": achieved, "tolerance": TOL_CPP,
            "pass": bool(run.returncode == 0 and achieved is not None
                         and achieved <= TOL_CPP),
            "per_channel": per_channel, "stdout": run.stdout.strip().splitlines()}


def main() -> int:
    GATE5.mkdir(parents=True, exist_ok=True)
    ev = evaluator_for(CHAMPION[0], CHAMPION[1])
    receipt = {
        "gate": "5.0",
        "generated": date.today().isoformat(),
        "member": {"name": "E62_champion", "m": CHAMPION[0], "omegas": list(CHAMPION[1]),
                   "p": int(ev.p), "n": int(ev.n),
                   "normalization": ev.normalization,
                   "cell_conditioning": ev.conditioning},
        "producers": {"python": "code/analysis/exponential_family/evaluator.py",
                      "cpp": "code/solver/exp_eval.h",
                      "entry_point": "code/analysis/exponential_family/gate5_evaluator.py"},
        "checks": {},
    }
    receipt["checks"]["i_integer_samples"] = check_integer_samples()
    receipt["checks"]["ii_offgrid_transfers"] = check_transfers()
    receipt["checks"]["iii_knot_smoothness"] = check_smoothness()
    receipt["checks"]["iv_partition_of_unity"] = check_partition()
    receipt["checks"]["v_polynomial_limit"] = check_polynomial_limit()
    receipt["member_file"] = extend_member_file()
    receipt["table"] = write_table()
    receipt["checks"]["vi_cpp_agreement"] = check_cpp()
    receipt["all_pass"] = all(c["pass"] for c in receipt["checks"].values())

    out = GATE5 / "evaluator_checks.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    for key, value in receipt["checks"].items():
        achieved = value["achieved"]
        shown = "none" if achieved is None else f"{achieved:.3e}"
        print(f"{key:26s} achieved {shown}  tolerance {value['tolerance']:.0e}  "
              f"{'ok' if value['pass'] else 'FAIL'}")
    print(f"receipt {out.relative_to(ROOT)}")
    return 0 if receipt["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
