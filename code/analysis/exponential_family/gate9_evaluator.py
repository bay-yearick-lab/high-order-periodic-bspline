"""Gate 9 section E4: the six verification checks of the Gate 8 evaluator.

Single entry point. It writes

  results/exponential_family/gate9/e4_evaluator_checks.json   the six checks,
  results/exponential_family/gate9/e4_eval_table_pert5.csv     10,000 rows
                                                               (x, psi, psi', psi''),

extends the two Gate 8 member files

  results/exponential_family/gate8/members/X8_n10_C_SLC1_pert5.json
  results/exponential_family/gate8/members/X8_n10_C_BMC11_SLC1992_pert12.json

with the flat "eval_*" block of version 2 that code/solver/exp_eval.h reads
(existing fields are never touched), and then builds and runs
code/solver/exp_eval_check to close check (vi). Exits nonzero if any check
fails.

The member carries the widened multiset of Gate 7 and the polynomial dial of
Gate 8, so psi = P(-D^2) phi with n = 10 exponents, degree p = 9 and smoothness
class C^2. The evaluator is general_evaluator.GeneralEvaluator; the Gate 0 path
of evaluator.py is untouched and check (v) is its regression.

The checks follow the Gate 5 contract with the Gate 8 routes substituted:

  (i)   integer samples equal the member file samples, which are the Gate 8
        alias route with the exact zeta tail, to 1e-13;
  (ii)  the off-grid transfers T_0, T_1, T_2 assembled from the shifted lattice
        samples of psi agree with the Gate 8 residue route to 1e-12, at 200
        angles and 2,000 random positions;
  (iii) psi, psi' and psi'' are continuous across every knot to 1e-13 and the
        jump of psi''' is reported beside them as the witness that the class is
        exactly C^2;
  (iv)  the translates partition unity to 1e-14 at 1,000 points;
  (v)   the champion table of Gate 5 and every number of the Gate 5 receipt are
        reproduced by the unchanged code path, to 1e-14 for the table and
        exactly for the receipt;
  (vi)  the C++ evaluator agrees with the Python one to 1e-14 on 10,000 rows.

Two conventions the contract leaves open are resolved here and recorded in the
receipt.

* Check (ii) assembles the transfers in the cancellation-free form of the Gate 0
  coupled channel, that is with sum_n psi'(n + xi) = 0 and
  sum_n psi''(n + xi) = sum_n n psi''(n + xi) = 0 removed analytically, and
  divides by the same sample symbol b(theta) that the residue route uses, so the
  residual measures the two routes to psi and not two cancellation patterns.
* Check (iii) takes the one-sided limits from the cell representation itself,
  the left limit being the left cell at xi = 1 and the right limit the right
  cell at xi = 0, so no finite difference or offset enters.

With no options the producer runs the two members above and writes the two
receipts named at the top. The options `--member <path>` (repeatable) and
`--out <json>` field any other Gate 8 member file instead, the first member
given being the gated one; `--table <csv>` names the evaluation table and
otherwise follows the name of the receipt. The smoothness gate reads the class
off the member, so a C^4 member is checked on orders 0 to 4 with the order-5
jump reported beside them.

Beside the six gates the receipt records the accuracy of the double evaluation
against the decimal reference route of general_evaluator.py at the working
precision, per channel and relative to that channel's own maximum, which is the
number the Gate 8 evaluator contract asks for.
"""

from __future__ import annotations

import argparse
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

import gate5_evaluator as g5  # noqa: E402
import gate8_family as g8  # noqa: E402
from evaluator import evaluator_for  # noqa: E402
from general_evaluator import GeneralEvaluator, evaluator_from_record  # noqa: E402

GATE8_MEMBERS = ROOT / "results" / "exponential_family" / "gate8" / "members"
GATE9 = ROOT / "results" / "exponential_family" / "gate9"
GATE5 = ROOT / "results" / "exponential_family" / "gate5"
SOLVER = CODE / "solver"

PRIMARY = GATE8_MEMBERS / "X8_n10_C_SLC1_pert5.json"
SECONDARY = GATE8_MEMBERS / "X8_n10_C_BMC11_SLC1992_pert12.json"
MEMBER_FILES = (("X8_n10_C_SLC1_pert5", PRIMARY),
                ("X8_n10_C_BMC11_SLC1992_pert12", SECONDARY))
TABLE = GATE9 / "e4_eval_table_pert5.csv"

TOL_SAMPLES = 1e-13
TOL_TRANSFERS = 1e-12
TOL_SMOOTH = 1e-13
TOL_PARTITION = 1e-14
TOL_REGRESSION = 1e-14
TOL_CPP = 1e-14
TOL_REFERENCE = 1e-13

SEED = 20260913
TABLE_ROWS = 10000
ANGLES = 200
POSITIONS = 2000
REFERENCE_POINTS = 600


def _members(member_files=MEMBER_FILES):
    """Return (name, record, generator, evaluator) for the listed Gate 8 members."""
    out = []
    for name, path in member_files:
        record = json.loads(path.read_text())
        out.append((name, path, record, g8.from_record(record),
                    evaluator_from_record(record)))
    return out


# ---------------------------------------------------------------------------
# (i) integer samples
# ---------------------------------------------------------------------------

def check_integer_samples(members) -> dict:
    """Return check (i): lattice samples against the stored alias-route samples."""
    detail = []
    worst = 0.0
    for name, _, record, _, ev in members:
        lattice = np.arange(0, int(record["radius"]) + 1).astype(float)
        devs = []
        for order, key in enumerate(("beta", "dbeta", "ddbeta")):
            reference = np.asarray(record[key], dtype=float)
            devs.append(float(np.max(np.abs(ev.derivative(lattice, order) - reference))))
        worst = max(worst, *devs)
        detail.append({"member": name, "n": int(ev.n), "p": int(ev.p),
                       "smoothness": int(ev.smoothness),
                       "assembly_conditioning": ev.conditioning,
                       "representation_conditioning": ev.representation_conditioning,
                       "max_dev_psi": devs[0], "max_dev_d1": devs[1],
                       "max_dev_d2": devs[2]})
    return {"achieved": worst, "tolerance": TOL_SAMPLES, "pass": worst <= TOL_SAMPLES,
            "route": "alias with the exact zeta tail, as stored in the member file",
            "members": detail}


# ---------------------------------------------------------------------------
# (ii) off-grid transfers
# ---------------------------------------------------------------------------

def assemble_transfers(ev: GeneralEvaluator, gen, theta: np.ndarray, xi: np.ndarray):
    """Return T_0, T_1, T_2 built from the evaluator's shifted lattice samples."""
    radius = (ev.n + 1) // 2 + 2
    lattice = np.arange(-radius, radius + 1).astype(float)
    s = [np.empty((lattice.size, xi.size)) for _ in range(3)]
    for i, node in enumerate(lattice):
        for order in range(3):
            s[order][i] = ev.derivative(node + xi, order)
    arg = np.outer(theta, lattice)
    expo = np.exp(-1j * arg)
    b = gen.symbols(theta)[0]
    shift = np.exp(-1j * np.outer(theta, xi))
    T0 = shift * (expo @ s[0]) / b[:, None]
    T1 = shift * (-1j) * ((expo - 1.0) @ s[1]) / (theta[:, None] * b[:, None])
    T2 = shift * (-1.0) * ((expo - 1.0 + 1j * arg) @ s[2]) / (theta[:, None] ** 2 * b[:, None])
    return T0, T1, T2


def direct_alias_moments(gen, theta: np.ndarray, xi: np.ndarray, kmax: int = 400000,
                         chunk: int = 40000) -> np.ndarray:
    """Return S_0, S_1, S_2 by summing the aliases term by term over |k| <= kmax.

    This is the arbiter of the third route: it truncates rather than resums, so
    its own error is the neglected tail, which for psi_hat = O(|xi|^{-(n-2q)})
    is O(kmax^{-(n-2q-1-j)}) in the channel j and is negligible for j = 0 and
    j = 1 at the kmax used here.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    S = np.zeros((3, theta.size, xi.size), dtype=complex)
    k0 = -kmax
    while k0 <= kmax:
        k1 = min(kmax, k0 + chunk - 1)
        ks = np.arange(k0, k1 + 1, dtype=float)
        X = theta[:, None] + 2.0 * np.pi * ks[None, :]
        P = gen.fourier(X)
        E = np.exp(2j * np.pi * np.outer(ks, xi))
        S[0] += P @ E
        S[1] += (X * P) @ E
        S[2] += (X * X * P) @ E
        k0 = k1 + 1
    return S


def evaluator_moments(ev: GeneralEvaluator, theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Return S_0, S_1, S_2 from the evaluator's shifted lattice samples."""
    radius = (ev.n + 1) // 2 + 2
    lattice = np.arange(-radius, radius + 1).astype(float)
    s = [np.array([ev.derivative(node + xi, order) for node in lattice])
         for order in range(3)]
    arg = np.outer(theta, lattice)
    expo = np.exp(-1j * arg)
    shift = np.exp(-1j * np.outer(theta, xi))
    return np.array([shift * (expo @ s[0]),
                     shift * (-1j) * ((expo - 1.0) @ s[1]),
                     shift * (-1.0) * ((expo - 1.0 + 1j * arg) @ s[2])])


def transfer_attribution(gen, ev: GeneralEvaluator, theta: float, xi: np.ndarray) -> dict:
    """Return which of the two routes to S_q carries the larger error.

    Both are compared with the term-by-term alias sum at one angle, so a gap
    between the assembled transfers and the residue route is attributed rather
    than left with the evaluator by default.
    """
    angle = np.array([float(theta)])
    brute = direct_alias_moments(gen, angle, xi)
    residue = gen.offgrid_moments(angle, xi)
    assembled = evaluator_moments(ev, angle, xi)
    out = {"theta": float(theta), "positions": [float(v) for v in xi],
           "b_at_theta": float(gen.symbols(angle)[0][0])}
    for q in range(3):
        out[f"residue_versus_direct_S{q}"] = float(np.max(np.abs(residue[q] - brute[q])))
        out[f"evaluator_versus_direct_S{q}"] = float(np.max(np.abs(assembled[q] - brute[q])))
    return out


def check_transfers(members) -> dict:
    """Return check (ii): assembled transfers against the Gate 8 residue route.

    The gate is the contracted member. The second member is measured beside it
    and attributed, because the residue route divides S_q by a sample symbol
    whose minimum over the angles is b_max / cond, so an absolute error of the
    reference route itself is amplified by the conditioning of the member.
    """
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-6.0, 6.0, POSITIONS)
    xi = x - np.round(x)
    theta = np.pi * np.arange(1, ANGLES + 1) / ANGLES
    detail = []
    gated = 0.0
    for index, (name, _, _, gen, ev) in enumerate(members):
        T = assemble_transfers(ev, gen, theta, xi)
        R = gen.offgrid_transfers(theta, xi)
        dev = [float(np.max(np.abs(T[q] - R[q]))) for q in range(3)]
        row = {"member": name, "gated": index == 0, "max_dev_T0": dev[0],
               "max_dev_T1": dev[1], "max_dev_T2": dev[2]}
        if max(dev) > TOL_TRANSFERS:
            worst = np.abs(T[0] - R[0])
            angle = float(theta[int(np.argmax(worst.max(axis=1)))])
            probe = xi[np.argsort(worst[int(np.argmax(worst.max(axis=1)))])[-4:]]
            row["attribution"] = transfer_attribution(gen, ev, angle, probe)
        detail.append(row)
        if index == 0:
            gated = max(dev)
    return {"achieved": gated, "tolerance": TOL_TRANSFERS,
            "pass": gated <= TOL_TRANSFERS, "angles": ANGLES,
            "positions": int(xi.size),
            "gated_member": members[0][0],
            "reference": "gate8_family.PolynomialGenerator.offgrid_transfers, residue route",
            "members": detail}


# ---------------------------------------------------------------------------
# (iii) smoothness
# ---------------------------------------------------------------------------

def check_smoothness(members) -> dict:
    """Return check (iii): the one-sided limits at the knots up to the class.

    The orders that the class of the member declares continuous are gated and
    the first order beyond them is reported as the witness that the class is
    exactly that one, so a C^2 member is gated on orders 0 to 2 with the
    order-3 jump beside it and a C^4 member on orders 0 to 4 with the order-5
    jump beside it.
    """
    detail = []
    worst = 0.0
    smallest_jump = float("inf")
    classes = set()
    for name, _, _, _, ev in members:
        knots = range(-(ev.n // 2), ev.n // 2 + 1)
        per_order = []
        for order in range(0, ev.smoothness + 1):
            jump = max(abs(ev.one_sided(k, order, "left") - ev.one_sided(k, order, "right"))
                       for k in knots)
            per_order.append(float(jump))
        top = max(abs(ev.one_sided(k, ev.smoothness + 1, "left")
                      - ev.one_sided(k, ev.smoothness + 1, "right")) for k in knots)
        worst = max(worst, max(per_order))
        smallest_jump = min(smallest_jump, float(top))
        classes.add(int(ev.smoothness))
        row = {"member": name, "smoothness_class": int(ev.smoothness),
               f"max_jump_orders_0_to_{ev.smoothness}": max(per_order)}
        for order, jump in enumerate(per_order):
            row[f"max_jump_order_{order}"] = jump
        row[f"max_jump_order_{ev.smoothness + 1}"] = float(top)
        detail.append(row)
    if len(classes) == 1:
        smoothness = classes.pop()
        label = (f"exactly C^{smoothness}: orders 0 to {smoothness} continuous, "
                 f"order {smoothness + 1} discontinuous")
        jump_key = f"smallest_order_{smoothness + 1}_jump"
    else:
        label = ("exactly the class each member declares: the declared orders "
                 "continuous, the next order discontinuous")
        jump_key = "smallest_first_discontinuous_jump"
    return {"achieved": worst, "tolerance": TOL_SMOOTH,
            "pass": bool(worst <= TOL_SMOOTH and smallest_jump > TOL_SMOOTH),
            jump_key: smallest_jump,
            "class": label,
            "members": detail}


# ---------------------------------------------------------------------------
# (iv) partition of unity
# ---------------------------------------------------------------------------

def check_partition(members) -> dict:
    """Return check (iv): the translates partition unity at 1,000 points."""
    rng = np.random.default_rng(SEED + 1)
    detail = []
    worst = 0.0
    for name, _, _, _, ev in members:
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
        detail.append({"member": name, "max_dev_sum_psi": d0,
                       "max_sum_d1": d1, "max_sum_d2": d2})
    return {"achieved": worst, "tolerance": TOL_PARTITION,
            "pass": worst <= TOL_PARTITION, "points": 1000, "members": detail}


# ---------------------------------------------------------------------------
# (v) the Gate 5 regression
# ---------------------------------------------------------------------------

def check_regression() -> dict:
    """Return check (v): the Gate 5 champion table and receipt, recomputed."""
    ev = evaluator_for(*g5.CHAMPION)
    rows = []
    for line in g5.TABLE.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("x,"):
            continue
        rows.append([float(v) for v in line.split(",")])
    table = np.asarray(rows, dtype=float)
    x = table[:, 0]
    channels = [float(np.max(np.abs(ev.derivative(x, q) - table[:, q + 1]))) for q in range(3)]
    table_worst = max(channels)

    stored = json.loads((GATE5 / "evaluator_checks.json").read_text())
    recomputed = {
        "i_integer_samples": g5.check_integer_samples(),
        "ii_offgrid_transfers": g5.check_transfers(),
        "iii_knot_smoothness": g5.check_smoothness(),
        "iv_partition_of_unity": g5.check_partition(),
        "v_polynomial_limit": g5.check_polynomial_limit(),
    }
    differences = []
    for key, value in recomputed.items():
        want = stored["checks"][key]
        if float(want["achieved"]) != float(value["achieved"]):
            differences.append({"check": key, "stored": want["achieved"],
                                "recomputed": value["achieved"]})
    member = stored["member"]
    for key, value in (("normalization", ev.normalization),
                       ("cell_conditioning", ev.conditioning)):
        if float(member[key]) != float(value):
            differences.append({"check": f"member.{key}", "stored": member[key],
                                "recomputed": value})
    return {"achieved": table_worst, "tolerance": TOL_REGRESSION,
            "pass": bool(table_worst <= TOL_REGRESSION and not differences),
            "table": str(g5.TABLE.relative_to(ROOT)), "rows": int(x.size),
            "max_dev_phi": channels[0], "max_dev_d1": channels[1],
            "max_dev_d2": channels[2],
            "receipt": str((GATE5 / "evaluator_checks.json").relative_to(ROOT)),
            "receipt_numbers_recomputed": {k: v["achieved"] for k, v in recomputed.items()},
            "receipt_differences": differences}


# ---------------------------------------------------------------------------
# (vi) the C++ evaluator
# ---------------------------------------------------------------------------

def write_table(ev: GeneralEvaluator, name: str = "X8_n10_C_SLC1_pert5",
                table: Path = TABLE) -> dict:
    """Write the 10,000-row (x, psi, psi', psi'') table the C++ check reads."""
    half = 0.5 * ev.n
    knots = np.concatenate([np.arange(-ev.n, ev.n + 1) * 0.5,
                            np.array([-half, half, -half - 0.25, half + 0.25])])
    rng = np.random.default_rng(SEED + 2)
    rest = rng.uniform(-half - 0.5, half + 0.5, TABLE_ROWS - knots.size)
    x = np.concatenate([knots, rest])
    v0 = ev.value(x)
    v1 = ev.d1(x)
    v2 = ev.d2(x)
    GATE9.mkdir(parents=True, exist_ok=True)
    lines = ["# Gate 9 section E4 check (vi): the Python evaluator of the Gate 8 member",
             f"# {name} at 10,000 points.",
             "# The first rows are the integer and half integer knots and the support ends.",
             "x,psi,dpsi,ddpsi"]
    for i in range(x.size):
        lines.append(f"{float(x[i]):.17g},{float(v0[i]):.17g},"
                     f"{float(v1[i]):.17g},{float(v2[i]):.17g}")
    table.write_text("\n".join(lines) + "\n")
    return {"file": str(table.relative_to(ROOT)), "rows": int(x.size),
            "range": [float(x.min()), float(x.max())]}


def check_cpp(member: Path = PRIMARY, table: Path = TABLE) -> dict:
    """Return check (vi): build and run the C++ evaluator check on the member."""
    build = subprocess.run(["make", "exp_eval_check"], cwd=SOLVER,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return {"achieved": None, "tolerance": TOL_CPP, "pass": False,
                "error": "build failed", "stderr": build.stderr[-2000:]}
    run = subprocess.run(["./exp_eval_check", "--exp-member", str(member),
                          "--table", str(table)], cwd=SOLVER,
                         capture_output=True, text=True)
    achieved = None
    per_channel = {}
    for line in run.stdout.splitlines():
        if "max deviation" in line:
            achieved = float(line.split("max deviation")[1].split("(")[0].strip())
        if "max|C++ - Python| d" in line:
            tag = line.split("max|C++ - Python| d")[1]
            per_channel[f"d{tag[0]}"] = float(tag.split()[1])
    legacy = subprocess.run(["./exp_eval_check", "--exp-member", str(g5.MEMBER),
                             "--table", str(g5.TABLE)], cwd=SOLVER,
                            capture_output=True, text=True)
    legacy_worst = None
    for line in legacy.stdout.splitlines():
        if "max deviation" in line:
            legacy_worst = float(line.split("max deviation")[1].split("(")[0].strip())
    return {"achieved": achieved, "tolerance": TOL_CPP,
            "pass": bool(run.returncode == 0 and achieved is not None
                         and achieved <= TOL_CPP and legacy.returncode == 0
                         and legacy_worst is not None and legacy_worst <= TOL_CPP),
            "per_channel": per_channel,
            "version_1_block_still_reads": {"member": str(g5.MEMBER.relative_to(ROOT)),
                                            "max_deviation": legacy_worst},
            "stdout": run.stdout.strip().splitlines()}


# ---------------------------------------------------------------------------
# The accuracy of the double evaluation against the decimal reference
# ---------------------------------------------------------------------------

def reference_accuracy(members) -> dict:
    """Return the double-versus-decimal deviation per channel, per member.

    The points are drawn uniformly over the support with a quarter of them
    forced into the two end cells, where psi is tiny and the representation is
    most exposed. The deviation of each channel is reported relative to that
    channel's own maximum over the support.
    """
    rng = np.random.default_rng(SEED + 3)
    detail = []
    worst = 0.0
    for name, _, _, _, ev in members:
        half = 0.5 * ev.n
        x = np.concatenate([rng.uniform(-half, half, REFERENCE_POINTS // 2),
                            rng.uniform(-half, -half + 1.0, REFERENCE_POINTS // 4),
                            rng.uniform(half - 1.0, half, REFERENCE_POINTS // 4)])
        dense = np.linspace(-half, half, 20001)
        row = {"member": name, "points": int(x.size)}
        for order in range(3):
            gap = float(np.max(np.abs(ev.derivative(x, order) - ev.reference(x, order))))
            scale = float(np.max(np.abs(ev.derivative(dense, order))))
            row[f"max_abs_dev_d{order}"] = gap
            row[f"channel_max_d{order}"] = scale
            row[f"relative_d{order}"] = gap / scale
            worst = max(worst, gap / scale)
        detail.append(row)
    return {"achieved": worst, "tolerance": TOL_REFERENCE, "pass": worst <= TOL_REFERENCE,
            "reference": "general_evaluator.GeneralEvaluator.reference, decimal at "
                         "the working precision, unfolded cell form",
            "members": detail}


# ---------------------------------------------------------------------------
# The member files
# ---------------------------------------------------------------------------

def extend_member_file(path: Path, ev: GeneralEvaluator) -> dict:
    """Add the flat eval_* block to a member file, keeping every other field.

    The block is the only part of the file this producer owns, so its fields are
    written whether or not they are already there; every field outside the block
    is compared before and after and the write is refused if any of them moves.
    """
    original = json.loads(path.read_text())
    block = ev.cell_table()
    if any(not key.startswith("eval_") for key in block):
        raise RuntimeError("the evaluator block must only carry eval_* fields")
    merged = dict(original)
    added = [key for key in block if key not in original]
    rewritten = [key for key in block if key in original and original[key] != block[key]]
    merged.update(block)
    for key, value in original.items():
        if key not in block and merged[key] != value:
            raise RuntimeError(f"refusing to change the existing member field {key}")
    path.write_text(json.dumps(merged, indent=2) + "\n")
    return {"file": str(path.relative_to(ROOT)),
            "evaluator_fields": list(block.keys()),
            "fields_written_for_the_first_time": added,
            "evaluator_fields_rewritten": rewritten,
            "pre_existing_fields_unchanged": [k for k in original if k not in block]}


def member_files(paths) -> tuple:
    """Return the (name, path) pairs of the member files named on the line."""
    out = []
    for item in paths:
        path = Path(item).resolve()
        record = json.loads(path.read_text())
        out.append((str(record.get("name", path.stem)), path))
    return tuple(out)


def table_path(out: Path, table: str | None, default: bool) -> Path:
    """Return the evaluation table that goes with this receipt.

    The default run keeps the stored name. A receipt named on the line carries
    its own tag, so the table follows that name with the receipt stem's
    `e4_evaluator_checks` replaced by `e4_eval_table`.
    """
    if table is not None:
        return Path(table).resolve()
    if default:
        return TABLE
    stem = out.stem.replace("e4_evaluator_checks", "e4_eval_table")
    return out.parent / f"{stem}.csv"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", action="append", default=None, metavar="PATH",
                        help="a Gate 8 member file to check; repeatable; the "
                             "first one given is the gated member")
    parser.add_argument("--out", default=None, metavar="JSON",
                        help="the receipt to write")
    parser.add_argument("--table", default=None, metavar="CSV",
                        help="the evaluation table the C++ check reads")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    default = args.member is None
    files = MEMBER_FILES if default else member_files(args.member)
    out = (GATE9 / "e4_evaluator_checks.json" if args.out is None
           else Path(args.out).resolve())
    table = table_path(out, args.table, default)

    GATE9.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    members = _members(files)
    primary = members[0][4]
    receipt = {
        "gate": "9.E4",
        "generated": date.today().isoformat(),
        "members": [{"name": name, "file": str(path.relative_to(ROOT)),
                     "m": int(record["m"]), "omegas": list(record["omegas"]),
                     "real_pairs": list(record["real_pairs"]),
                     "quads": [list(q) for q in record["quads"]],
                     "polynomial": list(record["polynomial"]),
                     "n": int(ev.n), "p": int(ev.p), "smoothness": int(ev.smoothness),
                     "fold_order": int(ev.fold_order),
                     "normalization": ev.normalization,
                     "assembly_conditioning": ev.conditioning,
                     "representation_conditioning": ev.representation_conditioning,
                     "working_digits": int(ev.digits)}
                    for name, path, record, _, ev in members],
        "producers": {"python": "code/analysis/exponential_family/general_evaluator.py",
                      "cpp": "code/solver/exp_eval.h",
                      "entry_point": "code/analysis/exponential_family/gate9_evaluator.py"},
        "checks": {},
    }
    receipt["checks"]["i_integer_samples"] = check_integer_samples(members)
    receipt["checks"]["ii_offgrid_transfers"] = check_transfers(members)
    receipt["checks"]["iii_knot_smoothness"] = check_smoothness(members)
    receipt["checks"]["iv_partition_of_unity"] = check_partition(members)
    receipt["checks"]["v_gate5_regression"] = check_regression()
    receipt["member_files"] = [extend_member_file(path, ev)
                               for _, path, _, _, ev in members]
    receipt["table"] = write_table(primary, members[0][0], table)
    receipt["checks"]["vi_cpp_agreement"] = check_cpp(members[0][1], table)
    receipt["double_versus_decimal"] = reference_accuracy(members)
    receipt["all_pass"] = bool(all(c["pass"] for c in receipt["checks"].values())
                               and receipt["double_versus_decimal"]["pass"])

    out.write_text(json.dumps(receipt, indent=2) + "\n")
    for key, value in receipt["checks"].items():
        achieved = value["achieved"]
        shown = "none" if achieved is None else f"{achieved:.3e}"
        print(f"{key:26s} achieved {shown}  tolerance {value['tolerance']:.0e}  "
              f"{'ok' if value['pass'] else 'FAIL'}")
    ref = receipt["double_versus_decimal"]
    print(f"{'double_versus_decimal':26s} achieved {ref['achieved']:.3e}  "
          f"tolerance {ref['tolerance']:.0e}  {'ok' if ref['pass'] else 'FAIL'}")
    print(f"receipt {out.relative_to(ROOT)}")
    return 0 if receipt["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
