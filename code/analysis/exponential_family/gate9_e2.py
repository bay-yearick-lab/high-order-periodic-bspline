"""Gate 9 experiment E2: the matched truncation Taylor-Green test for pert5.

Single entry point for section E2 of the frozen contract in
results/exponential_family/gate9/contract.md. Four modes:

    --predict  P(route) at N = 64 and 128 from the same N dealiased
               pseudospectral spectrum at t* and the Gate 1 symbol curves,
               extended with the everything member X8_n10_C_SLC1_pert5.
               Written before either solver run starts.
    --run      tgv_re1600 for route exp with the pert5 member at N = 64 and
               N = 128, launched concurrently at four threads each, with the
               exact argument vector of the Gate 4c launcher.
    --reduce   the Gate 4e decision quantities for the two pert5 fields
               against the stored same N pseudospectral references, the
               contract's E2 decision rule, and the champion regression.
    --check    recompute the decision from the stored quantities in
               e2_runs.csv.

Two options carry the same procedure to another member. --member <path>
fields that member file instead of pert5, and --tag <name> puts _<name>
before the extension of every output name (e2_prediction_<name>.json,
e2_registry_<name>.csv, e2_runs_<name>.csv, e2_decision_<name>.json,
e2_logs_<name>/, e2_fields_<name>/, e2_work_<name>/) and makes <name> the
route label inside the receipts. Without them the bindings are the frozen
pert5 ones, so the stored pert5 receipts re-reduce bit for bit.

Nothing of Gate 4c or Gate 4e is modified. Both modules are imported and their
own functions carry every convention: energy_spectrum, load_field, reduce_cell,
reference_bundle, make_run, command_for and parse_log are used unchanged, so
the numbers of this experiment sit in the same code path as the stored ones.

The member enters the prediction through its own symbol. EXP_ROUTE.md fixes
the applied first derivative eigenvalue as -(i/h) a(theta)/b(theta) with

    b(theta) = beta(0) + 2 sum_{n>=1} beta(n)  cos(n theta)
    a(theta) =           2 sum_{n>=1} beta'(n) sin(n theta)

while the Gate 0 metric convention writes that eigenvalue as (i/h) kappa_1.
The two conventions therefore give kappa_1 = -a/b, and the prediction mode
checks that form against gate8_family.PolynomialGenerator.symbols to 1e-13
before any P value is formed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = HERE.parents[2]
for path in (str(CODE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate4c_tgv as g4c  # noqa: E402
import gate4e_matched_truncation as g4e  # noqa: E402
import gate8_family as g8  # noqa: E402
from gate4c_tgv import ROUTES, TARGET_TIME, sha256, utc_now  # noqa: E402

GATE = ROOT / "results" / "exponential_family" / "gate9"
GATE4C = ROOT / "results" / "exponential_family" / "gate4c"
GATE4E = ROOT / "results" / "exponential_family" / "gate4e"
DEFAULT_MEMBER = (ROOT / "results" / "exponential_family" / "gate8" / "members"
                  / "X8_n10_C_SLC1_pert5.json")
DEFAULT_LABEL = "pert5"
MEMBER = DEFAULT_MEMBER
SOLVER = g4c.SOLVER

GRIDS = (64, 128)
LABEL = DEFAULT_LABEL
TAG = ""
THREADS = 4
SYMBOL_TOLERANCE = 1e-13
LANDING_TOLERANCE = 1e-9
MECHANISM_FACTOR = 2.0
STORED_LABELS = ("sl", "exp", "bmc_p11", "cw12b")

REGISTRY_FIELDS = (
    "key", "label", "N", "route", "p", "dealias", "dt_scale", "cfl", "threads",
    "member", "member_sha256", "initial_condition", "command", "pid",
    "launch_time_utc", "completion_time_utc", "returncode", "landing_time",
    "landing_abs_difference", "landed_on_tstar", "verdict", "verdict_reason",
    "wall_s", "steps", "wall_per_step_s", "log", "field", "field_bytes",
    "field_sha256",
)

RUN_FIELDS = ()
RULE = ""
WORK = WORK_DATA = LOGS = FIELDS = None
PREDICTION = REGISTRY = RUNS = DECISION = None
K_TESTS = K_RATIO_SL = K_RATIO_CHAMPION = ""
K_MECHANISM_PREDICTED = K_MECHANISM_REALIZED = K_PREDICTED_P_OVER = ""


def configure(member=None, tag=None) -> None:
    """Bind the member file, the route label and every output name.

    Called once by main. Without arguments the bindings are the frozen pert5
    ones of the E2 receipt, so that the stored files re-reduce bit for bit.
    With a tag every output name carries _<tag> before its extension and the
    route label inside the receipts is the tag.
    """
    global MEMBER, LABEL, TAG, WORK, WORK_DATA, LOGS, FIELDS
    global PREDICTION, REGISTRY, RUNS, DECISION, RUN_FIELDS, RULE
    global K_TESTS, K_RATIO_SL, K_RATIO_CHAMPION
    global K_MECHANISM_PREDICTED, K_MECHANISM_REALIZED, K_PREDICTED_P_OVER
    MEMBER = Path(member).resolve() if member else DEFAULT_MEMBER
    TAG = str(tag) if tag else ""
    LABEL = TAG if TAG else DEFAULT_LABEL
    suffix = f"_{TAG}" if TAG else ""
    WORK = GATE / f"e2_work{suffix}"
    WORK_DATA = WORK / "data"
    LOGS = GATE / f"e2_logs{suffix}"
    FIELDS = GATE / f"e2_fields{suffix}"
    PREDICTION = GATE / f"e2_prediction{suffix}.json"
    REGISTRY = GATE / f"e2_registry{suffix}.csv"
    RUNS = GATE / f"e2_runs{suffix}.csv"
    DECISION = GATE / f"e2_decision{suffix}.json"
    K_TESTS = f"tests_{LABEL}_below_sl"
    K_RATIO_SL = f"ratio_sl_over_{LABEL}"
    K_RATIO_CHAMPION = f"ratio_{LABEL}_over_champion"
    K_MECHANISM_PREDICTED = f"predicted_P_sl_over_P_{LABEL}"
    K_MECHANISM_REALIZED = f"realized_e_full_sl_over_{LABEL}"
    K_PREDICTED_P_OVER = f"predicted_P_over_{LABEL}"
    RUN_FIELDS = (
        "key", "label", "N", "dealias", "dt_scale", "cfl",
        "e_full", "e_band", "e_mid", "e_outer",
        "predicted_P", K_PREDICTED_P_OVER,
        "reference_field", "reference_sha256", "field", "field_sha256",
        "source",
    )
    RULE = (
        f"WIN if e_full({LABEL}) < e_full(sl) at both N = 64 and N = 128. "
        "PARTIAL if at one N only. REJECT if at neither. Mechanism confirmed "
        "if the realized ratio "
        f"e_full(sl)/e_full({LABEL}) is within a factor of two of the "
        f"predicted ratio P(sl)/P({LABEL}) at both N; otherwise recorded as "
        "unexplained with the shell resolved error attached and no claim."
    )


configure()


def contract_sha256() -> str:
    """Return the digest of the frozen Gate 9 contract."""
    return sha256(GATE / "contract.md")


def member_record() -> dict:
    """Return the stored everything member record."""
    return json.loads(MEMBER.read_text())


# ------------------------------------------------------------------- symbols


def kappa1_from_samples(record: dict, theta) -> np.ndarray:
    """Return kappa_1(theta) of a member from its stored lattice samples.

    b and a are the trigonometric polynomials of EXP_ROUTE.md. The applied D1
    eigenvalue is -(i/h) a/b there and (i/h) kappa_1 in the Gate 0 metrics, so
    kappa_1 = -a/b.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    beta = np.asarray(record["beta"], dtype=float)
    dbeta = np.asarray(record["dbeta"], dtype=float)
    order = np.arange(1, beta.size, dtype=float)
    argument = np.outer(theta, order)
    b = beta[0] + 2.0 * (np.cos(argument) @ beta[1:])
    a = 2.0 * (np.sin(argument) @ dbeta[1:])
    return -a / b


def kappa1_from_generator(record: dict, theta) -> np.ndarray:
    """Return kappa_1(theta) of a member through the Gate 8 generator."""
    generator = g8.from_record(record)
    return np.asarray(generator.symbols(np.atleast_1d(
        np.asarray(theta, dtype=float)))[1], dtype=float)


def symbol_cross_check(record: dict, theta) -> dict:
    """Compare the two kappa_1 routes of a member on the given angles."""
    from_samples = kappa1_from_samples(record, theta)
    from_generator = kappa1_from_generator(record, theta)
    deviation = float(np.max(np.abs(from_samples - from_generator)))
    return {
        "points": int(np.size(theta)),
        "max_absolute_deviation": deviation,
        "tolerance": SYMBOL_TOLERANCE,
        "agree": bool(deviation <= SYMBOL_TOLERANCE),
        "route_a": ("kappa_1 = -a/b from the member JSON beta and dbeta "
                    "samples, EXP_ROUTE.md convention"),
        "route_b": "gate8_family.from_record(record).symbols(theta)[1]",
    }


# ------------------------------------------------------------------- predict


def predict(args) -> int:
    """Write the predeclared spectrum weighted phase error for E2."""
    GATE.mkdir(parents=True, exist_ok=True)
    record = member_record()
    curves = g4c.kappa1_curves()
    curves[LABEL] = lambda theta: kappa1_from_samples(record, theta)

    dense = np.linspace(1e-6, np.pi, 200001)
    payload = {
        "producer": "code/analysis/exponential_family/gate9_e2.py --predict",
        "written_utc": utc_now(),
        "contract": {"path": "results/exponential_family/gate9/contract.md",
                     "sha256": contract_sha256()},
        "definition": ("P(route) = sqrt( sum_m E(m) |e_1(theta_m)|^2 / sum_m E(m) ) "
                       "over shells 1 <= m < N/3 with theta_m = 2 pi m / N and "
                       "e_1 = kappa_1/theta - 1, weighted by the SAME N dealiased "
                       "pseudospectral energy spectrum at t*, the Gate 4e predict "
                       "formula verbatim"),
        "member": {"path": str(MEMBER.relative_to(ROOT)),
                   "sha256": sha256(MEMBER),
                   "name": record["name"], "cond": record["cond"],
                   "p": record["p"], "radius": record["radius"]},
        "champion": {"path": str(g4c.MEMBER.relative_to(ROOT)),
                     "sha256": sha256(g4c.MEMBER),
                     "label_in_stored_routes": "exp"},
        "symbol_cross_check_dense": symbol_cross_check(record, dense),
        "mechanism_agreement_factor": MECHANISM_FACTOR,
        "grids": {},
    }
    if not payload["symbol_cross_check_dense"]["agree"]:
        raise SystemExit(f"the two kappa_1 routes of {LABEL} disagree beyond "
                         f"{SYMBOL_TOLERANCE}; no prediction written")

    for n in GRIDS:
        path = g4e.reference_field_path(n)
        if not path.exists():
            raise SystemExit(f"missing pseudospectral reference {path}")
        _, spectrum = g4e.reference_bundle(n)
        shells = np.arange(1, n // 3)
        weights = spectrum[shells]
        theta = 2.0 * np.pi * shells / n
        entry = {
            "reference_field": str(path.relative_to(ROOT)),
            "reference_sha256": sha256(path),
            "shells": [int(shells[0]), int(shells[-1])],
            "theta_range": [float(theta[0]), float(theta[-1])],
            "reference_band_energy": float(weights.sum()),
            "symbol_cross_check_band": symbol_cross_check(record, theta),
            "P": {},
            "e1_band_max_absolute": {},
        }
        if not entry["symbol_cross_check_band"]["agree"]:
            raise SystemExit(f"the two kappa_1 routes of {LABEL} disagree on "
                             f"the N = {n} band beyond {SYMBOL_TOLERANCE}")
        for label in [lbl for lbl, _, _, _ in ROUTES] + [LABEL]:
            kappa1 = np.asarray(curves[label](theta), dtype=float)
            e1 = kappa1 / theta - 1.0
            entry["P"][label] = float(np.sqrt(np.sum(weights * e1**2)
                                              / np.sum(weights)))
            entry["e1_band_max_absolute"][label] = float(np.max(np.abs(e1)))
        entry["predicted_ordering_best_first"] = sorted(
            entry["P"], key=lambda k: entry["P"][k])
        entry[K_RATIO_SL] = entry["P"]["sl"] / entry["P"][LABEL]
        entry[K_RATIO_CHAMPION] = entry["P"][LABEL] / entry["P"]["exp"]
        entry["ratio_sl_over_champion"] = entry["P"]["sl"] / entry["P"]["exp"]
        for other in ("bmc_p11", "cw12b"):
            entry[f"ratio_{other}_over_{LABEL}"] = (entry["P"][other]
                                                    / entry["P"][LABEL])
        payload["grids"][str(n)] = entry

    PREDICTION.write_text(json.dumps(payload, indent=2) + "\n")
    for n, entry in payload["grids"].items():
        print(f"N = {n}: shells {entry['shells'][0]}..{entry['shells'][1]}")
        for label in entry["predicted_ordering_best_first"]:
            print(f"    {label:10s} P = {entry['P'][label]:.6e}")
        print(f"    P(sl)/P({LABEL})      = {entry[K_RATIO_SL]:.4f}")
        print(f"    P({LABEL})/P(champion)= {entry[K_RATIO_CHAMPION]:.4f}")
        print(f"    kappa_1 route deviation on the band "
              f"{entry['symbol_cross_check_band']['max_absolute_deviation']:.3e}")
    print(f"wrote {PREDICTION.relative_to(ROOT)}")
    return 0


# ----------------------------------------------------------------------- run


def e2_run_record(n: int) -> dict:
    """Return the Gate 4c run description of the member cell at grid n."""
    run = g4c.make_run("exp", "exp", 9, True, n, 1, 1.0)
    run["key"] = f"{LABEL}_N{n}_da1_dt100"
    run["label"] = LABEL
    run["log_name"] = f"{run['key']}.log"
    return run


def e2_command(run: dict, ic_path: Path) -> list[str]:
    """Return the Gate 4c argument vector with the fielded member substituted."""
    command = g4c.command_for(run, ic_path)
    index = command.index("--exp-member")
    command[index + 1] = str(MEMBER)
    return command


def launch(run: dict, ic_path: Path):
    """Start one detached solver process and return it with its log handle."""
    log_path = LOGS / run["log_name"]
    handle = log_path.open("w")
    environment = dict(os.environ, OMP_NUM_THREADS=str(THREADS),
                       OMP_PROC_BIND="false")
    command = e2_command(run, ic_path)
    process = subprocess.Popen(
        command, cwd=str(WORK), env=environment, stdout=handle,
        stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    run["pid"] = process.pid
    run["command"] = " ".join(command)
    run["threads"] = THREADS
    run["log"] = str(log_path.relative_to(ROOT))
    run["initial_condition"] = str(ic_path.relative_to(ROOT))
    run["member"] = str(MEMBER.relative_to(ROOT))
    run["member_sha256"] = sha256(MEMBER)
    run["launch_time_utc"] = utc_now()
    run["launch_monotonic"] = time.time()
    print(f"LAUNCHED pid={process.pid} key={run['key']} threads={THREADS} "
          f"log={log_path.relative_to(ROOT)}", flush=True)
    return process, handle


def harvest(run: dict, returncode: int) -> None:
    """Move one finished run's field into the receipt directory and parse its log."""
    run["returncode"] = returncode
    run["completion_time_utc"] = utc_now()
    run["wall_wrapper_s"] = time.time() - run.pop("launch_monotonic")
    source_field = WORK_DATA / run["field_name"]
    if source_field.exists():
        FIELDS.mkdir(parents=True, exist_ok=True)
        target = FIELDS / run["field_name"]
        source_field.replace(target)
        run["field"] = str(target.relative_to(ROOT))
        run["field_bytes"] = target.stat().st_size
        run["field_sha256"] = sha256(target)
    run.update(g4c.parse_log((LOGS / run["log_name"]).read_text()))
    landing = run.get("landing_time")
    if landing is not None:
        run["landing_abs_difference"] = abs(landing - TARGET_TIME)
        run["landed_on_tstar"] = bool(run["landing_abs_difference"]
                                      <= LANDING_TOLERANCE)
    else:
        run["landing_abs_difference"] = None
        run["landed_on_tstar"] = False


def run(args) -> int:
    """Launch both member runs concurrently and record the registry."""
    if not PREDICTION.exists():
        raise SystemExit(f"{PREDICTION.relative_to(ROOT)} must exist before --run")
    if not SOLVER.exists():
        raise SystemExit(f"missing solver binary {SOLVER}")
    WORK_DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    FIELDS.mkdir(parents=True, exist_ok=True)

    runs = [e2_run_record(n) for n in GRIDS]
    active = []
    for item in runs:
        ic_path = GATE4C / "ic" / f"ic_tgv_N{item['N']}.bin"
        if not ic_path.exists():
            raise SystemExit(f"missing initial condition {ic_path}")
        process, handle = launch(item, ic_path)
        active.append((process, handle, item))
    while active:
        time.sleep(5.0)
        still = []
        for process, handle, item in active:
            code = process.poll()
            if code is None:
                still.append((process, handle, item))
                continue
            handle.close()
            harvest(item, code)
            print(f"EXIT pid={process.pid} key={item['key']} rc={code} "
                  f"landing={item.get('landing_time')} "
                  f"steps={item.get('steps')} wall={item.get('wall_s')}",
                  flush=True)
        active = still

    with REGISTRY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        for item in runs:
            writer.writerow({k: item.get(k, "") for k in REGISTRY_FIELDS})
    print(f"wrote {REGISTRY.relative_to(ROOT)}")
    failed = [item["key"] for item in runs
              if item.get("returncode") != 0 or not item.get("landed_on_tstar")]
    if failed:
        print(f"FAILED OR OFF TARGET: {failed}")
        return 1
    return 0


# -------------------------------------------------------------------- reduce


def verify_references() -> dict:
    """Check the stored pseudospectral reference fields against the manifest."""
    manifest = json.loads((GATE4E / "reference" / "manifest.json").read_text())
    out = {}
    for n in GRIDS:
        key = f"N{n}_dt100"
        entry = manifest["entries"][key]
        path = g4e.reference_field_path(n)
        digest = sha256(path)
        out[key] = {
            "field": str(path.relative_to(ROOT)),
            "manifest_sha256": entry["field_sha256"],
            "recomputed_sha256": digest,
            "matches": bool(digest == entry["field_sha256"]),
            "landing_time": entry["landing_time"],
            "cfl": entry["cfl"],
        }
        if not out[key]["matches"]:
            raise SystemExit(f"reference field {path} does not match the manifest")
    return out


def stored_rows() -> dict:
    """Return the stored Gate 4e reduced rows by label and grid."""
    with (GATE4E / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        if float(row["dt_scale"]) != 1.0:
            continue
        out[(row["label"], int(row["N"]))] = {
            "e_full": float(row["e_full"]), "e_band": float(row["e_band"]),
            "e_mid": float(row["e_mid"]), "e_outer": float(row["e_outer"]),
            "field": row["field"], "field_sha256": row["field_sha256"],
        }
    return out


def decide(e_full: dict, prediction: dict) -> dict:
    """Apply the frozen E2 decision rule verbatim."""
    stored = stored_rows()
    wins = {}
    for n in GRIDS:
        wins[f"N{n}"] = bool(e_full[n] < stored[("sl", n)]["e_full"])
    count = sum(1 for v in wins.values() if v)
    outcome = "WIN" if count == 2 else ("PARTIAL" if count == 1 else "REJECT")

    mechanism = {}
    for n in GRIDS:
        predicted = prediction["grids"][str(n)][K_RATIO_SL]
        realized = stored[("sl", n)]["e_full"] / e_full[n]
        quotient = realized / predicted
        mechanism[f"N{n}"] = {
            K_MECHANISM_PREDICTED: predicted,
            K_MECHANISM_REALIZED: realized,
            "realized_over_predicted": quotient,
            "within_factor_two": bool((1.0 / MECHANISM_FACTOR) <= quotient
                                      <= MECHANISM_FACTOR),
        }
    confirmed = all(mechanism[f"N{n}"]["within_factor_two"] for n in GRIDS)
    mechanism["confirmed"] = bool(confirmed)
    mechanism["status"] = ("confirmed" if confirmed else
                           "unexplained: the shell resolved error is attached in "
                           "e2_decision.json, no claim is made")
    return {
        "rule": RULE,
        "outcome": outcome,
        K_TESTS: wins,
        "mechanism": mechanism,
        "mechanism_confirmed": mechanism["confirmed"],
    }


def regression(bundles: dict) -> dict:
    """Re-reduce the stored champion fields and compare to the Gate 4e numbers."""
    stored = stored_rows()
    out = {"note": ("the stored route exp champion fields of Gate 4c re-reduced "
                    "through this producer's code path; bit for bit equality of "
                    "e_full with results/exponential_family/gate4e/runs.csv is "
                    "required"),
           "cells": {}}
    for n in GRIDS:
        path = GATE4C / "fields" / f"tgvfield_exp_cic_da23_p9_N{n}.bin"
        if not path.exists():
            out["cells"][f"N{n}"] = {"present": False,
                                     "path": str(path.relative_to(ROOT))}
            continue
        row = g4e.reduce_cell("exp", n, 1.0, bundles[n], path)
        reference = stored[("exp", n)]
        out["cells"][f"N{n}"] = {
            "present": True,
            "field": str(path.relative_to(ROOT)),
            "field_sha256": row["field_sha256"],
            "field_sha256_matches_gate4e": bool(row["field_sha256"]
                                                == reference["field_sha256"]),
            "e_full_recomputed": row["e_full"],
            "e_full_stored": reference["e_full"],
            "e_full_bit_identical": bool(row["e_full"] == reference["e_full"]),
            "e_band_recomputed": row["e_band"],
            "e_band_stored": reference["e_band"],
            "e_band_bit_identical": bool(row["e_band"] == reference["e_band"]),
            "e_mid_bit_identical": bool(row["e_mid"] == reference["e_mid"]),
            "e_outer_bit_identical": bool(row["e_outer"] == reference["e_outer"]),
        }
    evaluated = [c for c in out["cells"].values() if c.get("present")]
    out["all_bit_identical"] = bool(evaluated and all(
        c["e_full_bit_identical"] and c["e_band_bit_identical"]
        and c["e_mid_bit_identical"] and c["e_outer_bit_identical"]
        for c in evaluated))
    return out


def reduce(args) -> int:
    """Reduce both member fields, apply the E2 rule, write the receipts."""
    if not PREDICTION.exists():
        raise SystemExit(f"{PREDICTION.relative_to(ROOT)} must exist before --reduce")
    if not REGISTRY.exists():
        raise SystemExit(f"{REGISTRY.relative_to(ROOT)} must exist before --reduce")
    logs = sorted(LOGS.glob(f"{LABEL}_N*_da1_dt100.log"))
    if not logs:
        raise SystemExit(f"no run log under {LOGS.relative_to(ROOT)}")
    prediction_mtime = PREDICTION.stat().st_mtime
    youngest_log = min(path.stat().st_mtime for path in logs)
    if prediction_mtime > youngest_log:
        raise SystemExit("the prediction file is younger than a run log; the "
                         "E2 contract forbids reducing against it")

    references = verify_references()
    prediction = json.loads(PREDICTION.read_text())
    with REGISTRY.open(newline="") as handle:
        registry = {int(row["N"]): row for row in csv.DictReader(handle)}

    bundles = {n: g4e.reference_bundle(n) for n in GRIDS}
    reduced = []
    for n in GRIDS:
        entry = registry[n]
        path = ROOT / entry["field"]
        if not path.exists():
            raise SystemExit(f"missing {LABEL} field {path}")
        if sha256(path) != entry["field_sha256"]:
            raise SystemExit(f"{LABEL} field {path} does not match the "
                             "registry digest")
        row = g4e.reduce_cell(LABEL, n, 1.0, bundles[n], path)
        row["source"] = "gate9_e2"
        reduced.append(row)
        print(f"{row['key']:26s} e_full {row['e_full']:.8e}  "
              f"e_band {row['e_band']:.8e}  e_mid {row['e_mid']:.8e}  "
              f"e_outer {row['e_outer']:.8e}", flush=True)

    stored = stored_rows()
    with RUNS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        for n in GRIDS:
            grid = prediction["grids"][str(n)]
            reference_path = g4e.reference_field_path(n)
            reference_sha = references[f"N{n}_dt100"]["recomputed_sha256"]
            row = next(r for r in reduced if r["N"] == n)
            writer.writerow({
                "key": row["key"], "label": LABEL, "N": n, "dealias": 1,
                "dt_scale": 1.0, "cfl": g4c.BASE_CFL,
                "e_full": row["e_full"], "e_band": row["e_band"],
                "e_mid": row["e_mid"], "e_outer": row["e_outer"],
                "predicted_P": grid["P"][LABEL], K_PREDICTED_P_OVER: 1.0,
                "reference_field": str(reference_path.relative_to(ROOT)),
                "reference_sha256": reference_sha,
                "field": row["field"], "field_sha256": row["field_sha256"],
                "source": "gate9_e2",
            })
            for label in STORED_LABELS:
                cell = stored[(label, n)]
                writer.writerow({
                    "key": f"{label}_N{n}_da1_dt100", "label": label, "N": n,
                    "dealias": 1, "dt_scale": 1.0, "cfl": g4c.BASE_CFL,
                    "e_full": cell["e_full"], "e_band": cell["e_band"],
                    "e_mid": cell["e_mid"], "e_outer": cell["e_outer"],
                    "predicted_P": grid["P"][label],
                    K_PREDICTED_P_OVER: grid["P"][label] / grid["P"][LABEL],
                    "reference_field": str(reference_path.relative_to(ROOT)),
                    "reference_sha256": reference_sha,
                    "field": cell["field"], "field_sha256": cell["field_sha256"],
                    "source": "results/exponential_family/gate4e/runs.csv",
                })

    e_full = {n: next(r for r in reduced if r["N"] == n)["e_full"] for n in GRIDS}
    decision = decide(e_full, prediction)
    decision.update({
        "producer": "code/analysis/exponential_family/gate9_e2.py --reduce",
        "written_utc": utc_now(),
        "contract_sha256": contract_sha256(),
        "member": prediction["member"],
        "grids": list(GRIDS),
        "e_full": {f"N{n}": dict({LABEL: e_full[n]},
                                 **{label: stored[(label, n)]["e_full"]
                                    for label in STORED_LABELS})
                   for n in GRIDS},
        "e_band": {f"N{n}": dict(
            {LABEL: next(r for r in reduced if r["N"] == n)["e_band"]},
            **{label: stored[(label, n)]["e_band"] for label in STORED_LABELS})
            for n in GRIDS},
        "mid_outer_split": {
            f"N{n}": dict(
                {LABEL: {"e_mid": next(r for r in reduced if r["N"] == n)["e_mid"],
                         "e_outer": next(r for r in reduced if r["N"] == n)["e_outer"]}},
                **{label: {"e_mid": stored[(label, n)]["e_mid"],
                           "e_outer": stored[(label, n)]["e_outer"]}
                   for label in STORED_LABELS})
            for n in GRIDS},
        "ratios": {f"N{n}": {
            f"e_full_sl_over_{LABEL}": stored[("sl", n)]["e_full"] / e_full[n],
            f"e_full_{LABEL}_over_champion":
                e_full[n] / stored[("exp", n)]["e_full"],
            f"e_full_bmc_p11_over_{LABEL}":
                stored[("bmc_p11", n)]["e_full"] / e_full[n],
            f"e_full_cw12b_over_{LABEL}":
                stored[("cw12b", n)]["e_full"] / e_full[n],
        } for n in GRIDS},
        "prediction": {
            "path": str(PREDICTION.relative_to(ROOT)),
            "sha256": sha256(PREDICTION),
            "P": {f"N{n}": prediction["grids"][str(n)]["P"] for n in GRIDS},
            K_RATIO_SL: {
                f"N{n}": prediction["grids"][str(n)][K_RATIO_SL]
                for n in GRIDS},
            K_RATIO_CHAMPION: {
                f"N{n}": prediction["grids"][str(n)][K_RATIO_CHAMPION]
                for n in GRIDS},
        },
        "reference_fields": references,
        "runs": {f"N{n}": {k: registry[n].get(k) for k in (
            "pid", "command", "launch_time_utc", "completion_time_utc",
            "returncode", "landing_time", "landing_abs_difference",
            "landed_on_tstar", "verdict", "wall_s", "steps", "wall_per_step_s",
            "field", "field_sha256", "log")} for n in GRIDS},
        "shell_resolved": {row["key"]: {k: row[k] for k in (
            "shells", "shell_spectrum", "shell_reference_spectrum",
            "shell_relative_error")} for row in reduced},
        "regression": regression(bundles),
    })
    DECISION.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"outcome": decision["outcome"],
                      K_TESTS: decision[K_TESTS],
                      "mechanism": {k: decision["mechanism"][k] for k in
                                    ("N64", "N128", "confirmed")},
                      "regression_all_bit_identical":
                          decision["regression"]["all_bit_identical"]}, indent=2))
    print(f"wrote {RUNS.relative_to(ROOT)} and {DECISION.relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------- check


def check(args) -> int:
    """Recompute the decision from the stored quantities in e2_runs.csv."""
    with RUNS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    e_full = {int(row["N"]): float(row["e_full"])
              for row in rows if row["label"] == LABEL}
    prediction = json.loads(PREDICTION.read_text())
    stored = json.loads(DECISION.read_text())
    recomputed = decide(e_full, prediction)
    agreement = {
        "outcome": recomputed["outcome"] == stored["outcome"],
        "tests": recomputed[K_TESTS] == stored[K_TESTS],
        "mechanism_confirmed": (recomputed["mechanism_confirmed"]
                                == stored["mechanism_confirmed"]),
        "ratios": all(
            abs(recomputed["mechanism"][f"N{n}"][K_MECHANISM_REALIZED]
                - stored["mechanism"][f"N{n}"][K_MECHANISM_REALIZED])
            <= 0.0 for n in GRIDS),
    }
    print(json.dumps({"stored_outcome": stored["outcome"],
                      "recomputed_outcome": recomputed["outcome"],
                      "agreement": agreement}, indent=2))
    return 0 if all(agreement.values()) else 1


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    """Dispatch the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--reduce", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--member", default=None,
                        help="member file to field; default the pert5 "
                             "everything member of Gate 8")
    parser.add_argument("--tag", default=None,
                        help="suffix of every output name and the route label "
                             "inside the receipts; default none, the frozen "
                             "pert5 names")
    args = parser.parse_args(argv)
    configure(args.member, args.tag)
    GATE.mkdir(parents=True, exist_ok=True)
    if args.predict:
        return predict(args)
    if args.run:
        return run(args)
    if args.reduce:
        return reduce(args)
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
