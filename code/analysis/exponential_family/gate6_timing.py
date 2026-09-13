"""Gate 6: a sequential wall time receipt at N = 128 for the base configuration.

The Gate 4c per step timings at N = 128 were collected with five solver
processes alive at once, so they measure contention as much as arithmetic:
in that registry cw12b is faster with two thirds dealiasing than without,
which cannot be a property of the scheme. This script reruns the same base
configuration one process at a time.

Contract of the receipt:

* routes exp (the E62 champion member), sl, opt, bmc p9, bmc p11;
* base configuration of Gate 4c: N = 128, dealias none, cfl 0.20, the common
  route agnostic initial condition of results/exponential_family/gate4c/ic,
  exact time landing at t*, command lines built by gate4c_tgv.command_for and
  asserted equal to the strings in gate4c/process_registry.csv;
* three replicates per route, launched in round robin over the routes so that
  any drift of the machine spreads across routes rather than into one of them;
* strictly one solver process at a time, OMP_NUM_THREADS = 4, nice 19;
* wall time per step read from the solver VERDICT line by
  gate4c_tgv.parse_log, together with the loop wall time and the step count;
* each replicate's landed field is hashed and compared with the Gate 4c
  registry digest for the same route, then deleted.

The timing quantity is the per step wall time of one route divided by that of
another, so the reported ratios are dimensionless cost factors at fixed grid,
fixed time step, and fixed step count.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (str(HERE), str(HERE.parents[1])):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate4c_tgv as g4  # noqa: E402

ROOT = g4.ROOT
GATE4C = g4.GATE
GATE6 = ROOT / "results" / "exponential_family" / "gate6"
REGISTRY = GATE4C / "process_registry.csv"
IC_PATH = g4.IC_DIR / "ic_tgv_N128.bin"

GRID = 128
DEALIAS = 0
DT_SCALE = 1.0
REPLICATES = 3
THREADS = 4
LABELS = ("exp", "sl", "opt", "bmc_p9", "bmc_p11")
RATIO_PAIRS = (("exp", "sl"), ("exp", "opt"), ("exp", "bmc_p9"), ("exp", "bmc_p11"))


def route_specs() -> list[tuple]:
    """Return the Gate 4c route tuples of this receipt in reporting order."""
    by_label = {label: (label, route, degree, needs)
                for label, route, degree, needs in g4.ROUTES}
    return [by_label[label] for label in LABELS]


def registry_rows() -> dict:
    """Return the Gate 4c registry rows of the base N = 128 runs by label."""
    rows = {}
    with REGISTRY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["N"] == str(GRID) and row["dealias"] == str(DEALIAS) \
                    and row["dt_scale"] == "1.0":
                rows[row["label"]] = row
    return rows


def one_run(run: dict, command: list[str], work: Path, log_path: Path) -> dict:
    """Execute one solver process to completion and return its parsed log."""
    environment = dict(os.environ, OMP_NUM_THREADS=str(THREADS),
                       OMP_PROC_BIND="false")
    with log_path.open("w") as handle:
        process = subprocess.Popen(
            command, cwd=str(work), env=environment,
            stdout=handle, stderr=subprocess.STDOUT, text=True)
        code = process.wait()
    parsed = g4.parse_log(log_path.read_text())
    parsed["returncode"] = code
    parsed["pid"] = process.pid
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True,
                        help="scratch directory for solver output and logs")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()

    work = Path(args.work).resolve()
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    GATE6.mkdir(parents=True, exist_ok=True)
    samples_path = work / "samples.jsonl"

    if not g4.SOLVER.exists():
        raise SystemExit(f"missing solver executable {g4.SOLVER}")
    if not IC_PATH.exists():
        raise SystemExit(f"missing common initial condition {IC_PATH}")

    registry = registry_rows()
    specs = route_specs()
    runs, commands = {}, {}
    for label, route, degree, needs_member in specs:
        run = g4.make_run(label, route, degree, needs_member, GRID, DEALIAS, DT_SCALE)
        command = g4.command_for(run, IC_PATH)
        expected = registry[label]["command"]
        if " ".join(command) != expected:
            raise SystemExit(f"command mismatch for {label}:\n"
                             f"  built    {' '.join(command)}\n"
                             f"  registry {expected}")
        runs[label] = run
        commands[label] = command

    done = {}
    if samples_path.exists():
        for line in samples_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[(record["route"], record["replicate"])] = record

    order = [(replicate, label) for replicate in range(args.replicates)
             for label in LABELS]
    with samples_path.open("a") as sink:
        for position, (replicate, label) in enumerate(order):
            if (label, replicate) in done:
                print(f"SKIP {label} r{replicate} (already recorded)", flush=True)
                continue
            run = runs[label]
            log_path = work / "logs" / f"{label}_r{replicate}.log"
            print(f"RUN {position + 1}/{len(order)} {label} replicate {replicate}",
                  flush=True)
            parsed = one_run(run, commands[label], work, log_path)
            if parsed["returncode"] != 0:
                raise SystemExit(f"{label} r{replicate} returned "
                                 f"{parsed['returncode']}")
            field = work / "data" / run["field_name"]
            if not field.exists():
                raise SystemExit(f"{label} r{replicate} produced no field")
            digest = g4.sha256(field)
            field.unlink()
            history = work / "data" / run["csv_name"]
            if history.exists():
                history.unlink()
            record = {
                "route": label,
                "replicate": replicate,
                "launch_position": position,
                "wall_per_step_s": parsed["wall_per_step_s"],
                "wall_s": parsed["wall_s"],
                "steps": parsed["steps"],
                "landing_time": parsed.get("landing_time"),
                "verdict": parsed.get("verdict"),
                "field_sha256": digest,
                "registry_sha256": registry[label]["field_sha256"],
                "registry_match": digest == registry[label]["field_sha256"],
                "pid": parsed["pid"],
                "completed_utc": g4.utc_now(),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            done[(label, replicate)] = record
            print(f"    {parsed['wall_per_step_s']:.6f} s/step over "
                  f"{parsed['steps']} steps, wall {parsed['wall_s']:.4f} s, "
                  f"registry_match {record['registry_match']}", flush=True)

    records = [done[(label, replicate)] for replicate in range(args.replicates)
               for label in LABELS if (label, replicate) in done]
    records.sort(key=lambda r: (LABELS.index(r["route"]), r["replicate"]))

    csv_path = GATE6 / "timing_N128.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "route", "replicate", "wall_per_step_s", "wall_s", "steps",
            "field_sha256", "registry_match"), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({
                "route": record["route"],
                "replicate": record["replicate"],
                "wall_per_step_s": f"{record['wall_per_step_s']:.6f}",
                "wall_s": f"{record['wall_s']:.4f}",
                "steps": record["steps"],
                "field_sha256": record["field_sha256"],
                "registry_match": str(record["registry_match"]).lower(),
            })

    per_route = {}
    for label in LABELS:
        subset = [r for r in records if r["route"] == label]
        per_step = [r["wall_per_step_s"] for r in subset]
        wall = [r["wall_s"] for r in subset]
        per_route[label] = {
            "replicates": len(subset),
            "wall_per_step_s": per_step,
            "wall_s": wall,
            "median_wall_per_step_s": float(np.median(per_step)),
            "min_wall_per_step_s": float(np.min(per_step)),
            "max_wall_per_step_s": float(np.max(per_step)),
            "spread_percent": float(100.0 * (np.max(per_step) - np.min(per_step))
                                    / np.median(per_step)),
            "median_wall_s": float(np.median(wall)),
            "steps": subset[0]["steps"],
            "field_sha256": sorted({r["field_sha256"] for r in subset}),
            "registry_sha256": registry[label]["field_sha256"],
            "all_replicates_match_registry": all(r["registry_match"] for r in subset),
            "gate4c_registry_wall_per_step_s": float(
                registry[label]["wall_per_step_s"]),
        }

    ratios = {}
    for numerator, denominator in RATIO_PAIRS:
        ratios[f"{numerator}_over_{denominator}"] = (
            per_route[numerator]["median_wall_per_step_s"]
            / per_route[denominator]["median_wall_per_step_s"])
    paired = {}
    for numerator, denominator in RATIO_PAIRS:
        per_round = []
        for replicate in range(args.replicates):
            top = [r for r in records
                   if r["route"] == numerator and r["replicate"] == replicate]
            bottom = [r for r in records
                      if r["route"] == denominator and r["replicate"] == replicate]
            if top and bottom:
                per_round.append(top[0]["wall_per_step_s"]
                                 / bottom[0]["wall_per_step_s"])
        paired[f"{numerator}_over_{denominator}"] = {
            "by_replicate": per_round,
            "median": float(np.median(per_round)),
        }

    gate4c_ratios = {
        f"{a}_over_{b}": (per_route[a]["gate4c_registry_wall_per_step_s"]
                          / per_route[b]["gate4c_registry_wall_per_step_s"])
        for a, b in RATIO_PAIRS}

    payload = {
        "producer": "code/analysis/exponential_family/gate6_timing.py",
        "written_utc": g4.utc_now(),
        "purpose": ("sequential wall time receipt at N = 128 replacing the "
                    "five at a time Gate 4c timings of the same configuration"),
        "configuration": {
            "N": GRID,
            "dealias": "none",
            "cfl": g4.BASE_CFL,
            "target_time": g4.TARGET_TIME,
            "reynolds": g4.REYNOLDS,
            "exact_time_landing": True,
            "initial_condition": {
                "path": str(IC_PATH.relative_to(ROOT)),
                "sha256": g4.sha256(IC_PATH),
            },
            "member": {
                "path": str(g4.MEMBER.relative_to(ROOT)),
                "sha256": g4.sha256(g4.MEMBER),
            },
            "solver": str(g4.SOLVER.relative_to(ROOT)),
            "solver_sha256": g4.sha256(g4.SOLVER),
            "solver_source_sha256": g4.sha256(g4.SOLVER_DIR / "tgv_re1600.cpp"),
            "threads_per_process": THREADS,
            "concurrent_processes": 1,
            "nice": 19,
            "replicates": args.replicates,
            "launch_order": "round robin over routes, replicate by replicate",
            "timing_source": "solver VERDICT line, parsed by gate4c_tgv.parse_log",
        },
        "commands": {label: " ".join(commands[label]) for label in LABELS},
        "commands_identical_to_gate4c_registry": True,
        "routes": per_route,
        "median_wall_per_step_s": {
            label: per_route[label]["median_wall_per_step_s"] for label in LABELS},
        "ratios_median_wall_per_step": ratios,
        "ratios_paired_within_replicate_round": paired,
        "gate4c_registry_ratios_for_comparison": gate4c_ratios,
        "bit_reproducibility": {
            "definition": ("each replicate's landed field SHA-256 equals the "
                           "Gate 4c registry digest of the same route"),
            "all_match": all(per_route[label]["all_replicates_match_registry"]
                             for label in LABELS),
        },
        "fields_retained": False,
    }
    (GATE6 / "timing_N128.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\nmedian wall per step at N = 128, base configuration, sequential")
    for label in LABELS:
        entry = per_route[label]
        print(f"  {label:8s} {entry['median_wall_per_step_s']:.6f} s/step  "
              f"spread {entry['spread_percent']:.2f} percent  "
              f"gate4c {entry['gate4c_registry_wall_per_step_s']:.6f}")
    for key, value in ratios.items():
        print(f"  ratio {key:18s} {value:.4f}  "
              f"(gate4c registry {gate4c_ratios[key]:.4f})")
    print(f"wrote {csv_path.relative_to(ROOT)} and "
          f"{(GATE6 / 'timing_N128.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
