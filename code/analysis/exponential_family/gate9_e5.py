"""Gate 9 section E5: transport on fewer points and the immersed wall.

Two producers are imported unchanged and driven for the two fielded Gate 8
members,

  pert5   X8_n10_C_SLC1_pert5, X_4(i 0.3628 pi; quad) x P_3, C^2,
  core6   X9_m6_C_SLC1_SLC1992_6IIq2, X_6(i 0.5870 pi, i 0.7757 pi) x P_2, C^4.

Semi-Lagrangian mission (gate5b_semilagrangian.py)
--------------------------------------------------

The mission, the packets K = 12 and 24, the shifts A and B, the horizons
M = 1, 8 and 32, the one percent channel bar, the 1e-12 mass bar and the two
ladders are the stored ones. A member is carried as one coefficient tensor, so
the mission class inherits every method of gate5b.SplineMission and replaces
only the evaluator and the generator with the general evaluator and the Gate 8
polynomial generator of the member record. The nine-diagonal recovery solve is
the member's own lattice samples, trimmed at the stored sample floor.

The symbol-level predicted minima are written to e5_prediction.json before any
realized mission runs. The realized minima of the two members are then climbed
on the refined ladder of Amendment 1, and the rungs of the original ladder that
the refined climb leaves open are run as well, so a minimum is reported on both
ladders. The minima of the stored routes are read from the stored receipt
results/exponential_family/gate5/5b/decision.json and are not recomputed, since
they are already certified there; what is recomputed for the stored routes is
the mission time at those minima, so that every time in this receipt is
measured on one machine in one session.

The timing protocol is the stored one: one process per measurement, one thread,
the three components advanced without diagnostics, the median of three repeats,
the process launched as a subprocess of this producer with
`--stage timing-worker`.

Immersed wall (gate5c_ibm.py)
-----------------------------

The certified harness code/analysis/ibm_cert/run_matrix.py is imported read
only. gate5c_ibm.py installs one dispatching hook on its module-level
build_coupling for the champion's exponential branch; this module installs a
hook of the same shape that forwards the champion to gate5c's own function
object and adds the two members, so no file of the certified harness and no
frozen producer is edited. The construction of a member's coupling is the
construction the champion used, with the champion's generator replaced by the
member's: one tensor sample symbol b(theta_x) b(theta_y), the composed
divergence-gradient Helmholtz denominator from the member's own nodal first
derivative symbol, the off-grid read matrices from the member's own generator,
and H = h^2 I with S = B^{-1} Phi^T W.

Regression. The stored champion rows of gate5/5b/runs.csv and gate5/5c/runs.csv
are recomputed through the imported code and compared field by field. Every
deterministic field is required to agree bit for bit; the wall-clock fields are
reported beside them and are not part of the comparison, a measured time being
a property of the machine and the moment rather than of the code.
"""

from __future__ import annotations

import argparse

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
CERT = CODE / "analysis" / "ibm_cert"
for _path in (str(CODE), str(HERE), str(CERT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate5b_semilagrangian as g5b  # noqa: E402
import gate8_family as g8  # noqa: E402
from evaluator import as_evaluator  # noqa: E402

GATE8_MEMBERS = ROOT / "results" / "exponential_family" / "gate8" / "members"
GATE9 = ROOT / "results" / "exponential_family" / "gate9"
GATE5B = ROOT / "results" / "exponential_family" / "gate5" / "5b"
GATE5C = ROOT / "results" / "exponential_family" / "gate5" / "5c"
HISTORIES = GATE9 / "e5_histories"

MEMBERS = {
    "pert5": GATE8_MEMBERS / "X8_n10_C_SLC1_pert5.json",
    "core6": GATE8_MEMBERS / "X9_m6_C_SLC1_SLC1992_6IIq2.json",
}
MEMBER_NAMES = ("pert5", "core6")
STORED_NAMES = ("champion", "BMC9", "BMC11", "slc_operator_L12")
TIMING_NAMES = MEMBER_NAMES + STORED_NAMES
TIMING_HORIZONS = (8, 32)

FINE_LADDER = g5b.AMENDMENT_LADDER
COARSE_LADDER = g5b.LADDER
KS = g5b.KS
SHIFTS = tuple(g5b.SHIFTS)
HORIZONS = g5b.HORIZONS

# Amendment 1 of the Gate 9 contract states the expectation for both members:
# a minimum admissible grid at or below the champion's on the one percent line
# at K = 12 and 24, and a wall error ratio against SLC q4 plus L12 above 100
# at the finest grid.
CONTRACT_WALL_RATIO = 100.0


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# the member mission
# ----------------------------------------------------------------------

class MemberMission(g5b.SplineMission):
    """One Gate 8 member carried as a single coefficient tensor."""

    kind = "one field"

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = Path(path)
        self.record = json.loads(self.path.read_text())
        self.generator = g8.from_record(self.record)
        self.evaluator = as_evaluator(self.generator)
        samples = self.evaluator.value(np.arange(0, self.evaluator.n // 2 + 1))
        nonzero = np.nonzero(np.abs(samples) > g5b.SAMPLE_FLOOR)[0]
        self.samples = samples[: nonzero[-1] + 1]
        self.taps = self.evaluator.n
        self.solves_per_step = 3
        self.fields = 1

    def description(self) -> str:
        ev = self.evaluator
        return (f"{self.record['name']}, {ev.label()}, degree {ev.p}, support "
                f"[-{ev.n / 2:g}, {ev.n / 2:g}], class C^{ev.smoothness}, "
                f"{self.taps} taps per axis, "
                f"{2 * (self.samples.size - 1) + 1} diagonal recovery solve")

    def member_block(self) -> dict:
        ev = self.evaluator
        return {"route": self.name, "member": self.record["name"],
                "file": str(self.path.relative_to(ROOT)),
                "m": int(self.record["m"]), "n": int(ev.n), "p": int(ev.p),
                "smoothness": int(ev.smoothness),
                "cond": float(self.record["cond"]),
                "bandwidth": int(2 * (self.samples.size - 1) + 1),
                "taps_per_axis": int(self.taps)}


# The stored route table is captured before this module installs its own, so
# that installing the wider table cannot make the wider table call itself.
STORED_BUILD_ROUTES = g5b.build_routes


def build_routes() -> dict:
    """Return every mission route by name, the stored ones from gate5b."""
    routes = dict(STORED_BUILD_ROUTES())
    for name in MEMBER_NAMES:
        routes[name] = MemberMission(name, MEMBERS[name])
    return routes


def build_member_routes():
    return [MemberMission(name, MEMBERS[name]) for name in MEMBER_NAMES]


# ----------------------------------------------------------------------
# part 5b: prediction
# ----------------------------------------------------------------------

def predicted_case(name: str, K: int, shift: str) -> dict:
    """Return the predicted rows and the minima of one case on both ladders."""
    route = build_routes()[name]
    modes, amps = g5b.packet_modes(K)
    delta = g5b.shift_vector(shift)
    rows = []
    by_n = {}
    for npts in FINE_LADDER:
        history = g5b.predicted_histories(route, npts, modes, amps, delta)
        by_n[npts] = history
        for horizon in HORIZONS:
            row = {"route": name, "K": K, "shift": shift, "N": npts, "M": horizon}
            for key in g5b.HISTORY_KEYS:
                row[key] = float(history[key][horizon - 1])
                row[f"{key}_max"] = float(np.max(history[key][:horizon]))
            row["admissible"] = bool(g5b.admissible_horizon(history, horizon))
            rows.append(row)
    fine = {}
    coarse = {}
    below = []
    for horizon in HORIZONS:
        npts = g5b.minimum_grid(by_n, horizon, FINE_LADDER)
        fine[f"{name}|{K}|{shift}|{horizon}"] = npts
        coarse[f"{name}|{K}|{shift}|{horizon}"] = g5b.minimum_grid(
            by_n, horizon, COARSE_LADDER)
        under = g5b.rung_below(npts, FINE_LADDER)
        entry = {"route": name, "K": K, "shift": shift, "M": horizon,
                 "minimum_N": npts, "rung_below": under}
        if under is None:
            entry.update({"worst_channel": None, "worst_value": None,
                          "multiple_of_tolerance": None})
        else:
            value, key = g5b.worst_channel(by_n[under], horizon)
            entry.update({"worst_channel": key, "worst_value": value,
                          "multiple_of_tolerance": value / g5b.TOL})
        below.append(entry)
    return {"rows": rows, "fine": fine, "coarse": coarse, "below": below}


def _predict_star(task):
    return predicted_case(*task)


def combine(per_shift: dict, names) -> dict:
    """Return the per-shift minima reduced to the worse shift of each case."""
    out = {}
    for name in names:
        for K in KS:
            for horizon in HORIZONS:
                values = [per_shift.get(f"{name}|{K}|{shift}|{horizon}")
                          for shift in SHIFTS]
                out[f"{name}|{K}|{horizon}"] = (None if any(v is None for v in values)
                                                else max(values))
    return out


def stage_predict(jobs: int) -> dict:
    tasks = [(name, K, shift) for name in MEMBER_NAMES for K in KS
             for shift in SHIFTS]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
            results = list(pool.map(_predict_star, tasks))
    else:
        results = [predicted_case(*task) for task in tasks]
    rows = []
    fine = {}
    coarse = {}
    below = []
    for block in results:
        rows += block["rows"]
        fine.update(block["fine"])
        coarse.update(block["coarse"])
        below += block["below"]
    stored = json.loads((GATE5B / "decision.json").read_text())
    payload = {
        "producer": "code/analysis/exponential_family/gate9_e5.py",
        "written_utc": now(),
        "gate": "9.E5",
        "part": "5b",
        "imports": "code/analysis/exponential_family/gate5b_semilagrangian.py, unchanged",
        "members": [route.member_block() for route in build_member_routes()],
        "statement": (
            "The repeated-step transfer of a grid mode is the product of the "
            "three axis symbols of the route at that mode's angles, raised to "
            "the power M in the value channel and with one derivative factor "
            "in place of the last value factor in the derivative channels. The "
            "predicted histories are the exact modal L2 errors that arithmetic "
            "implies, including the aliasing of the packet onto the coarse "
            "grids."),
        "packet": {"seed": g5b.PACKET_SEED, "K": list(KS)},
        "shifts": {name: list(g5b.shift_vector(name)) for name in SHIFTS},
        "fine_ladder": list(FINE_LADDER),
        "coarse_ladder": list(COARSE_LADDER),
        "horizons": list(HORIZONS),
        "tolerance": g5b.TOL,
        "mass_tolerance": g5b.MASS_TOL,
        "rows": rows,
        "predicted_minimum_N_per_shift_fine": fine,
        "predicted_minimum_N_fine": combine(fine, MEMBER_NAMES),
        "predicted_minimum_N_per_shift_coarse": coarse,
        "predicted_minimum_N_coarse": combine(coarse, MEMBER_NAMES),
        "rung_below_predicted_minimum": below,
        "stored_routes_minimum_N_fine": stored["amendment_1"]["minimum_admissible_N"],
        "stored_routes_minimum_N_coarse": stored["minimum_admissible_N"],
        "stored_receipt": str((GATE5B / "decision.json").relative_to(ROOT)),
        "contract_expectation": (
            "The minimum admissible grid of each member is at or below the "
            "champion's on the one percent line at K = 12 and 24."),
    }
    GATE9.mkdir(parents=True, exist_ok=True)
    (GATE9 / "e5_prediction.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ----------------------------------------------------------------------
# part 5b: realized accuracy
# ----------------------------------------------------------------------

def accuracy_worker(name: str, K: int, shift: str, npts: int, threads: int = 1) -> dict:
    """Run one mission and return the stored row schema.

    The body is gate5b.accuracy_worker with this module's route table and this
    gate's history directory, so the arithmetic is the stored producer's and no
    stored history file is touched.
    """
    g5b.build_routes = build_routes
    g5b.HISTORIES = HISTORIES
    return g5b.accuracy_worker(name, K, shift, npts, threads)


def _accuracy_star(task):
    return accuracy_worker(*task)


def climb(jobs: int) -> list:
    """Climb the refined ladder for the two members, then close the coarse one."""
    rows = {}
    pending = [(name, K, shift) for name in MEMBER_NAMES for K in KS
               for shift in SHIFTS]
    for npts in FINE_LADDER:
        if not pending:
            break
        tasks = [(name, K, shift, npts, 1) for (name, K, shift) in pending]
        workers = max(1, min(jobs, len(tasks), g5b.amendment_workers(npts)))
        started = time.perf_counter()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_accuracy_star, tasks))
        else:
            results = [accuracy_worker(*task) for task in tasks]
        for (name, K, shift), row in zip(pending, results):
            rows[(name, K, shift, npts)] = row
        print(f"fine N = {npts}: {len(tasks)} missions on {workers} workers, "
              f"{time.perf_counter() - started:.1f} s", flush=True)
        pending = [key for key in pending
                   if not rows[(key[0], key[1], key[2], npts)]["admissible_M32"]]

    # The coarse ladder of the original contract is closed by running the rungs
    # the refined climb left open, so a minimum is reported on both ladders.
    extra = []
    for (name, K, shift) in [(n, K, s) for n in MEMBER_NAMES for K in KS
                             for s in SHIFTS]:
        fine_minimum = None
        for npts in FINE_LADDER:
            row = rows.get((name, K, shift, npts))
            if row is not None and row["admissible_M32"]:
                fine_minimum = npts
                break
        for npts in COARSE_LADDER:
            if (name, K, shift, npts) in rows:
                if rows[(name, K, shift, npts)]["admissible_M32"]:
                    break
                continue
            if fine_minimum is not None and npts < fine_minimum:
                continue
            extra.append((name, K, shift, npts, 1))
            break
    while extra:
        workers = max(1, min(jobs, len(extra),
                             g5b.amendment_workers(max(t[3] for t in extra))))
        started = time.perf_counter()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_accuracy_star, extra))
        else:
            results = [accuracy_worker(*task) for task in extra]
        again = []
        for task, row in zip(extra, results):
            rows[(task[0], task[1], task[2], task[3])] = row
            if not row["admissible_M32"]:
                place = list(COARSE_LADDER).index(task[3])
                if place + 1 < len(COARSE_LADDER):
                    again.append((task[0], task[1], task[2],
                                  COARSE_LADDER[place + 1], 1))
        print(f"coarse rungs {[t[3] for t in extra]}: "
              f"{time.perf_counter() - started:.1f} s", flush=True)
        extra = again
    return [rows[key] for key in sorted(rows, key=lambda k: (k[0], k[1], k[2], k[3]))]


def realized_minima(rows: list, ladder) -> dict:
    by_key = {}
    for row in rows:
        if row["kind"] != "accuracy":
            continue
        by_key.setdefault((row["route"], row["K"], row["shift"]), {})[row["N"]] = row
    per_shift = {}
    for (name, K, shift), by_n in by_key.items():
        for horizon in HORIZONS:
            found = None
            for npts in ladder:
                row = by_n.get(npts)
                if row is not None and row[f"admissible_M{horizon}"]:
                    found = npts
                    break
            per_shift[f"{name}|{K}|{shift}|{horizon}"] = found
    return {"per_shift": per_shift, "combined": combine(per_shift, MEMBER_NAMES)}


def below_rows(rows: list, minima: dict, ladder) -> list:
    """Return the worst realized channel history on the rung below each minimum."""
    index = {(row["route"], row["K"], row["shift"], row["N"]): row
             for row in rows if row["kind"] == "accuracy"}
    out = []
    for name in MEMBER_NAMES:
        for K in KS:
            for shift in SHIFTS:
                for horizon in HORIZONS:
                    npts = minima["per_shift"].get(f"{name}|{K}|{shift}|{horizon}")
                    under = g5b.rung_below(npts, ladder)
                    entry = {"route": name, "K": K, "shift": shift, "M": horizon,
                             "minimum_N": npts, "rung_below": under}
                    row = index.get((name, K, shift, under))
                    if row is None:
                        entry.update({"worst_channel": None, "worst_value": None,
                                      "multiple_of_tolerance": None})
                    else:
                        pairs = [(row[f"{key}_M{horizon}"], key)
                                 for key in g5b.CHANNELS]
                        value, key = max(pairs)
                        entry.update({"worst_channel": key, "worst_value": value,
                                      "multiple_of_tolerance": value / g5b.TOL})
                    out.append(entry)
    return out


# ----------------------------------------------------------------------
# part 5b: timing
# ----------------------------------------------------------------------

def run_timing(name: str, K: int, shift: str, npts: int, steps: int) -> dict:
    """Run one timing measurement in its own single-threaded subprocess."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--stage",
               "timing-worker", "--route", name, "--K", str(K), "--shift", shift,
               "--N", str(npts), "--steps", str(steps)]
    proc = subprocess.run(command, env=env, capture_output=True, text=True,
                          check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def timing_plan(member_minima: dict, stored_minima: dict) -> list:
    """Return the timing plan: every route at its own realized minimum."""
    plan = []
    for name in TIMING_NAMES:
        for K in KS:
            for shift in SHIFTS:
                for horizon in TIMING_HORIZONS:
                    if name in MEMBER_NAMES:
                        npts = member_minima["per_shift"].get(
                            f"{name}|{K}|{shift}|{horizon}")
                    else:
                        npts = stored_minima.get(f"{name}|{K}|{shift}|{horizon}")
                    if npts is None:
                        npts = FINE_LADDER[-1]
                    plan.append((name, K, shift, npts, horizon))
    return plan


def stage_timing(plan: list) -> list:
    cache = {}
    rows = []
    for name, K, shift, npts, horizon in plan:
        key = (name, shift, npts, horizon)
        if key not in cache:
            started = time.perf_counter()
            cache[key] = run_timing(name, K, shift, npts, horizon)
            print(f"timing {name} shift {shift} N = {npts} M = {horizon}: "
                  f"{cache[key]['wall_seconds']:.3f} s of mission, "
                  f"{time.perf_counter() - started:.1f} s of wall", flush=True)
        row = dict(cache[key])
        row["K"] = K
        row["basis"] = "minimum"
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# part 5b: the regression on the stored champion rows
# ----------------------------------------------------------------------

DETERMINISTIC = ([f"{key}_final" for key in g5b.HISTORY_KEYS]
                 + [f"{key}_M{horizon}" for horizon in HORIZONS
                    for key in g5b.HISTORY_KEYS]
                 + [f"admissible_M{horizon}" for horizon in HORIZONS])


def regression_5b(jobs: int, cap: int = 128) -> dict:
    """Recompute the stored champion accuracy rows and compare field by field."""
    stored = [row for row in csv.DictReader((GATE5B / "runs.csv").open())
              if row["kind"] == "accuracy" and row["route"] == "champion"
              and int(row["N"]) <= cap]
    tasks = [(row["route"], int(row["K"]), row["shift"], int(row["N"]), 1)
             for row in stored]
    workers = max(1, min(jobs, len(tasks)))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_accuracy_star, tasks))
    else:
        results = [accuracy_worker(*task) for task in tasks]
    differences = []
    for want, got in zip(stored, results):
        for field in DETERMINISTIC:
            a = want[field]
            b = got[field]
            text = str(b) if not isinstance(b, bool) else str(b)
            if a != text:
                differences.append({"route": want["route"], "K": want["K"],
                                    "shift": want["shift"], "N": want["N"],
                                    "field": field, "stored": a,
                                    "recomputed": text})
    return {
        "file": str((GATE5B / "runs.csv").relative_to(ROOT)),
        "route": "champion",
        "grid_cap": cap,
        "rows_compared": len(stored),
        "fields_per_row": len(DETERMINISTIC),
        "identical": bool(not differences),
        "differences": differences[:12],
        "excluded_fields": ["wall_seconds", "wall_seconds_all", "threads",
                            "history_file", "peak_rss_increment_bytes",
                            "baseline_rss_bytes", "peak_rss_bytes"],
        "statement": ("Every deterministic field of the stored champion "
                      "accuracy rows at N at most the cap is reproduced "
                      "character for character by the imported producer. The "
                      "wall-clock and resident-memory fields are properties of "
                      "the machine and are excluded."),
    }


# ----------------------------------------------------------------------
# part 5b: the receipt
# ----------------------------------------------------------------------

CSV_FIELDS = (["kind", "route", "K", "shift", "N", "steps", "basis", "threads",
               "wall_seconds", "wall_seconds_all", "peak_rss_increment_bytes",
               "baseline_rss_bytes", "peak_rss_bytes", "state_bytes",
               "history_file"]
              + [f"{key}_final" for key in g5b.HISTORY_KEYS]
              + [f"{key}_M{horizon}" for horizon in HORIZONS
                 for key in g5b.HISTORY_KEYS]
              + [f"admissible_M{horizon}" for horizon in HORIZONS])


def write_runs(rows: list, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.setdefault("threads", 1)
            if isinstance(out.get("wall_seconds_all"), list):
                out["wall_seconds_all"] = ";".join(f"{v:.6f}"
                                                   for v in out["wall_seconds_all"])
            writer.writerow(out)


def fmt(value, spec="{:.4e}") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return spec.format(value)


def grid_text(value) -> str:
    return "none" if value is None else str(value)


def cell(text: str) -> str:
    """Return one markdown cell, the route separator escaped.

    Several certified route names carry a vertical bar, which is the column
    separator of a markdown table, so it is escaped wherever a name is printed.
    """
    return str(text).replace("|", "\\|")


def write_5b_markdown(payload: dict, path: Path) -> None:
    lines = ["# Gate 9 section E5, part 5b: the semi-Lagrangian mission", "",
             f"Produced {payload['written_utc']} by "
             "code/analysis/exponential_family/gate9_e5.py, which imports "
             "gate5b_semilagrangian.py unchanged.", "",
             "## Members", ""]
    for block in payload["members"]:
        lines.append(f"* {block['route']}: {block['member']}, n = {block['n']}, "
                     f"degree {block['p']}, class C^{block['smoothness']}, "
                     f"bandwidth {block['bandwidth']}, cond {block['cond']:.3f}")
    lines += ["", "## Minimum admissible grid per direction", "",
              "Refined ladder of Amendment 1, worse of the two shifts.", "",
              "| route | K = 12, M = 1 | M = 8 | M = 32 | K = 24, M = 1 | M = 8 "
              "| M = 32 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    fine = payload["minimum_admissible_N_fine"]
    for route in payload["route_order"]:
        cells = [grid_text(fine.get(f"{route}|{K}|{h}"))
                 for K in KS for h in HORIZONS]
        lines.append(f"| {route} | " + " | ".join(cells) + " |")
    lines += ["", "Original ladder of the Gate 5 contract, worse of the two "
              "shifts. The stored part 5b receipt carries the original ladder "
              "for the three routes of the original contract alone, so the two "
              "routes that Amendment 1 added read as absent there.", "",
              "| route | K = 12, M = 1 | M = 8 | M = 32 | K = 24, M = 1 | M = 8 "
              "| M = 32 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    coarse = payload["minimum_admissible_N_coarse"]
    for route in payload["route_order"]:
        cells = [grid_text(coarse.get(f"{route}|{K}|{h}"))
                 for K in KS for h in HORIZONS]
        lines.append(f"| {route} | " + " | ".join(cells) + " |")
    lines += ["", "Predicted against realized, refined ladder, per member:", "",
              "| route | K | M | predicted | realized |",
              "| --- | --- | --- | --- | --- |"]
    predicted = payload["predicted_minimum_N_fine"]
    for route in MEMBER_NAMES:
        for K in KS:
            for horizon in HORIZONS:
                lines.append(f"| {route} | {K} | {horizon} | "
                             f"{grid_text(predicted.get(f'{route}|{K}|{horizon}'))} | "
                             f"{grid_text(fine.get(f'{route}|{K}|{horizon}'))} |")
    lines += ["", "## Mission time at the minima", "",
              "One process, one thread, the median of three repeats, the "
              "measurement launched as a subprocess of the producer.", "",
              "| route | K | shift | N | M | seconds |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in payload["timing_rows"]:
        lines.append(f"| {row['route']} | {row['K']} | {row['shift']} | "
                     f"{row['N']} | {row['steps']} | "
                     f"{row['wall_seconds']:.4f} |")
    lines += ["", "## Regression", "",
              f"{payload['regression']['rows_compared']} stored champion "
              f"accuracy rows of {payload['regression']['file']} recomputed "
              f"through the imported producer over "
              f"{payload['regression']['fields_per_row']} deterministic fields "
              f"each: identical = {payload['regression']['identical']}.", ""]
    path.write_text("\n".join(lines) + "\n")


def stage_semilagrangian(jobs: int, skip_timing: bool = False) -> dict:
    GATE9.mkdir(parents=True, exist_ok=True)
    HISTORIES.mkdir(parents=True, exist_ok=True)
    prediction_file = GATE9 / "e5_prediction.json"
    if not prediction_file.exists():
        raise SystemExit("the prediction must be written before the realized stage")
    prediction = json.loads(prediction_file.read_text())

    started = now()
    rows = climb(jobs)
    fine = realized_minima(rows, FINE_LADDER)
    coarse = realized_minima(rows, COARSE_LADDER)

    stored = json.loads((GATE5B / "decision.json").read_text())
    stored_fine_per_shift = stored["amendment_1"]["minimum_admissible_N_per_shift"]
    stored_fine = dict(stored["amendment_1"]["minimum_admissible_N"])
    stored_coarse = dict(stored["minimum_admissible_N"])

    combined_fine = dict(fine["combined"])
    combined_fine.update(stored_fine)
    combined_coarse = dict(coarse["combined"])
    combined_coarse.update(stored_coarse)

    timing_rows = []
    if not skip_timing:
        plan = timing_plan(fine, stored_fine_per_shift)
        timing_rows = stage_timing(plan)

    write_runs(rows + timing_rows, GATE9 / "e5_runs.csv")

    champion_at_or_below = {}
    for name in MEMBER_NAMES:
        for K in KS:
            for horizon in HORIZONS:
                mine = combined_fine.get(f"{name}|{K}|{horizon}")
                his = combined_fine.get(f"champion|{K}|{horizon}")
                champion_at_or_below[f"{name}|{K}|{horizon}"] = {
                    "member_N": mine, "champion_N": his,
                    "at_or_below": (None if mine is None or his is None
                                    else bool(mine <= his))}

    payload = {
        "producer": "code/analysis/exponential_family/gate9_e5.py",
        "written_utc": now(),
        "realized_started_utc": started,
        "gate": "9.E5",
        "part": "5b",
        "prediction": str(prediction_file.relative_to(ROOT)),
        "prediction_written_utc": prediction["written_utc"],
        "prediction_before_runs": True,
        "members": prediction["members"],
        "route_order": list(MEMBER_NAMES) + list(STORED_NAMES),
        "fine_ladder": list(FINE_LADDER),
        "coarse_ladder": list(COARSE_LADDER),
        "tolerance": g5b.TOL,
        "mass_tolerance": g5b.MASS_TOL,
        "predicted_minimum_N_fine": prediction["predicted_minimum_N_fine"],
        "predicted_minimum_N_coarse": prediction["predicted_minimum_N_coarse"],
        "predicted_minimum_N_per_shift_fine":
            prediction["predicted_minimum_N_per_shift_fine"],
        "member_minimum_N_per_shift_fine": fine["per_shift"],
        "member_minimum_N_per_shift_coarse": coarse["per_shift"],
        "minimum_admissible_N_fine": combined_fine,
        "minimum_admissible_N_coarse": combined_coarse,
        "stored_routes_from": str((GATE5B / "decision.json").relative_to(ROOT)),
        "rung_below_minimum_fine": below_rows(rows, fine, FINE_LADDER),
        "member_at_or_below_the_champion": champion_at_or_below,
        "mass_drift_max": max(float(row["mass_final"]) for row in rows),
        "timing_rows": timing_rows,
        "timing_protocol": (
            "One process per measurement, one thread, the three components "
            "advanced without diagnostics, the median of three repeats, the "
            "process launched as a subprocess of this producer with --stage "
            "timing-worker, which is the stored protocol of part 5b."),
        "timing_machine_note": (
            "The times are measured on a shared machine and are comparable "
            "with each other within this receipt. They are not comparable with "
            "the stored part 5b times, which were measured in another session, "
            "which is why every route of this comparison is timed again here."),
        "regression": regression_5b(jobs),
        "runs": "results/exponential_family/gate9/e5_runs.csv",
    }
    (GATE9 / "e5_semilagrangian.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_5b_markdown(payload, GATE9 / "e5_semilagrangian.md")
    return payload


# ----------------------------------------------------------------------
# part 5c: the immersed wall
# ----------------------------------------------------------------------

def wall_module():
    import gate5c_ibm as g5c
    return g5c


def member_generator(name: str):
    return generator_from_file(MEMBERS[name])


def generator_from_file(path):
    """Return the Gate 8 generator of any member record on disk."""
    return g8.from_record(json.loads(Path(path).read_text()))


def member_route_entry(name: str) -> dict:
    """Return the harness route dictionary of one member, in the exp branch."""
    import hashlib
    record = json.loads(MEMBERS[name].read_text())
    generator = member_generator(name)
    route_id = f"EXP-{record['name']}"
    return {
        "primary_route_id": route_id,
        "kind": "exp",
        "member": name,
        "derivative_route": route_id,
        "formal_order": generator.m,
        "degree": generator.p,
        "m": generator.m,
        "omegas": list(record["omegas"]),
        "member_file": str(MEMBERS[name]),
        "logical_aliases": [route_id],
        "evaluator_hash": hashlib.sha256(
            f"gate8-member:{record['name']}:{json.dumps(record['omegas'])}:"
            f"{json.dumps(record['real_pairs'])}:{json.dumps(record['quads'])}:"
            f"{json.dumps(record['polynomial'])}".encode()).hexdigest(),
    }


def member_bulk_denominator(generator, npts: int, h: float, rm) -> np.ndarray:
    """Return the composed divergence-gradient Helmholtz symbol of a member."""
    theta = np.fft.fftfreq(npts, d=1.0) * rm.BOX
    theta = np.where(theta >= 0.0, theta, theta + rm.BOX)
    theta = np.where(theta <= np.pi, theta, theta - rm.BOX)
    theta = np.abs(theta)
    kprime = generator.symbols(np.asarray(theta, dtype=float))[1]
    symbol = (kprime / h) ** 2
    return symbol[:, None] + symbol[None, :] + rm.KAPPA ** 2


def member_symbol_2d(evaluator, npts: int) -> np.ndarray:
    """Return the tensor sample symbol b(theta_x) b(theta_y) of a member."""
    radius = (evaluator.n + 1) // 2
    lattice = np.arange(-radius, radius + 1).astype(float)
    samples = evaluator.value(lattice)
    column = np.zeros(npts)
    for offset, value in zip(lattice.astype(int), samples):
        column[offset % npts] += value
    symbol = np.real(np.fft.fft(column))
    return symbol[:, None] * symbol[None, :]


def build_member_phi(evaluator, points: np.ndarray, npts: int, h: float,
                     derivative: int = 0):
    """Return the sparse off-grid read matrix of a member's generator."""
    from scipy.sparse import csr_matrix

    radius = 0.5 * evaluator.n
    span = evaluator.n + 2
    order_x = derivative
    order_y = 0
    px = (points[:, 0] / h) % npts
    py = (points[:, 1] / h) % npts
    base_x = np.floor(px - radius).astype(np.intp)
    base_y = np.floor(py - radius).astype(np.intp)
    half = npts / 2.0
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for m in range(points.shape[0]):
        dx = np.array([px[m] - (base_x[m] + s) for s in range(span)])
        dy = np.array([py[m] - (base_y[m] + s) for s in range(span)])
        dx = np.where(dx > half, dx - npts, np.where(dx < -half, dx + npts, dx))
        dy = np.where(dy > half, dy - npts, np.where(dy < -half, dy + npts, dy))
        wx = evaluator.derivative(dx, order_x)
        wy = evaluator.derivative(dy, order_y)
        ix = (base_x[m] + np.arange(span)) % npts
        iy = (base_y[m] + np.arange(span)) % npts
        for a in range(span):
            if wx[a] == 0.0:
                continue
            for b in range(span):
                value = wx[a] * wy[b]
                if value == 0.0:
                    continue
                rows.append(m)
                columns.append(int(ix[a]) * npts + int(iy[b]))
                data.append(float(value))
    return csr_matrix((data, (rows, columns)),
                      shape=(points.shape[0], npts * npts))


def build_member_coupling(route: dict, case, rm) -> dict:
    """Return a member's coupling dictionary in the harness convention."""
    return coupling_from_generator(generator_from_file(route["member_file"]),
                                   case, rm)


def coupling_from_generator(generator, case, rm) -> dict:
    """Return the coupling of one exponential-family generator.

    The construction is the champion's, with the champion's generator replaced
    by the given one: one tensor sample symbol, that generator's own nodal first
    derivative in the composed divergence-gradient denominator, its own off-grid
    reads, and H = h^2 I with S = B^{-1} Phi^T W.
    """
    evaluator = as_evaluator(generator)
    npts = case.npts
    h = case.h
    weights = case.weights
    denominator = member_bulk_denominator(generator, npts, h, rm)

    def solve(rhs: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.fft2(rhs) / denominator))

    phi_m = build_member_phi(evaluator, case.markers, npts, h)
    phi_q = build_member_phi(evaluator, case.queries, npts, h)
    b2 = member_symbol_2d(evaluator, npts)
    g_denominator = b2 * b2 * denominator
    ab_denominator = b2 * denominator

    def apply_binv(field):
        return np.real(np.fft.ifft2(np.fft.fft2(field) / b2))

    def marker_eval(field):
        return phi_m @ apply_binv(field).ravel()

    def query_eval(field):
        return phi_q @ apply_binv(field).ravel()

    def spread_euclidean(values):
        raw = (phi_m.T @ (weights * values)).reshape(npts, npts)
        return apply_binv(raw)

    def schur_column(values):
        raw = (phi_m.T @ (weights * values)).reshape(npts, npts)
        response = np.real(np.fft.ifft2(np.fft.fft2(raw) / g_denominator))
        return phi_m @ response.ravel()

    def correction(values):
        raw = (phi_m.T @ (weights * values)).reshape(npts, npts)
        return np.real(np.fft.ifft2(np.fft.fft2(raw) / ab_denominator))

    inverse_kernel = np.real(np.fft.ifft(1.0 / np.sqrt(np.diag(b2))))
    inverse_l1 = float(np.sum(np.abs(inverse_kernel))) ** 2
    query_l1 = float(np.max(np.asarray(np.abs(phi_q).sum(axis=1)).ravel()))
    constant_residual = max(
        float(np.max(np.abs(marker_eval(np.ones((npts, npts))) - 1.0))),
        float(np.max(np.abs(query_eval(np.ones((npts, npts))) - 1.0))),
    )
    return {
        "solve": solve,
        "E_marker": marker_eval,
        "E_query": query_eval,
        "S_euclidean": spread_euclidean,
        "schur_column": schur_column,
        "correction": correction,
        "evaluation_amplification": query_l1 * inverse_l1,
        "constant_residual": constant_residual,
        "marker_nnz": int(phi_m.nnz),
        "query_nnz": int(phi_q.nnz),
    }


def install_wall_hook(g5c, rm) -> None:
    """Install the dispatching hook that adds the member branch.

    The champion's exp branch is forwarded to gate5c_ibm's own function object
    and every certified route kind to the certified one, so the stored routes
    run through literally the same code.
    """
    champion_branch = g5c.build_exp_coupling
    original = rm.build_coupling

    def dispatch(route, evaluator, case, q_table):
        if route.get("kind") == "exp":
            if route.get("member") is not None:
                return build_member_coupling(route, case, rm)
            return champion_branch(route, case)
        return original(route, evaluator, case, q_table)

    rm.build_coupling = dispatch


CHAMPION_MEMBER_FILE = (ROOT / "results" / "exponential_family" / "members"
                        / "E62_champion.json")


def cross_check_champion(harness, g5c, rm) -> dict:
    """Run the champion through the member branch and against its own branch.

    The member branch reaches phi through general_evaluator.GeneralEvaluator and
    the Gate 8 alias route, the champion's own branch through
    evaluator.ExponentialEvaluator and the Gate 0 route, so the two agree only
    to the rounding of the two code paths. Reporting that gap is what says that
    the member branch is the champion's construction with the generator
    exchanged, and not a different experiment.
    """
    shift = g5c.SHIFTS[0]
    npts = g5c.GRIDS[0]
    case = rm.prepare_case(harness.reference, npts, shift[0], (shift[1], shift[2]),
                           max(48, npts // 2))
    entry = {"primary_route_id": "CHAMPION-THROUGH-THE-MEMBER-BRANCH",
             "kind": "exp", "member": "champion",
             "member_file": str(CHAMPION_MEMBER_FILE),
             "derivative_route": "CHAMPION-THROUGH-THE-MEMBER-BRANCH",
             "formal_order": 6, "degree": 9, "m": 6,
             "logical_aliases": ["CHAMPION-THROUGH-THE-MEMBER-BRANCH"],
             "evaluator_hash": "0" * 64}
    got = {}
    for route, evaluator, tag in ((entry, None, "member_branch"),
                                  (harness.entry(g5c.CHAMPION_ID)[0], None,
                                   "champion_branch")):
        result = rm.timed_cell(route, evaluator, case, harness.q_table,
                               harness.certificate, 1)
        got[tag] = {"trace_error": float(result["trace_error"]),
                    "exterior_relative_l2": float(result["exterior_relative_l2"])}
    out = {"N": npts, "shift_id": shift[0], "member_file":
           str(CHAMPION_MEMBER_FILE.relative_to(ROOT)), **got}
    for key in ("trace_error", "exterior_relative_l2"):
        a = got["member_branch"][key]
        b = got["champion_branch"][key]
        out[f"relative_gap_{key}"] = abs(a - b) / b
        out[f"absolute_gap_{key}"] = abs(a - b)
    return out


def stage_wall(repeats: int | None = None) -> dict:
    g5c = wall_module()
    import run_matrix as rm

    g5c.install_hook()
    install_wall_hook(g5c, rm)
    harness = g5c.Harness()
    member_entries = {}
    for name in MEMBER_NAMES:
        entry = member_route_entry(name)
        harness.roster[entry["primary_route_id"]] = (entry, None)
        member_entries[name] = entry
    route_ids = (tuple(entry["primary_route_id"] for entry in member_entries.values())
                 + tuple(g5c.ROUTE_IDS))
    labels = dict(g5c.ROUTE_LABELS)
    for name, entry in member_entries.items():
        labels[entry["primary_route_id"]] = name
    repeats = g5c.REPEATS if repeats is None else repeats

    GATE9.mkdir(parents=True, exist_ok=True)
    prediction = {
        "producer": "code/analysis/exponential_family/gate9_e5.py",
        "written_utc": now(),
        "part": "5c",
        "routes": list(route_ids),
        "basis": ("The coupled off-grid symbols of part 5a: the value-channel "
                  "transfer residual of each route's own evaluator and the "
                  "composed divergence-gradient Helmholtz symbol residual of "
                  "each route's own nodal first derivative, both weighted by "
                  "the modal amplitudes of the same 768-square Fourier "
                  "reference."),
        "contract_expectation": (
            "The wall error ratio of each member against SLC q4 plus L12 is "
            "above 100 at the finest grid."),
        "contract_ratio": CONTRACT_WALL_RATIO,
    }
    (GATE9 / "e5_wall_prediction.json").write_text(
        json.dumps(prediction, indent=2) + "\n")

    cells = []
    for npts in g5c.GRIDS:
        markers_per_obstacle = max(48, npts // 2)
        for shift_id, sx, sy in g5c.SHIFTS:
            case = rm.prepare_case(harness.reference, npts, shift_id, (sx, sy),
                                   markers_per_obstacle)
            if case.reference_translation_residual > 1e-12:
                raise RuntimeError(
                    f"reference translation residual failed: N={npts} {shift_id}")
            for route_id in route_ids:
                route, evaluator = harness.entry(route_id)
                result = rm.timed_cell(route, evaluator, case, harness.q_table,
                                       harness.certificate, repeats)
                curves, recomputed = g5c.per_curve_trace(route, evaluator, case,
                                                         harness.q_table)
                if recomputed != result["trace_error"]:
                    raise RuntimeError(
                        f"per-curve pass diverged from the certified trace error "
                        f"for {route_id} N={npts} {shift_id}")
                cells.append({
                    "route": route_id,
                    "label": labels[route_id],
                    "N": npts,
                    "h": case.h,
                    "shift_id": shift_id,
                    "shift_cell_units": [sx, sy],
                    "n_markers_per_obstacle": markers_per_obstacle,
                    "n_markers_total": int(case.markers.shape[0]),
                    "result": result,
                    "per_curve_trace_error": curves,
                    "shared_case_hashes": case.hashes,
                })
                print(f"[E5 wall] N={npts} {shift_id} {route_id:24s} "
                      f"trace={result['trace_error']:.6e} "
                      f"exterior={result['exterior_relative_l2']:.6e} "
                      f"t={result['total_time_seconds']:.4f}s "
                      f"invariants="
                      f"{'pass' if result['invariants_pass'] else 'FAIL'}",
                      flush=True)
    (GATE9 / "e5_wall_cells.json").write_text(
        json.dumps(cells, indent=2, default=float, sort_keys=True) + "\n")

    index = g5c.cell_index(cells)
    comparisons = {}
    for name, entry in member_entries.items():
        mine = entry["primary_route_id"]
        for competitor in ("SLC-SL|common_L12", "BMC9", g5c.CHAMPION_ID):
            for metric, threshold in (("wall_trace", g5c.TRACE_FACTOR_THRESHOLD),
                                      ("exterior_relative_l2",
                                       g5c.EXTERIOR_FACTOR_THRESHOLD)):
                key = f"{name}|{competitor}|{metric}"
                comparisons[key] = g5c.metric_block(index, mine, competitor,
                                                    metric, threshold)

    finest = g5c.GRIDS[-1]
    finest_ratios = {}
    for name, entry in member_entries.items():
        mine = entry["primary_route_id"]
        for competitor in ("SLC-SL|common_L12", "BMC9", g5c.CHAMPION_ID):
            cells_ratio = {}
            for metric, (error_key, floor_key) in g5c.METRICS.items():
                per_shift = {}
                for shift_id, _, _ in g5c.SHIFTS:
                    a = index[(mine, finest, shift_id)]["result"]
                    b = index[(competitor, finest, shift_id)]["result"]
                    per_shift[shift_id] = {
                        "member": a[error_key], "competitor": b[error_key],
                        "competitor_over_member": b[error_key] / a[error_key],
                        "member_above_ten_times_floor":
                            bool(a[error_key] > 10.0 * a[floor_key]),
                        "competitor_above_ten_times_floor":
                            bool(b[error_key] > 10.0 * b[floor_key]),
                    }
                worst = min(v["competitor_over_member"] for v in per_shift.values())
                cells_ratio[metric] = {
                    "per_shift": per_shift,
                    "worst_shift_ratio": worst,
                    "above_contract_ratio": bool(worst > CONTRACT_WALL_RATIO),
                }
            finest_ratios[f"{name}|{competitor}"] = cells_ratio

    times = {}
    for route_id in route_ids:
        rows = [index[(route_id, npts, shift_id)]["result"]
                for npts in g5c.GRIDS for shift_id, _, _ in g5c.SHIFTS]
        times[route_id] = {
            "total_seconds_by_N": {
                str(npts): float(np.median([
                    index[(route_id, npts, shift_id)]["result"]["total_time_seconds"]
                    for shift_id, _, _ in g5c.SHIFTS]))
                for npts in g5c.GRIDS},
            "setup_seconds_by_N": {
                str(npts): float(np.median([
                    index[(route_id, npts, shift_id)]["result"]["timings"][
                        g5c.SETUP_KEY]
                    for shift_id, _, _ in g5c.SHIFTS]))
                for npts in g5c.GRIDS},
            "median_total_seconds": float(np.median(
                [row["total_time_seconds"] for row in rows])),
        }
    time_ratios = {}
    for name, entry in member_entries.items():
        mine = entry["primary_route_id"]
        for competitor in ("SLC-SL|common_L12", "BMC9", g5c.CHAMPION_ID):
            time_ratios[f"{name}|{competitor}"] = {
                str(npts): (times[mine]["total_seconds_by_N"][str(npts)]
                            / times[competitor]["total_seconds_by_N"][str(npts)])
                for npts in g5c.GRIDS}

    payload = {
        "producer": "code/analysis/exponential_family/gate9_e5.py",
        "written_utc": now(),
        "gate": "9.E5",
        "part": "5c",
        "imports": ("code/analysis/exponential_family/gate5c_ibm.py and "
                    "code/analysis/ibm_cert/run_matrix.py, both read only; one "
                    "dispatching hook on run_matrix.build_coupling that "
                    "forwards the champion to gate5c_ibm's own branch"),
        "members": [MemberMission(name, MEMBERS[name]).member_block()
                    for name in MEMBER_NAMES],
        "member_route_ids": {name: entry["primary_route_id"]
                             for name, entry in member_entries.items()},
        "routes": list(route_ids),
        "grids": list(g5c.GRIDS),
        "shifts": [list(s) for s in g5c.SHIFTS],
        "repeats": repeats,
        "prediction": "results/exponential_family/gate9/e5_wall_prediction.json",
        "contract_expectation": prediction["contract_expectation"],
        "contract_ratio": CONTRACT_WALL_RATIO,
        "paper_reference_factors": {
            "wall_trace": g5c.TRACE_FACTOR_THRESHOLD,
            "exterior_relative_l2": g5c.EXTERIOR_FACTOR_THRESHOLD},
        "factor_rule_comparisons": comparisons,
        "finest_grid_ratios": finest_ratios,
        "times": times,
        "time_ratios_member_over_competitor": time_ratios,
        "invariants_all_pass": bool(all(c["result"]["invariants_pass"]
                                        for c in cells)),
        "contract_check": {
            "expectation": ("Amendment 1 of the Gate 9 contract expects a wall "
                            "error ratio against SLC q4 plus L12 above 100 at "
                            "the finest grid; section E5 records a ratio below "
                            "10 as a corner."),
            "ratio_threshold": CONTRACT_WALL_RATIO,
            "corner_threshold": 10.0,
            "finest_grid": g5c.GRIDS[-1],
            "rows": [
                {"member": name, "metric": metric,
                 "worst_shift_ratio": finest_ratios[
                     f"{name}|SLC-SL|common_L12"][metric]["worst_shift_ratio"],
                 "above_100": bool(finest_ratios[
                     f"{name}|SLC-SL|common_L12"][metric]["worst_shift_ratio"]
                     > CONTRACT_WALL_RATIO),
                 "corner_below_10": bool(finest_ratios[
                     f"{name}|SLC-SL|common_L12"][metric]["worst_shift_ratio"]
                     < 10.0)}
                for name in MEMBER_NAMES
                for metric in ("wall_trace", "exterior_relative_l2")],
        },
        "cross_check_champion_through_the_member_branch":
            cross_check_champion(harness, g5c, rm),
        "regression": regression_5c(index, g5c),
        "cells": "results/exponential_family/gate9/e5_wall_cells.json",
        "runs": "results/exponential_family/gate9/e5_wall_runs.csv",
    }
    write_wall_runs(cells, GATE9 / "e5_wall_runs.csv", g5c)
    (GATE9 / "e5_wall.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_wall_markdown(payload, index, g5c, GATE9 / "e5_wall.md")
    return payload


WALL_DETERMINISTIC = ["trace_error", "trace_error_disk", "trace_error_trefoil",
                      "trace_error_kite", "exterior_relative_l2",
                      "trace_numerical_floor",
                      "exterior_relative_l2_numerical_floor",
                      "trace_above_ten_times_floor",
                      "exterior_above_ten_times_floor",
                      "marker_constraint_residual", "schur_condition",
                      "weighted_adjoint_residual", "fielded_constant_residual",
                      "invariants_pass"]


def regression_5c(index: dict, g5c) -> dict:
    """Compare every stored 5c row recomputed here, field by field."""
    stored = list(csv.DictReader((GATE5C / "runs.csv").open()))
    differences = []
    compared = 0
    for row in stored:
        key = (row["route"], int(row["N"]), row["shift_id"])
        cell = index.get(key)
        if cell is None:
            continue
        rebuilt = wall_row(cell, g5c)
        compared += 1
        for field in WALL_DETERMINISTIC:
            if row[field] != rebuilt[field]:
                differences.append({"route": row["route"], "N": row["N"],
                                    "shift_id": row["shift_id"], "field": field,
                                    "stored": row[field],
                                    "recomputed": rebuilt[field]})
    return {
        "file": str((GATE5C / "runs.csv").relative_to(ROOT)),
        "rows_compared": compared,
        "fields_per_row": len(WALL_DETERMINISTIC),
        "identical": bool(not differences),
        "differences": differences[:12],
        "routes": sorted({row["route"] for row in stored}),
        "excluded_fields": ["total_time_seconds", "setup_seconds",
                            "recurring_seconds", "bulk_solve_seconds",
                            "schur_assembly_seconds",
                            "schur_solve_and_refinement_seconds",
                            "reconstruction_seconds",
                            "marker_evaluation_seconds", "off_marker_seconds",
                            "exterior_error_seconds"],
        "statement": ("Every deterministic field of every stored part 5c row, "
                      "the champion's among them, is reproduced character for "
                      "character with the member branch installed beside the "
                      "champion's. The wall-clock fields are properties of the "
                      "machine and are excluded."),
    }


def wall_row(cell: dict, g5c) -> dict:
    result = cell["result"]
    timings = result["timings"]
    setup = timings[g5c.SETUP_KEY]
    return {
        "route": cell["route"], "label": cell["label"], "N": cell["N"],
        "h": f"{cell['h']:.17g}", "shift_id": cell["shift_id"],
        "shift_sx": cell["shift_cell_units"][0],
        "shift_sy": cell["shift_cell_units"][1],
        "n_markers_per_obstacle": cell["n_markers_per_obstacle"],
        "n_markers_total": cell["n_markers_total"],
        "trace_error": f"{result['trace_error']:.17g}",
        "trace_error_disk": f"{cell['per_curve_trace_error']['disk']:.17g}",
        "trace_error_trefoil": f"{cell['per_curve_trace_error']['trefoil']:.17g}",
        "trace_error_kite": f"{cell['per_curve_trace_error']['kite']:.17g}",
        "exterior_relative_l2": f"{result['exterior_relative_l2']:.17g}",
        "trace_numerical_floor": f"{result['trace_numerical_floor']:.17g}",
        "exterior_relative_l2_numerical_floor":
            f"{result['exterior_relative_l2_numerical_floor']:.17g}",
        "trace_above_ten_times_floor":
            str(int(result["trace_above_ten_times_floor"])),
        "exterior_above_ten_times_floor":
            str(int(result["exterior_above_ten_times_floor"])),
        "marker_constraint_residual":
            f"{result['marker_constraint_residual']:.17g}",
        "schur_condition": f"{result['schur_condition']:.17g}",
        "weighted_adjoint_residual": f"{result['weighted_adjoint_residual']:.17g}",
        "fielded_constant_residual": f"{result['fielded_constant_residual']:.17g}",
        "invariants_pass": str(int(result["invariants_pass"])),
        "total_time_seconds": f"{result['total_time_seconds']:.6f}",
        "setup_seconds": f"{setup:.6f}",
        "recurring_seconds": f"{result['total_time_seconds'] - setup:.6f}",
        "bulk_solve_seconds": f"{timings['bulk_solve']:.6f}",
        "schur_assembly_seconds": f"{timings['schur_assembly']:.6f}",
        "schur_solve_and_refinement_seconds":
            f"{timings['schur_solve_and_refinement']:.6f}",
        "reconstruction_seconds": f"{timings['reconstruction']:.6f}",
        "marker_evaluation_seconds": f"{timings['marker_evaluation']:.6f}",
        "off_marker_seconds":
            f"{timings['off_marker_construction_and_evaluation']:.6f}",
        "exterior_error_seconds": f"{timings['exterior_error_evaluation']:.6f}",
    }


def write_wall_runs(cells: list, path: Path, g5c) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=g5c.CSV_COLUMNS)
        writer.writeheader()
        for cell in cells:
            writer.writerow(wall_row(cell, g5c))


def write_wall_markdown(payload: dict, index: dict, g5c, path: Path) -> None:
    grids = payload["grids"]
    shifts = [s[0] for s in payload["shifts"]]
    lines = ["# Gate 9 section E5, part 5c: the immersed wall", "",
             f"Produced {payload['written_utc']} by "
             "code/analysis/exponential_family/gate9_e5.py, which imports "
             "gate5c_ibm.py and the certified harness read only.", "",
             "## Worst-shift wall trace and exterior errors", "",
             "| route | metric | " + " | ".join(f"N = {n}" for n in grids) + " |",
             "| --- | --- | " + " | ".join("---" for _ in grids) + " |"]
    for route_id in payload["routes"]:
        for metric, (error_key, _) in g5c.METRICS.items():
            values = [fmt(max(index[(route_id, n, s)]["result"][error_key]
                              for s in shifts)) for n in grids]
            lines.append(f"| {cell(route_id)} | {metric} | "
                         + " | ".join(values) + " |")
    lines += ["", "## Ratios at the finest grid", "",
              "Worst shift, competitor error over member error.", "",
              "| member | competitor | wall trace | exterior |",
              "| --- | --- | --- | --- |"]
    for key, block in payload["finest_grid_ratios"].items():
        member, competitor = key.split("|", 1)
        lines.append(f"| {member} | {cell(competitor)} | "
                     f"{block['wall_trace']['worst_shift_ratio']:.3f} | "
                     f"{block['exterior_relative_l2']['worst_shift_ratio']:.3f} |")
    lines += ["", "## Solve times", "",
              "Median over the four shifts of the complete one-solve time, "
              "itself the median of three charged solves.", "",
              "| route | " + " | ".join(f"N = {n}" for n in grids) + " |",
              "| --- | " + " | ".join("---" for _ in grids) + " |"]
    for route_id in payload["routes"]:
        values = [f"{payload['times'][route_id]['total_seconds_by_N'][str(n)]:.4f}"
                  for n in grids]
        lines.append(f"| {cell(route_id)} | " + " | ".join(values) + " |")
    lines += ["", "## The contract expectation", "",
              payload["contract_check"]["expectation"], "",
              "| member | metric | worst-shift ratio at the finest grid | above "
              "100 | corner below 10 |",
              "| --- | --- | --- | --- | --- |"]
    for row in payload["contract_check"]["rows"]:
        lines.append(f"| {row['member']} | {row['metric']} | "
                     f"{row['worst_shift_ratio']:.3f} | {row['above_100']} | "
                     f"{row['corner_below_10']} |")
    cross = payload["cross_check_champion_through_the_member_branch"]
    lines += ["", "## The member branch against the champion's own branch", "",
              f"The champion run through the member branch at N = {cross['N']}, "
              f"shift {cross['shift_id']}, gives wall trace "
              f"{cross['member_branch']['trace_error']:.10e} against "
              f"{cross['champion_branch']['trace_error']:.10e} from its own "
              f"branch, a relative gap of "
              f"{cross['relative_gap_trace_error']:.2e} and an absolute gap of "
              f"{cross['absolute_gap_trace_error']:.2e}. The two branches reach "
              "the generator by different code paths, so they agree to the "
              "rounding of those paths and not bit for bit.", ""]
    lines += ["", "## Regression", "",
              f"{payload['regression']['rows_compared']} stored rows of "
              f"{payload['regression']['file']} recomputed with the member "
              f"branch installed beside the champion's, over "
              f"{payload['regression']['fields_per_row']} deterministic fields "
              f"each: identical = {payload['regression']['identical']}.", ""]
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True,
                        choices=("predict", "semilagrangian", "wall",
                                 "reduce", "timing-worker"))
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--route")
    parser.add_argument("--K", type=int)
    parser.add_argument("--shift")
    parser.add_argument("--N", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    if args.stage == "timing-worker":
        g5b.build_routes = build_routes
        row = g5b.timing_worker(args.route, args.K, args.shift, args.N, args.steps)
        print(json.dumps(row))
        return 0
    if args.stage == "predict":
        payload = stage_predict(args.jobs)
        print(json.dumps(payload["predicted_minimum_N_fine"], indent=2))
        return 0
    if args.stage == "semilagrangian":
        payload = stage_semilagrangian(args.jobs, args.skip_timing)
        print(json.dumps(payload["minimum_admissible_N_fine"], indent=2))
        print(f"regression identical {payload['regression']['identical']}")
        return 0
    if args.stage == "reduce":
        sl = json.loads((GATE9 / "e5_semilagrangian.json").read_text())
        write_5b_markdown(sl, GATE9 / "e5_semilagrangian.md")
        g5c = wall_module()
        wall = json.loads((GATE9 / "e5_wall.json").read_text())
        cells = json.loads((GATE9 / "e5_wall_cells.json").read_text())
        write_wall_markdown(wall, g5c.cell_index(cells), g5c,
                            GATE9 / "e5_wall.md")
        print("markdown receipts rewritten")
        return 0
    payload = stage_wall()
    print(json.dumps(payload["finest_grid_ratios"], indent=2, default=float)[:2000])
    print(f"regression identical {payload['regression']['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
