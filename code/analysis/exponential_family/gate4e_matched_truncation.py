"""Gate 4e: the matched truncation Taylor-Green test (operator fidelity in a DNS).

Single entry point for the frozen contract in
results/exponential_family/gate4e/contract.md. Five modes:

    --validate   rerun the stored torch pseudospectral case with the numpy port
                 (N = 64, cfl 0.30, Tend 20) and compare E(t), enstrophy(t) and
                 eps_ens(t) to code/solver/psref/tgv64.json, and the t = 9 field
                 to code/solver/psref/tgvfield_ps_N64_t9.bin. Writes
                 reference/validation.json.
    --reference  produce the same N dealiased pseudospectral fields at t* with
                 cfl 0.20 (N = 64 and 128) and the cfl 0.10 field at N = 64 for
                 the temporal check, with hashes and histories.
    --predict    P(route) from the same N pseudospectral spectrum at t* and the
                 Gate 1 symbol curves, written before any reduction.
    --reduce     every decision quantity for the retained dealias = 1 fields of
                 Gate 4c, the frozen decision rule, runs.csv and decision.json.
    --check      recompute the decision from the stored quantities in runs.csv.

Why the reference is legitimate. With the two thirds rule applied to the
nonlinear product, every route integrates the same Galerkin truncated
Navier-Stokes system on the N^3 grid: the retained modes are closed under the
dealiased product, and the exact solution of that truncated system, up to the
time stepping, is the dealiased Fourier pseudospectral solution on the same
grid. The route error against that solution is therefore the spatial operator
error alone, with the resolution floor of the truncation removed.

Conventions taken verbatim from the Gate 4c harness
(code/analysis/exponential_family/gate4c_tgv.py):

* the shell summed kinetic energy spectrum uses the forward transform divided
  by N^3 and shell membership m - 1/2 <= |k| < m + 1/2 (energy_spectrum);
* the band is 1 <= m < N/3 and the split is at N/6;
* e_full is the relative root mean square of the three component difference
  over the whole grid;
* the symbol convention is e_1 = kappa_1/theta - 1 with theta = 2 pi m / N;
* the exact time landing is floor(Tend/dt) whole steps and one final partial
  step, the rule of tgv_re1600.cpp.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = HERE.parents[2]
for path in (str(CODE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate4c_tgv as g4c  # noqa: E402
from gate4c_tgv import (  # noqa: E402
    ROUTES, TARGET_TIME, TARGET_TIME_STR, energy_spectrum, kappa1_curves,
    left_symbols, load_field, sha256,
)

GATE = ROOT / "results" / "exponential_family" / "gate4e"
GATE4C = ROOT / "results" / "exponential_family" / "gate4c"
GATE4C_FIELDS = GATE4C / "fields"
REFDIR = GATE / "reference"
SOLVER_DIR = ROOT / "code" / "solver"
PORT = SOLVER_DIR / "ps_reference_numpy.py"
PSREF = SOLVER_DIR / "psref"
PYTHON = ROOT / ".venv" / "bin" / "python"

GRIDS = (64, 128)
BASE_CFL = 0.20
TEMPORAL_DT_SCALE = 0.5
VALIDATION_CFL = 0.30
VALIDATION_TEND = 20.0
VALIDATION_TARGET = 1e-10
FIELD_TARGET = 1e-12
MECHANISM_FACTOR = 2.0
TRIUMPH_FACTOR = 5.0
PARTIAL_FACTOR = 1.5
TEMPORAL_TOLERANCE = 0.20

# The other route conditions of the frozen rule.
OTHER_ROUTES = ("opt", "bmc_p11", "cw12b")


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp with second resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reference_field_path(n: int, dt_scale: float = 1.0) -> Path:
    """Return the path of the same N pseudospectral field at t*."""
    tag = "" if dt_scale == 1.0 else f"_dt{int(round(dt_scale * 100)):03d}"
    return REFDIR / f"tgvfield_psN{n}_tstar{tag}.bin"


def reference_history_path(n: int, dt_scale: float = 1.0) -> Path:
    """Return the path of the pseudospectral history beside its field."""
    tag = "" if dt_scale == 1.0 else f"_dt{int(round(dt_scale * 100)):03d}"
    return REFDIR / f"history_psN{n}_tstar{tag}.json"


def run_port(argv: list[str], threads: int, log: Path) -> float:
    """Run the numpy port as a subprocess and return its wall time."""
    environment = dict(os.environ, OMP_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads))
    command = ["nice", "-n", "19", str(PYTHON), str(PORT)] + argv
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("w") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        code = subprocess.run(command, cwd=str(SOLVER_DIR), env=environment,
                              stdout=handle, stderr=subprocess.STDOUT,
                              text=True).returncode
    wall = time.time() - start
    if code != 0:
        raise SystemExit(f"port run failed (rc={code}); see {log}")
    return wall


# ------------------------------------------------------------------ validate


def relative_series_error(a, b) -> dict:
    """Return absolute and relative deviation statistics of two histories."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"history length mismatch {a.shape} vs {b.shape}")
    absolute = np.abs(a - b)
    scale = np.maximum(np.abs(b), np.finfo(float).tiny)
    relative = absolute / scale
    index = int(np.argmax(relative))
    return {
        "samples": int(a.size),
        "max_absolute": float(absolute.max()),
        "max_relative": float(relative.max()),
        "max_relative_index": index,
        "l2_relative": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
    }


def validate(args) -> int:
    """Reproduce the stored torch run with the numpy port and record the gap."""
    REFDIR.mkdir(parents=True, exist_ok=True)
    stored = json.loads((PSREF / "tgv64.json").read_text())
    n = int(stored["N"])
    dt = float(stored["dt"])
    # The stored field dump came from a separate invocation of the torch driver
    # at the same cfl, so its landing step is reproducible from the stored dt:
    # the driver dumps at the first step whose time reaches the requested time.
    dump_step = int(np.ceil(9.0 / dt - 1e-12))
    dump_time = dump_step * dt
    out_json = REFDIR / "validate_psN64_cfl030.json"
    log = GATE / "logs" / "validate_psN64.log"
    # The dump is written into the receipt directory, not into psref/, so the
    # stored torch field it is compared against is never overwritten.
    wall = run_port([
        "tgv", str(n), f"{VALIDATION_TEND:.10g}", "--cfl", f"{VALIDATION_CFL:.10g}",
        "--threads", str(args.threads), "--json", str(out_json),
        "--dump-field", "9", "--dump-dir", str(REFDIR),
    ], args.threads, log)
    produced = json.loads(out_json.read_text())
    out_field = REFDIR / f"tgvfield_ps_N{n}_t9.bin"

    payload = {
        "producer": "code/analysis/exponential_family/gate4e_matched_truncation.py --validate",
        "written_utc": utc_now(),
        "contract_sha256": sha256(GATE / "contract.md"),
        "port": {"path": str(PORT.relative_to(ROOT)), "sha256": sha256(PORT)},
        "stored": {"path": "code/solver/psref/tgv64.json",
                   "sha256": sha256(PSREF / "tgv64.json"),
                   "device": stored["device"], "dt": dt, "steps": stored["steps"],
                   "cfl": VALIDATION_CFL, "Tend": stored["Tend"]},
        "port_run": {"dt": produced["dt"], "steps": produced["steps"],
                     "wall_s": wall, "threads": args.threads,
                     "device": produced["device"]},
        "target_relative": VALIDATION_TARGET,
        "history": {},
        "field": {},
    }
    payload["dt_identical"] = bool(produced["dt"] == dt)
    payload["steps_identical"] = bool(produced["steps"] == stored["steps"])
    payload["time_grid_max_absolute_difference"] = float(
        np.max(np.abs(np.asarray(produced["t"]) - np.asarray(stored["t"]))))
    for key in ("E", "enstrophy", "eps_ens", "eps_dEdt", "divmax"):
        if key == "divmax":
            continue
        payload["history"][key] = relative_series_error(produced[key], stored[key])
    worst = max(payload["history"][k]["max_relative"] for k in ("E", "enstrophy", "eps_ens"))
    payload["history_worst_max_relative"] = worst
    payload["history_meets_target"] = bool(worst <= VALIDATION_TARGET)
    payload["peak_eps_ens"] = {
        "stored": stored["peak_eps_ens"], "port": produced["peak_eps_ens"],
        "relative_difference": abs(produced["peak_eps_ens"] - stored["peak_eps_ens"])
        / stored["peak_eps_ens"],
        "stored_t_peak": stored["t_peak"], "port_t_peak": produced["t_peak"],
    }

    stored_field = PSREF / "tgvfield_ps_N64_t9.bin"
    produced_field = out_field
    a = load_field(produced_field, n)
    b = load_field(stored_field, n)
    numerator = np.sqrt(np.mean(sum((x - y) ** 2 for x, y in zip(a, b, strict=True))))
    denominator = np.sqrt(np.mean(sum(y**2 for y in b)))
    payload["field"] = {
        "stored_path": "code/solver/psref/tgvfield_ps_N64_t9.bin",
        "stored_sha256": sha256(stored_field),
        "port_path": str(produced_field.relative_to(ROOT)),
        "port_sha256": sha256(produced_field),
        "dump_rule": ("the torch driver dumps at the first step whose time "
                      "reaches the requested time"),
        "dump_step_from_stored_dt": dump_step,
        "dump_time_from_stored_dt": dump_time,
        "dump_step_realized": produced["field_dumps"][0]["step"],
        "dump_time_realized": produced["field_dumps"][0]["t_actual"],
        "dump_time_reproducible": bool(produced["field_dumps"][0]["step"] == dump_step),
        "relative_l2_error": float(numerator / denominator),
        "max_absolute_difference": float(max(np.abs(x - y).max()
                                             for x, y in zip(a, b, strict=True))),
        "target_relative": FIELD_TARGET,
        "meets_target": bool(float(numerator / denominator) <= FIELD_TARGET),
    }
    (REFDIR / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "dt_identical", "steps_identical", "history_worst_max_relative",
        "history_meets_target")}, indent=2))
    print(json.dumps(payload["field"], indent=2))
    print(f"wrote {(REFDIR / 'validation.json').relative_to(ROOT)}")
    return 0


# ----------------------------------------------------------------- reference


def check_common_ic(n: int) -> dict:
    """Compare the Gate 4c common initial field to the analytic Taylor-Green one.

    The Taylor-Green field u = sin x cos y cos z, v = -cos x sin y cos z, w = 0
    has only the eight modes with |k_a| = 1 in every component, so it lies in
    the grid's Fourier space exactly and is already divergence free. The Leray
    projection applied when the common field was built is therefore the
    identity up to round off, which this check confirms.
    """
    path = GATE4C / "ic" / f"ic_tgv_N{n}.bin"
    if not path.exists():
        return {"present": False, "path": str(path.relative_to(ROOT))}
    u, v, w = load_field(path, n)
    h = 2.0 * np.pi / n
    axis = np.arange(n) * h
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    ua = np.sin(x) * np.cos(y) * np.cos(z)
    va = -np.cos(x) * np.sin(y) * np.cos(z)
    wa = np.zeros_like(ua)
    scale = float(np.sqrt(np.mean(ua**2 + va**2 + wa**2)))
    difference = float(np.sqrt(np.mean((u - ua) ** 2 + (v - va) ** 2 + (w - wa) ** 2)))
    return {
        "present": True,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "max_absolute_difference": float(max(np.abs(u - ua).max(),
                                             np.abs(v - va).max(),
                                             np.abs(w - wa).max())),
        "relative_l2_difference": difference / scale,
        "analytic_to_roundoff": bool(difference / scale < 1e-14),
        "E0_file": 0.5 * float(np.mean(u * u + v * v + w * w)),
        "E0_analytic": 0.5 * float(np.mean(ua * ua + va * va + wa * wa)),
    }


def make_reference(n: int, dt_scale: float, threads: int) -> dict:
    """Produce one same N pseudospectral field landed on t*."""
    REFDIR.mkdir(parents=True, exist_ok=True)
    cfl = BASE_CFL * dt_scale
    field = reference_field_path(n, dt_scale)
    history = reference_history_path(n, dt_scale)
    ic = GATE4C / "ic" / f"ic_tgv_N{n}.bin"
    tag = "" if dt_scale == 1.0 else f"_dt{int(round(dt_scale * 100)):03d}"
    log = GATE / "logs" / f"reference_psN{n}{tag}.log"
    argv = ["tgv", str(n), "--cfl", f"{cfl:.10g}",
            "--exact-time", TARGET_TIME_STR, "--threads", str(threads),
            "--dump", str(field), "--json", str(history)]
    if ic.exists():
        argv += ["--load-ic", str(ic)]
    wall = run_port(argv, threads, log)
    payload = json.loads(history.read_text())
    landing = payload["field_dumps"][-1]["t_actual"]
    return {
        "N": n, "cfl": cfl, "dt_scale": dt_scale, "dt": payload["dt"],
        "whole_steps": payload["whole_steps"], "dt_last": payload["dt_last"],
        "steps": payload["steps"], "landing_time": landing,
        "landing_abs_difference": abs(landing - TARGET_TIME),
        "field": str(field.relative_to(ROOT)), "field_sha256": sha256(field),
        "field_bytes": field.stat().st_size,
        "history": str(history.relative_to(ROOT)),
        "history_sha256": sha256(history),
        "initial_condition": (str(ic.relative_to(ROOT)) if ic.exists()
                              else "analytic"),
        "wall_s": wall, "threads": threads,
        "E_final": payload["E"][-1], "enstrophy_final": payload["enstrophy"][-1],
        "eps_ens_final": payload["eps_ens"][-1], "divmax_max": max(payload["divmax"]),
        "command": " ".join(["nice", "-n", "19", str(PYTHON.relative_to(ROOT)),
                             str(PORT.relative_to(ROOT))] + argv),
    }


def reference(args) -> int:
    """Produce the requested same N pseudospectral references at t*."""
    REFDIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REFDIR / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else
                {"producer": "code/analysis/exponential_family/"
                             "gate4e_matched_truncation.py --reference",
                 "target_time": TARGET_TIME, "base_cfl": BASE_CFL,
                 "port": str(PORT.relative_to(ROOT)), "entries": {}})
    manifest["port_sha256"] = sha256(PORT)
    manifest["written_utc"] = utc_now()
    manifest.setdefault("common_initial_condition", {})
    requested = []
    if args.N in ("64", "all"):
        requested.append((64, 1.0))
        requested.append((64, TEMPORAL_DT_SCALE))
    if args.N in ("128", "all"):
        requested.append((128, 1.0))
    for n, dt_scale in requested:
        key = f"N{n}_dt{int(round(dt_scale * 100)):03d}"
        if reference_field_path(n, dt_scale).exists() and not args.force:
            print(f"{key}: field already present, skipping (use --force to redo)")
            continue
        manifest["common_initial_condition"][str(n)] = check_common_ic(n)
        entry = make_reference(n, dt_scale, args.threads)
        manifest["entries"][key] = entry
        print(f"{key}: landed t={entry['landing_time']!r} "
              f"(|dt| {entry['landing_abs_difference']:.3e}) "
              f"steps={entry['steps']} wall={entry['wall_s']:.1f}s "
              f"sha256={entry['field_sha256'][:16]}")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for n in GRIDS:
        if str(n) not in manifest["common_initial_condition"]:
            manifest["common_initial_condition"][str(n)] = check_common_ic(n)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    return 0


# ------------------------------------------------------------------- predict


def reference_bundle(n: int, dt_scale: float = 1.0):
    """Return the same N pseudospectral fields and spectrum at t*."""
    path = reference_field_path(n, dt_scale)
    if not path.exists():
        raise SystemExit(f"missing pseudospectral reference {path}; run --reference")
    fields = load_field(path, n)
    return fields, energy_spectrum(fields, n)


def predict(args) -> int:
    """Write the predeclared spectrum weighted phase error from the same N spectrum."""
    GATE.mkdir(parents=True, exist_ok=True)
    curves = kappa1_curves()
    lefts = left_symbols()
    prior = json.loads((GATE4C / "prediction.json").read_text())
    payload = {
        "producer": "code/analysis/exponential_family/gate4e_matched_truncation.py --predict",
        "written_utc": utc_now(),
        "contract": {"path": "results/exponential_family/gate4e/contract.md",
                     "sha256": sha256(GATE / "contract.md")},
        "definition": ("P(route) = sqrt( sum_m E(m) |e_1(theta_m)|^2 / sum_m E(m) ) "
                       "over shells 1 <= m < N/3 with theta_m = 2 pi m / N and "
                       "e_1 = kappa_1/theta - 1, weighted by the SAME N dealiased "
                       "pseudospectral energy spectrum at t*"),
        "prior": {"path": "results/exponential_family/gate4c/prediction.json",
                  "sha256": sha256(GATE4C / "prediction.json"),
                  "reference": "384^3 restricted spectrum",
                  "ratio_sl_over_exp": {n: prior["grids"][n]["ratio_sl_over_exp"]
                                        for n in prior["grids"]}},
        "member": prior["member"],
        "mechanism_agreement_factor": MECHANISM_FACTOR,
        "grids": {},
    }
    for n in GRIDS:
        path = reference_field_path(n)
        if not path.exists():
            print(f"N = {n}: pseudospectral reference absent, skipping")
            continue
        _, spectrum = reference_bundle(n)
        shells = np.arange(1, n // 3)
        weights = spectrum[shells]
        theta = 2.0 * np.pi * shells / n
        entry = {
            "reference_field": str(path.relative_to(ROOT)),
            "reference_sha256": sha256(path),
            "shells": [int(shells[0]), int(shells[-1])],
            "theta_range": [float(theta[0]), float(theta[-1])],
            "reference_band_energy": float(weights.sum()),
            "reference_spectrum_band": [float(v) for v in weights],
            "P": {}, "left_symbol_band_range": {},
        }
        for label, _, _, _ in ROUTES:
            kappa1 = np.asarray(curves[label](theta), dtype=float)
            e1 = kappa1 / theta - 1.0
            entry["P"][label] = float(np.sqrt(np.sum(weights * e1**2) / np.sum(weights)))
            left = np.asarray(lefts[label](theta), dtype=float)
            entry["left_symbol_band_range"][label] = [float(left.min()), float(left.max())]
        entry["predicted_ordering_best_first"] = sorted(entry["P"],
                                                        key=lambda k: entry["P"][k])
        entry["ratio_sl_over_exp"] = entry["P"]["sl"] / entry["P"]["exp"]
        for other in OTHER_ROUTES:
            entry[f"ratio_{other}_over_exp"] = entry["P"][other] / entry["P"]["exp"]
        payload["grids"][str(n)] = entry
    out = GATE / "prediction.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for n, entry in payload["grids"].items():
        print(f"N = {n}: shells {entry['shells'][0]}..{entry['shells'][1]}")
        for label in entry["predicted_ordering_best_first"]:
            print(f"    {label:10s} P = {entry['P'][label]:.6e}")
        print(f"    predicted P(sl)/P(exp) = {entry['ratio_sl_over_exp']:.4f} "
              f"(prior with the 384^3 spectrum "
              f"{payload['prior']['ratio_sl_over_exp'][n]:.4f})")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


# -------------------------------------------------------------------- reduce


def field_name(label: str, n: int, dt_scale: float) -> str:
    """Return the Gate 4c dump file name of one dealias = 1 matrix cell."""
    route, degree = next((r, p) for lbl, r, p, _ in ROUTES if lbl == label)
    tag = "" if dt_scale == 1.0 else f"_dt{int(round(dt_scale * 100)):03d}"
    return f"tgvfield_{route}_cic_da23{tag}_p{degree}_N{n}.bin"


# Gate 4c's own reduction retains only the route exp fields and deletes the
# rest, so a Gate 4e input may survive only as a byte identical copy elsewhere.
# Each candidate directory is searched in order and the file that is used has
# its SHA-256 checked against the digest Gate 4c recorded for that matrix cell
# in process_registry.csv, so a substitute is accepted only when it is the same
# bytes as the run the contract names.
FIELD_SOURCES = (
    GATE4C_FIELDS,
    GATE4C / "timing_work" / "data",
    GATE / "rerun" / "data",
)


def registry_hashes() -> dict:
    """Return the Gate 4c recorded field digest of every matrix cell by key."""
    registry = GATE4C / "process_registry.csv"
    if not registry.exists():
        return {}
    with registry.open(newline="") as handle:
        return {row["key"]: row["field_sha256"] for row in csv.DictReader(handle)
                if row.get("field_sha256")}


def cell_key(label: str, n: int, dt_scale: float) -> str:
    """Return the Gate 4c registry key of one dealias = 1 matrix cell."""
    return f"{label}_N{n}_da1_dt{int(round(dt_scale * 100)):03d}"


def gate4c_field(label: str, n: int, dt_scale: float, hashes: dict | None = None):
    """Return an existing byte verified field for one cell, or None.

    The digest recorded by Gate 4c is authoritative: a file found outside the
    Gate 4c fields directory is returned only when its digest equals it.
    """
    name = field_name(label, n, dt_scale)
    expected = (hashes or {}).get(cell_key(label, n, dt_scale))
    for index, directory in enumerate(FIELD_SOURCES):
        candidate = directory / name
        if not candidate.exists():
            continue
        if index == 0 and not expected:
            return candidate
        digest = sha256(candidate)
        if expected and digest != expected:
            continue
        return candidate
    return None


def reduce_cell(label: str, n: int, dt_scale: float, bundle, path: Path) -> dict:
    """Compute every decision quantity for one route field against the reference."""
    reference_fields, reference_spectrum = bundle
    fields = load_field(path, n)
    spectrum = energy_spectrum(fields, n)
    band = slice(1, n // 3)
    mid = slice(1, n // 6)
    outer = slice(n // 6, n // 3)

    def spectrum_error(sl):
        return float(np.sqrt(np.sum((spectrum[sl] - reference_spectrum[sl]) ** 2)
                             / np.sum(reference_spectrum[sl] ** 2)))

    numerator = np.sqrt(np.mean(sum((a - b) ** 2 for a, b
                                    in zip(fields, reference_fields, strict=True))))
    denominator = np.sqrt(np.mean(sum(b**2 for b in reference_fields)))
    shells = np.arange(1, n // 3)
    return {
        "key": cell_key(label, n, dt_scale),
        "label": label, "N": n, "dealias": 1, "dt_scale": dt_scale,
        "cfl": BASE_CFL * dt_scale,
        "field": str(path.relative_to(ROOT)), "field_sha256": sha256(path),
        "e_full": float(numerator / denominator),
        "e_band": spectrum_error(band),
        "e_mid": spectrum_error(mid),
        "e_outer": spectrum_error(outer),
        "shells": [int(s) for s in shells],
        "shell_spectrum": [float(v) for v in spectrum[band]],
        "shell_reference_spectrum": [float(v) for v in reference_spectrum[band]],
        "shell_relative_error": [float((spectrum[band][i] - reference_spectrum[band][i])
                                       / reference_spectrum[band][i])
                                 for i in range(shells.size)],
    }


def pick(reduced, label, n, dt_scale=1.0):
    """Return the reduced row of one matrix cell."""
    for row in reduced:
        if (row["label"] == label and row["N"] == n
                and abs(row["dt_scale"] - dt_scale) < 1e-12):
            return row
    raise KeyError(f"no reduced row for {label} N={n} dt_scale={dt_scale}")


def has(reduced, label, n, dt_scale=1.0) -> bool:
    """Report whether one matrix cell was reduced."""
    try:
        pick(reduced, label, n, dt_scale)
        return True
    except KeyError:
        return False


def decide(reduced: list[dict], prediction: dict, grids) -> dict:
    """Apply the frozen Gate 4e decision rule verbatim."""
    tests = {}
    e_full = {}
    for n in GRIDS:
        e_full[f"N{n}"] = {row["label"]: row["e_full"]
                           for row in reduced if row["N"] == n and row["dt_scale"] == 1.0}
    for n in grids:
        exp = pick(reduced, "exp", n)["e_full"]
        sl = pick(reduced, "sl", n)["e_full"]
        tests[f"N{n}_exp_at_most_sl_over_5"] = bool(exp <= sl / TRIUMPH_FACTOR)
        tests[f"N{n}_exp_at_most_sl_over_1.5"] = bool(exp <= sl / PARTIAL_FACTOR)
        others = {other: pick(reduced, other, n)["e_full"] for other in OTHER_ROUTES}
        tests[f"N{n}_exp_at_most_min_other"] = bool(exp <= min(others.values()))
        for other, value in others.items():
            tests[f"N{n}_exp_at_most_{other}"] = bool(exp <= value)

    # Decision quantity 5, the ratio R, and the mechanism comparison need only
    # exp and sl, so they are reported wherever that pair exists even when the
    # other route condition cannot be evaluated on that grid.
    ratio_grids = [n for n in GRIDS if has(reduced, "exp", n) and has(reduced, "sl", n)]
    ratios = {f"N{n}": pick(reduced, "sl", n)["e_full"] / pick(reduced, "exp", n)["e_full"]
              for n in ratio_grids}
    base = ratios.get("N64")
    half_row_exp = None
    try:
        if base is None:
            raise KeyError("N = 64 is not among the decided grids")
        half = (pick(reduced, "sl", 64, TEMPORAL_DT_SCALE)["e_full"]
                / pick(reduced, "exp", 64, TEMPORAL_DT_SCALE)["e_full"])
        half_row_exp = pick(reduced, "exp", 64, TEMPORAL_DT_SCALE)["e_full"]
        temporal_change = abs(half - base) / base
        temporal_pass = bool(temporal_change < TEMPORAL_TOLERANCE)
    except KeyError:
        half, temporal_change, temporal_pass = None, None, False
    temporal = {
        "definition": ("R = e_full(sl)/e_full(exp) at N = 64 recomputed at dt "
                       "scale 0.5 against the cfl 0.10 pseudospectral reference; "
                       "the relative change must be below 20 percent"),
        "R_dt_scale_1.0": base,
        "R_dt_scale_0.5": half,
        "relative_change": temporal_change,
        "tolerance": TEMPORAL_TOLERANCE,
        "e_full_exp_dt_scale_0.5": half_row_exp,
        "pass": temporal_pass,
    }

    triumph_core = all(tests[f"N{n}_exp_at_most_sl_over_5"]
                       and tests[f"N{n}_exp_at_most_min_other"] for n in grids)
    partial_core = all(tests[f"N{n}_exp_at_most_sl_over_1.5"] for n in grids)
    # The frozen rule is quantified over BOTH N = 64 and N = 128. With either
    # grid absent no branch of it can be evaluated, so no outcome is recorded.
    complete = all(n in grids for n in GRIDS)
    if not complete:
        outcome = "UNDECIDABLE_INPUTS_MISSING"
    elif triumph_core and temporal_pass:
        outcome = "TRIUMPH"
    elif partial_core:
        outcome = "PARTIAL"
    else:
        outcome = "REJECT"

    mechanism = {}
    for n in ratio_grids:
        predicted = prediction["grids"].get(str(n), {}).get("ratio_sl_over_exp")
        realized = ratios[f"N{n}"]
        if predicted is None:
            mechanism[f"N{n}"] = {"predicted_P_sl_over_P_exp": None,
                                  "realized_e_full_sl_over_exp": realized,
                                  "realized_over_predicted": None,
                                  "within_factor_two": False}
            continue
        quotient = realized / predicted
        mechanism[f"N{n}"] = {
            "predicted_P_sl_over_P_exp": predicted,
            "realized_e_full_sl_over_exp": realized,
            "realized_over_predicted": quotient,
            "within_factor_two": bool((1.0 / MECHANISM_FACTOR) <= quotient
                                      <= MECHANISM_FACTOR),
        }
    evaluated = [v for k, v in mechanism.items() if k.startswith("N")]
    mechanism["grids_evaluated"] = [int(n) for n in ratio_grids]
    mechanism["confirmed"] = (bool(all(v["within_factor_two"] for v in evaluated))
                              if evaluated else None)
    if mechanism["confirmed"] is None:
        mechanism["status"] = "not evaluated: no grid has both exp and sl"
    elif not mechanism["confirmed"]:
        mechanism["status"] = ("unexplained: the shell resolved error is attached "
                               "in shell_errors.json, no claim is made")
    else:
        mechanism["status"] = "confirmed"

    return {
        "producer": "code/analysis/exponential_family/gate4e_matched_truncation.py --reduce",
        "written_utc": utc_now(),
        "contract_sha256": sha256(GATE / "contract.md"),
        "grids_decided": [int(n) for n in grids],
        "rule": ("TRIUMPH if at both N = 64 and N = 128: e_full(exp) <= "
                 "e_full(sl)/5 and e_full(exp) <= min(e_full(opt), "
                 "e_full(bmc p11), e_full(cw12b)), and the N = 64 ratio changes "
                 "by less than 20 percent under dt scale 0.5. PARTIAL if "
                 "e_full(exp) <= e_full(sl)/1.5 at both N but the 5x or the "
                 "other route conditions fail. REJECT otherwise. Mechanism "
                 "confirmed if the realized ratio is within a factor of two of "
                 "the predicted ratio."),
        "outcome": outcome,
        "rule_inputs_complete": bool(complete),
        "tests": tests,
        "triumph_core_conditions": bool(triumph_core),
        "partial_core_conditions": bool(partial_core),
        "ratio_sl_over_exp": ratios,
        "temporal_check": temporal,
        "mechanism": mechanism,
        "mechanism_confirmed": mechanism["confirmed"],
        "e_full": e_full,
    }


def reduce(args) -> int:
    """Reduce every retained dealias = 1 field, apply the frozen rule, write receipts."""
    if not (GATE / "prediction.json").exists():
        raise SystemExit("prediction.json must exist before --reduce")
    prediction = json.loads((GATE / "prediction.json").read_text())

    hashes = registry_hashes()
    cells = [(label, n, 1.0) for n in GRIDS for label, _, _, _ in ROUTES]
    cells += [(label, 64, TEMPORAL_DT_SCALE) for label in ("exp", "sl")]
    found, missing = {}, []
    for label, n, dt_scale in cells:
        path = gate4c_field(label, n, dt_scale, hashes)
        if path is None:
            missing.append(cell_key(label, n, dt_scale))
        else:
            found[(label, n, dt_scale)] = path
    for n in GRIDS:
        if not reference_field_path(n).exists():
            missing.append(f"pseudospectral_reference_N{n}")
    if not reference_field_path(64, TEMPORAL_DT_SCALE).exists():
        missing.append("pseudospectral_reference_N64_dt050")
    if missing:
        print("MISSING INPUTS (Gate 4c retains only the route exp fields; the "
              "rest were deleted by its own reduction):")
        for item in missing:
            print(f"    {item}")

    # A grid can be decided only when every route the frozen rule names is
    # present: exp and sl for the ratio, and opt, bmc p11, cw12b for the
    # other route condition.
    required = ("exp", "sl") + OTHER_ROUTES
    decidable = [n for n in GRIDS
                 if reference_field_path(n).exists()
                 and all((label, n, 1.0) in found for label in required)]

    reduced = []
    for (label, n, dt_scale), path in sorted(
            found.items(), key=lambda kv: (kv[0][1], -kv[0][2], kv[0][0])):
        bundle = reference_bundle(n, dt_scale)
        row = reduce_cell(label, n, dt_scale, bundle, path)
        row["field_source"] = str(path.parent.relative_to(ROOT))
        row["field_matches_gate4c_registry"] = bool(
            hashes.get(cell_key(label, n, dt_scale)) == row["field_sha256"])
        reduced.append(row)
        print(f"{row['key']:26s} e_full {row['e_full']:.8e}  "
              f"e_band {row['e_band']:.8e}  e_mid {row['e_mid']:.8e}  "
              f"e_outer {row['e_outer']:.8e}  "
              f"registry_match {row['field_matches_gate4c_registry']}", flush=True)
    if not reduced:
        raise SystemExit("no usable route field was found")

    manifest = json.loads((REFDIR / "manifest.json").read_text())
    fields = ("key", "label", "N", "dealias", "dt_scale", "cfl",
              "e_full", "e_band", "e_mid", "e_outer",
              "predicted_P", "predicted_P_over_exp",
              "reference_field", "reference_sha256", "field", "field_sha256",
              "field_source", "field_matches_gate4c_registry")
    with (GATE / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(reduced, key=lambda r: (r["N"], -r["dt_scale"], r["label"])):
            record = {k: row.get(k, "") for k in fields}
            grid = prediction["grids"].get(str(row["N"]), {})
            value = grid.get("P", {}).get(row["label"])
            record["predicted_P"] = value if value is not None else ""
            record["predicted_P_over_exp"] = (
                value / grid["P"]["exp"] if value is not None else "")
            reference_path = reference_field_path(row["N"], row["dt_scale"])
            record["reference_field"] = str(reference_path.relative_to(ROOT))
            record["reference_sha256"] = sha256(reference_path)
            writer.writerow(record)

    shell_payload = {
        "producer": "code/analysis/exponential_family/gate4e_matched_truncation.py --reduce",
        "written_utc": utc_now(),
        "note": ("Shell resolved kinetic energy spectrum error against the same N "
                 "dealiased pseudospectral solution at t*, with the mid and outer "
                 "split at N/6."),
        "cells": {row["key"]: {k: row[k] for k in (
            "shells", "shell_spectrum", "shell_reference_spectrum",
            "shell_relative_error", "e_band", "e_mid", "e_outer", "e_full")}
            for row in reduced},
    }
    (GATE / "shell_errors.json").write_text(json.dumps(shell_payload, indent=2) + "\n")

    decision = decide(reduced, prediction, decidable)
    decision["reference_manifest"] = {k: {"field": v["field"],
                                          "field_sha256": v["field_sha256"],
                                          "landing_time": v["landing_time"],
                                          "cfl": v["cfl"]}
                                      for k, v in manifest["entries"].items()}
    decision["missing_inputs"] = missing
    decision["mid_outer_split"] = {
        row["key"]: {"e_mid": row["e_mid"], "e_outer": row["e_outer"],
                     "outer_over_mid": (row["e_outer"] / row["e_mid"]
                                        if row["e_mid"] > 0 else None)}
        for row in reduced}
    (GATE / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"outcome": decision["outcome"],
                      "grids_decided": decision["grids_decided"],
                      "ratio_sl_over_exp": decision["ratio_sl_over_exp"],
                      "temporal_check": decision["temporal_check"],
                      "mechanism_confirmed": decision["mechanism_confirmed"]}, indent=2))
    print(f"wrote {(GATE / 'runs.csv').relative_to(ROOT)}, "
          f"{(GATE / 'decision.json').relative_to(ROOT)} and "
          f"{(GATE / 'shell_errors.json').relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------- check


def check(args) -> int:
    """Recompute the decision from the stored quantities in runs.csv."""
    with (GATE / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reduced = [{
        "key": row["key"], "label": row["label"], "N": int(row["N"]),
        "dealias": int(row["dealias"]), "dt_scale": float(row["dt_scale"]),
        "e_full": float(row["e_full"]), "e_band": float(row["e_band"]),
        "e_mid": float(row["e_mid"]), "e_outer": float(row["e_outer"]),
    } for row in rows]
    stored = json.loads((GATE / "decision.json").read_text())
    prediction = json.loads((GATE / "prediction.json").read_text())
    grids = tuple(stored["grids_decided"])
    recomputed = decide(reduced, prediction, grids)
    agree = {
        "outcome": recomputed["outcome"] == stored["outcome"],
        "tests": recomputed["tests"] == stored["tests"],
        "temporal_pass": (recomputed["temporal_check"]["pass"]
                          == stored["temporal_check"]["pass"]),
        "mechanism_confirmed": (recomputed["mechanism_confirmed"]
                                == stored["mechanism_confirmed"]),
    }
    print(json.dumps({"stored_outcome": stored["outcome"],
                      "recomputed_outcome": recomputed["outcome"],
                      "agreement": agree}, indent=2))
    return 0 if all(agree.values()) else 1


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    """Dispatch the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--reference", action="store_true")
    group.add_argument("--predict", action="store_true")
    group.add_argument("--reduce", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--N", choices=("64", "128", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    (GATE / "logs").mkdir(parents=True, exist_ok=True)
    if args.validate:
        return validate(args)
    if args.reference:
        return reference(args)
    if args.predict:
        return predict(args)
    if args.reduce:
        return reduce(args)
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
