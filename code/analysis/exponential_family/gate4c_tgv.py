"""Gate 4c: the Taylor-Green vortex at Re = 1600 with the Gate 1 champion.

Single entry point for the frozen contract in
results/exponential_family/gate4c/contract.md. Four modes:

    --predict   spectrum weighted linearized phase error P(route) for every
                route at N = 64 and 128, from the 384^3 reference spectrum at
                t* and the Gate 1 symbol curves. Written before any run.
    --run       the contract matrix of solver runs, plus the temporal checks
                and the wall time replicates.
    --reduce    every decision quantity per run, the frozen decision rule, and
                the predeclared diagnosis when the outcome is not TRIUMPH.
    --check     recompute the decision from the stored quantities.

Conventions taken verbatim from the existing harness:

* resolved band error e_band is the certification code path of
  code/analysis/certify_n128_exact_time_frontier.py: the 384^3 reference is
  restricted to the signed N mode cube, the shell summed kinetic energy
  spectrum uses the forward transform divided by N^3 and shell membership
  m - 1/2 <= |k| < m + 1/2, and
      e_band = sqrt( sum_band (S - S_ref)^2 / sum_band S_ref^2 )
  over shells 1 <= m < N/3.
* the exact time output rule is the one of tgv_probe.cpp: floor(Tend/dt) whole
  steps of length dt and one final partial step of length Tend - nst dt, so the
  dumped field sits at t* to round off.
* the common initial condition is the route agnostic field of the "cic"
  variant: the analytic Taylor-Green velocity sampled on the grid and passed
  through an exact Fourier Leray projection that uses no scheme's operator, fed
  to every route with --load-ic so the t = 0 field is byte identical.
* the symbol convention is that of the Gate 0 metrics: the first derivative
  symbol is (i/h) kappa_1(theta) and the relative error is
  e_1 = kappa_1/theta - 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
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

from comparators import (  # noqa: E402
    CW12B_D1, SLC_D1, Q8_D1, compact_kappa1, compact_left,
)
from family import ExponentialGenerator  # noqa: E402

GATE = ROOT / "results" / "exponential_family" / "gate4c"
MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
CHAMPIONS = ROOT / "results" / "exponential_family" / "gate1" / "champions.json"
SOLVER_DIR = ROOT / "code" / "solver"
SOLVER = SOLVER_DIR / "tgv_re1600"
PSREF = SOLVER_DIR / "psref"
REFERENCE = PSREF / "tgvfield_ps_N384_t9.bin"
REFERENCE_SHA256 = "1b1ec72d30358050e240a7b0687affadef1f7abf6da3583e0948f865fa92be77"
GDIAG = PSREF / "spectral_Re1600_512.gdiag"

WORK = GATE / "work"
WORK_DATA = WORK / "data"
TIMING_DIR = GATE / "timing_work"
IC_DIR = GATE / "ic"
FIELDS = GATE / "fields"
HISTORIES = GATE / "epsilon_histories"
LOGS = GATE / "logs"
CACHE = GATE / "cache"

TARGET_TIME = 9.00262644794325
TARGET_TIME_STR = "9.00262644794325"
REYNOLDS = 1600
REFERENCE_N = 384
BASE_CFL = 0.20
GRIDS = (64, 128)
DEALIAS_MODES = (0, 1)
MAX_CONCURRENT = 5
THREADS = 4
TIMING_REPLICATES = 3
MECHANISM_FACTOR = 2.0

# label, solver route token, positional degree, member file needed
ROUTES = (
    ("exp", "exp", 9, True),
    ("sl", "sl", 5, False),
    ("opt", "opt", 5, False),
    ("bmc_p9", "bmc", 9, False),
    ("bmc_p11", "bmc", 11, False),
    ("c10", "c10", 5, False),
    ("c12", "c12", 5, False),
    ("cw12b", "cw12b", 5, False),
)
TIMING_LABELS = ("exp", "sl", "opt", "bmc_p9")
TEMPORAL_LABELS = ("exp", "sl")
TEMPORAL_DT_SCALE = 0.5

# Maximum order tridiagonal Pade pairs, the literal constants of
# code/solver/bm_apply.h. D1 left side is 1 + 2 alpha cos(theta) and the right
# side is antisymmetric with the coefficients a_s.
C10_D1 = {"alpha": (2.0 / 5.0,),
          "a": (39.0 / 50.0, 1.0 / 15.0, -1.0 / 210.0, 1.0 / 4200.0)}
C12_D1 = {"alpha": (5.0 / 12.0,),
          "a": (7.0 / 9.0, 5.0 / 63.0, -5.0 / 672.0, 1.0 / 1512.0, -1.0 / 30240.0)}

# Gate 6 per step cost model: the number of cyclic band solves needed for both
# derivatives and the half bandwidth of the left side that is solved.
COST_MODEL = {
    "exp": {"solves": 1, "half_bandwidth": 4},
    "bmc_p9": {"solves": 1, "half_bandwidth": 4},
    "bmc_p11": {"solves": 1, "half_bandwidth": 5},
    "sl": {"solves": 2, "half_bandwidth": 2},
    "opt": {"solves": 2, "half_bandwidth": 1},
    "c10": {"solves": 2, "half_bandwidth": 1},
    "c12": {"solves": 2, "half_bandwidth": 1},
    "cw12b": {"solves": 2, "half_bandwidth": 2},
}


# ----------------------------------------------------------------- utilities


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp with second resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    """Compute a file SHA-256 digest in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def champion_omegas() -> list[float]:
    """Return the Gate 1 champion frequencies at band B0 and budget C_SLC1."""
    payload = json.loads(CHAMPIONS.read_text())
    return [float(w) for w in payload["champions"]["B0"]["C_SLC1"]["champion"]["omegas"]]


# --------------------------------------------------------- reference spectra


def load_field(path: Path, n: int):
    """Load three contiguous binary64 velocity components of an n^3 field."""
    raw = np.fromfile(path, dtype=np.float64)
    expected = 3 * n**3
    if raw.size != expected:
        raise ValueError(f"{path}: found {raw.size} values, expected {expected}")
    block = n**3
    return tuple(raw[i * block:(i + 1) * block].reshape(n, n, n) for i in range(3))


def restrict_reference(n: int):
    """Restrict the N=384 Fourier field to the signed n mode cube."""
    source = np.fromfile(REFERENCE, dtype=np.float64)
    expected = 3 * REFERENCE_N**3
    if source.size != expected:
        raise ValueError(f"reference contains {source.size} values, expected {expected}")
    coarse_modes = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    selector = np.ix_(coarse_modes, coarse_modes, coarse_modes)
    restricted = []
    block = REFERENCE_N**3
    for component in range(3):
        field = source[component * block:(component + 1) * block]
        field = field.reshape(REFERENCE_N, REFERENCE_N, REFERENCE_N)
        coefficients = np.fft.fftn(field) / REFERENCE_N**3
        restricted.append(np.real(np.fft.ifftn(coefficients[selector] * n**3)))
    return tuple(restricted)


def energy_spectrum(fields, n: int) -> np.ndarray:
    """Form the shell summed kinetic energy spectrum with campaign normalization."""
    modes = np.fft.fftfreq(n, d=1.0 / n)
    kz, ky, kx = np.meshgrid(modes, modes, modes, indexing="ij")
    magnitude = np.sqrt(kx**2 + ky**2 + kz**2)
    density = sum(np.abs(np.fft.fftn(field) / n**3) ** 2 for field in fields) * 0.5
    spectrum = np.zeros(n // 2)
    for shell in range(n // 2):
        mask = (magnitude >= shell - 0.5) & (magnitude < shell + 0.5)
        spectrum[shell] = float(density[mask].sum())
    return spectrum


def reference_bundle(n: int):
    """Return the restricted reference fields and spectrum at grid n, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    field_cache = CACHE / f"ref_restricted_N{n}.npy"
    spectrum_cache = CACHE / f"ref_spectrum_N{n}.npy"
    if field_cache.exists() and spectrum_cache.exists():
        fields = tuple(np.load(field_cache))
        return fields, np.load(spectrum_cache)
    fields = restrict_reference(n)
    spectrum = energy_spectrum(fields, n)
    np.save(field_cache, np.asarray(fields))
    np.save(spectrum_cache, spectrum)
    return fields, spectrum


# ------------------------------------------------------------ symbol curves


def kappa1_curves() -> dict:
    """Return kappa_1(theta) for every route of the contract."""
    omegas = champion_omegas()
    exp_generator = ExponentialGenerator(6, omegas)
    curves = {
        "exp": lambda theta: exp_generator.symbols(np.asarray(theta, float))[1],
        "sl": compact_kappa1(SLC_D1["alpha"], SLC_D1["a"]),
        "opt": compact_kappa1(Q8_D1["alpha"], Q8_D1["a"]),
        "cw12b": compact_kappa1(CW12B_D1["alpha"], CW12B_D1["a"]),
        "c10": compact_kappa1(C10_D1["alpha"], C10_D1["a"]),
        "c12": compact_kappa1(C12_D1["alpha"], C12_D1["a"]),
    }
    for label, degree in (("bmc_p9", 9), ("bmc_p11", 11)):
        generator = ExponentialGenerator(degree + 1, ())
        curves[label] = (lambda g: lambda theta:
                         g.symbols(np.asarray(theta, float))[1])(generator)
    return curves


def left_symbols() -> dict:
    """Return the left side symbol L(theta) for every route of the contract."""
    omegas = champion_omegas()
    exp_generator = ExponentialGenerator(6, omegas)
    symbols = {
        "exp": lambda theta: exp_generator.symbols(np.asarray(theta, float))[0],
        "sl": compact_left(SLC_D1["alpha"]),
        "opt": compact_left(Q8_D1["alpha"]),
        "cw12b": compact_left(CW12B_D1["alpha"]),
        "c10": compact_left(C10_D1["alpha"]),
        "c12": compact_left(C12_D1["alpha"]),
    }
    for label, degree in (("bmc_p9", 9), ("bmc_p11", 11)):
        generator = ExponentialGenerator(degree + 1, ())
        symbols[label] = (lambda g: lambda theta:
                          g.symbols(np.asarray(theta, float))[0])(generator)
    return symbols


# ------------------------------------------------------------------ predict


def predict(args) -> int:
    """Compute and write the predeclared spectrum weighted phase error."""
    GATE.mkdir(parents=True, exist_ok=True)
    reference_hash = sha256(REFERENCE)
    if reference_hash != REFERENCE_SHA256:
        raise SystemExit("the 384^3 reference hash does not match the certified file")
    curves = kappa1_curves()
    payload = {
        "producer": "code/analysis/exponential_family/gate4c_tgv.py --predict",
        "written_utc": utc_now(),
        "contract": {
            "path": "results/exponential_family/gate4c/contract.md",
            "sha256": sha256(GATE / "contract.md"),
        },
        "definition": ("P(route) = sqrt( sum_m E(m) |e_1(theta_m)|^2 / sum_m E(m) ) "
                       "over shells 1 <= m < N/3 with theta_m = 2 pi m / N and "
                       "e_1 = kappa_1/theta - 1"),
        "reference": {
            "path": str(REFERENCE.relative_to(ROOT)),
            "sha256": reference_hash,
            "grid": REFERENCE_N,
            "time": TARGET_TIME,
            "restriction": "signed Fourier mode cube",
        },
        "member": {
            "path": str(MEMBER.relative_to(ROOT)),
            "sha256": sha256(MEMBER),
            "omegas": champion_omegas(),
        },
        "grids": {},
    }
    for n in GRIDS:
        _, spectrum = reference_bundle(n)
        shells = np.arange(1, n // 3)
        weights = spectrum[shells]
        theta = 2.0 * np.pi * shells / n
        entry = {"shells": [int(shells[0]), int(shells[-1])],
                 "theta_range": [float(theta[0]), float(theta[-1])],
                 "reference_band_energy": float(weights.sum()),
                 "P": {}}
        for label, _, _, _ in ROUTES:
            kappa1 = np.asarray(curves[label](theta), dtype=float)
            e1 = kappa1 / theta - 1.0
            entry["P"][label] = float(np.sqrt(np.sum(weights * e1**2) / np.sum(weights)))
        order = sorted(entry["P"], key=lambda k: entry["P"][k])
        entry["predicted_ordering_best_first"] = order
        entry["ratio_sl_over_exp"] = entry["P"]["sl"] / entry["P"]["exp"]
        entry["ratio_opt_over_exp"] = entry["P"]["opt"] / entry["P"]["exp"]
        entry["ratio_bmc_p11_over_exp"] = entry["P"]["bmc_p11"] / entry["P"]["exp"]
        payload["grids"][str(n)] = entry
    payload["mechanism_agreement_factor"] = MECHANISM_FACTOR
    out = GATE / "prediction.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for n in GRIDS:
        entry = payload["grids"][str(n)]
        print(f"N = {n}: shells {entry['shells'][0]}..{entry['shells'][1]}")
        for label in entry["predicted_ordering_best_first"]:
            print(f"    {label:10s} P = {entry['P'][label]:.6e}")
        print(f"    predicted P(sl)/P(exp)      = {entry['ratio_sl_over_exp']:.4f}")
        print(f"    predicted P(opt)/P(exp)     = {entry['ratio_opt_over_exp']:.4f}")
        print(f"    predicted P(bmc_p11)/P(exp) = {entry['ratio_bmc_p11_over_exp']:.4f}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


# ----------------------------------------------------------- run definitions


def run_matrix() -> list[dict]:
    """Return every run of the contract matrix in launch order."""
    runs = []
    for n in GRIDS:
        for dealias in DEALIAS_MODES:
            for label, route, degree, needs_member in ROUTES:
                runs.append(make_run(label, route, degree, needs_member, n, dealias, 1.0))
    for dealias in DEALIAS_MODES:
        for label, route, degree, needs_member in ROUTES:
            if label in TEMPORAL_LABELS:
                runs.append(make_run(label, route, degree, needs_member, 64, dealias,
                                     TEMPORAL_DT_SCALE))
    return runs


def make_run(label, route, degree, needs_member, n, dealias, dt_scale) -> dict:
    """Return the full description of one solver run."""
    tag = "" if dt_scale == 1.0 else f"_dt{int(round(dt_scale * 100)):03d}"
    suffix = "_cic" + ("_da23" if dealias == 1 else "") + tag
    stem = f"{route}{suffix}_p{degree}_N{n}"
    key = f"{label}_N{n}_da{dealias}_dt{int(round(dt_scale * 100)):03d}"
    return {
        "key": key,
        "label": label,
        "route": route,
        "p": degree,
        "needs_member": needs_member,
        "N": n,
        "dealias": dealias,
        "dt_scale": dt_scale,
        "cfl": BASE_CFL * dt_scale,
        "tag": tag,
        "csv_name": f"tgv_{stem}_re{REYNOLDS}.csv",
        "field_name": f"tgvfield_{stem}.bin",
        "log_name": f"{key}.log",
    }


def command_for(run: dict, ic_path: Path) -> list[str]:
    """Return the argument vector of one solver run."""
    dealias_word = "twothirds" if run["dealias"] == 1 else "none"
    command = [
        "nice", "-n", "19", str(SOLVER),
        str(run["N"]), run["route"], str(REYNOLDS), TARGET_TIME_STR, str(run["p"]),
        f"{run['cfl']:.10g}", TARGET_TIME_STR, dealias_word,
        run["tag"] if run["tag"] else "-",
        "--load-ic", str(ic_path), "--exact-time",
    ]
    if run["needs_member"]:
        command += ["--exp-member", str(MEMBER)]
    return command


# ----------------------------------------------------- common initial field


def build_common_ic(n: int) -> Path:
    """Write the route agnostic Taylor-Green initial field for grid n.

    The analytic field u = sin x cos y cos z, v = -cos x sin y cos z, w = 0 is
    sampled on the grid and passed through the exact Fourier Leray projection
    u_hat <- u_hat - k (k . u_hat) / |k|^2, which uses no scheme's operator. The
    layout is u then v then w, each n^3 float64 in C order [z, y, x].
    """
    IC_DIR.mkdir(parents=True, exist_ok=True)
    path = IC_DIR / f"ic_tgv_N{n}.bin"
    if path.exists():
        return path
    h = 2.0 * np.pi / n
    axis = np.arange(n) * h
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    u = np.sin(x) * np.cos(y) * np.cos(z)
    v = -np.cos(x) * np.sin(y) * np.cos(z)
    w = np.zeros_like(u)
    modes = np.fft.fftfreq(n, d=1.0 / n)
    kz, ky, kx = np.meshgrid(modes, modes, modes, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    uh, vh, wh = (np.fft.fftn(f) for f in (u, v, w))
    divergence = kx * uh + ky * vh + kz * wh
    uh = uh - kx * divergence / k2
    vh = vh - ky * divergence / k2
    wh = wh - kz * divergence / k2
    for field in (uh, vh, wh):
        field[0, 0, 0] = 0.0
    u = np.real(np.fft.ifftn(uh))
    v = np.real(np.fft.ifftn(vh))
    w = np.real(np.fft.ifftn(wh))
    residual = 1j * (kx * np.fft.fftn(u) + ky * np.fft.fftn(v) + kz * np.fft.fftn(w))
    max_divergence = float(np.abs(np.real(np.fft.ifftn(residual))).max())
    raw = np.concatenate([u.ravel(order="C"), v.ravel(order="C"), w.ravel(order="C")])
    raw.astype(np.float64).tofile(path)
    energy = 0.5 * float(np.mean(u * u + v * v + w * w))
    print(f"  common IC N={n}: {path.name} spectral divergence {max_divergence:.3e} "
          f"E0 {energy:.10f} sha256 {sha256(path)}")
    return path


# ----------------------------------------------------------------- launching


def launch(run: dict, ic_path: Path):
    """Start one detached solver process and return it with its log handle."""
    log_path = LOGS / run["log_name"]
    handle = log_path.open("w")
    environment = dict(os.environ, OMP_NUM_THREADS=str(THREADS),
                       OMP_PROC_BIND="false")
    process = subprocess.Popen(
        command_for(run, ic_path), cwd=str(WORK), env=environment,
        stdout=handle, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    run["pid"] = process.pid
    run["launch_time_utc"] = utc_now()
    print(f"LAUNCHED pid={process.pid} key={run['key']} threads={THREADS} "
          f"log={log_path.relative_to(ROOT)}", flush=True)
    return process, handle


def harvest(run: dict, returncode: int) -> None:
    """Move one finished run's outputs into the receipt directories."""
    run["returncode"] = returncode
    run["completion_time_utc"] = utc_now()
    source_csv = WORK_DATA / run["csv_name"]
    source_field = WORK_DATA / run["field_name"]
    if source_csv.exists():
        target = HISTORIES / run["csv_name"]
        target.write_bytes(source_csv.read_bytes())
        run["csv"] = str(target.relative_to(ROOT))
    if source_field.exists():
        target = FIELDS / run["field_name"]
        source_field.replace(target)
        run["field"] = str(target.relative_to(ROOT))
        run["field_bytes"] = target.stat().st_size
        run["field_sha256"] = sha256(target)
    log_text = (LOGS / run["log_name"]).read_text()
    run.update(parse_log(log_text))


def parse_log(text: str) -> dict:
    """Extract the landing time, verdict, and timing from one run log."""
    out = {}
    match = re.search(r"dumped field at exactly t=([0-9.]+)", text)
    if match:
        out["landing_time"] = float(match.group(1))
    verdict = re.search(
        r"VERDICT,(\w+),t=([0-9.]+),reason=(\w+),loop=([0-9.]+),steps=(\d+),"
        r"wall_per_step=([0-9.eE+-]+)", text)
    if verdict:
        out["verdict"] = verdict.group(1)
        out["verdict_reason"] = verdict.group(3)
        out["wall_s"] = float(verdict.group(4))
        out["steps"] = int(verdict.group(5))
        out["wall_per_step_s"] = float(verdict.group(6))
    return out


def run_wave(runs: list[dict], ic_paths: dict, concurrent: int) -> None:
    """Run a wave of solver processes with a bounded number of them alive."""
    pending = list(runs)
    active = []
    while pending or active:
        while pending and len(active) < concurrent:
            run = pending.pop(0)
            process, handle = launch(run, ic_paths[run["N"]])
            active.append((process, handle, run))
        time.sleep(2.0)
        still = []
        for process, handle, run in active:
            code = process.poll()
            if code is None:
                still.append((process, handle, run))
                continue
            handle.close()
            harvest(run, code)
            print(f"EXIT pid={process.pid} key={run['key']} rc={code} "
                  f"landing={run.get('landing_time')} "
                  f"wall_per_step={run.get('wall_per_step_s')}", flush=True)
        active = still


def verify_pilot(run: dict) -> None:
    """Check the exact time landing and the band error path of the pilot run."""
    if run.get("returncode") != 0:
        raise SystemExit(f"pilot run {run['key']} failed with rc={run['returncode']}")
    landing = run.get("landing_time")
    if landing is None or abs(landing - round(TARGET_TIME, 10)) > 0.5e-10:
        raise SystemExit(f"pilot run {run['key']} did not land on t*: {landing}")
    field = ROOT / run["field"]
    n = run["N"]
    fields = load_field(field, n)
    reference_fields, reference_spectrum = reference_bundle(n)
    spectrum = energy_spectrum(fields, n)
    band = slice(1, n // 3)
    e_band = float(np.sqrt(np.sum((spectrum[band] - reference_spectrum[band]) ** 2)
                           / np.sum(reference_spectrum[band] ** 2)))
    numerator = np.sqrt(np.mean(sum((a - b) ** 2 for a, b
                                    in zip(fields, reference_fields, strict=True))))
    denominator = np.sqrt(np.mean(sum(b**2 for b in reference_fields)))
    if not np.isfinite(e_band) or e_band <= 0.0:
        raise SystemExit(f"pilot band error is not a usable number: {e_band}")
    print(f"PILOT {run['key']}: landing {landing:.10f} (target {TARGET_TIME}), "
          f"e_band {e_band:.8f}, e_full {float(numerator / denominator):.8f}")


def timing_replicates(ic_paths: dict) -> dict:
    """Time three replicates per route at N = 64 on four threads, one at a time."""
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    (TIMING_DIR / "data").mkdir(parents=True, exist_ok=True)
    out = {}
    for label, route, degree, needs_member in ROUTES:
        if label not in TIMING_LABELS:
            continue
        run = make_run(label, route, degree, needs_member, 64, 1, 1.0)
        samples = []
        for replicate in range(TIMING_REPLICATES):
            log_path = LOGS / f"timing_{label}_r{replicate}.log"
            handle = log_path.open("w")
            environment = dict(os.environ, OMP_NUM_THREADS=str(THREADS),
                               OMP_PROC_BIND="false")
            process = subprocess.Popen(
                command_for(run, ic_paths[64]), cwd=str(TIMING_DIR),
                env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
            code = process.wait()
            handle.close()
            if code != 0:
                raise SystemExit(f"timing replicate {label} r{replicate} rc={code}")
            parsed = parse_log(log_path.read_text())
            samples.append(parsed["wall_per_step_s"])
            print(f"TIMING {label} replicate {replicate}: "
                  f"{parsed['wall_per_step_s']:.6f} s/step (pid {process.pid})",
                  flush=True)
        out[label] = {
            "replicates_s_per_step": samples,
            "median_s_per_step": float(np.median(samples)),
            "steps": parsed["steps"],
            "threads": THREADS,
            "N": 64,
            "dealias": 1,
        }
    return out


def run(args) -> int:
    """Execute the contract matrix, the temporal checks, and the timings."""
    if not (GATE / "prediction.json").exists():
        raise SystemExit("prediction.json must exist before any run")
    if not SOLVER.exists():
        raise SystemExit(f"missing solver executable {SOLVER}")
    for directory in (WORK_DATA, FIELDS, HISTORIES, LOGS, IC_DIR, CACHE):
        directory.mkdir(parents=True, exist_ok=True)

    ic_paths = {n: build_common_ic(n) for n in GRIDS}
    runs = run_matrix()
    by_key = {r["key"]: r for r in runs}

    provenance = {
        "solver": str(SOLVER.relative_to(ROOT)),
        "solver_sha256": sha256(SOLVER),
        "solver_source_sha256": sha256(SOLVER_DIR / "tgv_re1600.cpp"),
        "member_sha256": sha256(MEMBER),
        "reference_sha256": sha256(REFERENCE),
        "ic": {str(n): {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
               for n, p in ic_paths.items()},
        "threads_per_process": THREADS,
        "max_concurrent": MAX_CONCURRENT,
        "nice": 19,
        "launch_time_utc": utc_now(),
    }

    pilot_key = "exp_N64_da1_dt100"
    print("=== pilot: one exp N=64 run to completion ===", flush=True)
    run_wave([by_key[pilot_key]], ic_paths, 1)
    verify_pilot(by_key[pilot_key])

    wave64 = [r for r in runs if r["N"] == 64 and r["key"] != pilot_key]
    print("=== wave 1: remaining N=64 runs ===", flush=True)
    run_wave(wave64, ic_paths, MAX_CONCURRENT)

    wave128 = [r for r in runs if r["N"] == 128]
    print("=== wave 2: N=128 runs ===", flush=True)
    run_wave(wave128, ic_paths, MAX_CONCURRENT)

    print("=== wall time replicates at N=64, four threads, sequential ===", flush=True)
    timings = timing_replicates(ic_paths)

    registry = GATE / "process_registry.csv"
    with registry.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "key", "label", "route", "p", "N", "dealias", "dt_scale", "cfl",
            "pid", "launch_time_utc", "completion_time_utc", "returncode",
            "verdict", "verdict_reason", "landing_time", "wall_s", "steps",
            "wall_per_step_s", "field_bytes", "field_sha256", "command",
        ), lineterminator="\n")
        writer.writeheader()
        for r in runs:
            row = {k: r.get(k, "") for k in writer.fieldnames}
            row["command"] = " ".join(command_for(r, ic_paths[r["N"]]))
            writer.writerow(row)
    provenance["timings"] = timings
    provenance["completion_time_utc"] = utc_now()
    (GATE / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    failed = [r["key"] for r in runs if r.get("returncode") != 0]
    print(f"wrote {registry.relative_to(ROOT)}")
    if failed:
        raise SystemExit(f"run failures: {', '.join(failed)}")
    return 0


# ------------------------------------------------------------------- reduce


def read_history(path: Path) -> dict:
    """Read one solver CSV history into named columns."""
    with path.open() as handle:
        header = handle.readline().strip().split(",")
    body = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return {name: body[:, i] for i, name in enumerate(header)}


def quad_peak(t: np.ndarray, y: np.ndarray):
    """Return the quadratic peak through the three samples around the maximum."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(t[i]), float(y[i])
    t0, t1, t2 = t[i - 1], t[i], t[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d1, d2 = t1 - t0, t2 - t1
    a = ((y2 - y1) / d2 - (y1 - y0) / d1) / (t2 - t0)
    b = (y1 - y0) / d1 - a * (t1 + t0)
    if abs(a) < 1e-30:
        return float(t1), float(y1)
    tstar = -b / (2 * a)
    if not (t0 <= tstar <= t2):
        return float(t1), float(y1)
    c = y1 - a * t1 * t1 - b * t1
    return float(tstar), float(a * tstar * tstar + b * tstar + c)


def reference_epsilon_peak():
    """Return the 512^3 reference -dE/dt peak time and value."""
    table = np.loadtxt(GDIAG, comments="#")
    return quad_peak(table[:, 0], table[:, 2])


def reduce_run(run: dict, bundles: dict) -> dict:
    """Compute every decision quantity of one run."""
    n = run["N"]
    reference_fields, reference_spectrum = bundles[n]
    row = dict(run)
    field_path = ROOT / run["field"] if run.get("field") else None
    if field_path is None or not field_path.exists():
        raise SystemExit(f"missing field for {run['key']}")
    fields = load_field(field_path, n)
    spectrum = energy_spectrum(fields, n)
    band = slice(1, n // 3)
    mid = slice(1, n // 6)
    outer = slice(n // 6, n // 3)
    row["e_band"] = float(np.sqrt(np.sum((spectrum[band] - reference_spectrum[band]) ** 2)
                                  / np.sum(reference_spectrum[band] ** 2)))
    row["e_mid"] = float(np.sqrt(np.sum((spectrum[mid] - reference_spectrum[mid]) ** 2)
                                 / np.sum(reference_spectrum[mid] ** 2)))
    row["e_outer"] = float(np.sqrt(np.sum((spectrum[outer] - reference_spectrum[outer]) ** 2)
                                   / np.sum(reference_spectrum[outer] ** 2)))
    numerator = np.sqrt(np.mean(sum((a - b) ** 2 for a, b
                                    in zip(fields, reference_fields, strict=True))))
    denominator = np.sqrt(np.mean(sum(b**2 for b in reference_fields)))
    row["e_full"] = float(numerator / denominator)
    row["shell_spectrum"] = [float(v) for v in spectrum[band]]

    history = read_history(ROOT / run["csv"])
    t = history["t"]
    energy = history["E"]
    epsilon = history["eps_dEdt"]
    peak_time, peak_value = quad_peak(t[1:], epsilon[1:])
    row["eps_peak_time"] = peak_time
    row["eps_peak_value"] = peak_value
    # The history stops at t*, so a run whose dissipation is still rising there
    # reports the endpoint rather than an interior maximum.
    row["eps_peak_at_horizon"] = bool(int(np.argmax(epsilon[1:])) == epsilon[1:].size - 1)
    row["divmax_max"] = float(np.max(history["divmax"]))
    increments = np.diff(energy)
    row["energy_max_increase"] = float(np.max(increments)) if increments.size else 0.0
    row["energy_monotone"] = bool(row["energy_max_increase"] <= 0.0)
    row["csv_final_time"] = float(t[-1])
    landing = run.get("landing_time")
    row["landing_abs_difference"] = (abs(float(landing) - round(TARGET_TIME, 10))
                                     if landing is not None else float("nan"))
    row["records"] = int(t.size)
    return row


def reduce(args) -> int:
    """Reduce every run, apply the frozen rule, and write the receipts."""
    registry = GATE / "process_registry.csv"
    if not registry.exists():
        raise SystemExit("process_registry.csv is missing; run --run first")
    with registry.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    runs = []
    for row in rows:
        run = make_run(row["label"], row["route"], int(row["p"]),
                       row["label"] == "exp", int(row["N"]), int(row["dealias"]),
                       float(row["dt_scale"]))
        run.update({
            "pid": int(row["pid"]) if row["pid"] else None,
            "returncode": int(row["returncode"]) if row["returncode"] else None,
            "verdict": row["verdict"],
            "verdict_reason": row["verdict_reason"],
            "landing_time": float(row["landing_time"]) if row["landing_time"] else None,
            "wall_s": float(row["wall_s"]) if row["wall_s"] else None,
            "steps": int(row["steps"]) if row["steps"] else None,
            "wall_per_step_s": float(row["wall_per_step_s"]) if row["wall_per_step_s"] else None,
            "field_sha256": row["field_sha256"],
            "field": f"results/exponential_family/gate4c/fields/{run['field_name']}",
            "csv": f"results/exponential_family/gate4c/epsilon_histories/{run['csv_name']}",
        })
        runs.append(run)

    bundles = {n: reference_bundle(n) for n in GRIDS}
    reference_peak_time, reference_peak_value = reference_epsilon_peak()
    reduced = []
    for run in runs:
        row = reduce_run(run, bundles)
        row["eps_peak_reference_value"] = reference_peak_value
        row["eps_peak_reference_time"] = reference_peak_time
        row["eps_peak_absolute_error"] = abs(row["eps_peak_value"] - reference_peak_value)
        row["eps_peak_relative_error"] = row["eps_peak_absolute_error"] / reference_peak_value
        row["eps_peak_time_error"] = row["eps_peak_time"] - reference_peak_time
        reduced.append(row)
        print(f"{row['key']:26s} e_band {row['e_band']:.8f}  e_full {row['e_full']:.6f}  "
              f"eps_peak {row['eps_peak_value']:.6e} at t {row['eps_peak_time']:.4f}",
              flush=True)

    provenance = json.loads((GATE / "run_provenance.json").read_text())
    timings = provenance.get("timings", {})
    fields = (
        "key", "label", "route", "p", "N", "dealias", "dt_scale", "cfl",
        "pid", "returncode", "verdict", "verdict_reason",
        "landing_time", "landing_abs_difference", "csv_final_time", "records",
        "e_band", "e_mid", "e_outer", "e_full",
        "eps_peak_value", "eps_peak_time", "eps_peak_at_horizon",
        "eps_peak_reference_value",
        "eps_peak_reference_time", "eps_peak_absolute_error",
        "eps_peak_relative_error", "eps_peak_time_error",
        "divmax_max", "energy_monotone", "energy_max_increase",
        "wall_s", "steps", "wall_per_step_s", "wall_per_step_median_s",
        "cost_model_solves", "cost_model_half_bandwidth",
        "field_sha256", "field_retained", "csv",
    )
    with (GATE / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in reduced:
            record = {k: row.get(k, "") for k in fields}
            record["cost_model_solves"] = COST_MODEL[row["label"]]["solves"]
            record["cost_model_half_bandwidth"] = COST_MODEL[row["label"]]["half_bandwidth"]
            median = timings.get(row["label"], {}).get("median_s_per_step")
            record["wall_per_step_median_s"] = (
                median if (median is not None and row["N"] == 64
                           and row["dealias"] == 1 and row["dt_scale"] == 1.0) else "")
            record["field_retained"] = row["label"] == "exp"
            writer.writerow(record)

    decision = decide(reduced, timings)
    (GATE / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision["outcome"], indent=2))

    if decision["outcome"] != "TRIUMPH":
        diagnosis = diagnose(reduced, decision)
        (GATE / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
        print(f"wrote {(GATE / 'diagnosis.json').relative_to(ROOT)}")

    prune_fields(reduced)
    print(f"wrote {(GATE / 'runs.csv').relative_to(ROOT)} and "
          f"{(GATE / 'decision.json').relative_to(ROOT)}")
    return 0


def prune_fields(reduced: list[dict]) -> None:
    """Keep the t* fields of route exp only and hash what is retained.

    The retention marker file suspends the deletion so that the reduction can
    be repeated from the same fields before the non exp ones are discarded.
    """
    marker = GATE / "retain_all_fields"
    lines = []
    for row in sorted(reduced, key=lambda r: r["key"]):
        path = ROOT / row["field"]
        if row["label"] == "exp":
            if path.exists():
                lines.append(f"{sha256(path)}  {row['field']}")
        elif path.exists() and not marker.exists():
            path.unlink()
    (GATE / "field_hashes.sha256").write_text("\n".join(lines) + "\n")
    if marker.exists():
        print(f"retention marker present: non exp fields kept, remove "
              f"{marker.relative_to(ROOT)} and rerun --reduce to prune them")


# ------------------------------------------------------------ decision rule


def pick(reduced, label, n, dealias, dt_scale=1.0):
    """Return the reduced row of one matrix cell."""
    for row in reduced:
        if (row["label"] == label and row["N"] == n and row["dealias"] == dealias
                and row["dt_scale"] == dt_scale):
            return row
    raise KeyError(f"no run for {label} N={n} dealias={dealias} dt_scale={dt_scale}")


def decide(reduced: list[dict], timings: dict) -> dict:
    """Apply the frozen decision rule verbatim."""
    tests = {}
    for n in GRIDS:
        exp = pick(reduced, "exp", n, 1)["e_band"]
        tests[f"N{n}_da1_exp_at_most_sl_over_1.5"] = (
            exp <= pick(reduced, "sl", n, 1)["e_band"] / 1.5)
        tests[f"N{n}_da1_exp_at_most_opt"] = exp <= pick(reduced, "opt", n, 1)["e_band"]
        tests[f"N{n}_da1_exp_at_most_bmc_p11"] = (
            exp <= pick(reduced, "bmc_p11", n, 1)["e_band"])
        tests[f"N{n}_da0_exp_at_most_sl"] = (
            pick(reduced, "exp", n, 0)["e_band"] <= pick(reduced, "sl", n, 0)["e_band"])
    for n in GRIDS:
        tests[f"N{n}_da1_exp_beats_sl"] = (
            pick(reduced, "exp", n, 1)["e_band"] < pick(reduced, "sl", n, 1)["e_band"])
    dealias1_conditions = all(v for k, v in tests.items()
                              if "_da1_" in k and not k.endswith("_beats_sl"))
    base_conditions = all(v for k, v in tests.items() if "_da0_" in k)
    beats_sl_da1 = all(tests[f"N{n}_da1_exp_beats_sl"] for n in GRIDS)

    margins = {}
    for n in GRIDS:
        for dealias in DEALIAS_MODES:
            margins[f"N{n}_da{dealias}"] = abs(
                pick(reduced, "sl", n, dealias)["e_band"]
                - pick(reduced, "exp", n, dealias)["e_band"])
    smallest_margin = min(margins.values())
    temporal = {}
    for label in TEMPORAL_LABELS:
        for dealias in DEALIAS_MODES:
            base = pick(reduced, label, 64, dealias, 1.0)["e_band"]
            half = pick(reduced, label, 64, dealias, TEMPORAL_DT_SCALE)["e_band"]
            shift = abs(half - base)
            temporal[f"{label}_N64_da{dealias}"] = {
                "e_band_dt_scale_1.0": base,
                "e_band_dt_scale_0.5": half,
                "shift": shift,
                "allowance": 0.5 * smallest_margin,
                "pass": shift < 0.5 * smallest_margin,
            }
    temporal_pass = all(v["pass"] for v in temporal.values())

    if not beats_sl_da1:
        outcome = "REJECT"
    elif dealias1_conditions and temporal_pass and base_conditions:
        outcome = "TRIUMPH"
    elif dealias1_conditions and not base_conditions:
        outcome = "PARTIAL"
    elif dealias1_conditions and base_conditions and not temporal_pass:
        # The frozen rule names no outcome for this case; the contract says the
        # margin is then reported as not temporally resolved.
        outcome = "TRIUMPH_CONDITIONS_MET_BUT_NOT_TEMPORALLY_RESOLVED"
    else:
        outcome = "UNCLASSIFIED_BY_FROZEN_RULE"

    prediction = json.loads((GATE / "prediction.json").read_text())
    mechanism = {}
    for n in GRIDS:
        realized = (pick(reduced, "sl", n, 1)["e_band"]
                    / pick(reduced, "exp", n, 1)["e_band"])
        predicted = prediction["grids"][str(n)]["ratio_sl_over_exp"]
        ratio = realized / predicted
        mechanism[f"N{n}"] = {
            "predicted_P_sl_over_P_exp": predicted,
            "realized_e_band_sl_over_exp": realized,
            "realized_over_predicted": ratio,
            "within_factor_two": (1.0 / MECHANISM_FACTOR) <= ratio <= MECHANISM_FACTOR,
        }
    mechanism["confirmed"] = all(v["within_factor_two"] for k, v in mechanism.items()
                                 if k.startswith("N"))

    return {
        "producer": "code/analysis/exponential_family/gate4c_tgv.py --reduce",
        "written_utc": utc_now(),
        "contract_sha256": sha256(GATE / "contract.md"),
        "rule": ("TRIUMPH if, at both N = 64 and N = 128 with dealias = 1, "
                 "e_band(exp) <= e_band(sl)/1.5 and e_band(exp) <= e_band(opt), "
                 "e_band(bmc11), and the temporal check passes; and, with "
                 "dealias = 0, e_band(exp) <= e_band(sl) at both grids. PARTIAL "
                 "if the dealias = 1 conditions hold but the base runs do not. "
                 "REJECT if exp does not beat sl at both grids with dealias = 1."),
        "outcome": outcome,
        "tests": tests,
        "dealias1_conditions": dealias1_conditions,
        "base_conditions": base_conditions,
        "exp_beats_sl_dealias1_both_grids": beats_sl_da1,
        "temporal_check": {
            "definition": ("exp and sl at N = 64, both dealiasing modes, rerun at "
                           "dt scale 0.5; each resolved band error must move by "
                           "less than half the smallest exp versus sl e_band "
                           "margin of the matrix"),
            "exp_versus_sl_margins": margins,
            "smallest_margin": smallest_margin,
            "allowance": 0.5 * smallest_margin,
            "cells": temporal,
            "pass": temporal_pass,
        },
        "mechanism": mechanism,
        "e_band": {f"{label}_N{n}_da{d}": pick(reduced, label, n, d)["e_band"]
                   for label, _, _, _ in ROUTES for n in GRIDS for d in DEALIAS_MODES},
        "timings": timings,
    }


def diagnose(reduced: list[dict], decision: dict) -> dict:
    """Run the predeclared diagnosis: shell resolved error and epsilon peaks."""
    out = {
        "producer": "code/analysis/exponential_family/gate4c_tgv.py --reduce",
        "written_utc": utc_now(),
        "trigger": decision["outcome"],
        "note": ("Predeclared diagnosis only: shell resolved error and epsilon(t) "
                 "peak comparison locating the loss in the mid band, the outer "
                 "band, or the aliased tail. No threshold is changed and no run "
                 "is repeated."),
        "cells": {},
    }
    for n in GRIDS:
        for dealias in DEALIAS_MODES:
            cell = {}
            for label, _, _, _ in ROUTES:
                row = pick(reduced, label, n, dealias)
                cell[label] = {
                    "e_band": row["e_band"],
                    "e_mid": row["e_mid"],
                    "e_outer": row["e_outer"],
                    "e_full": row["e_full"],
                    "eps_peak_value": row["eps_peak_value"],
                    "eps_peak_time": row["eps_peak_time"],
                    "eps_peak_relative_error": row["eps_peak_relative_error"],
                    "shell_spectrum": row["shell_spectrum"],
                }
            exp = cell["exp"]
            sl = cell["sl"]
            cell["exp_versus_sl"] = {
                "e_mid_ratio": sl["e_mid"] / exp["e_mid"],
                "e_outer_ratio": sl["e_outer"] / exp["e_outer"],
                "e_band_ratio": sl["e_band"] / exp["e_band"],
                "eps_peak_error_ratio": (sl["eps_peak_relative_error"]
                                         / exp["eps_peak_relative_error"]),
            }
            out["cells"][f"N{n}_da{dealias}"] = cell
    return out


# -------------------------------------------------------------------- check


def check(args) -> int:
    """Recompute the decision from the stored quantities in runs.csv."""
    with (GATE / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reduced = []
    for row in rows:
        reduced.append({
            "key": row["key"], "label": row["label"], "N": int(row["N"]),
            "dealias": int(row["dealias"]), "dt_scale": float(row["dt_scale"]),
            "e_band": float(row["e_band"]), "e_mid": float(row["e_mid"]),
            "e_outer": float(row["e_outer"]), "e_full": float(row["e_full"]),
            "eps_peak_value": float(row["eps_peak_value"]),
            "eps_peak_time": float(row["eps_peak_time"]),
            "eps_peak_relative_error": float(row["eps_peak_relative_error"]),
            "shell_spectrum": [],
        })
    stored = json.loads((GATE / "decision.json").read_text())
    recomputed = decide(reduced, stored.get("timings", {}))
    agree = {
        "outcome": recomputed["outcome"] == stored["outcome"],
        "tests": recomputed["tests"] == stored["tests"],
        "temporal_pass": (recomputed["temporal_check"]["pass"]
                          == stored["temporal_check"]["pass"]),
        "mechanism_confirmed": (recomputed["mechanism"]["confirmed"]
                                == stored["mechanism"]["confirmed"]),
    }
    print(json.dumps({"stored_outcome": stored["outcome"],
                      "recomputed_outcome": recomputed["outcome"],
                      "agreement": agree}, indent=2))
    return 0 if all(agree.values()) else 1


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    """Dispatch the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--reduce", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.predict:
        return predict(args)
    if args.run:
        return run(args)
    if args.reduce:
        return reduce(args)
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
