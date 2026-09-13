#!/usr/bin/env python3
"""Gate 5 part 5c: the immersed smooth wall against the strongest competitor.

The paper's Section 6.5 experiment is the periodic constrained Helmholtz
problem

    (-div_h grad_h + kappa^2) u = f   on [0, 2 pi)^2,      kappa = 2,
    E u = g                           on Gamma,

with three immersed curves, periodized Gaussian sources of width 0.22, a
768-square Fourier reference, N in {96, 128, 160, 192}, four cell-unit shifts,
and the exact adjoint pair H S = E^T W with H = h^2 I. This part reruns that
experiment for four routes:

  champion        E_{6,2}(1.7701927977475826, 2.0606855355641316), degree 9.
                  One coefficient field carries the bulk first derivative, the
                  off-grid wall reads of value and derivatives, and the adjoint
                  spread, all from the same generator.
  BMC9            E_{10,0}, degree 9, the paper's accuracy-first route.
  C10 plus L12    the paper's strongest admitted tridiagonal Pade route.
  SLC q4 plus L12  SLC 1992 nodal first derivative with the same common
                  twelve-point Lagrange wall evaluator.

Producer discipline
-------------------

The certified producer of the paper's cells is
code/analysis/ibm_cert/run_matrix.py. It binds every checkpoint to its own
SHA256, so editing it would invalidate the stored receipts, and it is not one
of the two files this part may edit. This module therefore imports it read only
and installs one dispatching hook on its module-level build_coupling: the hook
forwards every existing route kind to the original function object, so the
controls run through literally the same code, and it adds the exponential
branch for the champion. No file of the certified harness is modified.

Route construction of the champion
----------------------------------

Let phi be the generator of E_{6,2}(omega) with support half width n/2 = 5 and
lattice sample symbol b(theta) = sum_l phi(l) exp(-i l theta). On the periodic
grid the sample matrix B is circulant with symbol b(theta_x) b(theta_y), so

    E   = Phi B^{-1},          Phi_{m,(i,j)} = phi(x_m/h - i) phi(y_m/h - j),
    S   = B^{-1} Phi^T W,      H = h^2 I,        H (h^{-2} S) = E^T W,

which is the exact adjoint pair the contract requires, built from one
generator. The bulk operator is the composed divergence-gradient form with the
generator's own nodal first-derivative symbol kappa_1(theta), so the Helmholtz
denominator is (kappa_1(theta_x)^2 + kappa_1(theta_y)^2)/h^2 + kappa^2, exactly
the convention the paper uses for every local route. The off-grid weights come
from evaluator.py: the value weights are phi(delta - o) for the twelve offsets
o = -5, ..., 6, which is the matrix form of evaluate_periodic_field, and the
first and second derivative reads use the same coefficient vector through
phi' and phi''.

Prediction
----------

Section 5a measured the coupled off-grid transfers of the same routes. The
value channel of a route with grid symbol b and off-grid weights w_o(delta) has
transfer residual

    r(theta, delta) = sum_o w_o(delta) exp(i o theta) / b(theta)
                      - exp(i delta theta),

with b = 1 for the nodal Lagrange routes, and the bulk Helmholtz response
carries the separate symbol residual

    s(kx, ky) = (kx^2 + ky^2 + kappa^2)
                / ((kappa_1(theta_x)^2 + kappa_1(theta_y)^2)/h^2 + kappa^2) - 1.

Weighting both by the modal amplitudes of the same 768-square reference gives a
predicted wall-trace magnitude per route, and the prediction records only the
ordering it implies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
CERT = CODE / "analysis" / "ibm_cert"
for path in (str(CODE), str(HERE), str(CERT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scipy.sparse import csr_matrix  # noqa: E402

import run_matrix as rm  # noqa: E402
from evaluator import as_evaluator  # noqa: E402
from evaluator_common import OFFSETS, lagrange_weights  # noqa: E402
from family import ExponentialGenerator  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate5" / "5c"
FIGURES = OUT / "figures"
CONTRACT = ROOT / "results" / "exponential_family" / "gate5" / "contract.md"
CERT_RUN = (ROOT / "results" / "ibm_cert" / "runs"
            / "ibm_cert_final_floor_20260813T135207" / "checkpoints" / "primary")

CHAMPION_M = 6
CHAMPION_OMEGAS = (1.7701927977475826, 2.0606855355641316)
CHAMPION_ID = "EXP-E62"

ROUTE_IDS = (CHAMPION_ID, "BMC9", "C10|common_L12", "SLC-SL|common_L12")
ROUTE_LABELS = {
    CHAMPION_ID: "champion E_{6,2}",
    "BMC9": "BMC9",
    "C10|common_L12": "C10 + L12",
    "SLC-SL|common_L12": "SLC q4 + L12",
}
CURVES = ("disk", "trefoil", "kite")

GRIDS = rm.PRIMARY_GRIDS
SHIFTS = rm.SHIFTS
REPEATS = 3
SETUP_KEY = "evaluator_and_coupling_construction"

TRACE_FACTOR_THRESHOLD = 62.707527291670914
EXTERIOR_FACTOR_THRESHOLD = 32.16591996072041
PREMIUM_LIMIT = 2.0

METRICS = {
    "wall_trace": ("trace_error", "trace_numerical_floor"),
    "exterior_relative_l2": ("exterior_relative_l2",
                             "exterior_relative_l2_numerical_floor"),
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# the champion route inside the certified harness
# ----------------------------------------------------------------------

def champion_generator() -> ExponentialGenerator:
    return ExponentialGenerator(CHAMPION_M, CHAMPION_OMEGAS)


def champion_route() -> dict:
    generator = champion_generator()
    return {
        "primary_route_id": CHAMPION_ID,
        "kind": "exp",
        "derivative_route": CHAMPION_ID,
        "formal_order": generator.m,
        "degree": generator.p,
        "m": generator.m,
        "omegas": list(CHAMPION_OMEGAS),
        "logical_aliases": [CHAMPION_ID],
        "evaluator_hash": hashlib.sha256(
            f"exponential-generator-E{{{CHAMPION_M},2}}:{CHAMPION_OMEGAS}".encode()
        ).hexdigest(),
    }


def exp_first_derivative_symbol(theta: np.ndarray) -> np.ndarray:
    """Return kappa_1(theta) of the champion's nodal first derivative."""
    return champion_generator().symbols(np.asarray(theta, dtype=float))[1]


def exp_bulk_denominator(npts: int, h: float) -> np.ndarray:
    """Return the composed divergence-gradient Helmholtz symbol of the champion."""
    theta = np.fft.fftfreq(npts, d=1.0) * rm.BOX
    theta = np.where(theta >= 0.0, theta, theta + rm.BOX)
    theta = np.where(theta <= np.pi, theta, theta - rm.BOX)
    theta = np.abs(theta)
    kprime = exp_first_derivative_symbol(theta)
    symbol = (kprime / h) ** 2
    return symbol[:, None] + symbol[None, :] + rm.KAPPA**2


def exp_symbol_2d(npts: int) -> np.ndarray:
    """Return the tensor sample symbol b(theta_x) b(theta_y) of the champion."""
    evaluator = as_evaluator(champion_generator())
    radius = (evaluator.n + 1) // 2
    lattice = np.arange(-radius, radius + 1).astype(float)
    samples = evaluator.value(lattice)
    column = np.zeros(npts)
    for offset, value in zip(lattice.astype(int), samples):
        column[offset % npts] += value
    symbol = np.real(np.fft.fft(column))
    return symbol[:, None] * symbol[None, :]


def build_exp_phi(points: np.ndarray, npts: int, h: float,
                  derivative: int = 0) -> csr_matrix:
    """Return the sparse off-grid read matrix of the champion generator.

    Row m holds the tensor-product weights phi^{(a)}(x_m/h - i) phi^{(b)}(y_m/h - j)
    over the twelve-wide local footprint, which is the matrix form of
    evaluator.evaluate_periodic_field before the factor h^{-derivative}.
    """
    evaluator = as_evaluator(champion_generator())
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


def build_exp_coupling(route: dict, case) -> dict:
    """Return the champion's coupling dictionary in the harness convention."""
    npts = case.npts
    h = case.h
    weights = case.weights
    denominator = exp_bulk_denominator(npts, h)

    def solve(rhs: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.fft2(rhs) / denominator))

    phi_m = build_exp_phi(case.markers, npts, h)
    phi_q = build_exp_phi(case.queries, npts, h)
    b2 = exp_symbol_2d(npts)
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


_ORIGINAL_BUILD_COUPLING = rm.build_coupling


def _dispatch_build_coupling(route, evaluator, case, q_table):
    """Forward every certified route kind unchanged and add the champion branch."""
    if route.get("kind") == "exp":
        return build_exp_coupling(route, case)
    return _ORIGINAL_BUILD_COUPLING(route, evaluator, case, q_table)


def install_hook() -> None:
    rm.build_coupling = _dispatch_build_coupling


# ----------------------------------------------------------------------
# the harness context
# ----------------------------------------------------------------------

class Harness:
    def __init__(self):
        from evaluator_common import load_q_table

        outdir = rm.OUTDIR
        self.contract_hash = rm.sha256_file(outdir / "frozen_contract.json")
        self.certificate_hash = rm.sha256_file(outdir / "evaluator_certificate.json")
        self.candidate_hash = rm.sha256_file(outdir / "evaluator_candidate.npz")
        self.certificate = json.loads(
            (outdir / "evaluator_certificate.json").read_text()
        )
        self.q_table = load_q_table(outdir / "evaluator_candidate.npz")
        roster = rm.route_roster(self.candidate_hash)
        self.roster = {item[0]["primary_route_id"]: item for item in roster}
        self.roster[CHAMPION_ID] = (champion_route(), None)
        self.reference = rm.SpectralReference()

    def entry(self, route_id: str):
        return self.roster[route_id]


def per_curve_trace(route: dict, evaluator: dict | None, case,
                    q_table) -> tuple[dict, float]:
    """Return the per-curve off-marker trace errors of one repeated solve.

    The solve repeats the certified sequence of run_matrix.solve_once, and the
    caller compares the recomputed global maximum with the certified one, so a
    divergence in either code path is detected rather than hidden.
    """
    operators = rm.build_coupling(route, evaluator, case, q_table)
    particular = operators["solve"](case.forcing)
    marker_count = case.markers.shape[0]
    schur = np.empty((marker_count, marker_count))
    basis = np.zeros(marker_count)
    for column in range(marker_count):
        basis[column] = 1.0
        schur[:, column] = operators["schur_column"](basis)
        basis[column] = 0.0
    right_hand_side = operators["E_marker"](particular) - case.g_markers
    multiplier = np.linalg.solve(schur, right_hand_side)
    for _ in range(2):
        residual = right_hand_side.astype(np.longdouble) - (
            schur.astype(np.longdouble) @ multiplier.astype(np.longdouble)
        )
        multiplier += np.linalg.solve(schur, np.asarray(residual, dtype=float))
    solution = particular - operators["correction"](multiplier)
    residual_query = np.abs(operators["E_query"](solution) - case.g_queries)
    block = case.n_markers_per_obstacle
    per_curve = {
        name: float(np.max(residual_query[k * block:(k + 1) * block]))
        for k, name in enumerate(CURVES)
    }
    return per_curve, float(np.max(residual_query))


# ----------------------------------------------------------------------
# prediction from the 5a coupled off-grid symbols
# ----------------------------------------------------------------------

def spline_value_transfer(generator, theta: np.ndarray,
                          delta: np.ndarray) -> np.ndarray:
    """Return the off-grid value transfer residual of a one-field spline route."""
    evaluator = as_evaluator(generator)
    b = generator.symbols(theta)[0]
    offsets = OFFSETS.astype(float)
    residual = np.zeros((delta.size, theta.size), dtype=complex)
    for j, d in enumerate(delta):
        weights = evaluator.value(d - offsets)
        phase = np.exp(1j * np.outer(offsets, theta))
        residual[j] = (weights @ phase) / b - np.exp(1j * d * theta)
    return residual


def lagrange_value_transfer(order: int, theta: np.ndarray,
                            delta: np.ndarray) -> np.ndarray:
    """Return the off-grid value transfer residual of the nodal Lagrange read."""
    offsets = OFFSETS.astype(float)
    residual = np.zeros((delta.size, theta.size), dtype=complex)
    for j, d in enumerate(delta):
        weights = np.asarray(lagrange_weights(order, float(d)), dtype=float)
        phase = np.exp(1j * np.outer(offsets, theta))
        residual[j] = weights @ phase - np.exp(1j * d * theta)
    return residual


def route_kappa1(route_id: str, theta: np.ndarray) -> np.ndarray:
    import spline_analysis as sa

    if route_id == CHAMPION_ID:
        return exp_first_derivative_symbol(theta)
    if route_id == "BMC9":
        return sa.kprime_collocation(theta, 9)
    if route_id == "C10|common_L12":
        return sa.kprime_compact(theta, 10)
    if route_id == "SLC-SL|common_L12":
        slc = rm.COMPACT_ROUTES["SLC-SL"]
        return rm.compact_kprime(theta, slc["alpha"], slc["a"])
    raise ValueError(route_id)


def route_offgrid_residual(route_id: str, theta: np.ndarray,
                           delta: np.ndarray) -> np.ndarray:
    if route_id == CHAMPION_ID:
        return spline_value_transfer(champion_generator(), theta, delta)
    if route_id == "BMC9":
        return spline_value_transfer(ExponentialGenerator(10, ()), theta, delta)
    return lagrange_value_transfer(12, theta, delta)


def predict(harness: Harness, npts: int = 96, delta_points: int = 65) -> dict:
    """Return the predicted wall-trace ordering from the symbol residuals."""
    h = rm.BOX / npts
    reference = harness.reference
    modes = reference.modes.astype(float)
    keep = np.abs(modes) <= npts / 2
    kx = modes[keep]
    amplitude = np.abs(reference.uhat[np.ix_(keep, keep)]) / (rm.NREF * rm.NREF)
    theta = np.abs(kx) * h
    delta = np.linspace(0.0, 1.0, delta_points, endpoint=False)

    rows = []
    for route_id in ROUTE_IDS:
        kappa1 = route_kappa1(route_id, theta)
        symbol = (kappa1 / h) ** 2
        discrete = symbol[:, None] + symbol[None, :] + rm.KAPPA**2
        exact = kx[:, None] ** 2 + kx[None, :] ** 2 + rm.KAPPA**2
        bulk_residual = np.abs(exact / discrete - 1.0)
        bulk = float(np.sum(amplitude * bulk_residual))

        residual = np.abs(route_offgrid_residual(route_id, theta, delta))
        worst = residual.max(axis=0)
        offgrid = float(np.sum(amplitude * (worst[:, None] + worst[None, :])))
        rows.append({
            "route": route_id,
            "label": ROUTE_LABELS[route_id],
            "amplitude_weighted_bulk_symbol_error": bulk,
            "amplitude_weighted_offgrid_value_error": offgrid,
            "predicted_trace_magnitude": bulk + offgrid,
        })

    order = [row["route"] for row in
             sorted(rows, key=lambda item: item["predicted_trace_magnitude"])]
    return {
        "grid_used_for_the_prediction": npts,
        "delta_points": delta_points,
        "rows": rows,
        "predicted_trace_ordering_best_first": order,
        "predicted_champion_over_slc_trace_factor": float(
            [r for r in rows if r["route"] == "SLC-SL|common_L12"][0][
                "predicted_trace_magnitude"]
            / [r for r in rows if r["route"] == CHAMPION_ID][0][
                "predicted_trace_magnitude"]
        ),
    }


def write_prediction(harness: Harness) -> dict:
    payload = {
        "producer": "code/analysis/exponential_family/gate5c_ibm.py",
        "written_utc": now(),
        "part": "5c",
        "contract": {
            "path": "results/exponential_family/gate5/contract.md",
            "sha256": sha256(CONTRACT),
        },
        "routes": list(ROUTE_IDS),
        "basis": (
            "Section 5a coupled off-grid symbols: the value-channel transfer "
            "residual of each route's own evaluator, and the composed "
            "divergence-gradient Helmholtz symbol residual of each route's own "
            "nodal first derivative, both weighted by the modal amplitudes of "
            "the same 768-square Fourier reference restricted to the N = 96 grid."
        ),
        "prediction": predict(harness),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prediction.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ----------------------------------------------------------------------
# the matrix
# ----------------------------------------------------------------------

def run_matrix_cells(harness: Harness) -> list[dict]:
    cells = []
    for npts in GRIDS:
        markers_per_obstacle = max(48, npts // 2)
        for shift_id, sx, sy in SHIFTS:
            case = rm.prepare_case(harness.reference, npts, shift_id, (sx, sy),
                                   markers_per_obstacle)
            if case.reference_translation_residual > 1e-12:
                raise RuntimeError(
                    f"reference translation residual failed: N={npts} {shift_id}"
                )
            for route_id in ROUTE_IDS:
                route, evaluator = harness.entry(route_id)
                result = rm.timed_cell(route, evaluator, case, harness.q_table,
                                       harness.certificate, REPEATS)
                curves, recomputed = per_curve_trace(route, evaluator, case,
                                                     harness.q_table)
                if recomputed != result["trace_error"]:
                    raise RuntimeError(
                        f"per-curve pass diverged from the certified trace error "
                        f"for {route_id} N={npts} {shift_id}"
                    )
                cell = {
                    "route": route_id,
                    "label": ROUTE_LABELS[route_id],
                    "N": npts,
                    "h": case.h,
                    "shift_id": shift_id,
                    "shift_cell_units": [sx, sy],
                    "n_markers_per_obstacle": markers_per_obstacle,
                    "n_markers_total": int(case.markers.shape[0]),
                    "result": result,
                    "per_curve_trace_error": curves,
                    "shared_case_hashes": case.hashes,
                }
                cells.append(cell)
                print(
                    f"[5c] N={npts} {shift_id} {route_id:20s} "
                    f"trace={result['trace_error']:.6e} "
                    f"exterior={result['exterior_relative_l2']:.6e} "
                    f"t={result['total_time_seconds']:.4f}s "
                    f"invariants={'pass' if result['invariants_pass'] else 'FAIL'}",
                    flush=True,
                )
    return cells


def cell_index(cells: list[dict]) -> dict:
    return {(c["route"], c["N"], c["shift_id"]): c for c in cells}


def factor_receipt(index: dict, champion: str, competitor: str, shift: str,
                   metric_name: str, threshold: float) -> dict:
    """Apply the certified factor rule of reduce_results.factor_receipt."""
    error_key, floor_key = METRICS[metric_name]
    admissible = []
    for npts in reversed(GRIDS):
        b = index[(champion, npts, shift)]["result"]
        c = index[(competitor, npts, shift)]["result"]
        if b[error_key] > 10.0 * b[floor_key] and c[error_key] > 10.0 * c[floor_key]:
            admissible.append(npts)
    if admissible:
        npts = admissible[0]
        bcell = index[(champion, npts, shift)]
        ccell = index[(competitor, npts, shift)]
        factor = ccell["result"][error_key] / bcell["result"][error_key]
        status = "exact_pass" if factor >= threshold else "exact_fail"
        lower_bound = None
        decision_factor = factor
    else:
        npts = GRIDS[-1]
        bcell = index[(champion, npts, shift)]
        ccell = index[(competitor, npts, shift)]
        b = bcell["result"]
        c = ccell["result"]
        if c[error_key] > 10.0 * c[floor_key] and b[error_key] <= 10.0 * b[floor_key]:
            lower_bound = c[error_key] / (10.0 * b[floor_key])
            if lower_bound >= threshold:
                status = "lower_bound_pass"
                decision_factor = lower_bound
            else:
                status = "indeterminate_floor"
                decision_factor = None
        else:
            lower_bound = None
            status = "indeterminate_floor"
            decision_factor = None
        factor = c[error_key] / b[error_key]
    return {
        "champion_route": champion,
        "competitor_route": competitor,
        "shift_id": shift,
        "metric": metric_name,
        "factor_threshold": threshold,
        "N": npts,
        "status": status,
        "observed_factor": float(factor),
        "reportable_lower_bound": None if lower_bound is None else float(lower_bound),
        "decision_factor": None if decision_factor is None else float(decision_factor),
        "champion_error": bcell["result"][error_key],
        "champion_floor": bcell["result"][floor_key],
        "competitor_error": ccell["result"][error_key],
        "competitor_floor": ccell["result"][floor_key],
        "champion_total_time_seconds": bcell["result"]["total_time_seconds"],
        "competitor_total_time_seconds": ccell["result"]["total_time_seconds"],
        "champion_over_competitor_time_ratio": (
            bcell["result"]["total_time_seconds"]
            / ccell["result"]["total_time_seconds"]
        ),
        "champion_setup_seconds": bcell["result"]["timings"][SETUP_KEY],
        "competitor_setup_seconds": ccell["result"]["timings"][SETUP_KEY],
        "champion_over_competitor_setup_ratio": (
            bcell["result"]["timings"][SETUP_KEY]
            / ccell["result"]["timings"][SETUP_KEY]
        ),
        "champion_over_competitor_recurring_ratio": (
            (bcell["result"]["total_time_seconds"]
             - bcell["result"]["timings"][SETUP_KEY])
            / (ccell["result"]["total_time_seconds"]
               - ccell["result"]["timings"][SETUP_KEY])
        ),
    }


def metric_block(index: dict, champion: str, competitor: str,
                 metric_name: str, threshold: float) -> dict:
    comparisons = [
        factor_receipt(index, champion, competitor, shift_id, metric_name, threshold)
        for shift_id, _, _ in SHIFTS
    ]
    exact = [c for c in comparisons if c["status"].startswith("exact_")]
    exact_failures = [c for c in comparisons if c["status"] == "exact_fail"]
    lower_bound_passes = [c for c in comparisons if c["status"] == "lower_bound_pass"]
    indeterminate = [c for c in comparisons if c["status"] == "indeterminate_floor"]
    resolved = exact + lower_bound_passes
    all_pass = (not exact_failures and not indeterminate
                and len(resolved) == len(comparisons))
    strongest = (min(resolved, key=lambda item: item["decision_factor"])
                 if not indeterminate and resolved else None)
    return {
        "metric": metric_name,
        "factor_threshold": threshold,
        "passes_factor_requirement": bool(all_pass),
        "worst_shift_blocker": strongest,
        "minimum_exact_factor": (min(c["observed_factor"] for c in exact)
                                 if exact else None),
        "minimum_decision_factor": (min(c["decision_factor"] for c in resolved)
                                    if resolved else None),
        "exact_comparison_count": len(exact),
        "exact_failure_count": len(exact_failures),
        "lower_bound_pass_count": len(lower_bound_passes),
        "indeterminate_comparison_count": len(indeterminate),
        "comparisons": comparisons,
    }


def premium_stability(harness: Harness, blockers: list[dict],
                      replicates: int = 7) -> dict:
    """Re-time each blocker cell as paired replicates of the certified cell.

    The matrix cell already reports the median of three complete charged
    solves. This receipt repeats that whole estimator, alternating which route
    runs first so a warm-cache bias cannot favour one route, and reports the
    spread of the resulting one-solve ratio.
    """
    competitor = "SLC-SL|common_L12"
    receipts = {}
    for blocker in blockers:
        key = f"{blocker['metric']}@N{blocker['N']}__{blocker['shift_id']}"
        if key in receipts:
            continue
        shift = next(s for s in SHIFTS if s[0] == blocker["shift_id"])
        case = rm.prepare_case(harness.reference, blocker["N"], shift[0],
                               (shift[1], shift[2]),
                               max(48, blocker["N"] // 2))
        totals = {CHAMPION_ID: [], competitor: []}
        setups = {CHAMPION_ID: [], competitor: []}
        for replicate in range(replicates):
            order = ((CHAMPION_ID, competitor) if replicate % 2 == 0
                     else (competitor, CHAMPION_ID))
            for route_id in order:
                route, evaluator = harness.entry(route_id)
                result = rm.timed_cell(route, evaluator, case, harness.q_table,
                                       harness.certificate, REPEATS)
                totals[route_id].append(result["total_time_seconds"])
                setups[route_id].append(result["timings"][SETUP_KEY])
        ratios = [a / b for a, b in zip(totals[CHAMPION_ID], totals[competitor])]
        receipts[key] = {
            "metric": blocker["metric"],
            "N": blocker["N"],
            "shift_id": blocker["shift_id"],
            "replicates": replicates,
            "matrix_cell_ratio": blocker["champion_over_competitor_time_ratio"],
            "paired_ratios": [float(v) for v in ratios],
            "ratio_median": float(np.median(ratios)),
            "ratio_min": float(min(ratios)),
            "ratio_max": float(max(ratios)),
            "champion_total_median_seconds": float(np.median(totals[CHAMPION_ID])),
            "competitor_total_median_seconds": float(np.median(totals[competitor])),
            "champion_setup_median_seconds": float(np.median(setups[CHAMPION_ID])),
            "competitor_setup_median_seconds": float(np.median(setups[competitor])),
            "all_replicates_below_premium_limit": bool(max(ratios) < PREMIUM_LIMIT),
            "all_replicates_above_premium_limit": bool(min(ratios) >= PREMIUM_LIMIT),
        }
    pooled = [value for receipt in receipts.values()
              for value in receipt["paired_ratios"]]
    return {
        "premium_limit_time_ratio": PREMIUM_LIMIT,
        "blocker_cells": receipts,
        "pooled_ratio_median": float(np.median(pooled)),
        "pooled_ratio_min": float(min(pooled)),
        "pooled_ratio_max": float(max(pooled)),
        "premium_separable_from_the_limit": bool(
            max(pooled) < PREMIUM_LIMIT or min(pooled) >= PREMIUM_LIMIT
        ),
        "verdict_if_the_rule_used_the_pooled_median": (
            "TRIUMPH" if float(np.median(pooled)) < PREMIUM_LIMIT else "PARTIAL"
        ),
    }


def decide(cells: list[dict], prediction: dict, harness: Harness | None = None,
           stability: dict | None = None) -> dict:
    index = cell_index(cells)
    competitor = "SLC-SL|common_L12"
    trace = metric_block(index, CHAMPION_ID, competitor, "wall_trace",
                         TRACE_FACTOR_THRESHOLD)
    exterior = metric_block(index, CHAMPION_ID, competitor,
                            "exterior_relative_l2", EXTERIOR_FACTOR_THRESHOLD)
    control_trace = metric_block(index, "BMC9", "C10|common_L12", "wall_trace",
                                 TRACE_FACTOR_THRESHOLD)
    control_exterior = metric_block(index, "BMC9", "C10|common_L12",
                                    "exterior_relative_l2",
                                    EXTERIOR_FACTOR_THRESHOLD)

    factors_hold = bool(trace["passes_factor_requirement"]
                        and exterior["passes_factor_requirement"])
    premiums = []
    for block in (trace, exterior):
        blocker = block["worst_shift_blocker"]
        if blocker is not None:
            premiums.append(blocker["champion_over_competitor_time_ratio"])
    premium_ok = bool(premiums and max(premiums) < PREMIUM_LIMIT)
    blockers = [block["worst_shift_blocker"] for block in (trace, exterior)
                if block["worst_shift_blocker"] is not None]
    if stability is None and harness is not None and blockers:
        stability = premium_stability(harness, blockers)
    matrix_verdict = ("TRIUMPH" if factors_hold and premium_ok
                      else "PARTIAL" if factors_hold else "REJECT")
    # The rule requires the premium to be below 100 percent. A ratio that the
    # paired replicates do not separate from the limit has not met that
    # requirement, so the recorded verdict uses the replicate receipt when one
    # exists and the matrix cells alone otherwise.
    if stability is None:
        premium_established = premium_ok
    else:
        premium_established = bool(
            premium_ok and stability["pooled_ratio_max"] < PREMIUM_LIMIT
        )
    if factors_hold and premium_established:
        verdict = "TRIUMPH"
    elif factors_hold:
        verdict = "PARTIAL"
    else:
        verdict = "REJECT"

    realized_order = sorted(
        ROUTE_IDS,
        key=lambda rid: max(
            index[(rid, 96, shift_id)]["result"]["trace_error"]
            for shift_id, _, _ in SHIFTS
        ),
    )
    return {
        "producer": "code/analysis/exponential_family/gate5c_ibm.py",
        "written_utc": now(),
        "part": "5c",
        "contract": {
            "path": "results/exponential_family/gate5/contract.md",
            "sha256": sha256(CONTRACT),
        },
        "rule": (
            "The champion must reduce both the worst-shift off-marker wall-trace "
            "error and the exterior relative l2 error against SLC q4 plus the "
            "common L12 evaluator by at least the factors the paper reports for "
            "BMC9 against C10 plus common L12, 62.707527291670914 and "
            "32.16591996072041, at a complete one-solve premium below 100 "
            "percent. PARTIAL if the error factors hold but the premium exceeds "
            "100 percent. REJECT otherwise."
        ),
        "verdict": verdict,
        "verdict_from_the_admitted_matrix_cells_alone": matrix_verdict,
        "factors_hold": factors_hold,
        "premium_below_100_percent_in_the_matrix_cells": premium_ok,
        "premium_established_below_100_percent": premium_established,
        "premium_limit_time_ratio": PREMIUM_LIMIT,
        "observed_blocker_time_ratios": premiums,
        "premium_margin_receipt": {
            "note": (
                "The rule is applied verbatim to the admitted matrix cells, "
                "whose one-solve times are already the median of three complete "
                "charged solves. The margin against the 100 percent limit is "
                "small, so a paired timing-replicate receipt is recorded beside "
                "it and the verdict is not claimed to be separable from the "
                "limit unless that receipt says so."
            ),
            "verdict_robust_to_timing_replicates": bool(
                stability is not None
                and stability["premium_separable_from_the_limit"]
            ),
            "stability": stability,
        },
        "champion_versus_slc_q4_plus_L12": {
            "wall_trace": trace,
            "exterior_relative_l2": exterior,
        },
        "control_bmc9_versus_c10_plus_L12": {
            "wall_trace": control_trace,
            "exterior_relative_l2": control_exterior,
        },
        "paper_reference_factors": {
            "wall_trace": TRACE_FACTOR_THRESHOLD,
            "exterior_relative_l2": EXTERIOR_FACTOR_THRESHOLD,
            "total_time_ratios": [1.699800439571433, 1.7582010277920994],
        },
        "prediction_before_runs": True,
        "predicted_trace_ordering_best_first":
            prediction["prediction"]["predicted_trace_ordering_best_first"],
        "realized_trace_ordering_best_first_at_N96_worst_shift": realized_order,
        "prediction_ordering_correct": bool(
            realized_order
            == prediction["prediction"]["predicted_trace_ordering_best_first"]
        ),
        "invariants_all_pass": bool(
            all(c["result"]["invariants_pass"] for c in cells)
        ),
    }


# ----------------------------------------------------------------------
# receipts
# ----------------------------------------------------------------------

CSV_COLUMNS = [
    "route", "label", "N", "h", "shift_id", "shift_sx", "shift_sy",
    "n_markers_per_obstacle", "n_markers_total",
    "trace_error", "trace_error_disk", "trace_error_trefoil", "trace_error_kite",
    "exterior_relative_l2", "trace_numerical_floor",
    "exterior_relative_l2_numerical_floor",
    "trace_above_ten_times_floor", "exterior_above_ten_times_floor",
    "marker_constraint_residual", "schur_condition",
    "weighted_adjoint_residual", "fielded_constant_residual", "invariants_pass",
    "total_time_seconds", "setup_seconds", "recurring_seconds",
    "bulk_solve_seconds", "schur_assembly_seconds",
    "schur_solve_and_refinement_seconds", "reconstruction_seconds",
    "marker_evaluation_seconds", "off_marker_seconds", "exterior_error_seconds",
]


def write_runs_csv(cells: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for cell in cells:
            result = cell["result"]
            timings = result["timings"]
            setup = timings[SETUP_KEY]
            writer.writerow({
                "route": cell["route"],
                "label": cell["label"],
                "N": cell["N"],
                "h": f"{cell['h']:.17g}",
                "shift_id": cell["shift_id"],
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
                "trace_above_ten_times_floor": int(result["trace_above_ten_times_floor"]),
                "exterior_above_ten_times_floor":
                    int(result["exterior_above_ten_times_floor"]),
                "marker_constraint_residual":
                    f"{result['marker_constraint_residual']:.17g}",
                "schur_condition": f"{result['schur_condition']:.17g}",
                "weighted_adjoint_residual": f"{result['weighted_adjoint_residual']:.17g}",
                "fielded_constant_residual":
                    f"{result['fielded_constant_residual']:.17g}",
                "invariants_pass": int(result["invariants_pass"]),
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
            })


def write_figure(cells: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    index = cell_index(cells)
    colors = {CHAMPION_ID: "#2f6d8c", "BMC9": "#7a2d2d",
              "C10|common_L12": "#e0a95e", "SLC-SL|common_L12": "#4c8055"}
    markers = {CHAMPION_ID: "o", "BMC9": "s", "C10|common_L12": "^",
               "SLC-SL|common_L12": "v"}

    from matplotlib.ticker import NullFormatter, NullLocator

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4), constrained_layout=True)
    panels = [(name, f"{name}: worst-shift wall trace") for name in CURVES]
    panels.append(("__exterior__", "exterior relative $\\ell^2$ (all curves)"))
    handles = None
    for ax, (key, title) in zip(axes, panels):
        for route_id in ROUTE_IDS:
            values = []
            for npts in GRIDS:
                if key == "__exterior__":
                    values.append(max(
                        index[(route_id, npts, shift_id)]["result"][
                            "exterior_relative_l2"]
                        for shift_id, _, _ in SHIFTS))
                else:
                    values.append(max(
                        index[(route_id, npts, shift_id)][
                            "per_curve_trace_error"][key]
                        for shift_id, _, _ in SHIFTS))
            ax.loglog(GRIDS, values, marker=markers[route_id],
                      color=colors[route_id], lw=1.5, ms=5.0,
                      label=ROUTE_LABELS[route_id])
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("$N$")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(list(GRIDS))
        ax.set_xticklabels([str(n) for n in GRIDS])
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
        if handles is None:
            handles, _ = ax.get_legend_handles_labels()
    axes[0].set_ylabel("worst-shift error")
    fig.legend([ROUTE_LABELS[r] for r in ROUTE_IDS] and handles,
               [ROUTE_LABELS[r] for r in ROUTE_IDS],
               loc="lower center", ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Gate 5c: immersed smooth wall, worst shift over the four registered "
        "cell-unit shifts", fontsize=12.0)
    fig.savefig(FIGURES / "fig_wall.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("predict", "run", "reduce"),
                        required=True)
    args = parser.parse_args()

    install_hook()
    harness = Harness()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "predict":
        payload = write_prediction(harness)
        print(json.dumps(payload["prediction"], indent=2))
        return

    prediction = json.loads((OUT / "prediction.json").read_text())
    if args.stage == "run":
        cells = run_matrix_cells(harness)
        (OUT / "cells.json").write_text(
            json.dumps(cells, indent=2, default=float, sort_keys=True) + "\n"
        )
    else:
        cells = json.loads((OUT / "cells.json").read_text())
    write_runs_csv(cells)
    decision = decide(cells, prediction, harness)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    write_figure(cells)
    print(json.dumps({k: v for k, v in decision.items()
                      if k not in ("champion_versus_slc_q4_plus_L12",
                                   "control_bmc9_versus_c10_plus_L12")},
                     indent=2))


if __name__ == "__main__":
    main()
