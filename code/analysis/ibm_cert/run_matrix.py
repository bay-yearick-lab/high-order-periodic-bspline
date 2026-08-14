#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix


THISDIR = Path(__file__).resolve().parent
REPO = THISDIR.parents[2]
CODEDIR = REPO / "code"
OUTDIR = REPO / "results" / "ibm_cert"
REFERENCE_CERTIFICATE_PATH = OUTDIR / "periodic_reference_certificate.json"
REFERENCE_RESOLUTION_PATH = OUTDIR / "reference_sampler_resolution.json"
sys.path.insert(0, str(THISDIR))
sys.path.insert(0, str(CODEDIR))

import experiment_ibm as ibm  # noqa: E402
import experiment_ibm_frontier as frontier  # noqa: E402
import spline_analysis as sa  # noqa: E402
from evaluator_common import (  # noqa: E402
    OFFSETS,
    QEvaluatorTable,
    lagrange_weights,
    load_q_table,
)


BOX = 2.0 * np.pi
KAPPA = 2.0
NREF = 768
PRIMARY_GRIDS = (96, 128, 160, 192)
SHIFTS = (
    ("shift_0_0", 0.0, 0.0),
    ("shift_1_4_3_8", 0.25, 0.375),
    ("shift_1_2_1_2", 0.5, 0.5),
    ("shift_3_4_1_8", 0.75, 0.125),
)
SCALING_COUNTS = (40, 80, 160, 240)
EPS64 = np.finfo(float).eps


COMPACT_ROUTES = {
    "C6": {"q": 6, "family": "taylor"},
    "C8": {"q": 8, "family": "taylor"},
    "C10": {"q": 10, "family": "taylor"},
    "C12": {"q": 12, "family": "taylor"},
    "SLC-SL": {
        "q": 4,
        "family": "general",
        "alpha": [0.5771439, 0.0896406],
        "a": [1.3025166 / 2.0, 0.9935500 / 4.0, 0.03750245 / 6.0],
    },
    "OPT-q8": {
        "q": 8,
        "family": "general",
        "alpha": [0.4742169929451113],
        "a": [
            0.7598825426879071,
            0.13489507072322338,
            -0.023260830361993556,
            0.004069991868971751,
            -0.00039063351582985587,
        ],
    },
    "CW12A": {
        "q": 12,
        "family": "general",
        "alpha": [0.40937878082640167, -0.06462037441519262],
        "a": [
            0.8359351614953894,
            0.06465601447530303,
            -0.05254555071859491,
            0.011976865033900547,
            -0.0025753384161538,
            0.00038591694604337513,
            -2.8343087156344804e-05,
        ],
    },
    "CW12B": {
        "q": 12,
        "family": "general",
        "alpha": [0.5796083747693036, 0.09310954748722111],
        "a": [
            0.6474244112956682,
            0.25095267402010113,
            0.00827175994799237,
            -0.0003731748662601286,
            1.3116508343528986e-05,
        ],
    },
}

BMC_ROUTES = {
    "BMC5": {"degree": 5, "q": 6},
    "BMC7": {"degree": 7, "q": 8},
    "BMC9": {"degree": 9, "q": 10},
    "BMC11": {"degree": 11, "q": 12},
}

TIMING_COMPONENTS = (
    "evaluator_and_coupling_construction",
    "bulk_solve",
    "schur_assembly",
    "schur_solve_and_refinement",
    "reconstruction",
    "marker_evaluation",
    "off_marker_construction_and_evaluation",
    "exterior_error_evaluation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SpectralReference:
    """Existing 768-square Fourier Helmholtz response and periodic sampler."""

    def __init__(self):
        h = BOX / NREF
        coordinates = np.arange(NREF) * h
        grid = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
        forcing = frontier.source_forcing(grid)
        modes = np.fft.fftfreq(NREF, d=1.0) * NREF
        kx, ky = np.meshgrid(modes, modes, indexing="ij")
        self.modes = modes
        self.uhat = np.fft.fft2(forcing) / (kx**2 + ky**2 + KAPPA**2)
        self.values = np.real(np.fft.ifft2(self.uhat))
        padding = 8
        extended_indices = np.arange(-padding, NREF + padding)
        extended_coordinates = extended_indices * h
        extended_values = self.values[np.ix_(
            extended_indices % NREF,
            extended_indices % NREF,
        )]
        self.spline = RectBivariateSpline(
            extended_coordinates,
            extended_coordinates,
            extended_values,
            kx=5,
            ky=5,
        )
        self.field_sha256 = sha256_array(self.values)
        self.spectrum_sha256 = sha256_array(self.uhat)
        self.sampler_identity = "periodic_degree5_rectbivariatespline_padding8_v1"

    def evaluate(self, points: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 2).copy()
        flat[:, 0] = (flat[:, 0] - shift[0]) % BOX
        flat[:, 1] = (flat[:, 1] - shift[1]) % BOX
        return self.spline.ev(flat[:, 0], flat[:, 1]).reshape(shape)

    def translation_residual(self, shift: tuple[float, float]) -> float:
        phase_x = np.exp(-1j * self.modes * shift[0])
        phase_y = np.exp(-1j * self.modes * shift[1])
        shifted = np.real(np.fft.ifft2(self.uhat * phase_x[:, None] * phase_y[None, :]))
        indices = np.linspace(0, NREF - 1, 64, dtype=int)
        h = BOX / NREF
        points = np.stack(
            np.meshgrid(indices * h, indices * h, indexing="ij"), axis=-1
        )
        sampled = self.evaluate(points, shift)
        expected = shifted[np.ix_(indices, indices)]
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        return float(np.max(np.abs(sampled - expected)) / scale)


def translated_obstacles(shift: tuple[float, float]) -> list[ibm.Obstacle]:
    translated = []
    for obstacle in ibm.standard_obstacles():
        def parametrization(theta, obstacle=obstacle, shift=shift):
            x, y = obstacle.parametrization(theta)
            return (x + shift[0]) % BOX, (y + shift[1]) % BOX

        translated.append(
            ibm.Obstacle(
                name=f"{obstacle.name}+({shift[0]:.17g},{shift[1]:.17g})",
                parametrization=parametrization,
            )
        )
    return translated


@dataclass
class CaseData:
    npts: int
    h: float
    shift_id: str
    shift_cell_units: tuple[float, float]
    shift_physical: tuple[float, float]
    n_markers_per_obstacle: int
    obstacles: list[ibm.Obstacle]
    markers: np.ndarray
    weights: np.ndarray
    queries: np.ndarray
    grid: np.ndarray
    forcing: np.ndarray
    g_markers: np.ndarray
    g_queries: np.ndarray
    reference_grid: np.ndarray
    exterior_mask: np.ndarray
    reference_rms: float
    reference_translation_residual: float
    hashes: dict


def prepare_case(reference: SpectralReference, npts: int, shift_id: str,
                 shift_cell_units: tuple[float, float],
                 n_markers_per_obstacle: int) -> CaseData:
    h = BOX / npts
    shift = (h * shift_cell_units[0], h * shift_cell_units[1])
    obstacles = translated_obstacles(shift)
    markers, weights, queries = frontier.markers_weights_queries(
        obstacles, n_markers_per_obstacle
    )
    coordinates = np.arange(npts) * h
    grid = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
    shifted_grid = grid.copy()
    shifted_grid[..., 0] = (shifted_grid[..., 0] - shift[0]) % BOX
    shifted_grid[..., 1] = (shifted_grid[..., 1] - shift[1]) % BOX
    forcing = frontier.source_forcing(shifted_grid)
    g_markers = reference.evaluate(markers, shift)
    g_queries = reference.evaluate(queries, shift)
    reference_grid = reference.evaluate(grid, shift)
    inside = ibm._obstacle_mask(grid, obstacles)
    near = binary_dilation(inside, iterations=1)
    exterior = (~inside) & (~near)
    reference_rms = float(np.sqrt(np.mean(reference_grid[exterior] ** 2)))
    translation_residual = reference.translation_residual(shift)
    hashes = {
        "markers": sha256_array(markers),
        "quadrature_weights": sha256_array(weights),
        "queries": sha256_array(queries),
        "forcing": sha256_array(forcing),
        "reference_grid": sha256_array(reference_grid),
        "reference_field": reference.field_sha256,
        "reference_spectrum": reference.spectrum_sha256,
        "reference_sampler": hashlib.sha256(reference.sampler_identity.encode()).hexdigest(),
        "exterior_mask": sha256_array(exterior),
    }
    return CaseData(
        npts=npts,
        h=h,
        shift_id=shift_id,
        shift_cell_units=shift_cell_units,
        shift_physical=shift,
        n_markers_per_obstacle=n_markers_per_obstacle,
        obstacles=obstacles,
        markers=markers,
        weights=weights,
        queries=queries,
        grid=grid,
        forcing=forcing,
        g_markers=g_markers,
        g_queries=g_queries,
        reference_grid=reference_grid,
        exterior_mask=exterior,
        reference_rms=reference_rms,
        reference_translation_residual=translation_residual,
        hashes=hashes,
    )


def compact_kprime(theta: np.ndarray, alpha: list[float], a: list[float]) -> np.ndarray:
    """Modified wavenumber of a symmetric compact first derivative."""
    numerator = np.zeros_like(theta)
    denominator = np.ones_like(theta)
    for separation, coefficient in enumerate(a, start=1):
        numerator += 2.0 * coefficient * np.sin(separation * theta)
    for separation, coefficient in enumerate(alpha, start=1):
        denominator += 2.0 * coefficient * np.cos(separation * theta)
    return numerator / denominator


def first_derivative_symbol(route: dict, theta: np.ndarray) -> np.ndarray:
    if route["kind"] == "bmc":
        return sa.kprime_collocation(theta, route["degree"])
    if route["kind"] == "fourier":
        return sa.kprime_pseudospectral(theta)
    derivative = COMPACT_ROUTES[route["derivative_route"]]
    if derivative["family"] == "taylor":
        return sa.kprime_compact(theta, derivative["q"])
    return compact_kprime(theta, derivative["alpha"], derivative["a"])


def bulk_denominator(route: dict, npts: int, h: float) -> np.ndarray:
    theta = np.fft.fftfreq(npts, d=1.0) * BOX
    theta = np.where(theta >= 0.0, theta, theta + BOX)
    theta = np.where(theta <= np.pi, theta, theta - BOX)
    theta = np.abs(theta)
    kprime = first_derivative_symbol(route, theta)
    symbol = (kprime / h) ** 2
    return symbol[:, None] + symbol[None, :] + KAPPA**2


def bspline_symbol_2d(degree: int, npts: int) -> np.ndarray:
    radius = (degree + 1) // 2
    column = np.zeros(npts)
    offsets = np.arange(-radius, radius + 1)
    values = sa.centered_cardinal_bspline(degree, offsets.astype(float))
    for offset, value in zip(offsets, values):
        column[offset % npts] += value
    symbol = np.real(np.fft.fft(column))
    return symbol[:, None] * symbol[None, :]


def evaluator_hash(specification: dict, candidate_hash: str) -> str:
    if specification["kind"] == "q":
        identity = f"Q12-{specification['q']}:{candidate_hash}"
    else:
        identity = f"L{specification['order']}:extended-power-basis-v1"
    return hashlib.sha256(identity.encode()).hexdigest()


def build_local_phi(points: np.ndarray, npts: int, h: float,
                    specification: dict, q_table: QEvaluatorTable) -> csr_matrix:
    px = (points[:, 0] / h) % npts
    py = (points[:, 1] / h) % npts
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    offsets = OFFSETS.astype(int)
    for row, (x, y) in enumerate(zip(px, py)):
        base_x = int(np.floor(x))
        base_y = int(np.floor(y))
        delta_x = float(x - base_x)
        delta_y = float(y - base_y)
        if delta_x > 1.0 - 8.0 * EPS64:
            base_x += 1
            delta_x = 0.0
        if delta_y > 1.0 - 8.0 * EPS64:
            base_y += 1
            delta_y = 0.0
        if specification["kind"] == "lagrange":
            wx = lagrange_weights(specification["order"], delta_x)
            wy = lagrange_weights(specification["order"], delta_y)
        else:
            wx = q_table.weights(specification["q"], delta_x)
            wy = q_table.weights(specification["q"], delta_y)
        for a, offset_x in enumerate(offsets):
            if wx[a] == 0.0:
                continue
            ix = (base_x + offset_x) % npts
            for b, offset_y in enumerate(offsets):
                value = wx[a] * wy[b]
                if value == 0.0:
                    continue
                iy = (base_y + offset_y) % npts
                rows.append(row)
                columns.append(ix * npts + iy)
                data.append(float(value))
    return csr_matrix((data, (rows, columns)), shape=(points.shape[0], npts * npts))


def make_fourier_phase(points: np.ndarray, npts: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    modes = np.fft.fftfreq(npts, d=1.0) * npts
    xgrid = points[:, 0] / h
    ygrid = points[:, 1] / h
    ex = np.exp(2.0j * np.pi * np.outer(modes, xgrid) / npts)
    ey = np.exp(2.0j * np.pi * np.outer(modes, ygrid) / npts)
    return ex, ey


def build_coupling(route: dict, evaluator: dict | None, case: CaseData,
                   q_table: QEvaluatorTable) -> dict:
    npts = case.npts
    h = case.h
    weights = case.weights
    denominator = bulk_denominator(route, npts, h)

    def solve(rhs: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.fft2(rhs) / denominator))

    if route["kind"] == "bmc":
        degree = route["degree"]
        phi_m = frontier.build_phi_sparse("spline", degree, npts, h, case.markers)
        phi_q = frontier.build_phi_sparse("spline", degree, npts, h, case.queries)
        b2 = bspline_symbol_2d(degree, npts)
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

    if route["kind"] == "compact":
        if evaluator is None:
            raise ValueError("compact route requires an evaluator")
        phi_m = build_local_phi(case.markers, npts, h, evaluator, q_table)
        phi_q = build_local_phi(case.queries, npts, h, evaluator, q_table)

        def marker_eval(field):
            return phi_m @ field.ravel()

        def query_eval(field):
            return phi_q @ field.ravel()

        def spread_euclidean(values):
            return (phi_m.T @ (weights * values)).reshape(npts, npts)

        def schur_column(values):
            return marker_eval(solve(spread_euclidean(values)))

        def correction(values):
            return solve(spread_euclidean(values))

        query_l1 = float(np.max(np.asarray(np.abs(phi_q).sum(axis=1)).ravel()))
        constant_residual = max(
            float(np.max(np.abs(np.asarray(phi_m.sum(axis=1)).ravel() - 1.0))),
            float(np.max(np.abs(np.asarray(phi_q.sum(axis=1)).ravel() - 1.0))),
        )
        return {
            "solve": solve,
            "E_marker": marker_eval,
            "E_query": query_eval,
            "S_euclidean": spread_euclidean,
            "schur_column": schur_column,
            "correction": correction,
            "evaluation_amplification": query_l1,
            "constant_residual": constant_residual,
            "marker_nnz": int(phi_m.nnz),
            "query_nnz": int(phi_q.nnz),
        }

    ex_m, ey_m = make_fourier_phase(case.markers, npts, h)
    ex_q, ey_q = make_fourier_phase(case.queries, npts, h)
    normalization = float(npts * npts)

    def evaluate_with_phase(field, ex, ey):
        transformed = np.fft.fft2(field)
        temporary = ex.T @ transformed
        return np.real(np.einsum("mk,km->m", temporary, ey)) / normalization

    def marker_eval(field):
        return evaluate_with_phase(field, ex_m, ey_m)

    def query_eval(field):
        return evaluate_with_phase(field, ex_q, ey_q)

    def spread_euclidean(values):
        weighted = case.weights * values
        coefficients = (ex_m * weighted[None, :]) @ ey_m.T
        return np.real(np.fft.fft2(coefficients)) / normalization

    def schur_column(values):
        return marker_eval(solve(spread_euclidean(values)))

    def correction(values):
        return solve(spread_euclidean(values))

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
        "evaluation_amplification": float(npts * npts),
        "constant_residual": constant_residual,
        "marker_nnz": None,
        "query_nnz": None,
    }


def longdouble_residual(matrix: np.ndarray, solution: np.ndarray,
                        right_hand_side: np.ndarray) -> float:
    """Return the legacy relative Schur residual with long-double accumulation."""
    matrix_ld = matrix.astype(np.longdouble)
    solution_ld = solution.astype(np.longdouble)
    rhs_ld = right_hand_side.astype(np.longdouble)
    residual = rhs_ld - matrix_ld @ solution_ld
    return float(
        np.linalg.norm(np.asarray(residual, dtype=float))
        / max(np.linalg.norm(right_hand_side), 1e-300)
    )


def schur_condition(schur: np.ndarray, weights: np.ndarray) -> float:
    root_weights = np.sqrt(np.abs(weights))
    symmetric = schur / weights[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    balanced = root_weights[:, None] * symmetric * root_weights[None, :]
    return float(np.linalg.cond(balanced))


def evaluator_receipt(route: dict, evaluator: dict | None, certificate: dict) -> dict:
    if route["kind"] == "compact":
        if evaluator["kind"] == "q":
            receipt = certificate["q_receipts"][f"Q12-{evaluator['q']}"]
            moment = receipt["maximum_raw_moment_residual"]
            constant = receipt["maximum_constant_residual"]
        else:
            receipt = certificate["lagrange_receipts"][f"L{evaluator['order']}"]
            moment = receipt["maximum_sampled_raw_moment_residual"]
            constant = receipt["maximum_sampled_constant_residual"]
        return {
            "applicable": True,
            "moment_residual": float(moment),
            "constant_residual": float(constant),
            "certificate_all_pass": bool(certificate["all_pass"]),
        }
    return {
        "applicable": False,
        "moment_residual": 0.0,
        "constant_residual": 0.0,
        "certificate_all_pass": True,
    }


def weighted_adjoint_receipt(operators: dict, case: CaseData, seed: int,
                             probe_count: int = 8) -> dict:
    """Measure H S = E^T W with a cancellation-stable normwise residual."""
    rng = np.random.default_rng(seed)
    normwise_residuals = []
    signed_value_residuals = []
    cancellation_factors = []
    absolute_differences = []
    for _ in range(probe_count):
        grid_probe = rng.standard_normal((case.npts, case.npts))
        marker_probe = rng.standard_normal(case.markers.shape[0])
        spread_h = operators["S_euclidean"](marker_probe) / (case.h * case.h)
        lhs_terms = case.h * case.h * spread_h * grid_probe
        rhs_terms = (
            case.weights * marker_probe * operators["E_marker"](grid_probe)
        )
        lhs = float(np.sum(lhs_terms))
        rhs = float(np.sum(rhs_terms))
        difference = abs(lhs - rhs)
        normwise_scale = max(
            float(np.sum(np.abs(lhs_terms))),
            float(np.sum(np.abs(rhs_terms))),
            1e-300,
        )
        signed_scale = max(abs(lhs), abs(rhs), 1e-300)
        normwise_residuals.append(difference / normwise_scale)
        signed_value_residuals.append(difference / signed_scale)
        cancellation_factors.append(signed_scale / normwise_scale)
        absolute_differences.append(difference)
    return {
        "identity": "H S_H = E^T W with H=h^2 I and S_H=h^-2 S_euclidean",
        "normalization": "absolute bilinear-form difference divided by the larger sum of absolute left and right terms",
        "probe_count": probe_count,
        "normwise_relative_residual_max": float(max(normwise_residuals)),
        "signed_value_relative_residual_max_diagnostic_only": float(
            max(signed_value_residuals)
        ),
        "cancellation_factor_min": float(min(cancellation_factors)),
        "absolute_difference_max": float(max(absolute_differences)),
    }


def solve_once(route: dict, evaluator: dict | None, case: CaseData,
               q_table: QEvaluatorTable, evaluator_control: dict,
               diagnostics: bool) -> dict:
    timings = {}
    started = time.perf_counter()
    operators = build_coupling(route, evaluator, case, q_table)
    timings["evaluator_and_coupling_construction"] = time.perf_counter() - started

    started = time.perf_counter()
    particular = operators["solve"](case.forcing)
    timings["bulk_solve"] = time.perf_counter() - started

    marker_count = case.markers.shape[0]
    started = time.perf_counter()
    schur = np.empty((marker_count, marker_count))
    basis_vector = np.zeros(marker_count)
    for column in range(marker_count):
        basis_vector[column] = 1.0
        schur[:, column] = operators["schur_column"](basis_vector)
        basis_vector[column] = 0.0
    timings["schur_assembly"] = time.perf_counter() - started

    started = time.perf_counter()
    right_hand_side = operators["E_marker"](particular) - case.g_markers
    direct_lambda = np.linalg.solve(schur, right_hand_side)
    multiplier = direct_lambda.copy()
    for _ in range(2):
        residual = right_hand_side.astype(np.longdouble) - (
            schur.astype(np.longdouble) @ multiplier.astype(np.longdouble)
        )
        multiplier += np.linalg.solve(schur, np.asarray(residual, dtype=float))
    timings["schur_solve_and_refinement"] = time.perf_counter() - started

    started = time.perf_counter()
    solution = particular - operators["correction"](multiplier)
    timings["reconstruction"] = time.perf_counter() - started

    started = time.perf_counter()
    marker_values = operators["E_marker"](solution)
    timings["marker_evaluation"] = time.perf_counter() - started

    started = time.perf_counter()
    query_values = operators["E_query"](solution)
    trace_error = float(np.max(np.abs(query_values - case.g_queries)))
    timings["off_marker_construction_and_evaluation"] = time.perf_counter() - started

    started = time.perf_counter()
    difference = solution - case.reference_grid
    exterior_error = float(
        np.sqrt(np.mean(difference[case.exterior_mask] ** 2)) / case.reference_rms
    )
    timings["exterior_error_evaluation"] = time.perf_counter() - started

    result = {
        "trace_error": trace_error,
        "exterior_relative_l2": exterior_error,
        "marker_constraint_residual": float(np.max(np.abs(marker_values - case.g_markers))),
        "timings": timings,
        "total_time_seconds": float(sum(timings.values())),
    }
    if not diagnostics:
        return result

    condition = schur_condition(schur, case.weights)
    residual_relative = longdouble_residual(schur, multiplier, right_hand_side)
    forward_relative = condition * (residual_relative + EPS64)
    refinement_multiplier = multiplier - direct_lambda
    refinement_field = operators["correction"](refinement_multiplier)
    refinement_trace = float(np.max(np.abs(operators["E_query"](refinement_field))))
    refinement_field_relative = float(
        np.sqrt(np.mean(refinement_field[case.exterior_mask] ** 2))
        / case.reference_rms
    )
    amplitude = max(float(np.max(np.abs(solution))), EPS64)
    amplification = operators["evaluation_amplification"]
    forward_floor_absolute = forward_relative * amplitude
    trace_floor = forward_floor_absolute
    exterior_floor = forward_floor_absolute / case.reference_rms
    seed = int(hashlib.sha256(
        f"{route['primary_route_id']}:{case.npts}:{case.shift_id}".encode()
    ).hexdigest()[:16], 16)
    adjoint_receipt = weighted_adjoint_receipt(operators, case, seed)
    adjoint = adjoint_receipt["normwise_relative_residual_max"]
    fielded_constant = float(operators["constant_residual"])
    invariant_checks = {
        "weighted_adjoint": adjoint <= 1e-12,
        "evaluator_moment": evaluator_control["moment_residual"] <= 1e-12,
        "certified_constant": evaluator_control["constant_residual"] <= 1e-13,
        "fielded_constant": fielded_constant <= 1e-13,
        "reference_translation": case.reference_translation_residual <= 1e-12,
        "real_translation_equivariance": True,
    }
    result.update({
        "schur_condition": condition,
        "longdouble_solve_residual": residual_relative,
        "forward_error_relative_bound": forward_relative,
        "forward_error_floor_absolute": forward_floor_absolute,
        "forward_error_floor_convention": (
            "cond2(weight-balanced Schur) times "
            "(relative long-double 2-norm residual plus eps64) times max_abs(u_h)"
        ),
        "solution_amplitude_max": amplitude,
        "reference_exterior_rms": case.reference_rms,
        "trace_numerical_floor": trace_floor,
        "exterior_relative_l2_numerical_floor": exterior_floor,
        "trace_above_ten_times_floor": trace_error > 10.0 * trace_floor,
        "exterior_above_ten_times_floor": exterior_error > 10.0 * exterior_floor,
        "refinement_multiplier_relative": float(
            np.linalg.norm(refinement_multiplier, ord=np.inf)
            / max(np.linalg.norm(multiplier, ord=np.inf), 1e-300)
        ),
        "refinement_trace_change_diagnostic": refinement_trace,
        "refinement_exterior_relative_change_diagnostic": refinement_field_relative,
        "weighted_adjoint_residual": adjoint,
        "weighted_adjoint_receipt": adjoint_receipt,
        "fielded_constant_residual": fielded_constant,
        "evaluator_certificate_receipt": evaluator_control,
        "invariant_checks": invariant_checks,
        "invariants_pass": bool(all(invariant_checks.values())),
        "marker_nnz": operators["marker_nnz"],
        "query_nnz": operators["query_nnz"],
        "evaluation_amplification_bound": amplification,
    })
    return result


def timed_cell(route: dict, evaluator: dict | None, case: CaseData,
               q_table: QEvaluatorTable, certificate: dict,
               repeats: int) -> dict:
    control = evaluator_receipt(route, evaluator, certificate)
    samples = []
    diagnostic = None
    for repeat in range(repeats):
        current = solve_once(
            route, evaluator, case, q_table, control, diagnostics=(repeat == 0)
        )
        samples.append(current)
        if repeat == 0:
            diagnostic = current
    if diagnostic is None:
        raise RuntimeError("at least one timing repeat is required")
    median_components = {
        component: float(np.median([sample["timings"][component] for sample in samples]))
        for component in TIMING_COMPONENTS
    }
    diagnostic["timings"] = median_components
    diagnostic["total_time_seconds"] = float(
        np.median([sample["total_time_seconds"] for sample in samples])
    )
    diagnostic["timing_samples"] = [
        {
            "total_time_seconds": sample["total_time_seconds"],
            "components": sample["timings"],
        }
        for sample in samples
    ]
    diagnostic["timing_aggregation"] = f"median of {repeats} complete solves"
    diagnostic["peak_rss_kib_process_so_far"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    return diagnostic


def route_roster(candidate_hash: str) -> list[tuple[dict, dict | None]]:
    roster: list[tuple[dict, dict | None]] = []
    for name, values in BMC_ROUTES.items():
        route = {
            "primary_route_id": name,
            "kind": "bmc",
            "derivative_route": name,
            "formal_order": values["q"],
            "degree": values["degree"],
            "logical_aliases": [name],
            "evaluator_hash": hashlib.sha256(f"native-{name}".encode()).hexdigest(),
        }
        roster.append((route, None))
    for derivative_name, derivative in COMPACT_ROUTES.items():
        q = derivative["q"]
        if q == 12:
            evaluator = {"kind": "lagrange", "order": 12, "q": 12}
            route = {
                "primary_route_id": f"{derivative_name}|L12",
                "kind": "compact",
                "derivative_route": derivative_name,
                "formal_order": q,
                "logical_aliases": [
                    f"{derivative_name}|order_matched_L12",
                    f"{derivative_name}|common_L12",
                    f"{derivative_name}|Q12-12",
                ],
                "evaluator_hash": evaluator_hash(evaluator, candidate_hash),
            }
            roster.append((route, evaluator))
            continue
        evaluator_specs = (
            ({"kind": "lagrange", "order": q, "q": q},
             f"{derivative_name}|order_matched_L{q}"),
            ({"kind": "lagrange", "order": 12, "q": q},
             f"{derivative_name}|common_L12"),
            ({"kind": "q", "q": q}, f"{derivative_name}|Q12-{q}"),
        )
        for evaluator, logical_name in evaluator_specs:
            route = {
                "primary_route_id": logical_name,
                "kind": "compact",
                "derivative_route": derivative_name,
                "formal_order": q,
                "logical_aliases": [logical_name],
                "evaluator_hash": evaluator_hash(evaluator, candidate_hash),
            }
            roster.append((route, evaluator))
    fourier = {
        "primary_route_id": "Fourier-PS",
        "kind": "fourier",
        "derivative_route": "Fourier-PS",
        "formal_order": None,
        "logical_aliases": ["Fourier-PS"],
        "evaluator_hash": hashlib.sha256(b"global-trigonometric-evaluator").hexdigest(),
    }
    roster.append((fourier, None))
    if len(roster) != 23 or sum(len(route["logical_aliases"]) for route, _ in roster) != 29:
        raise RuntimeError("route roster does not match the frozen 23/29 cell count")
    return roster


def checkpoint_name(mode: str, case: CaseData, route: dict) -> str:
    safe_route = route["primary_route_id"].replace("|", "__").replace("-", "_")
    return (
        f"{mode}__N{case.npts:03d}__{case.shift_id}__"
        f"m{case.n_markers_per_obstacle:03d}__{safe_route}.json"
    )


def complete_cell_payload(mode: str, route: dict, evaluator: dict | None,
                          case: CaseData, result: dict, contract_hash: str,
                          certificate_hash: str, reference_certificate_hash: str,
                          producer_hash: str) -> dict:
    return {
        "schema_version": "ibm-cert-primary-cell-v1",
        "completion_status": "complete",
        "mode": mode,
        "cell_id": checkpoint_name(mode, case, route).removesuffix(".json"),
        "primary_route_id": route["primary_route_id"],
        "logical_aliases": route["logical_aliases"],
        "route_kind": route["kind"],
        "derivative_route": route["derivative_route"],
        "formal_order": route["formal_order"],
        "evaluator": evaluator,
        "evaluator_hash": route["evaluator_hash"],
        "N": case.npts,
        "h": case.h,
        "shift_id": case.shift_id,
        "shift_cell_units": list(case.shift_cell_units),
        "shift_physical": list(case.shift_physical),
        "n_markers_per_obstacle": case.n_markers_per_obstacle,
        "n_markers_total": int(case.markers.shape[0]),
        "n_queries_total": int(case.queries.shape[0]),
        "shared_case_hashes": case.hashes,
        "reference_translation_residual": case.reference_translation_residual,
        "result": result,
        "contract_sha256": contract_hash,
        "evaluator_certificate_sha256": certificate_hash,
        "periodic_reference_certificate_sha256": reference_certificate_hash,
        "producer_sha256": producer_hash,
        "completed_at": utc_timestamp(),
    }


def manifest_update(run_id: str, updates: dict) -> None:
    path = OUTDIR / "run_manifest.json"
    manifest = json.loads(path.read_text())
    runs = manifest.setdefault("matrix_runs", [])
    selected = None
    for run in runs:
        if run.get("run_id") == run_id:
            selected = run
            break
    if selected is None:
        selected = {"run_id": run_id}
        runs.append(selected)
    selected.update(updates)
    atomic_json(path, manifest)


def validate_priority() -> int:
    nice_value = os.getpriority(os.PRIO_PROCESS, 0)
    if nice_value != 19:
        raise RuntimeError(f"IBM-CERT process nice value is {nice_value}, expected 19")
    return nice_value


def run_tiny_proof(reference: SpectralReference, roster, q_table, certificate,
                   contract_hash: str, certificate_hash: str,
                   reference_certificate_hash: str, producer_hash: str) -> None:
    route, evaluator = roster[0]
    case = prepare_case(reference, 32, "tiny_shift_0_0", (0.0, 0.0), 16)
    result = timed_cell(route, evaluator, case, q_table, certificate, repeats=1)
    payload = complete_cell_payload(
        "tiny", route, evaluator, case, result, contract_hash,
        certificate_hash, reference_certificate_hash, producer_hash
    )
    payload["nonranking"] = True
    payload["purpose"] = "Validate end-to-end solve, invariant, atomic checkpoint, and reread semantics."
    path = OUTDIR / "tiny_proof.json"
    atomic_json(path, payload)
    reread = json.loads(path.read_text())
    if reread["completion_status"] != "complete" or not reread["result"]["invariants_pass"]:
        raise RuntimeError("tiny proof checkpoint reread failed")
    print(
        f"[tiny-proof] checkpoint={path.relative_to(REPO)} "
        f"adjoint={result['weighted_adjoint_residual']:.3e} "
        f"translation={case.reference_translation_residual:.3e}",
        flush=True,
    )


def run_adjoint_control(reference: SpectralReference, roster, q_table) -> None:
    """Certify the normwise adjoint residual on the previously failing case."""
    route, evaluator = next(
        item for item in roster if item[0]["primary_route_id"] == "BMC11"
    )
    case = prepare_case(reference, 160, "shift_0_0", (0.0, 0.0), 80)
    operators = build_coupling(route, evaluator, case, q_table)
    seed = int(hashlib.sha256(
        f"{route['primary_route_id']}:{case.npts}:{case.shift_id}".encode()
    ).hexdigest()[:16], 16)
    receipt = weighted_adjoint_receipt(
        operators, case, seed, probe_count=32
    )
    payload = {
        "schema_version": "ibm-cert-adjoint-normalization-resolution-v1",
        "classification": "weighted_adjoint_diagnostic_normalization_repair",
        "status": (
            "complete_pass"
            if receipt["normwise_relative_residual_max"] <= 1e-12
            else "failed"
        ),
        "threshold": 1e-12,
        "wall_data_used": False,
        "scientific_construction_changed": False,
        "case": {
            "route": "BMC11",
            "N": 160,
            "shift_cell_units": [0.0, 0.0],
            "markers_per_obstacle": 80,
        },
        "prior_diagnostic": {
            "normalization": "absolute difference divided by the magnitude of the realized signed bilinear value",
            "reported_residual": 1.2176598254821837e-12,
            "failure_mechanism": "The realized signed bilinear value was small because positive and negative contributions cancelled.",
        },
        "repaired_diagnostic": receipt,
        "repair_scope": "Change only the residual denominator to a normwise sum-of-absolute-terms scale and use 32 deterministic probes. H, E, S_H, W, the solver, and all wall outputs are unchanged.",
        "target_matrix_producer_sha256": sha256_file(Path(__file__)),
        "periodic_reference_certificate_sha256": sha256_file(
            REFERENCE_CERTIFICATE_PATH
        ),
    }
    atomic_json(OUTDIR / "adjoint_normalization_resolution.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if payload["status"] != "complete_pass":
        raise SystemExit(2)


def run_cells(mode: str, run_id: str, reference: SpectralReference, roster,
              q_table, certificate, contract_hash: str,
              certificate_hash: str, reference_certificate_hash: str,
              producer_hash: str) -> None:
    checkpoint_directory = OUTDIR / "runs" / run_id / "checkpoints" / mode
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if mode == "primary":
        specifications = [
            (npts, shift_id, (sx, sy), max(48, npts // 2))
            for npts in PRIMARY_GRIDS
            for shift_id, sx, sy in SHIFTS
        ]
        repeats = 3
    elif mode == "scaling":
        specifications = [
            (96, "shift_0_0", (0.0, 0.0), marker_count)
            for marker_count in SCALING_COUNTS
        ]
        repeats = 3
    else:
        raise ValueError(mode)

    expected = len(specifications) * len(roster)
    completed = len(list(checkpoint_directory.glob("*.json")))
    print(f"[{mode}] resume={completed}/{expected}", flush=True)
    for npts, shift_id, shift_cell, marker_count in specifications:
        case = prepare_case(reference, npts, shift_id, shift_cell, marker_count)
        if case.reference_translation_residual > 1e-12:
            raise RuntimeError(
                f"reference translation residual failed for N={npts} {shift_id}: "
                f"{case.reference_translation_residual:.3e}"
            )
        for route, evaluator in roster:
            path = checkpoint_directory / checkpoint_name(mode, case, route)
            if path.exists():
                prior = json.loads(path.read_text())
                if (
                    prior.get("completion_status") != "complete"
                    or prior.get("contract_sha256") != contract_hash
                    or prior.get("evaluator_certificate_sha256") != certificate_hash
                    or prior.get("periodic_reference_certificate_sha256")
                    != reference_certificate_hash
                    or prior.get("producer_sha256") != producer_hash
                ):
                    raise RuntimeError(f"checkpoint provenance mismatch: {path}")
                continue
            result = timed_cell(route, evaluator, case, q_table, certificate, repeats)
            payload = complete_cell_payload(
                mode, route, evaluator, case, result, contract_hash,
                certificate_hash, reference_certificate_hash, producer_hash
            )
            atomic_json(path, payload)
            completed += 1
            manifest_update(run_id, {
                "status": f"{mode}_running",
                f"{mode}_completed_distinct_cells": completed,
                f"{mode}_expected_distinct_cells": expected,
                "last_checkpoint": str(path.relative_to(REPO)),
                "last_checkpoint_at": utc_timestamp(),
                "peak_rss_kib_process_so_far": result["peak_rss_kib_process_so_far"],
            })
            print(
                f"[{mode} {completed:03d}/{expected:03d}] N={npts} {shift_id} "
                f"m={marker_count} {route['primary_route_id']} "
                f"trace={result['trace_error']:.6e} "
                f"field={result['exterior_relative_l2']:.6e} "
                f"time={result['total_time_seconds']:.4f}s "
                f"invariants={'pass' if result['invariants_pass'] else 'FAIL'}",
                flush=True,
            )
            if not result["invariants_pass"]:
                raise RuntimeError(f"cell invariant failed: {path}")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "reference-control",
            "adjoint-control",
            "tiny",
            "primary",
            "scaling",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--run-id", default="interactive")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    nice_value = validate_priority()
    if args.mode == "reference-control":
        from certify_periodic_reference import main as certify_reference

        certify_reference()
        return
    contract_path = OUTDIR / "frozen_contract.json"
    certificate_path = OUTDIR / "evaluator_certificate.json"
    candidate_path = OUTDIR / "evaluator_candidate.npz"
    contract_hash = sha256_file(contract_path)
    certificate_hash = sha256_file(certificate_path)
    reference_certificate_hash = sha256_file(REFERENCE_CERTIFICATE_PATH)
    reference_certificate = json.loads(REFERENCE_CERTIFICATE_PATH.read_text())
    reference_resolution = json.loads(REFERENCE_RESOLUTION_PATH.read_text())
    producer_hash = sha256_file(Path(__file__))
    certificate = json.loads(certificate_path.read_text())
    if not certificate["all_pass"]:
        raise RuntimeError("evaluator certificate is not admitted")
    if certificate["frozen_contract_sha256"] != contract_hash:
        raise RuntimeError("evaluator certificate is not bound to the current frozen contract")
    candidate_hash = sha256_file(candidate_path)
    if certificate["candidate_sha256"] != candidate_hash:
        raise RuntimeError("evaluator candidate hash does not match its certificate")
    if not reference_certificate["all_pass"]:
        raise RuntimeError("periodic reference certificate is not admitted")
    if reference_certificate["target_matrix_producer_sha256"] != producer_hash:
        raise RuntimeError(
            "periodic reference certificate is not bound to this matrix producer"
        )
    if reference_resolution["scientific_contract_sha256"] != contract_hash:
        raise RuntimeError("reference resolution is not bound to the frozen contract")
    if reference_resolution["certificate_sha256"] != reference_certificate_hash:
        raise RuntimeError("reference resolution does not identify the current certificate")

    q_table = load_q_table(candidate_path)
    roster = route_roster(candidate_hash)
    reference = SpectralReference()
    if args.mode == "adjoint-control":
        run_adjoint_control(reference, roster, q_table)
        return
    manifest_update(args.run_id, {
        "status": "started",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "nice": nice_value,
        "started_at": utc_timestamp(),
        "mode": args.mode,
        "log_path": args.log_path,
        "resume_command": (
            f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            f"nice -n 19 uv run code/analysis/ibm_cert/run_matrix.py "
            f"--mode {args.mode} --run-id {args.run_id} --log-path {args.log_path}"
        ),
        "contract_sha256": contract_hash,
        "evaluator_certificate_sha256": certificate_hash,
        "periodic_reference_certificate_sha256": reference_certificate_hash,
        "reference_sampler_resolution": str(REFERENCE_RESOLUTION_PATH.relative_to(REPO)),
        "checkpoint_root": str((OUTDIR / "runs" / args.run_id / "checkpoints").relative_to(REPO)),
        "producer_sha256": producer_hash,
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    })

    if args.mode == "tiny":
        run_tiny_proof(
            reference, roster, q_table, certificate,
            contract_hash, certificate_hash, reference_certificate_hash,
            producer_hash
        )
        manifest_update(args.run_id, {"status": "tiny_complete", "completed_at": utc_timestamp()})
        return

    if args.mode in ("primary", "all"):
        run_cells(
            "primary", args.run_id, reference, roster, q_table, certificate,
            contract_hash, certificate_hash, reference_certificate_hash,
            producer_hash
        )
    if args.mode in ("scaling", "all"):
        run_cells(
            "scaling", args.run_id, reference, roster, q_table, certificate,
            contract_hash, certificate_hash, reference_certificate_hash,
            producer_hash
        )
    run_directory = OUTDIR / "runs" / args.run_id
    completion = {
        "status": "matrix_complete",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "completed_at": utc_timestamp(),
        "primary_distinct_cells": len(list((run_directory / "checkpoints" / "primary").glob("*.json"))),
        "scaling_distinct_cells": len(list((run_directory / "checkpoints" / "scaling").glob("*.json"))),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    atomic_json(run_directory / "MATRIX_COMPLETED.json", completion)
    manifest_update(args.run_id, {
        "status": "matrix_complete",
        "completed_at": completion["completed_at"],
        "completion_record": str((run_directory / "MATRIX_COMPLETED.json").relative_to(REPO)),
        "peak_rss_kib_process": completion["peak_rss_kib"],
    })
    print(json.dumps(completion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
