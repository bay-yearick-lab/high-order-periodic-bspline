#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RectBivariateSpline


THISDIR = Path(__file__).resolve().parent
REPO = THISDIR.parents[2]
CODEDIR = REPO / "code"
OUTDIR = REPO / "results" / "ibm_cert"
sys.path.insert(0, str(CODEDIR))

import experiment_ibm_frontier as frontier  # noqa: E402


BOX = 2.0 * np.pi
NREF = 768
KAPPA = 2.0
PADDING = 8
THRESHOLD = 1e-12
EXPECTED_FORCING_SHA256 = "8be12bcad0c00d2c88f6406460b5cf36f85ae0058177bc6ee86004bd4cd263fe"
EXPECTED_SPECTRUM_SHA256 = "6dd57a0b94702ed0ca76ec6cc5be2e303f0bfec1bd7350352ca855d3262bdaf0"
EXPECTED_FIELD_SHA256 = "a3a8941b64bc42f587a99c69621e8840a9d7f613595ece11095ee99111ac0d33"
GRIDS = (96, 128, 160, 192)
SHIFTS = (
    ("shift_0_0", 0.0, 0.0),
    ("shift_1_4_3_8", 0.25, 0.375),
    ("shift_1_2_1_2", 0.5, 0.5),
    ("shift_3_4_1_8", 0.75, 0.125),
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_reference() -> dict:
    spacing = BOX / NREF
    coordinates = np.arange(NREF) * spacing
    grid = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
    forcing = frontier.source_forcing(grid)
    modes = np.fft.fftfreq(NREF, d=1.0) * NREF
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    spectrum = np.fft.fft2(forcing) / (kx**2 + ky**2 + KAPPA**2)
    values = np.real(np.fft.ifft2(spectrum))

    old_sampler = RectBivariateSpline(
        coordinates, coordinates, values, kx=5, ky=5
    )
    extended_indices = np.arange(-PADDING, NREF + PADDING)
    extended_coordinates = extended_indices * spacing
    extended_values = values[np.ix_(
        extended_indices % NREF,
        extended_indices % NREF,
    )]
    periodic_sampler = RectBivariateSpline(
        extended_coordinates,
        extended_coordinates,
        extended_values,
        kx=5,
        ky=5,
    )
    return {
        "spacing": spacing,
        "coordinates": coordinates,
        "grid": grid,
        "forcing": forcing,
        "modes": modes,
        "spectrum": spectrum,
        "values": values,
        "old_sampler": old_sampler,
        "periodic_sampler": periodic_sampler,
    }


def periodic_evaluate(sampler: RectBivariateSpline, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2) % BOX
    return sampler.ev(flat[:, 0], flat[:, 1]).reshape(shape)


def direct_fourier(points: np.ndarray, modes: np.ndarray,
                   spectrum: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    shape = points.shape[:-1]
    flat = (points.reshape(-1, 2)) % BOX
    phase_x = np.exp(1j * np.outer(flat[:, 0], modes))
    phase_y = np.exp(1j * np.outer(flat[:, 1], modes))
    values = np.sum((phase_x @ spectrum) * phase_y, axis=1) / (NREF * NREF)
    return np.real(values).reshape(shape)


def relative_max(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(right))), 1e-300)
    return float(np.max(np.abs(left - right)) / scale)


def certify() -> dict:
    reference = build_reference()
    spacing = reference["spacing"]
    modes = reference["modes"]
    spectrum = reference["spectrum"]
    periodic_sampler = reference["periodic_sampler"]
    old_sampler = reference["old_sampler"]

    sampled_indices = np.linspace(0, NREF - 1, 64, dtype=int)
    sampled_coordinates = sampled_indices * spacing
    sampled_grid = np.stack(
        np.meshgrid(sampled_coordinates, sampled_coordinates, indexing="ij"),
        axis=-1,
    )
    arbitrary_points = np.array([
        [0.0, 0.0],
        [0.0, BOX - 0.25 * spacing],
        [BOX - 0.25 * spacing, 0.0],
        [BOX - 0.25 * spacing, BOX - 0.75 * spacing],
        [1.234, 2.345],
        [3.2, 4.1],
    ])
    interior_coordinates = np.linspace(0.25, BOX - 0.25, 25)
    interior_points = np.stack(
        np.meshgrid(interior_coordinates, interior_coordinates, indexing="ij"),
        axis=-1,
    )

    cases = []
    for npts in GRIDS:
        h = BOX / npts
        for shift_id, sx, sy in SHIFTS:
            shift = np.array([sx * h, sy * h])
            shifted_points = sampled_grid - shift
            periodic_values = periodic_evaluate(periodic_sampler, shifted_points)
            phase = (
                np.exp(-1j * modes * shift[0])[:, None]
                * np.exp(-1j * modes * shift[1])[None, :]
            )
            shifted_reference = np.real(np.fft.ifft2(spectrum * phase))
            exact_grid_values = shifted_reference[np.ix_(sampled_indices, sampled_indices)]
            covariance_residual = relative_max(periodic_values, exact_grid_values)

            arbitrary_shifted = arbitrary_points - shift
            arbitrary_periodic = periodic_evaluate(periodic_sampler, arbitrary_shifted)
            arbitrary_direct = direct_fourier(arbitrary_shifted, modes, spectrum)
            direct_residual = relative_max(arbitrary_periodic, arbitrary_direct)

            shifted_source_grid = reference["grid"].copy()
            shifted_source_grid[..., 0] = (
                shifted_source_grid[..., 0] - shift[0]
            ) % BOX
            shifted_source_grid[..., 1] = (
                shifted_source_grid[..., 1] - shift[1]
            ) % BOX
            shifted_forcing = frontier.source_forcing(shifted_source_grid)
            rebuilt = np.real(np.fft.ifft2(
                np.fft.fft2(shifted_forcing)
                / (
                    modes[:, None] ** 2
                    + modes[None, :] ** 2
                    + KAPPA**2
                )
            ))
            rebuilt_residual = relative_max(rebuilt, shifted_reference)

            plus_period = arbitrary_shifted.copy()
            plus_period[:, 0] += BOX
            periodicity_x = relative_max(
                periodic_evaluate(periodic_sampler, plus_period),
                arbitrary_periodic,
            )
            plus_period = arbitrary_shifted.copy()
            plus_period[:, 1] -= BOX
            periodicity_y = relative_max(
                periodic_evaluate(periodic_sampler, plus_period),
                arbitrary_periodic,
            )
            cases.append({
                "N": npts,
                "shift_id": shift_id,
                "shift_cell_units": [sx, sy],
                "shift_physical": shift.tolist(),
                "sample_grid_translation_residual": covariance_residual,
                "arbitrary_seam_and_interior_direct_fourier_residual": direct_residual,
                "rebuilt_shifted_source_reference_residual": rebuilt_residual,
                "periodicity_x_residual": periodicity_x,
                "periodicity_y_residual": periodicity_y,
                "pass": bool(max(
                    covariance_residual,
                    direct_residual,
                    rebuilt_residual,
                    periodicity_x,
                    periodicity_y,
                ) <= THRESHOLD),
            })

    periodic_interior = periodic_evaluate(periodic_sampler, interior_points)
    flat_interior = interior_points.reshape(-1, 2)
    old_interior = old_sampler.ev(flat_interior[:, 0], flat_interior[:, 1]).reshape(
        interior_points.shape[:-1]
    )
    old_interior_residual = relative_max(periodic_interior, old_interior)

    failing_shift = np.array([0.75 * BOX / 128, 0.125 * BOX / 128])
    failing_phase = (
        np.exp(-1j * modes * failing_shift[0])[:, None]
        * np.exp(-1j * modes * failing_shift[1])[None, :]
    )
    failing_exact = np.real(np.fft.ifft2(spectrum * failing_phase))[
        np.ix_(sampled_indices, sampled_indices)
    ]
    failing_flat = ((sampled_grid - failing_shift) % BOX).reshape(-1, 2)
    old_failing = old_sampler.ev(failing_flat[:, 0], failing_flat[:, 1]).reshape(64, 64)
    old_positive_control = relative_max(old_failing, failing_exact)

    maxima = {
        "sample_grid_translation_residual": max(
            case["sample_grid_translation_residual"] for case in cases
        ),
        "arbitrary_direct_fourier_residual": max(
            case["arbitrary_seam_and_interior_direct_fourier_residual"]
            for case in cases
        ),
        "rebuilt_shifted_source_reference_residual": max(
            case["rebuilt_shifted_source_reference_residual"] for case in cases
        ),
        "periodicity_residual": max(
            max(case["periodicity_x_residual"], case["periodicity_y_residual"])
            for case in cases
        ),
        "old_sampler_valid_interior_agreement": old_interior_residual,
    }
    forcing_hash = sha256_array(reference["forcing"])
    spectrum_hash = sha256_array(spectrum)
    field_hash = sha256_array(reference["values"])
    checks = {
        "all_16_grid_shift_cases": all(case["pass"] for case in cases),
        "old_sampler_valid_interior_agreement": old_interior_residual <= THRESHOLD,
        "old_sampler_seam_defect_positive_control": old_positive_control > 1e-6,
        "underlying_forcing_unchanged": forcing_hash == EXPECTED_FORCING_SHA256,
        "underlying_reference_spectrum_unchanged": (
            spectrum_hash == EXPECTED_SPECTRUM_SHA256
        ),
        "underlying_reference_field_unchanged": field_hash == EXPECTED_FIELD_SHA256,
        "no_wall_data_used": True,
    }
    return {
        "schema_version": "ibm-cert-periodic-reference-certificate-v1",
        "all_pass": bool(all(checks.values())),
        "wall_data_used": False,
        "threshold": THRESHOLD,
        "underlying_reference": {
            "grid": [NREF, NREF],
            "kappa": KAPPA,
            "field_sha256": field_hash,
            "spectrum_sha256": spectrum_hash,
            "forcing_sha256": forcing_hash,
            "expected_field_sha256": EXPECTED_FIELD_SHA256,
            "expected_spectrum_sha256": EXPECTED_SPECTRUM_SHA256,
            "expected_forcing_sha256": EXPECTED_FORCING_SHA256,
        },
        "repaired_sampler": {
            "identity": "periodic_degree5_rectbivariatespline_padding8_v1",
            "degree": 5,
            "periodic_padding_nodes_each_side": PADDING,
            "operation": "Wrap query coordinates modulo 2*pi and evaluate the unchanged reference samples on an eight-node periodic extension.",
        },
        "cases": cases,
        "maxima": maxima,
        "checks": checks,
        "old_sampler_seam_failure_positive_control": {
            "N": 128,
            "shift_cell_units": [0.75, 0.125],
            "relative_residual": old_positive_control,
        },
        "producer_sha256": sha256_file(Path(__file__)),
        "target_matrix_producer_path": "code/analysis/ibm_cert/run_matrix.py",
        "target_matrix_producer_sha256": sha256_file(
            THISDIR / "run_matrix.py"
        ),
        "runtime_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_core_sha256": sha256_file(CODEDIR / "experiment_ibm_frontier.py"),
        "frozen_contract_sha256": sha256_file(OUTDIR / "frozen_contract.json"),
    }


def main() -> None:
    certificate = certify()
    certificate_path = OUTDIR / "periodic_reference_certificate.json"
    atomic_json(certificate_path, certificate)
    certificate_hash = sha256_file(certificate_path)
    resolution = {
        "schema_version": "ibm-cert-reference-sampler-resolution-v1",
        "status": "admitted" if certificate["all_pass"] else "rejected",
        "classification": "reference_sampler_implementation_repair",
        "scientific_contract_sha256": certificate["frozen_contract_sha256"],
        "certificate_path": str(certificate_path.relative_to(REPO)),
        "certificate_sha256": certificate_hash,
        "repair_scope": "Sampling operation only. The 768-square Fourier field, spectrum, grid, PDE, source, geometry, routes, shifts, metrics, thresholds, evaluator pool, timing contract, and promotion rules are unchanged.",
        "implementation_repair_ledger": {
            "evaluator_continuum_enclosure": "certification_algorithm repair; evaluator candidate unchanged",
            "periodic_reference_sampler": "reference sampling implementation repair; underlying Fourier reference unchanged",
            "scientific_mechanism_family_changes": 0,
        },
        "checkpoint_dependency_decision": {
            "prior_primary_cells": 161,
            "reuse": False,
            "reason": "Exterior error values depend on seam reference samples, and all checkpoint provenance is bound to the prior producer and sampler hashes. A clean rerun is cheaper and less ambiguous than cellwise dependency admission.",
            "preserve_prior_cells": True,
            "new_checkpoint_policy": "Use a new run-local checkpoint tree for all 368 primary and 92 scaling cells.",
        },
        "supersedes_state": "blocked_by_specific_information",
        "superseding_condition": "The independent periodic reference certificate passes every residual at 1e-12 without wall data.",
    }
    atomic_json(OUTDIR / "reference_sampler_resolution.json", resolution)
    print(json.dumps({
        "all_pass": certificate["all_pass"],
        "maxima": certificate["maxima"],
        "old_sampler_positive_control": certificate[
            "old_sampler_seam_failure_positive_control"
        ],
        "certificate_sha256": certificate_hash,
    }, indent=2))
    if not certificate["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
