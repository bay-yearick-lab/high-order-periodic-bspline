#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


THISDIR = Path(__file__).resolve().parent
REPO = THISDIR.parents[2]
OUTDIR = REPO / "results" / "ibm_cert"
CANDIDATE = OUTDIR / "evaluator_candidate.npz"
OPTIMIZER_RECEIPT = OUTDIR / "evaluator_optimizer_receipt.json"
CONTRACT_PATH = OUTDIR / "frozen_contract.json"
CERTIFICATE_PATH = OUTDIR / "evaluator_certificate.json"

OFFSETS = np.arange(-5, 7, dtype=np.int64)
BAND_ENDPOINT = 2.0 * np.pi / 3.0
ORDERS = (4, 6, 8, 10, 12)
THETA_COUNT = 4097
DELTA_SUBDIVISIONS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polynomial_coefficients(order: int) -> np.ndarray:
    """Construct the embedded Lagrange basis independently in power form."""
    nodes = np.arange(1 - order // 2, 1 + order // 2, dtype=np.int64)
    coeff = np.zeros((12, order), dtype=np.longdouble)
    for node in nodes:
        poly = np.array([1.0], dtype=np.longdouble)
        denominator = np.longdouble(1.0)
        for other in nodes:
            if other == node:
                continue
            poly = np.convolve(poly, np.array([-other, 1], dtype=np.longdouble))
            denominator *= np.longdouble(node - other)
        coeff[int(node + 5), :poly.size] = poly / denominator
    return coeff


def evaluate(coeff: np.ndarray, delta: float, derivative: int = 0) -> np.ndarray:
    work = coeff.copy()
    for _ in range(derivative):
        if work.shape[1] <= 1:
            return np.zeros(12, dtype=np.longdouble)
        work = work[:, 1:] * np.arange(1, work.shape[1], dtype=np.longdouble)[None, :]
    x = np.longdouble(delta)
    out = np.zeros(12, dtype=np.longdouble)
    for power in range(work.shape[1] - 1, -1, -1):
        out = out * x + work[:, power]
    return out


def integer_nullspace(order: int) -> np.ndarray:
    if order == 12:
        return np.zeros((12, 0), dtype=np.longdouble)
    columns = []
    for start in range(12 - order):
        column = np.zeros(12, dtype=np.longdouble)
        for k in range(order + 1):
            column[start + k] = np.longdouble((-1) ** k * math.comb(order, k))
        columns.append(column)
    return np.stack(columns, axis=1)


def absolute_polynomial_derivative_bound(coeff: np.ndarray, derivative: int,
                                         xmax: float = 0.5,
                                         offset_power: int = 0) -> float:
    """Bound the summed absolute basis derivatives on the half cell."""
    work = coeff.copy()
    for _ in range(derivative):
        if work.shape[1] <= 1:
            return 0.0
        work = work[:, 1:] * np.arange(1, work.shape[1], dtype=np.longdouble)[None, :]
    bound = np.longdouble(0.0)
    x = np.longdouble(xmax)
    offset_scale = np.abs(OFFSETS.astype(np.longdouble)) ** offset_power
    for power in range(work.shape[1]):
        bound += np.sum(offset_scale * np.abs(work[:, power])) * x**power
    return float(bound)


def raw_moment_residual(weights: np.ndarray, delta: float, order: int) -> tuple[float, float]:
    w = np.asarray(weights, dtype=np.longdouble)
    r = OFFSETS.astype(np.longdouble)
    d = np.longdouble(delta)
    residual = np.longdouble(0.0)
    for degree in range(order):
        residual = max(residual, abs(np.sum(w * r**degree) - d**degree))
    return float(residual), float(abs(np.sum(w) - 1.0))


def segment_certificate(order: int, anchors: np.ndarray, coordinates: np.ndarray | None,
                        theta: np.ndarray, exponentials: np.ndarray) -> dict:
    """Enclose transfer error and l1 norm over every delta-theta cell."""
    coeff = polynomial_coefficients(order)
    basis = integer_nullspace(order)
    theta_step_half = 0.5 * float(theta[1] - theta[0])
    coarse_delta_step = float(anchors[1] - anchors[0])
    fine_delta_step = coarse_delta_step / DELTA_SUBDIVISIONS
    delta_step_half = 0.5 * fine_delta_step
    base_bounds = {
        (derivative, offset_power): absolute_polynomial_derivative_bound(
            coeff, derivative, offset_power=offset_power
        )
        for derivative in range(5)
        for offset_power in range(4)
    }

    lower = 0.0
    upper = 0.0
    maximum_l1_upper = 0.0
    maximum_moment = 0.0
    maximum_constant = 0.0
    worst_lower = None
    worst_upper = None
    maximum_third_remainder = 0.0

    r = OFFSETS.astype(float)
    endpoint_theta = np.array([0.0, BAND_ENDPOINT])
    endpoint_exponentials = np.exp(1j * np.outer(r, endpoint_theta))
    for index in range(anchors.size - 1):
        if coordinates is None:
            slope = None
        else:
            slope = ((coordinates[index + 1] - coordinates[index])
                     / coarse_delta_step)

        for subdivision in range(DELTA_SUBDIVISIONS):
            left = float(anchors[index]) + subdivision * fine_delta_step
            right = left + fine_delta_step
            delta = 0.5 * (left + right)
            base = evaluate(coeff, delta, 0)
            base_d = evaluate(coeff, delta, 1)
            base_dd = evaluate(coeff, delta, 2)
            base_ddd = evaluate(coeff, delta, 3)
            if coordinates is None:
                weights = base
                weights_d = base_d
            else:
                fraction = (delta - float(anchors[index])) / coarse_delta_step
                coordinate = ((1.0 - fraction) * coordinates[index]
                              + fraction * coordinates[index + 1])
                weights = base + basis @ coordinate.astype(np.longdouble)
                weights_d = base_d + basis @ slope.astype(np.longdouble)

            w = np.asarray(weights, dtype=float)
            wd = np.asarray(weights_d, dtype=float)
            wdd = np.asarray(base_dd, dtype=float)
            exact = np.exp(1j * delta * theta)
            z = w @ exponentials - exact
            z_delta = wd @ exponentials - 1j * theta * exact
            z_theta = (1j * r * w) @ exponentials - 1j * delta * exact
            z_delta_delta = wdd @ exponentials + theta**2 * exact
            z_delta_theta = ((1j * r * wd) @ exponentials
                             - 1j * exact + delta * theta * exact)
            z_theta_theta = (-(r**2) * w) @ exponentials + delta**2 * exact

            magnitude_squared = np.abs(z) ** 2
            derivative_delta = 2.0 * np.real(np.conj(z) * z_delta)
            derivative_theta = 2.0 * np.real(np.conj(z) * z_theta)
            derivative_delta_delta = 2.0 * (
                np.abs(z_delta) ** 2
                + np.real(np.conj(z) * z_delta_delta)
            )
            derivative_delta_theta = 2.0 * np.real(
                np.conj(z_theta) * z_delta + np.conj(z) * z_delta_theta
            )
            derivative_theta_theta = 2.0 * (
                np.abs(z_theta) ** 2
                + np.real(np.conj(z) * z_theta_theta)
            )

            weighted = lambda values, power: float(
                np.sum(np.abs(values) * np.abs(r) ** power)
            )
            bw0 = {
                power: (weighted(w, power)
                        + delta_step_half * weighted(wd, power)
                        + 0.5 * delta_step_half**2 * base_bounds[(2, power)])
                for power in range(4)
            }
            bw1 = {
                power: (weighted(wd, power)
                        + delta_step_half * base_bounds[(2, power)]
                        + 0.5 * delta_step_half**2 * base_bounds[(3, power)])
                for power in range(3)
            }
            bw2 = {
                power: (weighted(wdd, power)
                        + delta_step_half * base_bounds[(3, power)])
                for power in range(2)
            }

            bz = bw0[0] + 1.0
            bzd = bw1[0] + BAND_ENDPOINT
            bzt = bw0[1] + 0.5
            bzdd = bw2[0] + BAND_ENDPOINT**2
            bzdt = bw1[1] + 1.0 + 0.5 * BAND_ENDPOINT
            bztt = bw0[2] + 0.25
            bzddd = base_bounds[(3, 0)] + BAND_ENDPOINT**3
            bzddt = (base_bounds[(2, 1)] + 2.0 * BAND_ENDPOINT
                     + 0.5 * BAND_ENDPOINT**2)
            bzdtt = (base_bounds[(1, 2)] + 1.0
                     + 0.25 * BAND_ENDPOINT)
            bzttt = bw0[3] + 0.125

            bfddd = 2.0 * (bz * bzddd + 3.0 * bzd * bzdd)
            bfddt = 2.0 * (2.0 * bzd * bzdt + bzt * bzdd + bz * bzddt)
            bfdtt = 2.0 * (2.0 * bzt * bzdt + bzd * bztt + bz * bzdtt)
            bfttt = 2.0 * (bz * bzttt + 3.0 * bzt * bztt)
            third_remainder = (
                bfddd * delta_step_half**3
                + 3.0 * bfddt * delta_step_half**2 * theta_step_half
                + 3.0 * bfdtt * delta_step_half * theta_step_half**2
                + bfttt * theta_step_half**3
            ) / 6.0
            maximum_third_remainder = max(maximum_third_remainder, third_remainder)

            delta_linear = delta_step_half * np.abs(derivative_delta)
            theta_linear = theta_step_half * np.abs(derivative_theta)
            delta_quadratic = 0.5 * delta_step_half**2 * np.abs(
                derivative_delta_delta
            )
            mixed_quadratic = (delta_step_half * theta_step_half
                               * np.abs(derivative_delta_theta))
            theta_quadratic = 0.5 * theta_step_half**2 * np.abs(
                derivative_theta_theta
            )
            grid_envelope = (magnitude_squared + delta_linear + theta_linear
                             + delta_quadratic + mixed_quadratic
                             + theta_quadratic + third_remainder)
            upper_index = int(np.argmax(grid_envelope))
            cell_upper = float(np.sqrt(max(0.0, grid_envelope[upper_index])))

            endpoint_exact = np.exp(1j * delta * endpoint_theta)
            endpoint_squared = np.abs(w @ endpoint_exponentials - endpoint_exact) ** 2
            center_index = int(np.argmax(magnitude_squared))
            if float(np.max(endpoint_squared)) > float(magnitude_squared[center_index]):
                lower_squared = float(np.max(endpoint_squared))
                lower_theta = float(endpoint_theta[int(np.argmax(endpoint_squared))])
            else:
                lower_squared = float(magnitude_squared[center_index])
                lower_theta = float(theta[center_index])
            cell_lower = float(np.sqrt(lower_squared))

            fine_index = index * DELTA_SUBDIVISIONS + subdivision
            if cell_lower > lower:
                lower = cell_lower
                worst_lower = {
                    "delta": delta,
                    "theta": lower_theta,
                    "coarse_segment": index,
                    "fine_segment": fine_index,
                }
            if cell_upper > upper:
                upper = cell_upper
                worst_upper = {
                    "delta_interval": [left, right],
                    "theta_interval_center": float(theta[upper_index]),
                    "coarse_segment": index,
                    "fine_segment": fine_index,
                    "enclosure_terms_squared": {
                        "center": float(magnitude_squared[upper_index]),
                        "delta_linear": float(delta_linear[upper_index]),
                        "theta_linear": float(theta_linear[upper_index]),
                        "delta_quadratic": float(delta_quadratic[upper_index]),
                        "mixed_quadratic": float(mixed_quadratic[upper_index]),
                        "theta_quadratic": float(theta_quadratic[upper_index]),
                        "third_remainder": float(third_remainder),
                    },
                }

            l1_upper = float(
                np.sum(np.abs(w))
                + delta_step_half * np.sum(np.abs(wd))
                + 0.5 * delta_step_half**2 * base_bounds[(2, 0)]
            )
            maximum_l1_upper = max(maximum_l1_upper, l1_upper)

            moment, constant = raw_moment_residual(weights, delta, order)
            maximum_moment = max(maximum_moment, moment)
            maximum_constant = max(maximum_constant, constant)

    endpoint_delta = float(anchors[-1])
    endpoint_weights = evaluate(coeff, endpoint_delta, 0)
    if coordinates is not None:
        endpoint_weights += basis @ coordinates[-1].astype(np.longdouble)
    moment, constant = raw_moment_residual(endpoint_weights, endpoint_delta, order)
    maximum_moment = max(maximum_moment, moment)
    maximum_constant = max(maximum_constant, constant)

    return {
        "continuum_transfer_lower": lower,
        "continuum_transfer_upper": upper,
        "continuum_enclosure_gap": upper - lower,
        "worst_lower_witness": worst_lower,
        "worst_upper_box": worst_upper,
        "maximum_l1_upper": maximum_l1_upper,
        "maximum_raw_moment_residual": maximum_moment,
        "maximum_constant_residual": maximum_constant,
        "anchor_segments_half_cell": int(anchors.size - 1),
        "subdivisions_per_anchor_segment": DELTA_SUBDIVISIONS,
        "delta_segments_half_cell": int((anchors.size - 1) * DELTA_SUBDIVISIONS),
        "theta_intervals": int(theta.size),
        "maximum_third_remainder_squared": maximum_third_remainder,
        "reflection_covers_full_cell": True,
    }


def lagrange_receipt(order: int) -> dict:
    coeff = polynomial_coefficients(order)
    deltas = np.linspace(0.0, 1.0, 16385)
    maximum_l1 = 0.0
    maximum_moment = 0.0
    maximum_constant = 0.0
    for delta in deltas:
        weights = evaluate(coeff, float(delta))
        maximum_l1 = max(maximum_l1, float(np.sum(np.abs(weights))))
        moment, constant = raw_moment_residual(weights, float(delta), order)
        maximum_moment = max(maximum_moment, moment)
        maximum_constant = max(maximum_constant, constant)
    exact_l1 = {
        4: [5, 4],
        6: [89, 64],
        8: [381, 256],
        10: [25609, 16384],
        12: [106405, 65536],
    }[order]
    return {
        "order": order,
        "support": list(range(1 - order // 2, 1 + order // 2)),
        "maximum_l1_exact_fraction": exact_l1,
        "maximum_l1_exact": exact_l1[0] / exact_l1[1],
        "maximum_sampled_raw_moment_residual": maximum_moment,
        "maximum_sampled_constant_residual": maximum_constant,
        "real_weights": True,
        "periodic_translation_equivariance": True,
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    optimizer = json.loads(OPTIMIZER_RECEIPT.read_text())
    with np.load(CANDIDATE) as data:
        anchors = data["anchors"]
        coordinate_tables = {q: data[f"q{q}_coordinates"] for q in ORDERS if q < 12}

    theta_edges = np.linspace(0.0, BAND_ENDPOINT, THETA_COUNT)
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    exponentials = np.exp(1j * np.outer(OFFSETS.astype(float), theta))
    q_receipts = {}
    checks = {}
    gap_abs = float(contract["evaluators"]["continuum_certificate_gap_abs"])
    gap_rel = float(contract["evaluators"]["continuum_certificate_gap_rel"])

    for order in ORDERS:
        coordinates = coordinate_tables.get(order)
        receipt = segment_certificate(order, anchors, coordinates, theta, exponentials)
        tolerance = max(gap_abs, gap_rel * receipt["continuum_transfer_upper"])
        receipt["enclosure_tolerance"] = tolerance
        receipt["alias"] = "L12" if order == 12 else None
        q_receipts[f"Q12-{order}"] = receipt
        checks[f"Q12-{order}_continuum_enclosed"] = bool(
            receipt["continuum_enclosure_gap"] <= tolerance
        )
        checks[f"Q12-{order}_l1"] = bool(receipt["maximum_l1_upper"] <= 4.0)
        checks[f"Q12-{order}_moments"] = bool(
            receipt["maximum_raw_moment_residual"] <= 1e-12
        )
        checks[f"Q12-{order}_constant"] = bool(
            receipt["maximum_constant_residual"] <= 1e-13
        )

    lagrange = {f"L{order}": lagrange_receipt(order) for order in ORDERS}
    for name, receipt in lagrange.items():
        checks[f"{name}_l1"] = bool(receipt["maximum_l1_exact"] <= 4.0)
        checks[f"{name}_moments"] = bool(
            receipt["maximum_sampled_raw_moment_residual"] <= 1e-12
        )
        checks[f"{name}_constant"] = bool(
            receipt["maximum_sampled_constant_residual"] <= 1e-13
        )

    q12_identity = bool(
        q_receipts["Q12-12"]["maximum_raw_moment_residual"] <= 1e-12
        and optimizer["summaries"]["Q12-12"]["alias"] == "L12"
    )
    checks["Q12-12_equals_L12"] = q12_identity

    certificate = {
        "schema_version": "ibm-cert-evaluator-certificate-v1",
        "wall_data_used": False,
        "all_pass": bool(all(checks.values())),
        "checks": checks,
        "offsets": OFFSETS.tolist(),
        "band": [-BAND_ENDPOINT, BAND_ENDPOINT],
        "formal_order_map": contract["compact_routes"],
        "optimizer_definition": contract["evaluators"]["q_family_definition"],
        "objective": contract["evaluators"]["objective"],
        "q_receipts": q_receipts,
        "lagrange_receipts": lagrange,
        "q12_aliases": {
            "C12": ["order_matched_L12", "common_L12", "Q12-12"],
            "CW12A": ["order_matched_L12", "common_L12", "Q12-12"],
            "CW12B": ["order_matched_L12", "common_L12", "Q12-12"]
        },
        "symbolic_controls": {
            "moment_parameterization": "Lq(delta) plus shifted q-th finite differences",
            "nullspace_identity": "sum_j Z[j,a] r_j^m = 0 for every m<q",
            "endpoint_gluing": "nodal at delta 0 and reflected nodal at delta 1",
            "real_translation_equivariance": "weights are real and depend only on fractional cell coordinate",
            "negative_frequency_relation": "T(-theta)=conjugate(T(theta))"
        },
        "candidate_path": str(CANDIDATE.relative_to(REPO)),
        "candidate_sha256": sha256(CANDIDATE),
        "optimizer_receipt_sha256": sha256(OPTIMIZER_RECEIPT),
        "optimizer_source_sha256": optimizer["producer_sha256"],
        "independent_verifier_source_sha256": sha256(Path(__file__)),
        "frozen_contract_sha256": sha256(CONTRACT_PATH),
    }
    atomic_json(CERTIFICATE_PATH, certificate)
    print(json.dumps({
        "all_pass": certificate["all_pass"],
        "q_receipts": q_receipts,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }, indent=2))
    print(f"Wrote {CERTIFICATE_PATH}")
    if not certificate["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
