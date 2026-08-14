#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


THISDIR = Path(__file__).resolve().parent
REPO = THISDIR.parents[2]
OUTDIR = REPO / "results" / "ibm_cert"
sys.path.insert(0, str(THISDIR))

from evaluator_common import (  # noqa: E402
    BAND_ENDPOINT,
    OFFSETS,
    Q_ORDERS,
    exact_nullspace,
    lagrange_weights,
    moment_residual,
    normalized_nullspace,
    transfer_error,
)


ANCHOR_COUNT = 1025
THETA_COUNT = 97
PHASE_COUNT = 64
CANDIDATE_PATH = OUTDIR / "evaluator_candidate.npz"
RECEIPT_PATH = OUTDIR / "evaluator_optimizer_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointwise_minimax(order: int, delta: float) -> tuple[np.ndarray, dict]:
    """Solve the frozen polygonal minimax problem at one fractional coordinate."""
    base = np.asarray(lagrange_weights(order, delta), dtype=float)
    if order == 12 or delta == 0.0:
        theta = np.linspace(0.0, BAND_ENDPOINT, 4097)
        err = np.abs(transfer_error(base, delta, theta))
        return base.astype(np.longdouble), {
            "lp_objective": float(np.max(err)),
            "sampled_transfer_max": float(np.max(err)),
            "l1": float(np.sum(np.abs(base))),
        }

    basis, scales = normalized_nullspace(order)
    nfree = basis.shape[1]
    theta = np.linspace(0.0, BAND_ENDPOINT, THETA_COUNT)
    phases = np.linspace(0.0, 2.0 * np.pi, PHASE_COUNT, endpoint=False)
    exponentials = np.exp(1j * np.outer(theta, OFFSETS.astype(float)))
    base_error = exponentials @ base - np.exp(1j * theta * delta)
    free_transfer = exponentials @ basis

    rows = []
    rhs = []
    for phase in phases:
        rotation = np.exp(-1j * phase)
        rows.append(np.c_[
            np.real(rotation * free_transfer),
            np.zeros((theta.size, 12)),
            -np.ones(theta.size),
        ])
        rhs.append(-np.real(rotation * base_error))

    rows.append(np.c_[basis, -np.eye(12), np.zeros(12)])
    rhs.append(-base)
    rows.append(np.c_[-basis, -np.eye(12), np.zeros(12)])
    rhs.append(base)
    rows.append(np.r_[np.zeros(nfree), np.ones(12), 0.0][None, :])
    rhs.append(np.array([4.0]))

    objective = np.r_[np.zeros(nfree + 12), 1.0]
    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.concatenate(rhs),
        bounds=[(None, None)] * nfree + [(0.0, None)] * 12 + [(0.0, None)],
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"Q12-{order} LP failed at delta={delta}: {result.message}")

    normalized_coordinate = result.x[:nfree]
    weights = base + basis @ normalized_coordinate
    if abs(delta - 0.5) < 4.0 * np.finfo(float).eps:
        weights = 0.5 * (weights + weights[::-1])

    integer_basis = np.asarray(exact_nullspace(order), dtype=float)
    coordinate = normalized_coordinate / scales
    if abs(delta - 0.5) < 4.0 * np.finfo(float).eps:
        coordinate = np.linalg.lstsq(integer_basis, weights - base, rcond=None)[0]
    weights_long = np.asarray(lagrange_weights(order, delta), dtype=np.longdouble)
    weights_long += exact_nullspace(order) @ coordinate.astype(np.longdouble)

    dense_theta = np.linspace(0.0, BAND_ENDPOINT, 4097)
    dense_error = np.abs(transfer_error(weights_long, delta, dense_theta))
    polygon_factor = 1.0 / np.cos(np.pi / PHASE_COUNT)
    return weights_long, {
        "coordinate": coordinate,
        "lp_objective": float(result.fun),
        "polygon_sample_upper": float(result.fun * polygon_factor),
        "sampled_transfer_max": float(np.max(dense_error)),
        "l1": float(np.sum(np.abs(weights_long))),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchors = np.linspace(0.0, 0.5, ANCHOR_COUNT)
    arrays: dict[str, np.ndarray] = {"anchors": anchors}
    summaries = {}
    started = time.perf_counter()

    for order in Q_ORDERS:
        if order == 12:
            residual = []
            l1_values = []
            transfer = []
            for delta in anchors:
                weights = lagrange_weights(12, float(delta))
                mom, const = moment_residual(weights, float(delta), 12)
                residual.append(max(mom, const))
                l1_values.append(float(np.sum(np.abs(weights))))
                theta = np.linspace(0.0, BAND_ENDPOINT, 4097)
                transfer.append(float(np.max(np.abs(transfer_error(weights, float(delta), theta)))))
            summaries["Q12-12"] = {
                "alias": "L12",
                "maximum_anchor_moment_residual": max(residual),
                "maximum_anchor_l1": max(l1_values),
                "maximum_anchor_sampled_transfer_error": max(transfer),
            }
            continue

        coordinate_count = 12 - order
        coordinates = np.empty((ANCHOR_COUNT, coordinate_count), dtype=float)
        maximum_lp = 0.0
        maximum_sampled = 0.0
        maximum_l1 = 0.0
        maximum_moment = 0.0
        maximum_constant = 0.0
        order_started = time.perf_counter()
        for index, delta in enumerate(anchors):
            weights, result = pointwise_minimax(order, float(delta))
            coordinate = result.get("coordinate", np.zeros(coordinate_count))
            coordinates[index] = coordinate
            mom, const = moment_residual(weights, float(delta), order)
            maximum_lp = max(maximum_lp, result["lp_objective"])
            maximum_sampled = max(maximum_sampled, result["sampled_transfer_max"])
            maximum_l1 = max(maximum_l1, result["l1"])
            maximum_moment = max(maximum_moment, mom)
            maximum_constant = max(maximum_constant, const)
            if index and index % 128 == 0:
                print(f"Q12-{order}: {index}/{ANCHOR_COUNT - 1} anchors", flush=True)
        arrays[f"q{order}_coordinates"] = coordinates
        summaries[f"Q12-{order}"] = {
            "anchor_count_half_cell": ANCHOR_COUNT,
            "maximum_anchor_lp_objective": maximum_lp,
            "maximum_anchor_sampled_transfer_error": maximum_sampled,
            "maximum_anchor_l1": maximum_l1,
            "maximum_anchor_moment_residual": maximum_moment,
            "maximum_anchor_constant_residual": maximum_constant,
            "elapsed_seconds": time.perf_counter() - order_started,
        }

    temporary = CANDIDATE_PATH.with_suffix(".npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, CANDIDATE_PATH)

    receipt = {
        "schema_version": "ibm-cert-evaluator-optimizer-v1",
        "wall_data_used": False,
        "offsets": OFFSETS.tolist(),
        "band_endpoint": BAND_ENDPOINT,
        "anchor_count_half_cell": ANCHOR_COUNT,
        "theta_seed_count": THETA_COUNT,
        "phase_direction_count": PHASE_COUNT,
        "optimizer": "scipy.optimize.linprog method highs",
        "parameterization": "order-matched Lq plus exact shifted finite-difference nullspace",
        "interpolation": "piecewise linear nullspace coordinates with exact moment base",
        "reflection": "w_r(1-delta)=w_(1-r)(delta)",
        "summaries": summaries,
        "candidate_path": str(CANDIDATE_PATH.relative_to(REPO)),
        "candidate_sha256": sha256(CANDIDATE_PATH),
        "producer_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(RECEIPT_PATH, receipt)
    print(f"Wrote {CANDIDATE_PATH}")
    print(f"Wrote {RECEIPT_PATH}")


if __name__ == "__main__":
    main()
