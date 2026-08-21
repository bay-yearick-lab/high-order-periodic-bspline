# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Certify finite-periodic first and direct-second derivative identities.

Galerkin source rows are assembled by piecewise Gauss integration of the
degree-p basis. Collocation source rows are assembled separately from image
sums of the degree-(2p+1) basis and its derivatives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import spline_analysis as sa


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evidence" / "finite_periodic_identity.json"
DEGREES = range(1, 14)
GRIDS = range(2, 65)
TOL_SOURCE = 2.0e-12
TOL_OPERATOR = 5.0e-11
TOL_POSITIVE = 1.0e-13


def integrate_overlap(p: int, shift: int, kind: str) -> float:
    """Integrate one whole-line overlap on every polynomial knot interval."""
    half = 0.5 * (p + 1)
    left = max(-half, shift - half)
    right = min(half, shift + half)
    if right <= left:
        return 0.0
    knots = np.unique(np.concatenate((
        np.arange(-half, half + 1.0, 1.0),
        shift + np.arange(-half, half + 1.0, 1.0),
        np.array([left, right]),
    )))
    knots = knots[(knots >= left) & (knots <= right)]
    nodes, weights = leggauss(p + 3)
    total = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a:
            continue
        x = 0.5 * ((b - a) * nodes + a + b)
        basis = sa.centered_cardinal_bspline(p, x)
        if kind == "mass":
            values = basis * sa.centered_cardinal_bspline(p, x - shift)
        elif kind == "first":
            values = basis * sa.centered_cardinal_bspline_derivative(p, x - shift)
        elif kind == "second":
            values = -sa.centered_cardinal_bspline_derivative(
                p, x
            ) * sa.centered_cardinal_bspline_derivative(p, x - shift)
        else:
            raise ValueError(kind)
        total += 0.5 * (b - a) * float(np.dot(weights, values))
    return total


def circularize(samples: dict[int, float], n: int) -> np.ndarray:
    """Fold a finite whole-line row into one periodic row."""
    row = np.zeros(n)
    for shift, value in samples.items():
        row[shift % n] += value
    return row


def image_row(degree: int, n: int, derivative: int) -> np.ndarray:
    """Evaluate the circular image sum of a centered cardinal basis row."""
    radius = degree + 2
    samples: dict[int, float] = {}
    for shift in range(-radius, radius + 1):
        x = np.array([float(shift)])
        if derivative == 0:
            value = sa.centered_cardinal_bspline(degree, x)[0]
        elif derivative == 1:
            value = -sa.centered_cardinal_bspline_derivative(degree, x)[0]
        elif derivative == 2:
            value = sa.centered_cardinal_bspline_second_derivative(degree, x)[0]
        else:
            raise ValueError(derivative)
        samples[shift] = float(value)
    return circularize(samples, n)


def circulant(row: np.ndarray) -> np.ndarray:
    """Build a circulant matrix from its first row."""
    return np.vstack([np.roll(row, i) for i in range(len(row))])


def main() -> int:
    records = []
    worst_source = 0.0
    worst_first = 0.0
    worst_second = 0.0
    minimum_symbol = float("inf")
    for p in DEGREES:
        support = range(-(p + 1), p + 2)
        mass_samples = {s: integrate_overlap(p, s, "mass") for s in support}
        first_samples = {s: integrate_overlap(p, s, "first") for s in support}
        second_samples = {s: integrate_overlap(p, s, "second") for s in support}
        for n in GRIDS:
            mass_row = circularize(mass_samples, n)
            first_row = circularize(first_samples, n)
            second_row = circularize(second_samples, n)
            sample_row = image_row(2 * p + 1, n, 0)
            d1_row = image_row(2 * p + 1, n, 1)
            d2_row = image_row(2 * p + 1, n, 2)
            source_error = max(
                float(np.max(np.abs(mass_row - sample_row))),
                float(np.max(np.abs(first_row - d1_row))),
                float(np.max(np.abs(second_row - d2_row))),
            )
            symbol_min = float(np.min(np.real(np.fft.fft(sample_row))))
            mass = circulant(mass_row)
            bmat = circulant(sample_row)
            d1_bmg = np.linalg.solve(mass, circulant(first_row))
            d1_bmc = np.linalg.solve(bmat, circulant(d1_row))
            d2_bmg = np.linalg.solve(mass, circulant(second_row))
            d2_bmc = np.linalg.solve(bmat, circulant(d2_row))
            first_error = float(np.max(np.abs(d1_bmg - d1_bmc))) / max(
                float(np.max(np.abs(d1_bmg))), 1.0
            )
            second_error = float(np.max(np.abs(d2_bmg - d2_bmc))) / max(
                float(np.max(np.abs(d2_bmg))), 1.0
            )
            worst_source = max(worst_source, source_error)
            worst_first = max(worst_first, first_error)
            worst_second = max(worst_second, second_error)
            minimum_symbol = min(minimum_symbol, symbol_min)
            records.append({
                "bmg_degree": p,
                "bmc_degree": 2 * p + 1,
                "N": n,
                "source_row_max_abs_error": source_error,
                "first_operator_relative_max_error": first_error,
                "direct_second_operator_relative_max_error": second_error,
                "minimum_sample_symbol": symbol_min,
            })
    passed = (
        worst_source <= TOL_SOURCE
        and worst_first <= TOL_OPERATOR
        and worst_second <= TOL_OPERATOR
        and minimum_symbol > TOL_POSITIVE
    )
    payload = {
        "mathematical_range": "all integer degrees p >= 1 and all integers N >= 2",
        "regression_range": {"bmg_degree": [1, 13], "N": [2, 64]},
        "paper_facing_ladder": "BMC3 through BMC13, equivalently BMG1 through BMG6",
        "assembly_independence": {
            "bmg": "piecewise Gauss integration followed by circular folding",
            "bmc": "higher-degree cardinal basis image sums followed by circular folding",
        },
        "tolerances": {
            "source_row": TOL_SOURCE,
            "operator": TOL_OPERATOR,
            "strict_symbol_floor": TOL_POSITIVE,
        },
        "summary": {
            "cells": len(records),
            "worst_source_row_max_abs_error": worst_source,
            "worst_first_operator_relative_max_error": worst_first,
            "worst_direct_second_operator_relative_max_error": worst_second,
            "minimum_sample_symbol": minimum_symbol,
            "status": "PASS" if passed else "FAIL",
        },
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
