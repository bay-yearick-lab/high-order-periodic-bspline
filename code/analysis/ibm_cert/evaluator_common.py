from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


OFFSETS = np.arange(-5, 7, dtype=np.int64)
BAND_ENDPOINT = 2.0 * np.pi / 3.0
Q_ORDERS = (4, 6, 8, 10, 12)


@lru_cache(maxsize=None)
def lagrange_polynomial_coefficients(order: int) -> np.ndarray:
    """Return embedded power-basis coefficients for the centered order-q read."""
    if order % 2 or order < 2 or order > 12:
        raise ValueError(f"unsupported Lagrange order {order}")
    nodes = np.arange(1 - order // 2, 1 + order // 2, dtype=np.int64)
    coeff = np.zeros((12, order), dtype=np.longdouble)
    for node in nodes:
        poly = np.array([1.0], dtype=np.longdouble)
        denom = np.longdouble(1.0)
        for other in nodes:
            if other == node:
                continue
            poly = np.convolve(poly, np.array([-other, 1], dtype=np.longdouble))
            denom *= np.longdouble(node - other)
        coeff[int(node + 5), : poly.size] = poly / denom
    return coeff


def evaluate_polynomials(coeff: np.ndarray, delta: float, derivative: int = 0) -> np.ndarray:
    """Evaluate embedded Lagrange polynomials and their first derivatives."""
    work = np.asarray(coeff, dtype=np.longdouble).copy()
    for _ in range(derivative):
        if work.shape[1] <= 1:
            return np.zeros(work.shape[0], dtype=np.longdouble)
        powers = np.arange(1, work.shape[1], dtype=np.longdouble)
        work = work[:, 1:] * powers[None, :]
    x = np.longdouble(delta)
    out = np.zeros(work.shape[0], dtype=np.longdouble)
    for power in range(work.shape[1] - 1, -1, -1):
        out = out * x + work[:, power]
    return out


def lagrange_weights(order: int, delta: float) -> np.ndarray:
    """Evaluate the order-q periodic Lagrange weights in extended precision."""
    return evaluate_polynomials(lagrange_polynomial_coefficients(order), delta)


@lru_cache(maxsize=None)
def exact_nullspace(order: int) -> np.ndarray:
    """Return shifted finite-difference columns that annihilate q moments."""
    if order == 12:
        return np.zeros((12, 0), dtype=np.longdouble)
    columns = []
    for start in range(12 - order):
        column = np.zeros(12, dtype=np.longdouble)
        for k in range(order + 1):
            column[start + k] = np.longdouble((-1) ** k * math.comb(order, k))
        columns.append(column)
    return np.stack(columns, axis=1)


def normalized_nullspace(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a scaled nullspace for stable linear programming coordinates."""
    integer_basis = np.asarray(exact_nullspace(order), dtype=float)
    if integer_basis.shape[1] == 0:
        return integer_basis, np.ones(0)
    scales = np.linalg.norm(integer_basis, axis=0)
    return integer_basis / scales[None, :], scales


def moment_residual(weights: np.ndarray, delta: float, order: int) -> tuple[float, float]:
    """Return raw monomial and constant residuals in extended precision."""
    w = np.asarray(weights, dtype=np.longdouble)
    r = OFFSETS.astype(np.longdouble)
    d = np.longdouble(delta)
    residual = np.longdouble(0.0)
    for degree in range(order):
        residual = max(residual, abs(np.sum(w * r**degree) - d**degree))
    constant = abs(np.sum(w) - np.longdouble(1.0))
    return float(residual), float(constant)


def transfer_error(weights: np.ndarray, delta: float, theta: np.ndarray) -> np.ndarray:
    """Evaluate the complex one-dimensional value-transfer residual."""
    w = np.asarray(weights, dtype=float)
    t = np.asarray(theta, dtype=float)
    phase = np.exp(1j * np.outer(OFFSETS.astype(float), t))
    return w @ phase - np.exp(1j * float(delta) * t)


class QEvaluatorTable:
    """Moment-exact interpolant of pointwise minimax nullspace coordinates."""

    def __init__(self, anchors: np.ndarray, coordinates: dict[int, np.ndarray]):
        self.anchors = np.asarray(anchors, dtype=float)
        self.coordinates = {int(q): np.asarray(values, dtype=float)
                            for q, values in coordinates.items()}

    def weights(self, order: int, delta: float) -> np.ndarray:
        d = float(delta) % 1.0
        reflected = d > 0.5
        if reflected:
            d = 1.0 - d
        base = lagrange_weights(order, d)
        if order < 12:
            values = self.coordinates[order]
            position = d / self.anchors[-1] * (self.anchors.size - 1)
            left = min(int(np.floor(position)), self.anchors.size - 2)
            frac = np.longdouble(position - left)
            c0 = values[left].astype(np.longdouble)
            c1 = values[left + 1].astype(np.longdouble)
            coordinate = (np.longdouble(1.0) - frac) * c0 + frac * c1
            base = base + exact_nullspace(order) @ coordinate
        if reflected:
            base = base[::-1].copy()
        return base


def load_q_table(path) -> QEvaluatorTable:
    """Load the serialized Q12-q nullspace-coordinate tables."""
    with np.load(path) as data:
        anchors = data["anchors"]
        coordinates = {q: data[f"q{q}_coordinates"] for q in Q_ORDERS if q < 12}
    return QEvaluatorTable(anchors, coordinates)
