"""Centered Lagrange evaluation layers of even width, as transfers and as weights.

An even number n_L of consecutive nodes carries the unique polynomial of degree
n_L - 1 through the nodal values. Written against the cell coordinate
xi in [-1/2, 1/2] measured from the nearest node j, the layer delivers three
channels at the query x = (j + xi) h,

    value             sum_a L_a(xi) u_{j+a},
    first derivative  h^{-1} sum_a L_a'(xi) u_{j+a},
    second derivative h^{-2} sum_a L_a''(xi) u_{j+a},

where a runs over the offsets the floor of x/h selects. Taking the base node
from floor and then n_L consecutive nodes starting n_L/2 - 1 to its left, which
is the Milestone 10 convention and the convention of the twelve-point wall
evaluator, gives the offsets

    a = -(n_L/2 - 1), ..., +n_L/2   for xi >= 0,
    a = -n_L/2,       ..., +n_L/2 - 1 for xi < 0,

both measured from the nearest node. The stencil therefore switches at xi = 0,
where the interpolant is continuous but its derivatives are not, and xi = 0
itself takes the xi >= 0 stencil, which is what floor does.

Applied to the grid mode u_j = exp(i j theta) the layer has the carrier-kept
transfer

    hat L^{(q)}(theta, xi) = sum_a L_a^{(q)}(xi) exp(i a theta),

with the same offsets. The exact channel is (i theta)^q exp(i theta xi), so the
transfer of the Gate 0 contract is exp(-i theta xi) hat L^{(q)} / (i theta)^q.
This module returns hat L^{(q)} and leaves the normalization to the caller.

The cardinal functions are carried in the power basis of the local coordinate
that runs over [0, 1) at offset zero, that is, in t = xi for xi >= 0 and
t = xi + 1 for xi < 0. The nodes are small integers and t lies in [0, 1), so
the Horner evaluation of the basis and of its first two derivatives agrees with
exact rational arithmetic to a relative 7e-16 at n_L = 8, 12 and 16, which is
the reason the power basis is used rather than a barycentric form.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SUPPORTED_NODES = (8, 12, 16)


def check_nodes(n_nodes: int) -> int:
    """Return n_nodes when the module carries that width."""
    n_nodes = int(n_nodes)
    if n_nodes not in SUPPORTED_NODES:
        raise ValueError(f"unsupported Lagrange width {n_nodes}")
    return n_nodes


@lru_cache(maxsize=None)
def basis(n_nodes: int):
    """Return the offsets and the power basis coefficients of the n_L cardinals.

    Node a sits at the integer offset a - (n_L/2 - 1) from floor(x/h), so the
    local coordinate runs over [0, 1) at offset zero.
    """
    n_nodes = check_nodes(n_nodes)
    offsets = np.arange(n_nodes) - (n_nodes // 2 - 1)
    coefficients = []
    for a in range(n_nodes):
        others = [offsets[b] for b in range(n_nodes) if b != a]
        coeffs = np.poly(others)
        scale = float(np.prod([offsets[a] - o for o in others]))
        coefficients.append(coeffs / scale)
    return offsets, tuple(coefficients)


def offsets(n_nodes: int) -> np.ndarray:
    """Return the integer offsets of the n_L nodes from floor(x/h)."""
    return basis(n_nodes)[0]


def nodal_weights(n_nodes: int, local, order: int = 0) -> np.ndarray:
    """Return the n_L weights at local coordinates in [0, 1).

    The result has shape (local.size, n_L) and carries no factor of h; the
    caller divides by h^order.
    """
    n_nodes = check_nodes(n_nodes)
    _, polynomials = basis(n_nodes)
    local = np.atleast_1d(np.asarray(local, dtype=float))
    out = np.empty((local.size, n_nodes))
    for a, coeffs in enumerate(polynomials):
        derived = np.polyder(coeffs, order) if order else coeffs
        out[:, a] = np.polyval(derived, local)
    return out


def apply_nodal(n_nodes: int, values: np.ndarray, x: np.ndarray, h: float,
                order: int = 0) -> np.ndarray:
    """Return the Lagrange channel of a periodic nodal field at the queries x."""
    n_nodes = check_nodes(n_nodes)
    node_offsets, _ = basis(n_nodes)
    values = np.asarray(values, dtype=float)
    npts = values.shape[0]
    t = x / h
    base = np.floor(t).astype(np.intp)
    local = t - base
    weights = nodal_weights(n_nodes, local, order)
    out = np.zeros(local.shape, dtype=float)
    for a, offset in enumerate(node_offsets):
        out += weights[:, a] * values[(base + int(offset)) % npts]
    return out / h ** order


def transfer(n_nodes: int, theta: np.ndarray, xi: np.ndarray,
             order: int = 0) -> np.ndarray:
    """Return the carrier-kept transfer sum_a L_a^{(order)}(xi) exp(i a theta).

    The stencil is the one floor selects, so the offsets are measured from the
    nearest node and switch at xi = 0.
    """
    n_nodes = check_nodes(n_nodes)
    node_offsets, _ = basis(n_nodes)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    shift = np.where(xi >= 0.0, 0, -1)
    local = xi + np.where(xi >= 0.0, 0.0, 1.0)
    weights = nodal_weights(n_nodes, local, order)
    out = np.zeros((theta.size, xi.size), dtype=complex)
    for a, offset in enumerate(node_offsets):
        phase = np.exp(1j * np.outer(theta, (offset + shift).astype(float)))
        out += weights[None, :, a] * phase
    return out
