"""Gate 5 part 5b: the semi-Lagrangian matched-accuracy mission.

The physical problem is the exact periodic translation of a divergence-free
broadband packet on [0, 2 pi)^3. The packet

    u_K(x) = sum over |k_i| <= K of c_k exp(i k . x),   k . c_k = 0,
    c_{-k} = conj(c_k),   mean square of |u_K| equal to one,

is carried forward by a semi-Lagrangian step with a rigid shift: the new
nodal field is the old field read at the departure point x - delta, and at
every step the three first derivatives and the Laplacian are read at the same
departure points. Two shifts are used,

    delta_A = 2 pi (sqrt2 / 17, sqrt3 / 19, sqrt5 / 23),
    delta_B = 2 pi (sqrt7 / 29, sqrt11 / 31, sqrt13 / 37),

both irrational multiples of the mesh on every axis and every horizon, so no
step ever lands on a node.

Because the shift is rigid, every departure point carries the same cell
coordinate, so one step of every route is a tensor product of three periodic
one-dimensional operators. Writing s_d = delta_d / h = n_d + xi_d with n_d
integer, the axis operator is an integer shift by n_d followed by a short
stencil in xi_d, and its symbol at the grid angle theta is

    P_q(theta) = h^{-q} exp(-i n theta) (sum_l w_q[l] exp(-i l theta)) / b(theta)

for the one-field routes, with b the sample symbol of the generator and
w_q[l] = phi^{(q)}(l - xi), and

    I(theta) = exp(-i n theta) sum_l w[l] exp(-i l theta),
    P_1 = I (i kappa_1 / h),   P_2 = I (-kappa_2 / h^2)

for the operator-faithful SLC augmentation, whose derivative channels are
nodal SLC fields carried through the same degree-seven Lagrange layer. These
are the transfer functions of part 5a, evaluated at the cell coordinate the
shift selects; the test file checks that identification against
gate5a_offgrid.py directly.

The value channel of a step is the field that is fed forward, so after M steps
the value channel of grid mode m is amplified by A(m)^M with
A(m) = prod_d P_0(theta_d), while the derivative channels read the field of
step M - 1, giving the repeated-step transfers A^M and G_q A^{M-1} of the
contract.

Routes:

  champion        E_{6,2}(1.7701927977475826, 2.0606855355641316), degree 9,
                  one cyclic recovery per axis, then value, gradient and
                  Laplacian from the same coefficient tensor
  slc_operator   nodal SLC 1992 D1 for each gradient component and nodal
                  SLC D2 summed for the Laplacian, two pentadiagonal solves
                  per axis per field, then the degree-seven nearest-eight-node
                  Lagrange interpolant applied tensor-product to the value, the
                  three gradient and the Laplacian fields
  BMC9            E_{10,0}, degree 9, one field, the polynomial control

Six histories are recorded at every step: the relative L2 errors over the grid
of the value, of the three gradient components and of the Laplacian against the
exact translated packet, and the mass drift, the change of the discrete mean of
the carried field relative to its root mean square. A grid is admissible for a
horizon M when every one of the first five histories stays below one percent
through step M and the mass drift stays below 1e-12.

The prediction is the same modal arithmetic done with the transfers alone and
is written before any realized run.

Amendment 1. Three things above understate the competitor or the resolution of
the measurement. The augmentation carries an eight-point Lagrange layer while
the wall evaluator of the paper is the twelve-point layer, so the strongest
local evaluator available to the competitor was not the one run; the ladder
{32, 64, 128, 256, 512} doubles, so a minimum read off it overstates the node
separation whenever the true threshold sits just above a rung; and BMC11 was
never run. The amendment therefore adds, without changing anything above,

  * the augmentation at the Lagrange widths 12 and 16, built from the shared
    layer lagrange_layers.py, alongside the eight-point augmentation, which
    the amendment names slc_operator_L8 and which runs.csv keeps calling
    slc_operator;
  * BMC11, E_{12,0}, degree 11, one field;
  * the ladder N in {32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256,
    320, 384}, whose successive ratios are at most 1.25, so a minimum read off
    it is within that factor of the threshold it stands for;
  * the worst of the five channel histories on the rung below each realized
    minimum, as a multiple of the one percent bar, so the margin by which that
    rung fails is visible;
  * the timing at the realized minima of the champion, BMC9, BMC11 and the
    twelve- and sixteen-point augmentations, and of the eight-point one as
    well when the stage is asked for every route, together with the
    equal-grid timing of all six routes at N = 128;
  * the rule of the contract applied verbatim against each Lagrange width
    separately.

The amendment block of prediction.json is written before any amendment run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path

# The linear algebra of one mission runs on one thread, so that the parallelism
# of the accuracy stage is across processes and the timing stage is single
# threaded by construction. The transforms of the diagnostics take their thread
# count from their own argument.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.fft import irfftn, rfftn
from scipy.linalg import solveh_banded
from scipy.ndimage import convolve1d

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as cmp  # noqa: E402
import gate5a_offgrid as g5a  # noqa: E402
import lagrange_layers as ll  # noqa: E402
from evaluator import ExponentialEvaluator  # noqa: E402
from family import ExponentialGenerator  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate5" / "5b"
HISTORIES = OUT / "histories"
FIGURES = OUT / "figures"
CONTRACT = ROOT / "results" / "exponential_family" / "gate5" / "contract.md"
MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"

CHAMPION = (6, (1.7701927977475826, 2.0606855355641316))
SHIFTS = {
    "A": (np.sqrt(2.0) / 17.0, np.sqrt(3.0) / 19.0, np.sqrt(5.0) / 23.0),
    "B": (np.sqrt(7.0) / 29.0, np.sqrt(11.0) / 31.0, np.sqrt(13.0) / 37.0),
}
KS = (12, 24)
LADDER = (32, 64, 128, 256, 512)
HORIZONS = (1, 8, 32)
STEPS = max(HORIZONS)
ROUTE_NAMES = ("champion", "slc_operator", "BMC9")
CHANNELS = ("value", "grad_x", "grad_y", "grad_z", "laplacian")
HISTORY_KEYS = CHANNELS + ("mass",)
TOL = 1e-2
MASS_TOL = 1e-12
PACKET_SEED = 20260913
EQUAL_GRID = 128
TIMING_REPEATS = 3
AGREEMENT_FACTOR = 2.0
SAMPLE_FLOOR = 1e-14
FFT_WORKERS = 8
COHERENCE_GRID = 64
COHERENCE_EPS = 0.02

# Amendment 1. The eight-point augmentation keeps its original name in
# runs.csv and in the history files, so the amendment names it explicitly and
# translates that one name back.
LAGRANGE_WIDTHS = (8, 12, 16)
AMENDMENT_LADDER = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256,
                    320, 384)
AMENDMENT_ROUTE_NAMES = (("champion", "BMC9", "BMC11")
                         + tuple(f"slc_operator_L{n}" for n in LAGRANGE_WIDTHS))
AMENDMENT_TIMING_NAMES = ("champion", "BMC9", "BMC11", "slc_operator_L12",
                          "slc_operator_L16")
ORIGINAL_ALIAS = {"slc_operator_L8": "slc_operator"}
LAGRANGE_DEGREE_WORDS = {8: "seven", 12: "eleven", 16: "fifteen"}
LAGRANGE_NODE_WORDS = {8: "eight", 12: "twelve", 16: "sixteen"}


def csv_route(name: str) -> str:
    """Return the runs.csv name of a route, the original name where one exists."""
    return ORIGINAL_ALIAS.get(name, name)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shift_vector(name: str) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(SHIFTS[name], dtype=float)


def member_record() -> dict:
    """Return the frozen champion member and whether the route matches it."""
    data = json.loads(MEMBER.read_text())
    return {"file": str(MEMBER.relative_to(ROOT)), "sha256": sha256(MEMBER),
            "m": int(data["m"]), "r": int(data["r"]), "p": int(data["p"]),
            "omegas": [float(v) for v in data["omegas"]],
            "matches_route": (int(data["m"]) == CHAMPION[0]
                              and tuple(float(v) for v in data["omegas"]) == CHAMPION[1])}


# ----------------------------------------------------------------------
# the divergence-free broadband packet
# ----------------------------------------------------------------------

def packet_modes(K: int, seed: int = PACKET_SEED):
    """Return the wave vectors and amplitudes of the packet u_K.

    Every integer wave vector with |k_i| <= K other than the origin carries a
    complex amplitude; the amplitudes of the lexicographic half space are drawn
    once from the seeded generator and projected onto the plane orthogonal to
    k, and the other half is the conjugate, so the field is real and
    divergence-free. The overall scale sets the mean square of |u_K| to one.
    """
    rng = np.random.default_rng(seed + K)
    axis = np.arange(-K, K + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    k1, k2, k3 = grid[:, 0], grid[:, 1], grid[:, 2]
    keep = (k3 > 0) | ((k3 == 0) & ((k2 > 0) | ((k2 == 0) & (k1 > 0))))
    half = grid[keep]
    draw = (rng.standard_normal((half.shape[0], 3))
            + 1j * rng.standard_normal((half.shape[0], 3)))
    kk = half.astype(float)
    norm2 = np.sum(kk * kk, axis=1)
    draw -= kk * (np.sum(kk * draw, axis=1) / norm2)[:, None]
    scale = 1.0 / np.sqrt(2.0 * np.sum(np.abs(draw) ** 2))
    draw *= scale
    modes = np.concatenate([half, -half], axis=0)
    amps = np.concatenate([draw, np.conj(draw)], axis=0)
    return modes, amps


def grid_groups(modes: np.ndarray, npts: int):
    """Return the half-spectrum grouping of the modes by their grid alias.

    Two wave vectors that differ by npts on every axis are the same grid mode,
    so the realized field only ever sees the sum of their amplitudes. The
    selection keeps the modes whose third alias index lies in [0, npts/2],
    which is exactly the half spectrum of the real transform.
    """
    alias = modes % npts
    keep = alias[:, 2] <= npts // 2
    alias = alias[keep]
    flat = (alias[:, 0] * npts + alias[:, 1]) * (npts // 2 + 1) + alias[:, 2]
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    unique, start = np.unique(flat_sorted, return_index=True)
    group = np.empty(flat.shape, dtype=np.intp)
    group[order] = np.searchsorted(unique, flat_sorted)
    index = np.empty((unique.size, 3), dtype=np.intp)
    index[:, 2] = unique % (npts // 2 + 1)
    rest = unique // (npts // 2 + 1)
    index[:, 1] = rest % npts
    index[:, 0] = rest // npts
    weight = np.where((index[:, 2] == 0) | (index[:, 2] == npts // 2), 1.0, 2.0)
    return keep, group, index, weight


def half_sum(group: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Return the per-group sums of a complex array."""
    real = np.bincount(group, weights=values.real, minlength=count)
    imag = np.bincount(group, weights=values.imag, minlength=count)
    return real + 1j * imag


def synthesize(index: np.ndarray, coeffs: np.ndarray, npts: int,
               workers: int = 1) -> np.ndarray:
    """Return the real nodal field whose half-spectrum coefficients are coeffs."""
    spectrum = np.zeros((npts, npts, npts // 2 + 1), dtype=complex)
    spectrum[index[:, 0], index[:, 1], index[:, 2]] = coeffs * float(npts) ** 3
    return irfftn(spectrum, s=(npts, npts, npts), workers=workers)


def plane_weights(npts: int) -> np.ndarray:
    """Return the Parseval weight of each plane of the half spectrum.

    The real transform stores the modes with third index in [0, npts/2]; every
    plane but the first and the last stands for itself and its conjugate.
    """
    j = np.arange(npts // 2 + 1)
    return np.where((j == 0) | (j == npts // 2), 1.0, 2.0)


def weighted_sumsq(spectrum: np.ndarray, weight: np.ndarray) -> float:
    """Return the Parseval sum of |spectrum|^2 over the full grid of modes."""
    total = 0.0
    for j in range(spectrum.shape[2]):
        slab = spectrum[:, :, j]
        total += float(weight[j]) * float(np.vdot(slab, slab).real)
    return total


# ----------------------------------------------------------------------
# periodic banded solves
# ----------------------------------------------------------------------

class CyclicBanded:
    """The inverse of a symmetric banded circulant, by a banded solve.

    The circulant with first row a_0, a_1, ..., a_q is the banded Toeplitz
    matrix T plus a correction supported on the two q by q corners, which is
    U K U^T for U the columns of the identity on the first and last q indices.
    The Woodbury identity in the form

        A^{-1} = T^{-1} - G (I + K U^T G)^{-1} K U^T T^{-1},   G = T^{-1} U,

    never forms the inverse of K, so a vanishing outermost coefficient is
    harmless. T is positive definite because its symbol is the positive symbol
    of the circulant, so the banded solve is a Cholesky solve.
    """

    def __init__(self, coeffs, npts: int) -> None:
        coeffs = np.asarray(coeffs, dtype=float)
        self.coeffs = coeffs
        self.npts = int(npts)
        q = coeffs.size - 1
        self.q = q
        if self.npts < 2 * q + 1:
            raise ValueError("The grid is too small for the bandwidth.")
        ab = np.zeros((q + 1, npts))
        for d in range(q + 1):
            ab[q - d, d:] = coeffs[d]
        self.ab = ab
        rows = np.concatenate([np.arange(q), np.arange(npts - q, npts)])
        self.rows = rows
        block = np.zeros((2 * q, 2 * q))
        for i in range(q):
            for j in range(q):
                order = q - j + i
                if order <= q:
                    block[i, q + j] = coeffs[order]
                    block[q + j, i] = coeffs[order]
        self.block = block
        unit = np.zeros((npts, 2 * q))
        unit[rows, np.arange(2 * q)] = 1.0
        gmat = solveh_banded(ab, unit, lower=False)
        self.gmat = gmat
        middle = np.eye(2 * q) + block @ gmat[rows, :]
        self.inner = np.linalg.solve(middle, block)

    def solve(self, rhs: np.ndarray, overwrite: bool = False) -> np.ndarray:
        """Solve with right hand sides stored as rows of rhs, shape (nrhs, n)."""
        columns = np.asarray(rhs).T
        y = solveh_banded(self.ab, columns, lower=False, overwrite_b=overwrite)
        y -= self.gmat @ (self.inner @ y[self.rows, :])
        return y.T

    def symbol(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        out = np.full(theta.shape, self.coeffs[0])
        for d in range(1, self.coeffs.size):
            out = out + 2.0 * self.coeffs[d] * np.cos(d * theta)
        return out


def solve_axis(field: np.ndarray, axis: int, solver: CyclicBanded) -> np.ndarray:
    """Apply the inverse of a periodic banded operator along one axis."""
    moved = np.moveaxis(field, axis, -1)
    shape = moved.shape
    rhs = np.array(moved, order="C", copy=True).reshape(-1, shape[-1])
    out = solver.solve(rhs, overwrite=True).reshape(shape)
    return np.ascontiguousarray(np.moveaxis(out, -1, axis))


def apply_stencil(field: np.ndarray, axis: int, offsets: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """Return out[j] = sum_l weights[l] field[j - l] along one axis.

    The offsets are consecutive integers; convolve1d places weight index i at
    the shift i - center with center = len(weights) // 2 + origin, so the
    origin is fixed by the first offset.
    """
    lo = int(offsets[0])
    origin = -lo - weights.size // 2
    return convolve1d(field, weights, axis=axis, mode="wrap", origin=origin)


# ----------------------------------------------------------------------
# axis plans and routes
# ----------------------------------------------------------------------

@dataclass
class AxisPlan:
    roll: int
    offsets: np.ndarray
    weights: tuple


@dataclass
class RoutePlan:
    npts: int
    step: float
    rolls: tuple
    axes: list
    solvers: dict = dataclass_field(default_factory=dict)


class SplineMission:
    """One exponential-family member carried as a single coefficient tensor."""

    kind = "one field"

    def __init__(self, name: str, m: int, omegas=()) -> None:
        self.name = name
        self.evaluator = ExponentialEvaluator(m, omegas)
        self.generator = ExponentialGenerator(self.evaluator.m, self.evaluator.omegas)
        samples = self.evaluator.value(np.arange(0, self.evaluator.n // 2 + 1))
        nonzero = np.nonzero(np.abs(samples) > SAMPLE_FLOOR)[0]
        self.samples = samples[: nonzero[-1] + 1]
        self.taps = self.evaluator.n
        self.solves_per_step = 3
        self.fields = 1

    def description(self) -> str:
        return (f"{self.evaluator.label()}, degree {self.evaluator.p}, support "
                f"[-{self.evaluator.n / 2:g}, {self.evaluator.n / 2:g}], "
                f"{self.taps} taps per axis")

    def axis_plan(self, npts: int, delta: float) -> AxisPlan:
        step = 2.0 * np.pi / npts
        s = delta / step
        roll = int(np.rint(s))
        xi = s - roll
        radius = self.evaluator.n / 2.0
        lo = int(np.floor(xi - radius)) + 1
        offsets = lo + np.arange(self.evaluator.n)
        weights = tuple(self.evaluator.derivative(offsets - xi, q) / step ** q
                        for q in range(3))
        return AxisPlan(roll=roll, offsets=offsets, weights=weights)

    def plan(self, npts: int, delta: np.ndarray) -> RoutePlan:
        axes = [self.axis_plan(npts, float(delta[d])) for d in range(3)]
        plan = RoutePlan(npts=npts, step=2.0 * np.pi / npts,
                         rolls=tuple(a.roll for a in axes), axes=axes)
        plan.solvers["recovery"] = CyclicBanded(self.samples, npts)
        return plan

    def axis_symbols(self, plan: RoutePlan, axis: int) -> tuple:
        npts = plan.npts
        theta = 2.0 * np.pi * np.arange(npts) / npts
        ax = plan.axes[axis]
        carrier = np.exp(-1j * ax.roll * theta)
        b = plan.solvers["recovery"].symbol(theta)
        out = []
        for q in range(3):
            phase = np.exp(-1j * np.outer(theta, ax.offsets.astype(float)))
            out.append(carrier * (phase @ ax.weights[q]) / b)
        return tuple(out)

    def step_channels(self, plan: RoutePlan, values: np.ndarray, sink) -> np.ndarray:
        """Run one step, hand every channel to the sink and return the new field.

        The seven tensor terms of the five channels are assembled with the
        second-derivative factors taken first, so the axis whose pass is the
        most expensive carries three passes and the cheapest carries seven.
        """
        coeff = np.roll(values, plan.rolls, axis=(0, 1, 2))
        for d in range(3):
            coeff = solve_axis(coeff, d, plan.solvers["recovery"])
        w = [plan.axes[d].weights for d in range(3)]
        o = [plan.axes[d].offsets for d in range(3)]
        first = apply_stencil(coeff, 0, o[0], w[0][1])
        mid = apply_stencil(first, 1, o[1], w[1][0])
        del first
        grad = apply_stencil(mid, 2, o[2], w[2][0])
        del mid
        sink("grad_x", grad)
        del grad
        first = apply_stencil(coeff, 0, o[0], w[0][2])
        mid = apply_stencil(first, 1, o[1], w[1][0])
        del first
        lap = apply_stencil(mid, 2, o[2], w[2][0])
        del mid
        first = apply_stencil(coeff, 0, o[0], w[0][0])
        del coeff
        mid = apply_stencil(first, 1, o[1], w[1][1])
        grad = apply_stencil(mid, 2, o[2], w[2][0])
        del mid
        sink("grad_y", grad)
        del grad
        mid = apply_stencil(first, 1, o[1], w[1][2])
        lap += apply_stencil(mid, 2, o[2], w[2][0])
        del mid
        mid = apply_stencil(first, 1, o[1], w[1][0])
        del first
        value = apply_stencil(mid, 2, o[2], w[2][0])
        sink("value", value)
        grad = apply_stencil(mid, 2, o[2], w[2][1])
        sink("grad_z", grad)
        del grad
        lap += apply_stencil(mid, 2, o[2], w[2][2])
        del mid
        sink("laplacian", lap)
        return value


class SLCMission:
    """Nodal SLC 1992 derivatives carried through a Lagrange layer of width n_L.

    The width enters only through the layer: the nodal D1 and D2 fields are the
    same at every width, and the value, gradient and Laplacian channels are the
    centered n_L point interpolant of lagrange_layers.py applied to them.
    """

    kind = "three fields"

    def __init__(self, n_nodes: int = g5a.LAGRANGE_POINTS) -> None:
        self.n_nodes = ll.check_nodes(n_nodes)
        self.name = csv_route(f"slc_operator_L{self.n_nodes}")
        self.kappa1 = cmp.compact_kappa1(cmp.SLC_D1["alpha"], cmp.SLC_D1["a"])
        self.kappa2 = cmp.compact_kappa2(cmp.SLC_D2["alpha"], cmp.SLC_D2["a0"],
                                         cmp.SLC_D2["a"])
        self.left1 = np.concatenate([[1.0], np.asarray(cmp.SLC_D1["alpha"])])
        self.left2 = np.concatenate([[1.0], np.asarray(cmp.SLC_D2["alpha"])])
        a1 = np.asarray(cmp.SLC_D1["a"], dtype=float)
        self.right1_offsets = np.arange(-a1.size, a1.size + 1)
        self.right1 = np.concatenate([a1[::-1], [0.0], -a1])
        a2 = np.asarray(cmp.SLC_D2["a"], dtype=float)
        self.right2_offsets = np.arange(-a2.size, a2.size + 1)
        self.right2 = np.concatenate([a2[::-1], [float(cmp.SLC_D2["a0"])], a2])
        self.taps = self.n_nodes
        self.solves_per_step = 6
        self.fields = 3

    def description(self) -> str:
        degree = LAGRANGE_DEGREE_WORDS[self.n_nodes]
        nodes = LAGRANGE_NODE_WORDS[self.n_nodes]
        return ("SLC 1992 nodal D1 and D2, two pentadiagonal solves per axis per "
                f"field, then the degree {degree} Lagrange interpolant on the nearest "
                f"{nodes} nodes, {self.n_nodes} taps per axis per field")

    def axis_plan(self, npts: int, delta: float) -> AxisPlan:
        step = 2.0 * np.pi / npts
        s = delta / step
        roll = int(np.floor(s)) + 1
        local = 1.0 - (s - np.floor(s))
        weights = ll.nodal_weights(self.n_nodes, np.array([local]), 0)[0]
        offsets = (self.n_nodes // 2 - 1) - np.arange(self.n_nodes)
        order = np.argsort(offsets)
        return AxisPlan(roll=roll, offsets=offsets[order],
                        weights=(weights[order],))

    def plan(self, npts: int, delta: np.ndarray) -> RoutePlan:
        axes = [self.axis_plan(npts, float(delta[d])) for d in range(3)]
        plan = RoutePlan(npts=npts, step=2.0 * np.pi / npts,
                         rolls=tuple(a.roll for a in axes), axes=axes)
        plan.solvers["left1"] = CyclicBanded(self.left1, npts)
        plan.solvers["left2"] = CyclicBanded(self.left2, npts)
        return plan

    def axis_symbols(self, plan: RoutePlan, axis: int) -> tuple:
        npts = plan.npts
        step = plan.step
        theta = 2.0 * np.pi * np.arange(npts) / npts
        ax = plan.axes[axis]
        carrier = np.exp(-1j * ax.roll * theta)
        phase = np.exp(-1j * np.outer(theta, ax.offsets.astype(float)))
        inter = carrier * (phase @ ax.weights[0])
        k1 = np.asarray(self.kappa1(theta), dtype=float)
        k2 = np.asarray(self.kappa2(theta), dtype=float)
        return (inter, inter * 1j * k1 / step, inter * (-k2) / step ** 2)

    def _nodal_d1(self, plan: RoutePlan, field: np.ndarray, axis: int) -> np.ndarray:
        out = apply_stencil(field, axis, self.right1_offsets, self.right1)
        out = solve_axis(out, axis, plan.solvers["left1"])
        out /= plan.step
        return out

    def _nodal_d2(self, plan: RoutePlan, field: np.ndarray, axis: int) -> np.ndarray:
        out = apply_stencil(field, axis, self.right2_offsets, self.right2)
        out = solve_axis(out, axis, plan.solvers["left2"])
        out /= plan.step ** 2
        return out

    def _interpolate(self, plan: RoutePlan, field: np.ndarray) -> np.ndarray:
        out = field
        for d in (2, 1, 0):
            ax = plan.axes[d]
            out = apply_stencil(out, d, ax.offsets, ax.weights[0])
        return out

    def step_channels(self, plan: RoutePlan, values: np.ndarray, sink) -> np.ndarray:
        rolled = np.roll(values, plan.rolls, axis=(0, 1, 2))
        value = self._interpolate(plan, rolled)
        sink("value", value)
        for d, label in enumerate(("grad_x", "grad_y", "grad_z")):
            nodal = self._nodal_d1(plan, rolled, d)
            read = self._interpolate(plan, nodal)
            del nodal
            sink(label, read)
            del read
        lap = None
        for d in range(3):
            nodal = self._nodal_d2(plan, rolled, d)
            lap = nodal if lap is None else lap + nodal
            del nodal
        lap = self._interpolate(plan, lap)
        sink("laplacian", lap)
        return value


def build_routes() -> dict:
    """Return the routes by their runs.csv name, the original three first."""
    routes = {
        "champion": SplineMission("champion", *CHAMPION),
        "slc_operator": SLCMission(),
        "BMC9": SplineMission("BMC9", 10, ()),
        "BMC11": SplineMission("BMC11", 12, ()),
    }
    for n_nodes in LAGRANGE_WIDTHS:
        if n_nodes != g5a.LAGRANGE_POINTS:
            routes[f"slc_operator_L{n_nodes}"] = SLCMission(n_nodes)
    return routes


def mode_transfers(route, plan: RoutePlan, alias: np.ndarray) -> dict:
    """Return the one-step channel transfers of the listed grid modes.

    The alias array holds the grid index of each mode on each axis; the tensor
    product structure means the value transfer is the product of the three axis
    value symbols and the derivative transfers replace one factor.
    """
    p0, p1, p2 = [], [], []
    for d in range(3):
        s0, s1, s2 = route.axis_symbols(plan, d)
        p0.append(s0[alias[:, d]])
        p1.append(s1[alias[:, d]])
        p2.append(s2[alias[:, d]])
    value = p0[0] * p0[1] * p0[2]
    grads = [p1[0] * p0[1] * p0[2], p0[0] * p1[1] * p0[2], p0[0] * p0[1] * p1[2]]
    lap = (p2[0] * p0[1] * p0[2] + p0[0] * p2[1] * p0[2] + p0[0] * p0[1] * p2[2])
    return {"value": value, "grad_x": grads[0], "grad_y": grads[1],
            "grad_z": grads[2], "laplacian": lap}


# ----------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------

def exact_multipliers(modes: np.ndarray) -> dict:
    k = modes.astype(float)
    return {"value": np.ones(k.shape[0]),
            "grad_x": 1j * k[:, 0],
            "grad_y": 1j * k[:, 1],
            "grad_z": 1j * k[:, 2],
            "laplacian": -np.sum(k * k, axis=1)}


def predicted_histories(route, npts: int, modes: np.ndarray, amps: np.ndarray,
                        delta: np.ndarray, steps: int = STEPS) -> dict:
    """Return the six histories computed from the transfers and the amplitudes."""
    keep, group, index, weight = grid_groups(modes, npts)
    kept_modes = modes[keep]
    kept_amps = amps[keep]
    count = index.shape[0]
    plan = route.plan(npts, delta)
    transfers = mode_transfers(route, plan, index)
    mult = exact_multipliers(kept_modes)
    phase_one = np.exp(-1j * (kept_modes.astype(float) @ delta))
    carried = np.stack([half_sum(group, kept_amps[:, c], count) for c in range(3)])
    history = {key: np.zeros(steps) for key in HISTORY_KEYS}
    mean_start = carried[:, 0] if index[0].tolist() == [0, 0, 0] else np.zeros(3)
    rms = np.sqrt(sum(float(np.sum(weight * np.abs(carried[c]) ** 2))
                      for c in range(3)))
    amp = np.ones(count, dtype=complex)
    phase = np.ones(kept_modes.shape[0], dtype=complex)
    for step in range(1, steps + 1):
        previous = amp.copy()
        amp = amp * transfers["value"]
        phase = phase * phase_one
        for key in CHANNELS:
            factor = amp if key == "value" else previous * transfers[key]
            err = 0.0
            den = 0.0
            for c in range(3):
                exact = half_sum(group, kept_amps[:, c] * mult[key] * phase, count)
                num = factor * carried[c]
                err += float(np.sum(weight * np.abs(num - exact) ** 2))
                den += float(np.sum(weight * np.abs(exact) ** 2))
            history[key][step - 1] = np.sqrt(err / den) if den > 0.0 else 0.0
        if index[0].tolist() == [0, 0, 0]:
            drift = np.abs(amp[0] * carried[:, 0] - mean_start)
            history["mass"][step - 1] = float(np.max(drift)) / rms
        else:
            history["mass"][step - 1] = 0.0
    return history


def admissible_horizon(history: dict, horizon: int) -> bool:
    for key in CHANNELS:
        if float(np.max(history[key][:horizon])) >= TOL:
            return False
    return float(np.max(history["mass"][:horizon])) < MASS_TOL


def minimum_grid(histories_by_n: dict, horizon: int, ladder=LADDER):
    for npts in ladder:
        if npts in histories_by_n and admissible_horizon(histories_by_n[npts], horizon):
            return npts
    return None


def stage_predict() -> dict:
    routes = build_routes()
    table = []
    minima = {}
    for name in ROUTE_NAMES:
        route = routes[name]
        for K in KS:
            modes, amps = packet_modes(K)
            for shift in SHIFTS:
                delta = shift_vector(shift)
                by_n = {}
                for npts in LADDER:
                    history = predicted_histories(route, npts, modes, amps, delta)
                    by_n[npts] = history
                    for horizon in HORIZONS:
                        row = {"route": name, "K": K, "shift": shift, "N": npts,
                               "M": horizon}
                        for key in HISTORY_KEYS:
                            row[key] = float(history[key][horizon - 1])
                            row[f"{key}_max"] = float(np.max(history[key][:horizon]))
                        row["admissible"] = bool(admissible_horizon(history, horizon))
                        table.append(row)
                for horizon in HORIZONS:
                    minima[f"{name}|{K}|{shift}|{horizon}"] = minimum_grid(by_n, horizon)
    combined = {}
    for name in ROUTE_NAMES:
        for K in KS:
            for horizon in HORIZONS:
                per_shift = [minima[f"{name}|{K}|{shift}|{horizon}"] for shift in SHIFTS]
                worst = None if any(v is None for v in per_shift) else max(per_shift)
                combined[f"{name}|{K}|{horizon}"] = worst
    ratios = {}
    for K in KS:
        for horizon in HORIZONS:
            champ = combined[f"champion|{K}|{horizon}"]
            slc = combined[f"slc_operator|{K}|{horizon}"]
            key = f"{K}|{horizon}"
            if champ is None or slc is None:
                ratios[key] = None
            else:
                ratios[key] = {"champion_N": champ, "slc_N": slc,
                               "grid_ratio": slc / champ,
                               "node_ratio_slc_over_champion": (slc / champ) ** 3,
                               "champion_at_most_half": champ * 2 <= slc}
    payload = {
        "gate": "5b",
        "written": now(),
        "contract": str(CONTRACT.relative_to(ROOT)),
        "statement": (
            "The repeated-step transfer of a grid mode is the product of the three "
            "axis symbols of the route at that mode's angles, raised to the power M "
            "for the value channel and multiplied by one derivative factor in place "
            "of the last value factor for the derivative channels. The predicted "
            "histories below are the exact modal L2 errors that arithmetic implies, "
            "including the aliasing of the packet onto the coarse grids."),
        "packet": {"seed": PACKET_SEED, "K": list(KS),
                   "modes_per_K": {str(K): int(packet_modes(K)[0].shape[0])
                                   for K in KS}},
        "shifts": {name: list(shift_vector(name)) for name in SHIFTS},
        "ladder": list(LADDER),
        "horizons": list(HORIZONS),
        "tolerance": TOL,
        "mass_tolerance": MASS_TOL,
        "rows": table,
        "minimum_admissible_N_per_shift": minima,
        "minimum_admissible_N": combined,
        "node_count_ratio": ratios,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prediction.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ----------------------------------------------------------------------
# realized mission
# ----------------------------------------------------------------------

class Diagnostics:
    """Measure the six histories of one step against the exact translated packet."""

    def __init__(self, modes, amps, npts, delta, workers=FFT_WORKERS) -> None:
        self.npts = npts
        self.workers = workers
        keep, group, index, weight = grid_groups(modes, npts)
        self.group = group
        self.index = index
        self.weight = weight
        self.plane = plane_weights(npts)
        self.count = index.shape[0]
        self.kept_modes = modes[keep]
        self.kept_amps = amps[keep]
        self.mult = exact_multipliers(self.kept_modes)
        self.phase_one = np.exp(-1j * (self.kept_modes.astype(float) @ delta))
        self.phase = np.ones(self.kept_modes.shape[0], dtype=complex)
        self.carried = np.stack([half_sum(group, self.kept_amps[:, c], self.count)
                                 for c in range(3)])
        self.rms = np.sqrt(sum(float(np.sum(weight * np.abs(self.carried[c]) ** 2))
                               for c in range(3)))
        self.zero = (index[0].tolist() == [0, 0, 0])
        self.mean_start = np.real(self.carried[:, 0]) if self.zero else np.zeros(3)
        self.mean_now = np.zeros(3)
        self.reset()

    def initial_fields(self):
        return [synthesize(self.index, self.carried[c], self.npts, self.workers)
                for c in range(3)]

    def advance(self) -> None:
        self.phase = self.phase * self.phase_one

    def reset(self) -> None:
        self.error = {key: 0.0 for key in CHANNELS}
        self.denominator = {key: 0.0 for key in CHANNELS}

    def exact_coefficients(self, key: str, component: int) -> np.ndarray:
        weights = self.kept_amps[:, component] * self.mult[key] * self.phase
        return half_sum(self.group, weights, self.count)

    def measure(self, key: str, component: int, field: np.ndarray) -> None:
        spectrum = rfftn(field, workers=self.workers)
        exact = self.exact_coefficients(key, component) * float(self.npts) ** 3
        if key == "value" and component >= 0:
            self.mean_now[component] = float(spectrum[0, 0, 0].real) / self.npts ** 3
        spectrum[self.index[:, 0], self.index[:, 1], self.index[:, 2]] -= exact
        self.error[key] += weighted_sumsq(spectrum, self.plane)
        self.denominator[key] += float(np.sum(self.weight * np.abs(exact) ** 2))
        del spectrum

    def history_row(self) -> dict:
        row = {}
        for key in CHANNELS:
            den = self.denominator[key]
            row[key] = float(np.sqrt(self.error[key] / den)) if den > 0.0 else 0.0
        row["mass"] = float(np.max(np.abs(self.mean_now - self.mean_start))) / self.rms
        return row


def run_mission(route, npts: int, delta: np.ndarray, modes, amps, steps: int,
                diagnostics: bool = True, workers: int = FFT_WORKERS):
    """Run the full mission and return the per-step histories."""
    diag = Diagnostics(modes, amps, npts, delta, workers)
    fields = diag.initial_fields()
    plan = route.plan(npts, delta)
    history = {key: np.zeros(steps) for key in HISTORY_KEYS}
    for step in range(1, steps + 1):
        diag.advance()
        diag.reset()
        for component in range(3):
            if diagnostics:
                def sink(key, array, component=component):
                    diag.measure(key, component, array)
            else:
                def sink(key, array):
                    return None
            fields[component] = route.step_channels(plan, fields[component], sink)
        if diagnostics:
            row = diag.history_row()
            for key in HISTORY_KEYS:
                history[key][step - 1] = row[key]
    return history, fields


def stage_accuracy(jobs: int) -> list:
    """Climb the ladder for every route, packet and shift until a grid passes."""
    pending = [(name, K, shift) for name in ROUTE_NAMES for K in KS for shift in SHIFTS]
    rows = []
    for npts in LADDER:
        if not pending:
            break
        workers = min(jobs, len(pending) if npts <= 128 else (2 if npts == 256 else 1))
        threads = max(1, FFT_WORKERS // max(workers, 1))
        tasks = [(name, K, shift, npts, threads) for (name, K, shift) in pending]
        results = dispatch(tasks, workers)
        survivors = []
        for (name, K, shift), row in zip(pending, results):
            rows.append(row)
            if not row["admissible_M32"]:
                survivors.append((name, K, shift))
        pending = survivors
    return rows


def dispatch(tasks, workers: int) -> list:
    if workers <= 1:
        return [accuracy_worker(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_accuracy_star, tasks))


def _accuracy_star(task):
    return accuracy_worker(*task)


def accuracy_worker(name: str, K: int, shift: str, npts: int, threads: int = 1) -> dict:
    route = build_routes()[name]
    modes, amps = packet_modes(K)
    delta = shift_vector(shift)
    started = time.perf_counter()
    history, _ = run_mission(route, npts, delta, modes, amps, STEPS, True, threads)
    elapsed = time.perf_counter() - started
    HISTORIES.mkdir(parents=True, exist_ok=True)
    path = HISTORIES / f"{name}_K{K}_{shift}_N{npts}_M{STEPS}.npz"
    np.savez_compressed(path, step=np.arange(1, STEPS + 1),
                        **{key: history[key] for key in HISTORY_KEYS})
    row = {"kind": "accuracy", "route": name, "K": K, "shift": shift, "N": npts,
           "steps": STEPS, "wall_seconds": elapsed, "threads": threads,
           "history_file": path.name}
    for key in HISTORY_KEYS:
        row[f"{key}_final"] = float(history[key][-1])
    for horizon in HORIZONS:
        row[f"admissible_M{horizon}"] = bool(admissible_horizon(history, horizon))
        for key in HISTORY_KEYS:
            row[f"{key}_M{horizon}"] = float(history[key][horizon - 1])
    return row


# ----------------------------------------------------------------------
# coherence residuals
# ----------------------------------------------------------------------

FD_FIRST = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0


def coherence_residual(route, npts: int, K: int, shift: str) -> dict:
    """Compare the gradient channel with the derivative of the value channel.

    The value channel is the interpolant evaluated at x - delta, so shifting
    delta along one axis and differencing in that parameter is a difference of
    the interpolant itself. A one-field route returns the derivative of its own
    interpolant, so the residual is the finite-difference truncation; the
    operator-faithful augmentation returns a nodal derivative instead.
    """
    modes, amps = packet_modes(K)
    delta = shift_vector(shift)
    diag = Diagnostics(modes, amps, npts, delta, workers=1)
    fields = diag.initial_fields()
    step = 2.0 * np.pi / npts
    out = {}
    for axis, label in enumerate(("grad_x", "grad_y", "grad_z")):
        residual = 0.0
        scale = 0.0
        for component in range(1):
            got = {}

            def sink(key, array, store=got):
                if key == label:
                    store[key] = array

            route.step_channels(route.plan(npts, delta), fields[component], sink)
            approx = np.zeros_like(fields[component])
            for offset, weight in zip(range(-3, 4), FD_FIRST):
                if weight == 0.0:
                    continue
                moved = delta.copy()
                moved[axis] -= offset * COHERENCE_EPS * step
                shifted = {}

                def value_sink(key, array, store=shifted):
                    if key == "value":
                        store[key] = array

                route.step_channels(route.plan(npts, moved), fields[component],
                                    value_sink)
                approx += weight * shifted["value"]
            approx /= COHERENCE_EPS * step
            residual = max(residual, float(np.max(np.abs(approx - got[label]))))
            scale = max(scale, float(np.max(np.abs(got[label]))))
        out[label] = residual / scale
    return out


# ----------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------

def read_status(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1]) * 1024
    return 0


def reset_peak_rss() -> bool:
    try:
        Path("/proc/self/clear_refs").write_text("5\n")
        return True
    except OSError:
        return False


def timing_worker(name: str, K: int, shift: str, npts: int, steps: int) -> dict:
    route = build_routes()[name]
    modes, amps = packet_modes(K)
    delta = shift_vector(shift)
    diag = Diagnostics(modes, amps, npts, delta, workers=1)
    fields = diag.initial_fields()
    del diag, modes, amps
    plan = route.plan(npts, delta)
    gc.collect()
    reset_ok = reset_peak_rss()
    baseline = read_status("VmRSS:")
    times = []
    peak = 0

    def sink(key, array):
        return None

    for repeat in range(TIMING_REPEATS):
        state = [f.copy() for f in fields]
        gc.collect()
        started = time.perf_counter()
        for step in range(steps):
            for component in range(3):
                state[component] = route.step_channels(plan, state[component], sink)
        times.append(time.perf_counter() - started)
        peak = max(peak, read_status("VmHWM:"))
        del state
    return {"kind": "timing", "route": name, "K": K, "shift": shift, "N": npts,
            "steps": steps, "wall_seconds": float(np.median(times)),
            "wall_seconds_all": [float(v) for v in times],
            "peak_rss_increment_bytes": int(peak - baseline),
            "baseline_rss_bytes": int(baseline),
            "peak_rss_bytes": int(peak),
            "state_bytes": int(3 * npts ** 3 * 8),
            "peak_reset": bool(reset_ok)}


def run_timing(name: str, K: int, shift: str, npts: int, steps: int) -> dict:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", "timing-worker",
               "--route", name, "--K", str(K), "--shift", shift, "--N", str(npts),
               "--steps", str(steps)]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def stage_timing(accuracy_rows: list) -> list:
    minima = realized_minima(accuracy_rows)
    plan = []
    for name in ROUTE_NAMES:
        for K in KS:
            for shift in SHIFTS:
                for horizon in (8, STEPS):
                    npts = minima["per_shift"].get(f"{name}|{K}|{shift}|{horizon}")
                    if npts is None:
                        npts = LADDER[-1]
                    plan.append((name, K, shift, npts, horizon, "minimum"))
    for name in ROUTE_NAMES:
        for shift in SHIFTS:
            for horizon in (8, STEPS):
                plan.append((name, KS[-1], shift, EQUAL_GRID, horizon, "equal_grid"))
    cache = {}
    rows = []
    for name, K, shift, npts, horizon, kind in plan:
        key = (name, shift, npts, horizon)
        if key not in cache:
            cache[key] = run_timing(name, K, shift, npts, horizon)
        row = dict(cache[key])
        row["K"] = K
        row["basis"] = kind
        rows.append(row)
    return rows


def realized_minima(rows: list, ladder=LADDER, names=ROUTE_NAMES) -> dict:
    by_key = {}
    for row in rows:
        if row["kind"] != "accuracy":
            continue
        by_key.setdefault((row["route"], row["K"], row["shift"]), {})[row["N"]] = row
    per_shift = {}
    for (name, K, shift), by_n in by_key.items():
        for horizon in (1, 8, STEPS):
            found = None
            for npts in ladder:
                row = by_n.get(npts)
                if row is not None and row[f"admissible_M{horizon}"]:
                    found = npts
                    break
            per_shift[f"{name}|{K}|{shift}|{horizon}"] = found
    combined = {}
    for name in names:
        for K in KS:
            for horizon in (1, 8, STEPS):
                values = [per_shift.get(f"{name}|{K}|{shift}|{horizon}")
                          for shift in SHIFTS]
                combined[f"{name}|{K}|{horizon}"] = (None if any(v is None for v in values)
                                                     else max(values))
    return {"per_shift": per_shift, "combined": combined}


# ----------------------------------------------------------------------
# reduction
# ----------------------------------------------------------------------

CSV_FIELDS = (["kind", "route", "K", "shift", "N", "steps", "basis", "threads",
               "wall_seconds", "wall_seconds_all", "peak_rss_increment_bytes",
               "baseline_rss_bytes", "peak_rss_bytes", "state_bytes",
               "history_file"]
              + [f"{key}_final" for key in HISTORY_KEYS]
              + [f"{key}_M{horizon}" for horizon in HORIZONS for key in HISTORY_KEYS]
              + [f"admissible_M{horizon}" for horizon in HORIZONS])


def write_runs(rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.setdefault("threads", 1)
            if isinstance(out.get("wall_seconds_all"), list):
                out["wall_seconds_all"] = ";".join(f"{v:.6f}"
                                                   for v in out["wall_seconds_all"])
            writer.writerow(out)


def timing_lookup(rows: list) -> dict:
    out = {}
    for row in rows:
        if row["kind"] != "timing":
            continue
        out[(row["basis"], row["route"], row["K"], row["shift"], row["steps"])] = row
    return out


def stage_decide(accuracy_rows: list, timing_rows: list, prediction: dict,
                 coherence: dict) -> dict:
    minima = realized_minima(accuracy_rows)
    times = timing_lookup(timing_rows)
    node_rows = []
    time_rows = []
    node_ok = True
    time_ok = True
    for K in KS:
        for horizon in HORIZONS:
            champ = minima["combined"][f"champion|{K}|{horizon}"]
            slc = minima["combined"][f"slc_operator|{K}|{horizon}"]
            champ_eff = LADDER[-1] if champ is None else champ
            slc_eff = LADDER[-1] if slc is None else slc
            passes = (champ is not None) and (champ_eff * 2 <= slc_eff)
            node_ok = node_ok and passes
            node_rows.append({
                "K": K, "M": horizon,
                "champion_N": champ, "slc_N": slc,
                "slc_beyond_ladder": slc is None,
                "champion_beyond_ladder": champ is None,
                "grid_ratio": slc_eff / champ_eff,
                "node_ratio_slc_over_champion": (slc_eff / champ_eff) ** 3,
                "champion_at_most_half": bool(passes)})
            if horizon not in (8, STEPS):
                continue
            for shift in SHIFTS:
                champ_row = times.get(("minimum", "champion", K, shift, horizon))
                slc_row = times.get(("minimum", "slc_operator", K, shift, horizon))
                if champ_row is None or slc_row is None:
                    continue
                ratio = champ_row["wall_seconds"] / slc_row["wall_seconds"]
                below = ratio < 0.5
                time_ok = time_ok and below
                time_rows.append({
                    "K": K, "M": horizon, "shift": shift,
                    "champion_N": champ_row["N"], "slc_N": slc_row["N"],
                    "champion_seconds": champ_row["wall_seconds"],
                    "slc_seconds": slc_row["wall_seconds"],
                    "time_ratio": ratio, "below_half": bool(below)})
    if not time_rows:
        time_ok = False
    if node_ok and time_ok:
        verdict = "TRIUMPH"
    elif node_ok:
        verdict = "PARTIAL"
    else:
        verdict = "REJECT"
    compare = prediction_versus_realized(accuracy_rows, prediction, minima)
    payload = {
        "gate": "5b",
        "written": now(),
        "rule": ("TRIUMPH if at every (K, M) the champion's minimum admissible N is "
                 "at most half SLC's and the total-time ratio at the minima is below "
                 "0.5; PARTIAL if the node ratio holds but time does not; REJECT "
                 "otherwise."),
        "verdict": verdict,
        "member": member_record(),
        "node_rule_holds": bool(node_ok),
        "time_rule_holds": bool(time_ok),
        "minimum_admissible_N": minima["combined"],
        "minimum_admissible_N_per_shift": minima["per_shift"],
        "node_rows": node_rows,
        "time_rows": time_rows,
        "coherence_residuals": coherence,
        "mass_drift_max": max(float(row["mass_final"]) for row in accuracy_rows
                              if row["kind"] == "accuracy"),
        "prediction_versus_realized": compare,
    }
    (OUT / "decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def prediction_versus_realized(accuracy_rows: list, prediction: dict,
                               minima: dict) -> dict:
    predicted = {}
    for row in prediction["rows"]:
        predicted[(row["route"], row["K"], row["shift"], row["N"], row["M"])] = row
    rows = []
    worst = 0.0
    worst_row = None
    for run in accuracy_rows:
        if run["kind"] != "accuracy":
            continue
        for horizon in HORIZONS:
            key = (run["route"], run["K"], run["shift"], run["N"], horizon)
            ref = predicted.get(key)
            if ref is None:
                continue
            for channel in CHANNELS:
                got = run[f"{channel}_M{horizon}"]
                want = ref[channel]
                if want <= 0.0 and got <= 0.0:
                    ratio = 1.0
                elif want <= 0.0:
                    ratio = float("inf")
                else:
                    ratio = got / want
                entry = {"route": run["route"], "K": run["K"], "shift": run["shift"],
                         "N": run["N"], "M": horizon, "channel": channel,
                         "realized": got, "predicted": want, "ratio": ratio}
                rows.append(entry)
                deviation = max(ratio, 1.0 / ratio) if ratio > 0.0 else float("inf")
                if deviation > worst:
                    worst = deviation
                    worst_row = entry
    grids_agree = True
    grid_rows = []
    for key, value in minima["combined"].items():
        want = prediction["minimum_admissible_N"].get(key)
        grid_rows.append({"key": key, "realized": value, "predicted": want,
                          "agree": value == want})
        grids_agree = grids_agree and (value == want)
    return {"factor": AGREEMENT_FACTOR,
            "worst_deviation": worst,
            "worst_row": worst_row,
            "within_factor": bool(worst <= AGREEMENT_FACTOR),
            "minimum_grids_agree": bool(grids_agree),
            "minimum_grid_rows": grid_rows,
            "rows": rows}


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------

def load_history(name: str, K: int, shift: str, npts: int) -> dict:
    path = HISTORIES / f"{name}_K{K}_{shift}_N{npts}_M{STEPS}.npz"
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


FIGURE_COLORS = {"champion": "#1b3a6b", "BMC9": "#4d7c3a", "BMC11": "#9467bd",
                 "slc_operator": "#b2182b", "slc_operator_L12": "#8c564b",
                 "slc_operator_L16": "#7f7f7f"}
FIGURE_LABELS = {"champion": "champion E(6,2)", "BMC9": "BMC9", "BMC11": "BMC11",
                 "slc_operator_L8": "SLC plus L8",
                 "slc_operator_L12": "SLC plus L12",
                 "slc_operator_L16": "SLC plus L16"}
FIGURE_TICKS = (32, 48, 64, 96, 128, 192, 256, 384)


def figure_grid(accuracy_rows: list, minima: dict, name: str, K: int, shift: str,
                horizon: int):
    """Return the grid a route is drawn on, its minimum or its largest run."""
    for key in (f"{name}|{K}|{shift}|{horizon}",
                f"{csv_route(name)}|{K}|{shift}|{horizon}"):
        value = minima.get("per_shift", {}).get(key)
        if value is not None:
            return value
    runs = [row["N"] for row in accuracy_rows
            if row["kind"] == "accuracy" and row["route"] == csv_route(name)
            and row["K"] == K and row["shift"] == shift]
    return max(runs) if runs else None


def make_figure(accuracy_rows: list, minima: dict) -> None:
    """Draw the histories of the two headline routes and the error against N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = 24
    shift = "A"
    labels = {"value": "value", "grad_x": "gradient x", "grad_y": "gradient y",
              "grad_z": "gradient z", "laplacian": "Laplacian", "mass": "mass drift"}
    available = {row["route"] for row in accuracy_rows if row["kind"] == "accuracy"}
    competitor = ("slc_operator_L12" if "slc_operator_L12" in available
                  else "slc_operator_L8")
    drawn = [name for name in AMENDMENT_ROUTE_NAMES if csv_route(name) in available]
    fig = plt.figure(figsize=(11.0, 6.6))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[0, 3]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[1, 2:])]
    chosen = {}
    for name in ("champion", competitor):
        chosen[name] = figure_grid(accuracy_rows, minima, name, K, shift, STEPS)
    for ax, key in zip(axes[:6], HISTORY_KEYS):
        for name, npts in chosen.items():
            if npts is None:
                continue
            history = load_history(csv_route(name), K, shift, npts)
            ax.semilogy(history["step"], np.maximum(history[key], 1e-18),
                        color=FIGURE_COLORS[csv_route(name)], lw=1.4,
                        label=f"{FIGURE_LABELS[name]}, N = {npts}")
        ax.axhline(TOL if key != "mass" else MASS_TOL, color="0.35", ls="--", lw=0.9)
        ax.set_title(labels[key], fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    axes[0].set_ylabel("relative L2 error", fontsize=8)
    axes[4].set_ylabel("relative L2 error", fontsize=8)
    axes[5].legend(fontsize=6.5, loc="center left")
    panel = axes[6]
    for name in drawn:
        xs, ys = [], []
        for row in accuracy_rows:
            if (row["kind"] == "accuracy" and row["route"] == csv_route(name)
                    and row["K"] == K and row["shift"] == shift):
                xs.append(row["N"])
                ys.append(max(row[f"{key}_M{STEPS}"] for key in CHANNELS))
        if not xs:
            continue
        order = np.argsort(xs)
        panel.loglog(np.array(xs)[order], np.maximum(np.array(ys)[order], 1e-18),
                     "o-", color=FIGURE_COLORS[csv_route(name)], lw=1.4, ms=3.4,
                     label=FIGURE_LABELS[name])
    panel.axhline(TOL, color="0.35", ls="--", lw=0.9)
    grids = sorted({row["N"] for row in accuracy_rows
                    if row["kind"] == "accuracy" and row["K"] == K
                    and row["shift"] == shift})
    ticks = [value for value in FIGURE_TICKS
             if grids and min(grids) <= value <= max(grids)]
    panel.set_xticks(ticks or grids)
    panel.set_xticklabels([str(value) for value in (ticks or grids)])
    panel.set_xticks([], minor=True)
    panel.set_title(f"worst channel at M = {STEPS}, K = {K}, shift {shift}", fontsize=9)
    panel.set_xlabel("N", fontsize=8)
    panel.set_ylabel("relative L2 error", fontsize=8)
    panel.tick_params(labelsize=7)
    panel.grid(True, which="both", lw=0.3, alpha=0.4)
    panel.legend(fontsize=6.0, loc="lower left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_semilagrangian.pdf", bbox_inches="tight")
    plt.close(fig)
    listed = ", ".join(FIGURE_LABELS[name] for name in drawn)
    caption = ["# Figure captions, Gate 5 part 5b", ""]
    caption += wrapped(
        "`fig_semilagrangian.pdf`. Top row and the first two panels of the bottom "
        "row: the six histories of the semi-Lagrangian mission, the relative L2 "
        "errors of the value, of the three gradient components and of the Laplacian "
        "against the exact translated packet, and the mass drift, versus step, for "
        f"the packet K = {K} and shift {shift}, each route on its own minimum "
        f"admissible grid for the horizon M = {STEPS} (champion N = "
        f"{chosen['champion']}, {FIGURE_LABELS[competitor]} N = {chosen[competitor]}). "
        "The competitor drawn is the augmentation on the twelve-point Lagrange layer, "
        "the layer of the wall evaluator of the paper, which is the strongest of the "
        "three widths amendment 1 carries. The dashed line is the one percent "
        "admissibility threshold, and 1e-12 in the mass panel.")
    caption += [""]
    caption += wrapped(
        "Bottom right: the worst of the five channel errors at the final step versus "
        "the grid, for every route of amendment 1 on the same packet and shift "
        f"({listed}), each climbing the refined ladder to its own minimum. Produced by "
        "code/analysis/exponential_family/gate5b_semilagrangian.py; the numbers are "
        "in runs.csv, prediction.json and decision.json.")
    (FIGURES / "captions.md").write_text("\n".join(caption) + "\n")


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------

def fmt(value, spec="{:.4e}") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return spec.format(float(value))


def wrapped(text: str, width: int = 76) -> list:
    """Return one paragraph broken into lines of at most the given width."""
    return textwrap.wrap(" ".join(text.split()), width=width)


def grid_text(value) -> str:
    """Return a minimum admissible grid, or the mark of one beyond the ladder."""
    return f"> {AMENDMENT_LADDER[-1]}" if value is None else str(int(value))


def amendment_readme_lines(prediction: dict, amendment: dict, accuracy_rows: list,
                           timing_rows: list) -> list:
    """Return the amendment section of the README."""
    block = prediction["amendment_1"]
    coherence = amendment["coherence_residuals"]
    routes = amendment_routes()
    minima = amendment["minimum_admissible_N"]
    per_shift = amendment["minimum_admissible_N_per_shift"]
    names = {csv_route(name) for name in AMENDMENT_ROUTE_NAMES}
    lines = ["## Amendment 1", ""] + wrapped(block["scope"]) + ["",
             "The eight-point augmentation keeps its original name in `runs.csv` and",
             "in the history files, so `slc_operator_L8` below is the route the",
             "tables above call `slc_operator`, and its rows at N in {32, 64, 128,",
             "256} are the original runs, reused rather than repeated.", "",
             "### Routes", "",
             "| route | runs.csv name | construction | fields | taps per axis | "
             "solves per step per field |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in AMENDMENT_ROUTE_NAMES:
        route = routes[name]
        lines.append(f"| {name} | {csv_route(name)} | {route.description()} | "
                     f"{route.fields} | {route.taps} | {route.solves_per_step} |")
    lines += ["", "### Ladder", "",
              "N in {" + ", ".join(str(value) for value in AMENDMENT_LADDER) + "},",
              "whose successive ratios are at most 1.25. Each route, packet and",
              "shift climbs the ladder until its grid is admissible at the horizon",
              "32, which is the strictest of the three, so every rung below every",
              "minimum is run and the margin on the rung below is a measurement.",
              "", "### Minimum admissible grids", "",
              "| route | K | M | N shift A | N shift B | N worse shift | predicted | "
              "agree |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in AMENDMENT_ROUTE_NAMES:
        for K in KS:
            for horizon in HORIZONS:
                got = minima.get(f"{name}|{K}|{horizon}")
                want = block["minimum_admissible_N"].get(f"{name}|{K}|{horizon}")
                cells = [grid_text(per_shift.get(f"{name}|{K}|{shift}|{horizon}"))
                         for shift in SHIFTS]
                lines.append(f"| {name} | {K} | {horizon} | {cells[0]} | {cells[1]} | "
                             f"{grid_text(got)} | {grid_text(want)} | "
                             f"{'yes' if got == want else 'no'} |")
    predicted_below = {(row["route"], row["K"], row["shift"], row["M"]): row
                       for row in amendment["rung_below_minimum_predicted"]}
    lines += ["", "### The rung below the minimum", "",
              "The worst of the five channel histories through step M on the rung",
              "immediately below the minimum, as a multiple of the one percent bar.",
              "A multiple just above one is a rung that fails narrowly, so the",
              "minimum above it overstates what the route needs by less than the",
              "rung spacing.", "",
              "| route | K | shift | M | minimum N | rung below | worst channel | "
              "worst value | multiple of the bar | predicted multiple |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in amendment["rung_below_minimum"]:
        want = predicted_below.get((row["route"], row["K"], row["shift"], row["M"]))
        lines.append(
            f"| {row['route']} | {row['K']} | {row['shift']} | {row['M']} | "
            f"{grid_text(row['minimum_N'])} | "
            f"{'n/a' if row['rung_below'] is None else row['rung_below']} | "
            f"{row['worst_channel'] or 'n/a'} | {fmt(row['worst_value'])} | "
            f"{fmt(row['multiple_of_tolerance'], '{:.3f}')} | "
            f"{fmt(None if want is None else want['multiple_of_tolerance'], '{:.3f}')} |")
    lines += ["", "### Node counts against each Lagrange width", "",
              "| width | K | M | champion N | SLC N | grid ratio | node ratio | "
              "champion at most half |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for width in LAGRANGE_WIDTHS:
        for row in amendment["per_width"][str(width)]["node_rows"]:
            lines.append(
                f"| {row['width']} | {row['K']} | {row['M']} | "
                f"{grid_text(row['champion_N'])} | {grid_text(row['slc_N'])} | "
                f"{row['grid_ratio']:.3f} | "
                f"{row['node_ratio_slc_over_champion']:.3f} | "
                f"{fmt(row['champion_at_most_half'])} |")
    lines += ["", "The node ratio is the cube of the grid ratio. Where the champion's",
              "minimum is 32 the ladder has no coarser rung, so the ratio there is a",
              "lower bound on the separation rather than a measurement of it.", "",
              "### Timing at the amendment minima", ""]
    lines += wrapped(
        "Single thread, median of three, one process at a time, the mission only, "
        "with no diagnostics. The rows with basis `amendment_minimum` are the routes "
        + ", ".join(amendment["timing_routes_run"])
        + " each on its own minimum admissible grid for that horizon; the rows with "
        f"basis `amendment_equal_grid` are all six routes on the common grid N = "
        f"{EQUAL_GRID}. A mission costs the same whatever the packet carries, so one "
        "measurement serves both packets whenever the route, the shift, the grid and "
        "the horizon agree, and appears once per row it answers.")
    lines += ["",
              "| basis | route | K | shift | N | M | seconds | carried state MiB | "
              "peak RSS MiB | peak RSS increment MiB |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in timing_rows:
        if row.get("basis") not in (AMENDMENT_MINIMUM, AMENDMENT_EQUAL):
            continue
        lines.append(f"| {row['basis']} | {row['route']} | {row['K']} | {row['shift']} "
                     f"| {row['N']} | {row['steps']} | {row['wall_seconds']:.4f} | "
                     f"{row['state_bytes'] / 1048576:.1f} | "
                     f"{row['peak_rss_bytes'] / 1048576:.1f} | "
                     f"{row['peak_rss_increment_bytes'] / 1048576:.1f} |")
    lines += ["", "### Time ratios against each Lagrange width", "",
              "| width | K | M | shift | champion N | SLC N | champion s | SLC s | "
              "ratio | below 0.5 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for width in LAGRANGE_WIDTHS:
        rows = amendment["per_width"][str(width)]["time_rows"]
        if not rows:
            lines.append(f"| {width} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a "
                         "| n/a |")
            continue
        for row in rows:
            lines.append(f"| {row['width']} | {row['K']} | {row['M']} | {row['shift']} "
                         f"| {row['champion_N']} | {row['slc_N']} | "
                         f"{row['champion_seconds']:.4f} | {row['slc_seconds']:.4f} | "
                         f"{row['time_ratio']:.4f} | {fmt(row['below_half'])} |")
    lines += ["", "### The rule, width by width", ""]
    lines += wrapped(amendment["rule"]) + [""] + wrapped(amendment["rule_applied"])
    lines += ["",
              "| width | node rule | time rule | verdict |",
              "| --- | --- | --- | --- |"]
    for width in LAGRANGE_WIDTHS:
        entry = amendment["per_width"][str(width)]
        time_text = ("not evaluated" if entry["time_rule_holds"] is None
                     else fmt(entry["time_rule_holds"]))
        lines.append(f"| {width} | {fmt(entry['node_rule_holds'])} | {time_text} | "
                     f"{entry['verdict'] or 'not evaluated'} |")
    if any(amendment["per_width"][str(width)]["time_rule_holds"] is None
           for width in LAGRANGE_WIDTHS):
        lines += ["",
                  "A width whose node clause fails reads REJECT under the rule",
                  "whatever the time clause does, so a width with no time row is a",
                  "width the rule already decides without one."]
    compare = amendment["prediction_versus_realized"]
    lines += ["", "### Predicted against realized", ""]
    lines += wrapped(
        "The amendment block of `prediction.json` is the same modal arithmetic as the "
        "original block, evaluated for the three Lagrange widths, for BMC11 and on "
        "the refined ladder, and it was written before any amendment run. The minimum "
        "grids "
        + ("agree" if compare["minimum_grids_agree"] else "do not agree")
        + ", shift by shift "
        + ("as well" if compare["minimum_grids_agree_per_shift"] else "no more")
        + ", and the worst deviation of the ratio realized over predicted from one is "
        f"{compare['worst_deviation']:.4f} against the recorded factor "
        f"{AGREEMENT_FACTOR:g}.")
    lines += ["",
              "### Coherence residuals", "",
              "| route | grad x | grad y | grad z |", "| --- | --- | --- | --- |"]
    for name in AMENDMENT_ROUTE_NAMES:
        row = coherence[name]
        lines.append(f"| {name} | {fmt(row['grad_x'])} | {fmt(row['grad_y'])} | "
                     f"{fmt(row['grad_z'])} |")
    drift = amendment["mass_drift_max"]
    lines += ["", "### Mass", "",
              "The largest mass drift over every amendment accuracy run is "
              f"{drift:.3e}." if drift is not None else "", "",
              "### Realized ladder", "",
              "| route | K | shift | N | value | grad x | grad y | grad z | "
              "Laplacian | mass | M=1 | M=8 | M=32 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
              "--- | --- |"]
    order = {csv_route(name): place
             for place, name in enumerate(AMENDMENT_ROUTE_NAMES)}
    selected = [row for row in accuracy_rows
                if row["kind"] == "accuracy" and row["route"] in names
                and row["N"] in AMENDMENT_LADDER]
    for row in sorted(selected, key=lambda r: (order[r["route"]], r["K"], r["shift"],
                                               r["N"])):
        cells = " | ".join(fmt(row[f"{key}_final"]) for key in HISTORY_KEYS)
        flags = " | ".join(fmt(row[f"admissible_M{h}"]) for h in HORIZONS)
        lines.append(f"| {row['route']} | {row['K']} | {row['shift']} | {row['N']} | "
                     f"{cells} | {flags} |")
    lines += ["", "The columns are the values at the final step, step 32, and the",
              "three flags are admissibility at the horizons 1, 8 and 32.", ""]
    return lines

def write_readme(prediction: dict, accuracy_rows: list, timing_rows: list,
                 decision: dict, coherence: dict, amendment_accuracy=None,
                 amendment_timing=None, amendment=None) -> None:
    routes = build_routes()
    minima = realized_minima(accuracy_rows)
    lines = ["# Gate 5 part 5b: the semi-Lagrangian matched-accuracy mission", ""]
    lines += [
        "Producer `code/analysis/exponential_family/gate5b_semilagrangian.py`,",
        "which writes `prediction.json` before any realized run, then `runs.csv`,",
        "the per-step histories under `histories/`, `decision.json`, this file and",
        "`figures/`.", "",
        "## Mission", "",
        "Exact periodic translation of a divergence-free broadband packet on",
        "[0, 2 pi)^3. The packet carries every integer wave vector with |k_i| <= K",
        "other than the origin; the amplitudes of the lexicographic half space are",
        f"drawn once from seed {PACKET_SEED} + K, projected onto the plane orthogonal",
        "to k and scaled so the mean square of |u_K| is one, and the other half is",
        "the conjugate, so the nodal field is real and divergence-free. Each step",
        "the new nodal field is the old field read at the departure point x - delta,",
        "and the three first derivatives and the Laplacian are read at the same",
        "points. The shifts are", "",
        "    delta_A = 2 pi (sqrt2 / 17, sqrt3 / 19, sqrt5 / 23)",
        "    delta_B = 2 pi (sqrt7 / 29, sqrt11 / 31, sqrt13 / 37)", "",
        "and the horizons are M in {1, 8, 32} on the ladder N in",
        f"{{{', '.join(str(n) for n in LADDER)}}}. A grid is admissible for horizon M",
        "when the five channel histories stay below one percent through step M and",
        f"the mass drift stays below {MASS_TOL:g}.", "",
        "## Routes", "",
        "| route | construction | fields | taps per axis | solves per step per field |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ROUTE_NAMES:
        route = routes[name]
        lines.append(f"| {name} | {route.description()} | {route.fields} | "
                     f"{route.taps} | {route.solves_per_step} |")
    lines += ["",
              "Because the shift is rigid, every departure point carries the same cell",
              "coordinate, so one step of each route is a tensor product of three",
              "periodic one-dimensional operators: an integer shift, then the short",
              "stencil of the route, with the cyclic recovery of the one-field routes",
              "and the two pentadiagonal SLC solves per axis carried out as banded",
              "solves with a Woodbury correction for the periodic corners. The test",
              "file checks the separable form against a direct tensor-product",
              "evaluation at the departure points and against the transfer functions",
              "of part 5a.", ""]
    lines += ["## Predicted minimum admissible grids", "",
              "| route | K | M | predicted N | realized N | agree |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in ROUTE_NAMES:
        for K in KS:
            for horizon in HORIZONS:
                key = f"{name}|{K}|{horizon}"
                want = prediction["minimum_admissible_N"].get(key)
                got = minima["combined"].get(key)
                lines.append(f"| {name} | {K} | {horizon} | "
                             f"{'> 512' if want is None else want} | "
                             f"{'> 512' if got is None else got} | "
                             f"{'yes' if want == got else 'no'} |")
    lines += ["", "## Realized ladder", "",
              "| route | K | shift | N | value | grad x | grad y | grad z | Laplacian | "
              "mass | M=1 | M=8 | M=32 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
              "| --- |"]
    for row in sorted(accuracy_rows, key=lambda r: (r["route"], r["K"], r["shift"],
                                                    r["N"])):
        if row["kind"] != "accuracy":
            continue
        cells = " | ".join(fmt(row[f"{key}_final"]) for key in HISTORY_KEYS)
        flags = " | ".join(fmt(row[f"admissible_M{h}"]) for h in HORIZONS)
        lines.append(f"| {row['route']} | {row['K']} | {row['shift']} | {row['N']} | "
                     f"{cells} | {flags} |")
    lines += ["", "The columns are the values at the final step, step 32, and the",
              "three flags are admissibility at the horizons 1, 8 and 32. The ladder",
              "climbs until a grid passes at the horizon 32, so the grids above a",
              "route's minimum are not run. The wall times of the accuracy runs are in",
              "runs.csv but are not decision quantities: those runs carry the",
              "diagnostics and share the machine with up to eight of their own kind.",
              ""]
    lines += ["## Timing at the minima", "",
              "Single thread, median of three, one process at a time, the mission",
              "only, with no diagnostics. The rows with basis `minimum` are each route",
              "on its own minimum admissible grid for that horizon, which is the",
              "matched-accuracy comparison the rule asks for; the rows with basis",
              "`equal_grid` are the same missions on the common grid N = "
              f"{EQUAL_GRID} for",
              "reference. The carried state is the three nodal component fields; the",
              "peak resident set increment is measured over a process that already",
              "holds them, so it is the working set of one mission on top of its own",
              "input, and the peak resident set is the absolute high water mark.",
              "A mission costs the same whatever the packet carries, so one",
              "measurement serves both packets whenever the route, the shift, the",
              "grid and the horizon agree, and appears once per row it answers.",
              "", 
              "| basis | route | K | shift | N | M | seconds | carried state MiB | "
              "peak RSS MiB | peak RSS increment MiB |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in timing_rows:
        lines.append(f"| {row['basis']} | {row['route']} | {row['K']} | {row['shift']} "
                     f"| {row['N']} | {row['steps']} | {row['wall_seconds']:.4f} | "
                     f"{row['state_bytes'] / 1048576:.1f} | "
                     f"{row['peak_rss_bytes'] / 1048576:.1f} | "
                     f"{row['peak_rss_increment_bytes'] / 1048576:.1f} |")
    lines += ["", "## Decision", "",
              f"Verdict **{decision['verdict']}**.", "",
              "| K | M | champion N | SLC N | node ratio | champion at most half |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in decision["node_rows"]:
        lines.append(f"| {row['K']} | {row['M']} | "
                     f"{'> 512' if row['champion_N'] is None else row['champion_N']} | "
                     f"{'> 512' if row['slc_N'] is None else row['slc_N']} | "
                     f"{row['node_ratio_slc_over_champion']:.3f} | "
                     f"{fmt(row['champion_at_most_half'])} |")
    lines += ["",
              "The node ratio is the cube of the grid ratio. Where the champion's",
              "minimum is 32 the ladder has no coarser rung, so the ratio there is a",
              "lower bound on the separation rather than a measurement of it.", "",
              "| K | M | shift | champion s | SLC s | ratio | below 0.5 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in decision["time_rows"]:
        lines.append(f"| {row['K']} | {row['M']} | {row['shift']} | "
                     f"{row['champion_seconds']:.4f} | {row['slc_seconds']:.4f} | "
                     f"{row['time_ratio']:.4f} | {fmt(row['below_half'])} |")
    compare = decision["prediction_versus_realized"]
    member = decision["member"]
    lines += ["", "## Member", "",
              f"The champion is `{member['file']}`, sha256 `{member['sha256']}`,",
              f"m = {member['m']}, r = {member['r']}, p = {member['p']}, omega = "
              f"({member['omegas'][0]:.17g}, {member['omegas'][1]:.17g}); the route "
              "reads", "the same pair: "
              + ("yes" if member["matches_route"] else "no") + "."]
    lines += ["", "## Predicted against realized", "",
              "The prediction is the exact modal arithmetic of the transfers, written",
              "before any realized run. The minimum grids "
              + ("agree" if compare["minimum_grids_agree"] else "do not agree")
              + " and the worst deviation of the ratio realized over predicted from",
              f"one is {compare['worst_deviation']:.4f} against the recorded factor "
              f"{AGREEMENT_FACTOR:g}.", ""]
    lines += ["## Coherence residuals", "",
              "The gradient channel against a sixth-order difference of the value",
              "channel in the shift parameter, on the K = 12 packet at N = "
              f"{COHERENCE_GRID}, relative to the gradient itself. A one-field route",
              "returns the derivative of its own interpolant, so the residual is the",
              "difference truncation; the operator-faithful augmentation returns a",
              "nodal derivative instead.", "",
              "| route | grad x | grad y | grad z |", "| --- | --- | --- | --- |"]
    for name in ROUTE_NAMES:
        row = coherence[name]
        lines.append(f"| {name} | {fmt(row['grad_x'])} | {fmt(row['grad_y'])} | "
                     f"{fmt(row['grad_z'])} |")
    lines += ["", "## Mass", "",
              "Every route reproduces a constant exactly, so the zero mode of the step",
              "operator is one and the mass of the carried field is conserved to",
              "rounding. The largest mass drift over every accuracy run is",
              f"{decision['mass_drift_max']:.3e}.", ""]
    if amendment is not None:
        lines += amendment_readme_lines(
            prediction, amendment,
            accuracy_rows if amendment_accuracy is None else amendment_accuracy,
            timing_rows if amendment_timing is None else amendment_timing)
    lines += ["## Commands", "", "```",
              "PY=.venv/bin/python",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage predict",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage accuracy --jobs 8",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage timing",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage reduce",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage amendment-predict --jobs 6",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage amendment-accuracy --jobs 8",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage amendment-timing",
              "$PY code/analysis/exponential_family/gate5b_semilagrangian.py "
              "--stage amendment-reduce",
              "$PY -m pytest code/tests/test_gate5b.py -q",
              "```", "",
              "The accuracy stages use up to eight worker processes, each with one",
              "thread for the linear algebra; the timing stages run one mission at a",
              "time in a fresh single-threaded process. The amendment accuracy stage",
              "reuses every run `runs.csv` already holds, so the original runs are",
              "not repeated.", "",
              "## Hashes", "", "| file | sha256 |", "| --- | --- |"]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "README.md":
            lines.append(f"| {path.relative_to(OUT)} | {sha256(path)} |")
    producer = Path(__file__).resolve()
    lines.append(f"| {producer.relative_to(ROOT)} | {sha256(producer)} |")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# amendment 1
# ----------------------------------------------------------------------

AMENDMENT_MINIMUM = "amendment_minimum"
AMENDMENT_EQUAL = "amendment_equal_grid"
AMENDMENT_RULE = (
    "TRIUMPH if at every (K, M) the champion's minimum admissible N is at most "
    "half SLC's and the total time ratio at the minima is below 0.5; PARTIAL if "
    "the node ratio holds but time does not; REJECT otherwise.")
AMENDMENT_SCOPE = (
    "The augmentation is carried at the Lagrange widths 8, 12 and 16, the twelve "
    "point width being the wall evaluator of the paper; BMC11 is added to the "
    "controls; the ladder is refined so that successive rungs differ by at most "
    "a factor 1.25; the worst channel history on the rung below each realized "
    "minimum is reported as a multiple of the one percent bar; and the rule of "
    "the contract is applied verbatim against each Lagrange width separately.")


def amendment_routes() -> dict:
    """Return the six amendment routes by their amendment name."""
    built = build_routes()
    return {name: built[csv_route(name)] for name in AMENDMENT_ROUTE_NAMES}


def worst_channel(history: dict, horizon: int):
    """Return the worst of the five channel histories through step M and its name."""
    pairs = [(float(np.max(np.asarray(history[key])[:horizon])), key)
             for key in CHANNELS]
    value, key = max(pairs)
    return value, key


def rung_below(npts, ladder=AMENDMENT_LADDER):
    """Return the rung immediately below npts, the top rung when npts is None."""
    if npts is None:
        return ladder[-1]
    place = list(ladder).index(npts)
    return ladder[place - 1] if place > 0 else None


def amendment_workers(npts: int) -> int:
    """Return the process count one rung tolerates, set by its live arrays."""
    if npts <= 128:
        return 8
    if npts <= 192:
        return 4
    if npts <= 256:
        return 2
    return 1


def amendment_predict_case(name: str, K: int, shift: str) -> dict:
    """Return the predicted rows, minima and rung-below margins of one case."""
    route = amendment_routes()[name]
    modes, amps = packet_modes(K)
    delta = shift_vector(shift)
    rows = []
    by_n = {}
    for npts in AMENDMENT_LADDER:
        history = predicted_histories(route, npts, modes, amps, delta)
        by_n[npts] = history
        for horizon in HORIZONS:
            row = {"route": name, "K": K, "shift": shift, "N": npts, "M": horizon}
            for key in HISTORY_KEYS:
                row[key] = float(history[key][horizon - 1])
                row[f"{key}_max"] = float(np.max(history[key][:horizon]))
            row["admissible"] = bool(admissible_horizon(history, horizon))
            rows.append(row)
    minima = {}
    below = []
    for horizon in HORIZONS:
        npts = minimum_grid(by_n, horizon, AMENDMENT_LADDER)
        minima[f"{name}|{K}|{shift}|{horizon}"] = npts
        under = rung_below(npts)
        entry = {"route": name, "K": K, "shift": shift, "M": horizon,
                 "minimum_N": npts, "rung_below": under}
        if under is None:
            entry.update({"worst_channel": None, "worst_value": None,
                          "multiple_of_tolerance": None})
        else:
            value, key = worst_channel(by_n[under], horizon)
            entry.update({"worst_channel": key, "worst_value": value,
                          "multiple_of_tolerance": value / TOL})
        below.append(entry)
    return {"rows": rows, "minima": minima, "below": below}


def _predict_star(task):
    return amendment_predict_case(*task)


def combine_minima(per_shift: dict, names=AMENDMENT_ROUTE_NAMES) -> dict:
    """Return the per-shift minima reduced to the worse shift of each case."""
    combined = {}
    for name in names:
        for K in KS:
            for horizon in HORIZONS:
                values = [per_shift.get(f"{name}|{K}|{shift}|{horizon}")
                          for shift in SHIFTS]
                combined[f"{name}|{K}|{horizon}"] = (
                    None if any(v is None for v in values) else max(values))
    return combined


def width_node_rows(minima: dict, width: int, ladder=AMENDMENT_LADDER) -> list:
    """Return the node-count rows of the champion against one Lagrange width."""
    slc_name = f"slc_operator_L{width}"
    rows = []
    for K in KS:
        for horizon in HORIZONS:
            champ = minima[f"champion|{K}|{horizon}"]
            slc = minima[f"{slc_name}|{K}|{horizon}"]
            champ_eff = ladder[-1] if champ is None else champ
            slc_eff = ladder[-1] if slc is None else slc
            passes = (champ is not None) and (champ_eff * 2 <= slc_eff)
            rows.append({"width": width, "K": K, "M": horizon,
                         "champion_N": champ, "slc_N": slc,
                         "champion_beyond_ladder": champ is None,
                         "slc_beyond_ladder": slc is None,
                         "grid_ratio": slc_eff / champ_eff,
                         "node_ratio_slc_over_champion": (slc_eff / champ_eff) ** 3,
                         "champion_at_most_half": bool(passes)})
    return rows


def stage_amendment_predict(jobs: int = 1) -> dict:
    """Write the amendment block of prediction.json, before any amendment run."""
    path = OUT / "prediction.json"
    payload = json.loads(path.read_text())
    tasks = [(name, K, shift) for name in AMENDMENT_ROUTE_NAMES for K in KS
             for shift in SHIFTS]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_predict_star, tasks))
    else:
        results = [amendment_predict_case(*task) for task in tasks]
    rows = []
    per_shift = {}
    below = []
    for result in results:
        rows.extend(result["rows"])
        per_shift.update(result["minima"])
        below.extend(result["below"])
    combined = combine_minima(per_shift)
    node_rows = []
    for width in LAGRANGE_WIDTHS:
        node_rows.extend(width_node_rows(combined, width))
    routes = amendment_routes()
    block = {
        "written": now(),
        "scope": AMENDMENT_SCOPE,
        "statement": (
            "The arithmetic is the one the original block states, evaluated for "
            "the three Lagrange widths, for BMC11 and on the refined ladder. The "
            "rung below each predicted minimum carries the worst of the five "
            "channel histories through step M, reported as a multiple of the one "
            "percent bar, so the margin by which that rung fails is part of the "
            "prediction."),
        "routes": list(AMENDMENT_ROUTE_NAMES),
        "runs_csv_names": {name: csv_route(name) for name in AMENDMENT_ROUTE_NAMES},
        "route_description": {name: routes[name].description()
                              for name in AMENDMENT_ROUTE_NAMES},
        "lagrange_widths": list(LAGRANGE_WIDTHS),
        "ladder": list(AMENDMENT_LADDER),
        "horizons": list(HORIZONS),
        "tolerance": TOL,
        "mass_tolerance": MASS_TOL,
        "timing_routes": list(AMENDMENT_TIMING_NAMES),
        "equal_grid": EQUAL_GRID,
        "rows": rows,
        "minimum_admissible_N_per_shift": per_shift,
        "minimum_admissible_N": combined,
        "rung_below_minimum": below,
        "node_count_ratio": node_rows,
    }
    payload["amendment_1"] = block
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return block


def stage_amendment_accuracy(jobs: int, rows: list) -> list:
    """Climb the refined ladder, reusing every run runs.csv already holds.

    A case leaves the climb once its grid is admissible at the horizon 32, which
    is the strictest of the three, so every rung below every minimum is run.
    """
    existing = {(row["route"], row["K"], row["shift"], row["N"]): row
                for row in rows if row["kind"] == "accuracy"}
    pending = [(csv_route(name), K, shift) for name in AMENDMENT_ROUTE_NAMES
               for K in KS for shift in SHIFTS]
    fresh = []
    for npts in AMENDMENT_LADDER:
        if not pending:
            break
        tasks = [(name, K, shift, npts) for (name, K, shift) in pending
                 if (name, K, shift, npts) not in existing]
        if tasks:
            workers = max(1, min(jobs, len(tasks), amendment_workers(npts)))
            threads = max(1, FFT_WORKERS // workers)
            started = time.perf_counter()
            results = dispatch([task + (threads,) for task in tasks], workers)
            for task, row in zip(tasks, results):
                existing[task] = row
                fresh.append(row)
            print(f"N = {npts}: {len(tasks)} runs on {workers} workers, "
                  f"{time.perf_counter() - started:.1f} s", flush=True)
        survivors = [key for key in pending
                     if not existing[(key[0], key[1], key[2], npts)]["admissible_M32"]]
        pending = survivors
    return fresh


def amendment_realized_minima(rows: list) -> dict:
    """Return the realized minima of the amendment routes on the refined ladder."""
    csv_names = tuple(csv_route(name) for name in AMENDMENT_ROUTE_NAMES)
    found = realized_minima(rows, AMENDMENT_LADDER, csv_names)
    per_shift = {}
    for name in AMENDMENT_ROUTE_NAMES:
        for K in KS:
            for shift in SHIFTS:
                for horizon in HORIZONS:
                    key = f"{csv_route(name)}|{K}|{shift}|{horizon}"
                    per_shift[f"{name}|{K}|{shift}|{horizon}"] = found["per_shift"].get(key)
    return {"per_shift": per_shift, "combined": combine_minima(per_shift)}


def amendment_below_rows(minima: dict) -> list:
    """Return the worst realized history on the rung below each minimum."""
    out = []
    for name in AMENDMENT_ROUTE_NAMES:
        for K in KS:
            for shift in SHIFTS:
                for horizon in HORIZONS:
                    npts = minima["per_shift"].get(f"{name}|{K}|{shift}|{horizon}")
                    under = rung_below(npts)
                    entry = {"route": name, "K": K, "shift": shift, "M": horizon,
                             "minimum_N": npts, "rung_below": under}
                    path = (HISTORIES / f"{csv_route(name)}_K{K}_{shift}_N{under}"
                            f"_M{STEPS}.npz") if under is not None else None
                    if path is None or not path.exists():
                        entry.update({"worst_channel": None, "worst_value": None,
                                      "multiple_of_tolerance": None})
                    else:
                        history = load_history(csv_route(name), K, shift, under)
                        value, key = worst_channel(history, horizon)
                        entry.update({"worst_channel": key, "worst_value": value,
                                      "multiple_of_tolerance": value / TOL})
                    out.append(entry)
    return out


def stage_amendment_timing(accuracy_rows: list, existing_rows: list,
                           names=AMENDMENT_TIMING_NAMES, checkpoint=None) -> list:
    """Time every route at its realized minima and on the common grid."""
    minima = amendment_realized_minima(accuracy_rows)
    plan = []
    for name in names:
        for K in KS:
            for shift in SHIFTS:
                for horizon in (8, STEPS):
                    npts = minima["per_shift"].get(f"{name}|{K}|{shift}|{horizon}")
                    if npts is None:
                        npts = AMENDMENT_LADDER[-1]
                    plan.append((name, K, shift, npts, horizon, AMENDMENT_MINIMUM))
    for name in AMENDMENT_ROUTE_NAMES:
        for shift in SHIFTS:
            for horizon in (8, STEPS):
                plan.append((name, KS[-1], shift, EQUAL_GRID, horizon,
                             AMENDMENT_EQUAL))
    cache = {}
    for row in existing_rows:
        if row["kind"] != "timing" or row.get("basis") not in (AMENDMENT_MINIMUM,
                                                               AMENDMENT_EQUAL):
            continue
        cache[(row["route"], row["shift"], row["N"], row["steps"])] = row
    rows = []
    for name, K, shift, npts, horizon, kind in plan:
        key = (csv_route(name), shift, npts, horizon)
        measured = key not in cache
        if measured:
            started = time.perf_counter()
            cache[key] = run_timing(csv_route(name), K, shift, npts, horizon)
            print(f"timing {name} shift {shift} N = {npts} M = {horizon}: "
                  f"{time.perf_counter() - started:.1f} s", flush=True)
        row = dict(cache[key])
        row["kind"] = "timing"
        row["route"] = csv_route(name)
        row["K"] = K
        row["basis"] = kind
        rows.append(row)
        if measured and checkpoint is not None:
            checkpoint(list(rows))
    return rows


def amendment_time_rows(times: dict, width: int) -> list:
    """Return the time ratios of the champion against one Lagrange width."""
    slc_name = csv_route(f"slc_operator_L{width}")
    rows = []
    for K in KS:
        for horizon in (8, STEPS):
            for shift in SHIFTS:
                champ_row = times.get((AMENDMENT_MINIMUM, "champion", K, shift,
                                       horizon))
                slc_row = times.get((AMENDMENT_MINIMUM, slc_name, K, shift,
                                      horizon))
                if champ_row is None or slc_row is None:
                    continue
                ratio = champ_row["wall_seconds"] / slc_row["wall_seconds"]
                rows.append({"width": width, "K": K, "M": horizon, "shift": shift,
                             "champion_N": champ_row["N"], "slc_N": slc_row["N"],
                             "champion_seconds": champ_row["wall_seconds"],
                             "slc_seconds": slc_row["wall_seconds"],
                             "champion_peak_rss_bytes": champ_row["peak_rss_bytes"],
                             "slc_peak_rss_bytes": slc_row["peak_rss_bytes"],
                             "time_ratio": ratio, "below_half": bool(ratio < 0.5)})
    return rows


def amendment_agreement(accuracy_rows: list, block: dict, minima: dict) -> dict:
    """Return the amendment prediction against the amendment realized runs."""
    predicted = {}
    for row in block["rows"]:
        key = (csv_route(row["route"]), row["K"], row["shift"], row["N"], row["M"])
        predicted[key] = row
    rows = []
    worst = 0.0
    worst_row = None
    names = {csv_route(name) for name in AMENDMENT_ROUTE_NAMES}
    for run in accuracy_rows:
        if run["kind"] != "accuracy" or run["route"] not in names:
            continue
        if run["N"] not in AMENDMENT_LADDER:
            continue
        for horizon in HORIZONS:
            ref = predicted.get((run["route"], run["K"], run["shift"], run["N"],
                                 horizon))
            if ref is None:
                continue
            for channel in CHANNELS:
                got = run[f"{channel}_M{horizon}"]
                want = ref[channel]
                if want <= 0.0 and got <= 0.0:
                    ratio = 1.0
                elif want <= 0.0:
                    ratio = float("inf")
                else:
                    ratio = got / want
                entry = {"route": run["route"], "K": run["K"], "shift": run["shift"],
                         "N": run["N"], "M": horizon, "channel": channel,
                         "realized": got, "predicted": want, "ratio": ratio}
                rows.append(entry)
                deviation = max(ratio, 1.0 / ratio) if ratio > 0.0 else float("inf")
                if deviation > worst:
                    worst = deviation
                    worst_row = entry
    grid_rows = []
    grids_agree = True
    for key, value in minima["combined"].items():
        want = block["minimum_admissible_N"].get(key)
        grid_rows.append({"key": key, "realized": value, "predicted": want,
                          "agree": value == want})
        grids_agree = grids_agree and (value == want)
    shift_rows = []
    shifts_agree = True
    for key, value in minima["per_shift"].items():
        want = block["minimum_admissible_N_per_shift"].get(key)
        shift_rows.append({"key": key, "realized": value, "predicted": want,
                           "agree": value == want})
        shifts_agree = shifts_agree and (value == want)
    return {"factor": AGREEMENT_FACTOR, "worst_deviation": worst,
            "worst_row": worst_row, "within_factor": bool(worst <= AGREEMENT_FACTOR),
            "minimum_grids_agree": bool(grids_agree),
            "minimum_grids_agree_per_shift": bool(shifts_agree),
            "minimum_grid_rows": grid_rows,
            "minimum_grid_rows_per_shift": shift_rows,
            "rows": rows}


def stage_amendment_decide(accuracy_rows: list, timing_rows: list, prediction: dict,
                           coherence: dict) -> dict:
    """Return the amendment block of decision.json, one verdict per width."""
    block = prediction["amendment_1"]
    minima = amendment_realized_minima(accuracy_rows)
    times = timing_lookup(timing_rows)
    widths = {}
    for width in LAGRANGE_WIDTHS:
        node_rows = width_node_rows(minima["combined"], width)
        time_rows = amendment_time_rows(times, width)
        node_ok = all(row["champion_at_most_half"] for row in node_rows)
        if time_rows:
            time_ok = all(row["below_half"] for row in time_rows)
            expected = len(KS) * 2 * len(SHIFTS)
            complete = len(time_rows) == expected
        else:
            time_ok = None
            complete = False
        # The rule reads REJECT whenever the node clause fails, whatever the time
        # clause does, so a width whose node clause fails needs no timing.
        if not node_ok:
            verdict = "REJECT"
        elif time_ok is None:
            verdict = None
        elif time_ok:
            verdict = "TRIUMPH"
        else:
            verdict = "PARTIAL"
        widths[str(width)] = {
            "route": f"slc_operator_L{width}",
            "runs_csv_name": csv_route(f"slc_operator_L{width}"),
            "node_rule_holds": bool(node_ok),
            "time_rule_holds": None if time_ok is None else bool(time_ok),
            "time_rows_complete": bool(complete),
            "verdict": verdict,
            "node_rows": node_rows,
            "time_rows": time_rows,
        }
    memory_rows = []
    for row in timing_rows:
        if row.get("basis") not in (AMENDMENT_MINIMUM, AMENDMENT_EQUAL):
            continue
        memory_rows.append({"basis": row["basis"], "route": row["route"],
                            "K": row["K"], "shift": row["shift"], "N": row["N"],
                            "M": row["steps"],
                            "state_bytes": row["state_bytes"],
                            "peak_rss_bytes": row["peak_rss_bytes"],
                            "peak_rss_increment_bytes":
                                row["peak_rss_increment_bytes"]})
    timed = [name for name in AMENDMENT_ROUTE_NAMES
             if any(row.get("basis") == AMENDMENT_MINIMUM
                    and row["route"] == csv_route(name) for row in timing_rows)]
    names = {csv_route(name) for name in AMENDMENT_ROUTE_NAMES}
    mass = [float(row["mass_final"]) for row in accuracy_rows
            if row["kind"] == "accuracy" and row["route"] in names
            and row["N"] in AMENDMENT_LADDER]
    return {
        "written": now(),
        "scope": AMENDMENT_SCOPE,
        "rule": AMENDMENT_RULE,
        "rule_applied": ("The rule is applied verbatim against each Lagrange width "
                         "separately: the node clause asks the champion's minimum "
                         "admissible N to be at most half that width's at every "
                         "(K, M), and the time clause asks the total time ratio at "
                         "the realized minima to be below 0.5 at every (K, M) and "
                         "shift."),
        "routes": list(AMENDMENT_ROUTE_NAMES),
        "ladder": list(AMENDMENT_LADDER),
        "member": member_record(),
        "minimum_admissible_N": minima["combined"],
        "minimum_admissible_N_per_shift": minima["per_shift"],
        "rung_below_minimum": amendment_below_rows(minima),
        "rung_below_minimum_predicted": block["rung_below_minimum"],
        "per_width": widths,
        "verdict_per_width": {key: value["verdict"] for key, value in widths.items()},
        "timing_routes_declared": list(AMENDMENT_TIMING_NAMES),
        "timing_routes_run": timed,
        "memory_rows": memory_rows,
        "coherence_residuals": coherence,
        "mass_drift_max": max(mass) if mass else None,
        "prediction_versus_realized": amendment_agreement(accuracy_rows, block,
                                                          minima),
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def load_rows() -> list:
    path = OUT / "runs.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in ("kind", "route", "shift", "basis", "history_file",
                             "wall_seconds_all"):
                    row[key] = value
                elif key in ("K", "N", "steps", "threads", "peak_rss_increment_bytes",
                             "baseline_rss_bytes", "peak_rss_bytes", "state_bytes"):
                    row[key] = int(value)
                elif value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["predict", "accuracy", "timing", "reduce", "all",
                                 "timing-worker", "amendment-predict",
                                 "amendment-accuracy", "amendment-timing",
                                 "amendment-reduce"])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timing-routes", default="amendment",
                        choices=["amendment", "all"])
    parser.add_argument("--route", default="champion")
    parser.add_argument("--K", type=int, default=24)
    parser.add_argument("--shift", default="A")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "timing-worker":
        print(json.dumps(timing_worker(args.route, args.K, args.shift, args.N,
                                       args.steps)))
        return 0
    if args.stage in ("predict", "all"):
        stage_predict()
        print(f"prediction written {now()}")
    if args.stage in ("accuracy", "all"):
        if not (OUT / "prediction.json").exists():
            raise SystemExit("The prediction must be written before any realized run.")
        rows = [row for row in load_rows() if row["kind"] != "accuracy"]
        rows += stage_accuracy(args.jobs)
        write_runs(rows)
        print(f"accuracy runs written {now()}")
    if args.stage in ("timing", "all"):
        rows = load_rows()
        accuracy = [row for row in rows if row["kind"] == "accuracy"]
        rows = accuracy + stage_timing(accuracy)
        write_runs(rows)
        print(f"timing runs written {now()}")
    if args.stage in ("reduce", "all"):
        rows = load_rows()
        accuracy = [row for row in rows if row["kind"] == "accuracy"]
        timing = [row for row in rows if row["kind"] == "timing"]
        prediction = json.loads((OUT / "prediction.json").read_text())
        routes = build_routes()
        coherence = {name: coherence_residual(routes[name], COHERENCE_GRID, 12, "A")
                     for name in ROUTE_NAMES}
        decision = stage_decide(accuracy, timing, prediction, coherence)
        make_figure(accuracy, realized_minima(accuracy))
        write_readme(prediction, accuracy, timing, decision, coherence)
        print(f"reduction written {now()}: {decision['verdict']}")
    if args.stage == "amendment-predict":
        stage_amendment_predict(args.jobs)
        print(f"amendment prediction written {now()}")
    if args.stage == "amendment-accuracy":
        prediction = json.loads((OUT / "prediction.json").read_text())
        if "amendment_1" not in prediction:
            raise SystemExit("The amendment prediction must be written before any "
                             "amendment run.")
        rows = load_rows()
        rows = rows + stage_amendment_accuracy(args.jobs, rows)
        write_runs(rows)
        print(f"amendment accuracy runs written {now()}")
    if args.stage == "amendment-timing":
        rows = load_rows()
        accuracy = [row for row in rows if row["kind"] == "accuracy"]
        names = (AMENDMENT_ROUTE_NAMES if args.timing_routes == "all"
                 else AMENDMENT_TIMING_NAMES)
        base = [row for row in rows
                if not (row["kind"] == "timing"
                        and row.get("basis") in (AMENDMENT_MINIMUM, AMENDMENT_EQUAL))]
        fresh = stage_amendment_timing(accuracy, rows, names,
                                       lambda done: write_runs(base + done))
        write_runs(base + fresh)
        print(f"amendment timing runs written {now()}")
    if args.stage == "amendment-reduce":
        rows = load_rows()
        accuracy = [row for row in rows if row["kind"] == "accuracy"]
        timing = [row for row in rows if row["kind"] == "timing"]
        prediction = json.loads((OUT / "prediction.json").read_text())
        routes = build_routes()
        coherence = {csv_route(name): coherence_residual(routes[csv_route(name)],
                                                         COHERENCE_GRID, 12, "A")
                     for name in AMENDMENT_ROUTE_NAMES}
        base_accuracy = [row for row in accuracy
                         if row["route"] in ROUTE_NAMES and row["N"] in LADDER]
        base_timing = [row for row in timing
                       if row.get("basis") in ("minimum", "equal_grid")]
        base_coherence = {name: coherence[name] for name in ROUTE_NAMES}
        decision = stage_decide(base_accuracy, base_timing, prediction,
                                base_coherence)
        amend_coherence = {name: coherence[csv_route(name)]
                           for name in AMENDMENT_ROUTE_NAMES}
        amendment = stage_amendment_decide(accuracy, timing, prediction,
                                           amend_coherence)
        decision["amendment_1"] = amendment
        (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        make_figure(accuracy, amendment_realized_minima(accuracy))
        write_readme(prediction, base_accuracy, base_timing, decision,
                     base_coherence, accuracy, timing, amendment)
        verdicts = ", ".join(f"L{width} {amendment['verdict_per_width'][width]}"
                             for width in sorted(amendment["verdict_per_width"],
                                                 key=int))
        print(f"amendment reduction written {now()}: {verdicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
