"""Gate 5 part 5a: coupled off-grid capability, at the symbol level and realized.

Five routes answer the same question, namely what a solver can read at an
arbitrary point between the nodes: the value, the first derivative and the
literal second derivative of the field it is carrying.

  champion        E_{6,2}(1.7701927977475826, 2.0606855355641316), one field
  BMC9            E_{10,0}, one field
  BMC11           E_{12,0}, one field
  slc_operator   SLC 1992 nodal D1 and D2, then the degree-seven barycentric
                  Lagrange interpolant on the nearest eight nodes applied to
                  the value field and to each of the two nodal derivative
                  fields (the Milestone 10 construction)
  slc_field      the same Lagrange value interpolant, differentiated

Symbol level. With the query at x = (j + xi) h and the carrier e^{i theta xi}
kept explicit, write hat T_q(theta, xi) for the response of channel q to the
grid mode e^{i j theta}, normalized by the exact channel, and
T_q = e^{-i theta xi} hat T_q for the transfer of the Gate 0 contract. The
coupled error is

    e_c(theta) = max over xi in [-1/2, 1/2] of max(|T_0 - 1|, |T_1 - 1|, |T_2 - 1|)

on 129 uniform xi. The coherence residuals are

    C_1 = hat T_1 - (1/(i theta)) d hat T_0 / d xi,
    C_2 = hat T_2 + (1/theta^2) d^2 hat T_0 / d xi^2,

which vanish identically when the three channels are read from one field, and
which the operator-faithful augmentation cannot satisfy because its derivative
channels come from operators rather than from the interpolant.

Realized. The Gate 0 anchor packet

    u(x) = sin(3x + 0.17) + 0.35 cos(7x - 0.31)
         + 0.15 sin(11x + 0.28) + 0.07 cos(15x - 0.41)

on N in {64, 128, 256} of the 2 pi torus, read at 4,096 uniformly random query
positions (seed 20260912, one position set shared by every grid and route), the
three channel errors taken against the analytic field, and the orders fitted
across the three grids. The prediction is written before any realized run.

Amendment 1. That packet has its top mode at k = 15, that is at 0.469 pi on
N = 64 and 0.234 pi on N = 128, so the realized test above measures formal
order rather than the design regime, and the augmentation above uses an
eight-node Lagrange layer while the wall experiment of the paper uses the
twelve-node evaluator. The amendment therefore adds, without changing anything
above,

  * the symbol level of the two augmentations at Lagrange widths 8, 12 and 16,
    alongside the champion, BMC9 and BMC11 rows unchanged;
  * a resolution-scaled realized test on N in {48, 64, 96, 128, 192, 256} with
    the packet u(x) = sum_{k=1}^{floor(N/3)} k^{-1} cos(k x + varphi_k), fixed
    phases, whose top mode sits at the dealiased band edge on every grid, so
    the points per wavelength at the band edge are fixed as N grows;
  * the realized order of the second derivative channel as the all-grid fit and
    as the successive pairwise fits, with the pairs whose endpoints sit at the
    double precision floor marked;
  * the well-resolved packet kept as a second table, so both regimes are
    visible, with the ratio of every route to the champion in both.

The amendment block of prediction.json is written before any amendment
realized run. The shared evaluation layer is lagrange_layers.py.

Two conventions the contract leaves open are resolved here.

* Lagrange stencil centering. The Milestone 10 bolt-on
  (code/solver/offgrid_turbulence.cpp) takes the eight nodes
  floor(x/h) - 3, ..., floor(x/h) + 4. Written against the cell coordinate
  xi in [-1/2, 1/2] of the transfer conventions, with j the nearest node, that
  is the offsets -3, ..., +4 for xi >= 0 and -4, ..., +3 for xi < 0. The
  stencil therefore switches at xi = 0, where the interpolant is continuous but
  its derivatives are not, and e_c is computed with both halves. The value
  xi = 0 itself is assigned the xi >= 0 stencil, which is what floor does.
* Transfer assembly. The identity of A6 is stated as T_1 = (1/(i theta))
  dT_0/dxi; with T_q defined after the carrier e^{i theta xi} has been divided
  out, that identity reads T_1 = T_0 + (1/(i theta)) dT_0/dxi, so the residual
  C_1 above is written with the carrier kept, which is the form that is
  identically zero for a one-field route. Likewise for C_2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for path in (str(CODE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import comparators as cmp  # noqa: E402
import lagrange_layers as ll  # noqa: E402
import metrics as mt  # noqa: E402
from evaluator import (ExponentialEvaluator, evaluate_periodic_field,  # noqa: E402
                       periodic_coefficients)
from family import ExponentialGenerator  # noqa: E402

OUT = ROOT / "results" / "exponential_family" / "gate5" / "5a"
FIGURES = OUT / "figures"
CONTRACT = ROOT / "results" / "exponential_family" / "gate5" / "contract.md"

CHAMPION = (6, (1.7701927977475826, 2.0606855355641316))
XI_POINTS = 129
THETA_POINTS = 20001
THETA_CHUNK = 1000
BANDS = mt.BANDS
REACH_TOL = 1e-2
COHERENCE_FLOOR_THETA = 0.01 * np.pi
PREDICTION_FLOOR = 1e-12
LAGRANGE_POINTS = 8
GRIDS = (64, 128, 256)
QUERIES = 4096
SEED = 20260912
PACKET = ((3.0, 1.0, 0.17, "sin"), (7.0, 0.35, -0.31, "cos"),
          (11.0, 0.15, 0.28, "sin"), (15.0, 0.07, -0.41, "cos"))
AGREEMENT_FACTOR = 2.0

# Amendment 1. The widths of the Lagrange evaluation layer that are compared,
# the grids of the resolution-scaled packet, the band fraction that fixes its
# top mode and the amplitude law and phase seed of that packet.
LAGRANGE_WIDTHS = (8, 12, 16)
SCALED_GRIDS = (48, 64, 96, 128, 192, 256)
SCALED_BAND_DIVISOR = 3
SCALED_AMPLITUDE_EXPONENT = -1.0
SCALED_PHASE_SEED = 20260913
SCALED_TRIPLE = (64, 128, 256)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# the anchor packet
# ----------------------------------------------------------------------

def packet_modes():
    """Return the complex amplitudes c_k with u(x) = 2 Re sum_k c_k exp(i k x)."""
    modes = []
    for wavenumber, amplitude, phase, kind in PACKET:
        if kind == "sin":
            c = amplitude * np.exp(1j * phase) / (2j)
        else:
            c = amplitude * np.exp(-1j * phase) / 2.0
        modes.append((wavenumber, complex(c)))
    return modes


def modal_field(modes, x: np.ndarray, derivative: int = 0) -> np.ndarray:
    """Return 2 Re sum_k c_k (i k)^d exp(i k x) for the listed modes."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape, dtype=float)
    for wavenumber, c in modes:
        out += 2.0 * np.real(c * (1j * wavenumber) ** derivative * np.exp(1j * wavenumber * x))
    return out


def packet(x: np.ndarray, derivative: int = 0) -> np.ndarray:
    """Return the analytic packet or one of its derivatives."""
    return modal_field(packet_modes(), x, derivative)


def scaled_packet_modes(npts: int):
    """Return the modes of the resolution-scaled packet on npts nodes.

    Every integer wavenumber 1 <= k <= floor(npts / 3) is present with
    amplitude k^{-1} and a phase drawn once, so

        u(x) = sum_k k^{-1} cos(k x + varphi_k),

    whose top mode sits at theta = floor(npts/3) (2 pi / npts), that is, at the
    dealiased band edge 2 pi / 3 up to the rounding of floor. The phases are
    the leading entries of one draw, so the packet of a coarse grid is the
    low-wavenumber part of the packet of every finer grid. With u written as
    2 Re sum_k c_k exp(i k x) the amplitude is c_k = exp(i varphi_k) / (2 k).
    """
    kmax = npts // SCALED_BAND_DIVISOR
    rng = np.random.default_rng(SCALED_PHASE_SEED)
    phases = rng.uniform(0.0, 2.0 * np.pi, max(SCALED_GRIDS) // SCALED_BAND_DIVISOR)
    modes = []
    for k in range(1, kmax + 1):
        amplitude = float(k) ** SCALED_AMPLITUDE_EXPONENT
        modes.append((float(k), complex(0.5 * amplitude * np.exp(1j * phases[k - 1]))))
    return modes


# ----------------------------------------------------------------------
# the Lagrange layer, from the shared module
# ----------------------------------------------------------------------
#
# lagrange_layers.py carries the centered layer of any even width in
# LAGRANGE_WIDTHS with the same stencil convention. The four names below are
# the eight-node case, kept so that the callers of this module keep their
# signatures.

LAGRANGE_OFFSETS = ll.offsets(LAGRANGE_POINTS)
LAGRANGE_POLY = list(ll.basis(LAGRANGE_POINTS)[1])


def lagrange_basis():
    """Return the offsets and the coefficients of the eight cardinal functions."""
    offsets, polynomials = ll.basis(LAGRANGE_POINTS)
    return offsets, list(polynomials)


def lagrange_weights(local: np.ndarray, order: int = 0) -> np.ndarray:
    """Return the eight interpolation weights at local coordinates in [0, 1)."""
    return ll.nodal_weights(LAGRANGE_POINTS, local, order)


def lagrange_apply(values: np.ndarray, x: np.ndarray, h: float, order: int = 0) -> np.ndarray:
    """Return the Lagrange interpolant of a periodic nodal field at x."""
    return ll.apply_nodal(LAGRANGE_POINTS, values, x, h, order)


def lagrange_transfer(theta: np.ndarray, xi: np.ndarray, order: int = 0) -> np.ndarray:
    """Return the carrier-kept eight-node Lagrange transfer."""
    return ll.transfer(LAGRANGE_POINTS, theta, xi, order)


# ----------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------

class SplineRoute:
    """One exponential-family member read as a single field."""

    def __init__(self, name: str, m: int, omegas=()) -> None:
        self.name = name
        self.evaluator = ExponentialEvaluator(m, omegas)
        self.generator = ExponentialGenerator(self.evaluator.m, self.evaluator.omegas)
        self.taps_per_query = int(self.evaluator.n)
        self.solves_per_field = 1
        self.fields = 1
        self.kind = "one field"
        self._samples = {}

    def description(self) -> str:
        return (f"{self.evaluator.label()}, degree {self.evaluator.p}, support "
                f"[-{self.evaluator.n / 2:g}, {self.evaluator.n / 2:g}]")

    def sample_block(self, xi: np.ndarray):
        """Return the lattice and the three generator sample blocks at xi.

        The blocks phi^{(q)}(l + xi) depend on the cell coordinate alone, so
        one query set is sampled once and reused by every grid mode.
        """
        key = (xi.shape, xi.tobytes())
        got = self._samples.get(key)
        if got is not None:
            return got
        ev = self.evaluator
        radius = (ev.n + 1) // 2 + 2
        lattice = np.arange(-radius, radius + 1).astype(float)
        block = [np.empty((lattice.size, xi.size)) for _ in range(3)]
        for i, node in enumerate(lattice):
            for q in range(3):
                block[q][i] = ev.derivative(node + xi, q)
        if len(self._samples) >= 24:
            self._samples.clear()
        self._samples[key] = (lattice, block)
        return lattice, block

    def hat_transfers(self, theta: np.ndarray, xi: np.ndarray):
        """Return hat T_0, hat T_1, hat T_2 and the two xi derivatives of hat T_0."""
        lattice, s = self.sample_block(xi)
        arg = np.outer(theta, lattice)
        expo = np.exp(-1j * arg)
        b = self.generator.symbols(theta)[0][:, None]
        hat0 = (expo @ s[0]) / b
        hat1 = (-1j) * ((expo - 1.0) @ s[1]) / (theta[:, None] * b)
        hat2 = (-1.0) * ((expo - 1.0 + 1j * arg) @ s[2]) / (theta[:, None] ** 2 * b)
        d1 = (expo @ s[1]) / b
        d2 = (expo @ s[2]) / b
        return hat0, hat1, hat2, d1, d2

    def realized(self, values: np.ndarray, x: np.ndarray, h: float):
        coefficients = periodic_coefficients(values, self.evaluator)
        return tuple(evaluate_periodic_field(coefficients, x, h, self.evaluator,
                                             derivative=q) for q in range(3))


WORDS = {8: "seven", 12: "eleven", 16: "fifteen"}
NODE_WORDS = {8: "eight", 12: "twelve", 16: "sixteen"}


class SLCRoute:
    """The SLC 1992 pair with a centered Lagrange layer of width n_L bolted on."""

    def __init__(self, name: str, faithful: str, n_nodes: int = LAGRANGE_POINTS) -> None:
        self.name = name
        self.faithful = faithful
        self.n_nodes = ll.check_nodes(n_nodes)
        self.kappa1 = cmp.compact_kappa1(cmp.SLC_D1["alpha"], cmp.SLC_D1["a"])
        self.kappa2 = cmp.compact_kappa2(cmp.SLC_D2["alpha"], cmp.SLC_D2["a0"],
                                         cmp.SLC_D2["a"])
        if faithful == "operator":
            self.taps_per_query = 3 * self.n_nodes
            self.solves_per_field = 2
            self.fields = 3
            self.kind = "three fields"
        else:
            self.taps_per_query = self.n_nodes
            self.solves_per_field = 0
            self.fields = 1
            self.kind = "one field"

    def description(self) -> str:
        degree = WORDS[self.n_nodes]
        nodes = NODE_WORDS[self.n_nodes]
        if self.faithful == "operator":
            return (f"SLC 1992 nodal D1 and D2, degree {degree} Lagrange on the nearest "
                    f"{nodes} nodes for all three channels")
        return (f"degree {degree} Lagrange value interpolant on the nearest {nodes} "
                "nodes, differentiated analytically")

    def hat_transfers(self, theta: np.ndarray, xi: np.ndarray):
        hat0 = ll.transfer(self.n_nodes, theta, xi, 0)
        d1 = ll.transfer(self.n_nodes, theta, xi, 1)
        d2 = ll.transfer(self.n_nodes, theta, xi, 2)
        if self.faithful == "operator":
            k1 = (np.asarray(self.kappa1(theta), dtype=float) / theta)[:, None]
            k2 = (np.asarray(self.kappa2(theta), dtype=float) / theta ** 2)[:, None]
            hat1 = k1 * hat0
            hat2 = k2 * hat0
        else:
            hat1 = d1 / (1j * theta[:, None])
            hat2 = -d2 / (theta[:, None] ** 2)
        return hat0, hat1, hat2, d1, d2

    def realized(self, values: np.ndarray, x: np.ndarray, h: float):
        npts = values.shape[0]
        n = self.n_nodes
        value = ll.apply_nodal(n, values, x, h, 0)
        if self.faithful == "field":
            return (value, ll.apply_nodal(n, values, x, h, 1),
                    ll.apply_nodal(n, values, x, h, 2))
        theta = 2.0 * np.pi * np.fft.fftfreq(npts)
        spectrum = np.fft.fft(values)
        nodal1 = np.real(np.fft.ifft(spectrum * 1j * np.asarray(self.kappa1(theta),
                                                                dtype=float) / h))
        nodal2 = np.real(np.fft.ifft(spectrum * (-np.asarray(self.kappa2(theta),
                                                             dtype=float)) / (h * h)))
        return (value, ll.apply_nodal(n, nodal1, x, h, 0),
                ll.apply_nodal(n, nodal2, x, h, 0))


def build_routes():
    """Return the five routes of the original part 5a, in the original order."""
    return [
        SplineRoute("champion", *CHAMPION),
        SplineRoute("BMC9", 10, ()),
        SplineRoute("BMC11", 12, ()),
        SLCRoute("slc_operator", "operator"),
        SLCRoute("slc_field", "field"),
    ]


# The amendment names the eight-node augmentations explicitly, so that
# slc_operator_L8 is the route the original tables call slc_operator and
# slc_field_L8 is the route they call slc_field.
SPLINE_NAMES = ("champion", "BMC9", "BMC11")
HEADLINE_NAMES = SPLINE_NAMES + tuple(f"slc_operator_L{n}" for n in LAGRANGE_WIDTHS)
ORIGINAL_ALIAS = {"slc_operator_L8": "slc_operator", "slc_field_L8": "slc_field"}


def build_amendment_routes():
    """Return the nine routes of amendment 1, splines first then the augmentations."""
    routes = [
        SplineRoute("champion", *CHAMPION),
        SplineRoute("BMC9", 10, ()),
        SplineRoute("BMC11", 12, ()),
    ]
    for faithful in ("operator", "field"):
        for n_nodes in LAGRANGE_WIDTHS:
            routes.append(SLCRoute(f"slc_{faithful}_L{n_nodes}", faithful, n_nodes))
    return routes


# ----------------------------------------------------------------------
# symbol level
# ----------------------------------------------------------------------

def symbol_level(route) -> dict:
    """Return e_c, the coherence residuals and the band quantities of one route."""
    theta = mt.theta_grid(THETA_POINTS)
    xi = np.linspace(-0.5, 0.5, XI_POINTS)
    ec = np.empty(theta.size)
    c1 = np.empty(theta.size)
    c2 = np.empty(theta.size)
    channels = np.empty((3, theta.size))
    for start in range(0, theta.size, THETA_CHUNK):
        block = theta[start:start + THETA_CHUNK]
        hat0, hat1, hat2, d0, dd0 = route.hat_transfers(block, xi)
        carrier = np.exp(-1j * np.outer(block, xi))
        worst = None
        for q, hat in enumerate((hat0, hat1, hat2)):
            err = np.abs(carrier * hat - 1.0)
            channels[q, start:start + block.size] = err.max(axis=1)
            worst = err if worst is None else np.maximum(worst, err)
        ec[start:start + block.size] = worst.max(axis=1)
        r1 = hat1 - d0 / (1j * block[:, None])
        r2 = hat2 + dd0 / (block[:, None] ** 2)
        c1[start:start + block.size] = np.abs(r1).max(axis=1)
        c2[start:start + block.size] = np.abs(r2).max(axis=1)
    row = {"route": route.name, "description": route.description(),
           "kind": route.kind, "fields": route.fields,
           "taps_per_query": route.taps_per_query,
           "solves_per_field": route.solves_per_field}
    for key, edge in BANDS.items():
        row[f"ec_max_{key}"] = mt.band_max(theta, ec, edge)
        row[f"c1_max_{key}"] = mt.band_max(theta, c1, edge)
        row[f"c2_max_{key}"] = mt.band_max(theta, c2, edge)
        for q in range(3):
            row[f"e{q}_max_{key}"] = mt.band_max(theta, channels[q], edge)
    reach = mt.reach(theta, ec, REACH_TOL)
    row["reach_ec_1e-2_over_pi"] = reach["linear"] / np.pi
    row["reach_ec_1e-2_saturated"] = bool(reach["saturated"])
    row["c1_max_full"] = float(np.max(c1))
    row["c2_max_full"] = float(np.max(c2))
    # The identity carries 1/theta and 1/theta^2, so at the bottom of the angle
    # grid a residual that is exactly zero in exact arithmetic still reads as
    # the rounding of the transfers divided by theta^2. The restricted maxima
    # below cut that amplification off at theta = 0.01 pi.
    keep = theta >= COHERENCE_FLOOR_THETA
    row["c1_max_theta_above_floor"] = float(np.max(c1[keep]))
    row["c2_max_theta_above_floor"] = float(np.max(c2[keep]))
    row["coherence_floor_theta_over_pi"] = float(COHERENCE_FLOOR_THETA / np.pi)
    return row, theta, ec


# ----------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------

def query_positions(h: float, npts: int):
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0.0, 2.0 * np.pi, QUERIES)
    return x


def predicted_errors(route, npts: int, x: np.ndarray, modes=None) -> dict:
    """Return the modal prediction of the three realized channel errors.

    The packet is a finite sum of grid modes, so the numerical channel q at the
    query x = (j + xi) h is exactly 2 Re sum_k c_k (i k)^q T_q(k h, xi) e^{i k x}
    and the error is the same sum with T_q - 1. That is the modal_exact entry.
    The modal_bound entry is the triangle inequality applied mode by mode with
    the worst xi, which is what a reader with only the transfer plot can form.
    """
    if modes is None:
        modes = packet_modes()
    h = 2.0 * np.pi / npts
    t = x / h
    xi = t - np.round(t)
    xi_grid = np.linspace(-0.5, 0.5, XI_POINTS)
    out = {"modal_exact": {}, "modal_bound": {}}
    errors = [np.zeros(x.size) for _ in range(3)]
    bounds = [0.0, 0.0, 0.0]
    scales = [np.max(np.abs(modal_field(modes, x, q))) for q in range(3)]
    for wavenumber, c in modes:
        theta = np.array([wavenumber * h])
        hat0, hat1, hat2, _, _ = route.hat_transfers(theta, xi)
        carrier = np.exp(-1j * theta[:, None] * xi)
        gap = [(carrier * hat - 1.0)[0] for hat in (hat0, hat1, hat2)]
        hat0g, hat1g, hat2g, _, _ = route.hat_transfers(theta, xi_grid)
        carrier_g = np.exp(-1j * theta[:, None] * xi_grid)
        gap_grid = [(carrier_g * hat - 1.0)[0] for hat in (hat0g, hat1g, hat2g)]
        for q in range(3):
            factor = c * (1j * wavenumber) ** q
            errors[q] += 2.0 * np.real(factor * gap[q] * np.exp(1j * wavenumber * x))
            bounds[q] += 2.0 * abs(factor) * float(np.max(np.abs(gap_grid[q])))
    for q in range(3):
        out["modal_exact"][f"channel_{q}"] = float(np.max(np.abs(errors[q])) / scales[q])
        out["modal_bound"][f"channel_{q}"] = float(bounds[q] / scales[q])
    return out


# ----------------------------------------------------------------------
# realized
# ----------------------------------------------------------------------

def realized_errors(route, npts: int, x: np.ndarray, modes=None) -> dict:
    if modes is None:
        modes = packet_modes()
    h = 2.0 * np.pi / npts
    nodes = h * np.arange(npts)
    values = modal_field(modes, nodes)
    got = route.realized(values, x, h)
    row = {}
    for q in range(3):
        exact = modal_field(modes, x, q)
        scale = float(np.max(np.abs(exact)))
        diff = got[q] - exact
        row[f"channel_{q}_max_abs"] = float(np.max(np.abs(diff)))
        row[f"channel_{q}_rel_sup"] = float(np.max(np.abs(diff)) / scale)
        row[f"channel_{q}_rel_l2"] = float(np.sqrt(np.mean(diff ** 2))
                                           / np.sqrt(np.mean(exact ** 2)))
    return row


def fitted_order(grids, errors) -> float:
    mask = np.array(errors) > 0.0
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope = np.polyfit(np.log(np.array(grids, dtype=float)[mask]),
                       np.log(np.array(errors)[mask]), 1)[0]
    return float(-slope)


def pairwise_orders(grids, errors):
    """Return the successive two-grid slopes log(e_i/e_{i+1}) / log(N_{i+1}/N_i)."""
    return [fitted_order(grids[i:i + 2], errors[i:i + 2]) for i in range(len(grids) - 1)]


# ----------------------------------------------------------------------
# amendment 1
# ----------------------------------------------------------------------
#
# The anchor packet has a fixed top wavenumber k = 15, so its top angle
# k h = 15 (2 pi / N) falls like 1/N: 0.469 pi at N = 64 and 0.234 pi at
# N = 128, both well below the dealiased band edge 2 pi / 3. A refinement
# sequence at fixed k therefore walks the packet down into the range where
# every e_c(theta) is the leading Taylor term, and what is fitted there is the
# formal order.
#
# The resolution-scaled packet fixes the top angle instead of the top
# wavenumber: every integer mode 1 <= k <= floor(N/3) is present, so the top
# angle is floor(N/3) (2 pi / N), which is 2 pi / 3 up to the rounding of
# floor, and the points per wavelength of the hardest mode are h-independent.
#
# The Lagrange evaluation layer of the augmentation is carried at the three
# widths n_L = 8, 12 and 16, whose interpolation errors fall like h^{n_L} in
# the value channel and h^{n_L - q} in channel q. The twelve-node layer is the
# one the wall experiment uses.

CHANNEL_NAMES = ("value", "d1", "d2")


def scaled_top_mode(npts: int) -> dict:
    """Return the top wavenumber of the resolution-scaled packet and its angle."""
    kmax = npts // SCALED_BAND_DIVISOR
    return {"N": npts, "k_max": int(kmax), "modes": int(kmax),
            "theta_top_over_pi": float(kmax * 2.0 / npts),
            "points_per_wavelength_top": float(npts / kmax)}


def anchor_top_mode(npts: int) -> dict:
    """Return the top wavenumber of the Gate 0 anchor packet and its angle."""
    kmax = max(int(wavenumber) for wavenumber, _ in packet_modes())
    return {"N": npts, "k_max": kmax, "modes": len(PACKET),
            "theta_top_over_pi": float(kmax * 2.0 / npts),
            "points_per_wavelength_top": float(npts / kmax)}


def amendment_symbol_rows(routes, base_rows, base_curves):
    """Return the symbol rows and e_c curves of the amendment routes.

    The rows of champion, BMC9, BMC11 and the two eight-node augmentations are
    the rows already computed for the original five routes, carried over under
    the explicit amendment names, so those numbers are literally unchanged.
    """
    rows = []
    curves = []
    for route in routes:
        alias = ORIGINAL_ALIAS.get(route.name, route.name)
        if alias in base_rows:
            row = dict(base_rows[alias])
            row["route"] = route.name
            row["original_5a_route"] = alias
            theta, ec = base_curves[alias]
        else:
            row, theta, ec = symbol_level(route)
            row["original_5a_route"] = None
        rows.append(row)
        curves.append((route.name, theta, ec))
    return rows, curves


def spectral_cost_model(grids) -> dict:
    """Return the cost model of the excluded global spectral interpolant.

    Reading one query point from the N nodal values of a periodic field by its
    trigonometric interpolant touches every node, so the stencil is the whole
    torus: N taps per query point for the value, and N again for each
    derivative channel unless the three are accumulated in one pass, which is
    3 N multiply-adds over the N/2 + 1 modes. Amortized over Q query points the
    forward transform costs O(N log N) once and the nonuniform evaluation
    O(N Q), or O(N log N + Q log(1/eps)) with a nonuniform fast transform whose
    constant is set by the requested tolerance.
    """
    return {
        "excluded": True,
        "reason": ("The comparison of part 5a is between evaluators of compact "
                   "support at a stated bandwidth. The trigonometric interpolant "
                   "is not compact: its stencil is the entire periodic grid, so "
                   "it has no support half width, no taps per query that is "
                   "independent of N, and no restriction to a wall or to a "
                   "subdomain without global communication. Its coupled error "
                   "e_c(theta) is zero to rounding at every theta in (0, pi) by "
                   "construction, so it does not sit on the same axis as the "
                   "routes compared here; it is the exact answer whose cost the "
                   "compact routes are trading against."),
        "taps_per_query": {str(n): int(n) for n in grids},
        "taps_per_query_law": "N, the whole grid, for every channel",
        "setup": "one real forward transform, O(N log N), shared by all queries",
        "per_query": ("O(N) for the direct sum over the N/2 + 1 modes, or "
                      "O(log(1/eps)) after an O(N log N) nonuniform fast "
                      "transform spreading step"),
        "locality": "global; no immersed wall or distributed subdomain use "
                    "without an all-to-all",
    }


def amendment_prediction(routes, x, scaled_modes) -> dict:
    """Return the amendment block of prediction.json, written before any run."""
    return {
        "written_utc": now(),
        "scope": ("Repairs the two defects of the original part 5a: the realized "
                  "packet is rebuilt so that its top mode sits at the dealiased "
                  "band edge on every grid, and the Lagrange evaluation layer of "
                  "the SLC augmentation is carried at widths 8, 12 and 16 so "
                  "that the twelve-node evaluator of the wall experiment is "
                  "among the competitors."),
        "lagrange_widths": list(LAGRANGE_WIDTHS),
        "resolution_scaled_packet": {
            "law": ("u(x) = sum_{k=1}^{floor(N/3)} k^{-1} cos(k x + varphi_k), "
                    "varphi_k drawn once from U[0, 2 pi)"),
            "amplitude_law": "a_k = k^{-1}",
            "amplitude_exponent": SCALED_AMPLITUDE_EXPONENT,
            "phase_seed": SCALED_PHASE_SEED,
            "phases_nested": ("one draw of floor(max N / 3) phases, the leading "
                              "entries reused on every grid"),
            "band_divisor": SCALED_BAND_DIVISOR,
            "grids": list(SCALED_GRIDS),
            "queries": QUERIES,
            "query_seed": SEED,
            "top_mode": [scaled_top_mode(n) for n in SCALED_GRIDS],
        },
        "well_resolved_packet": {
            "law": "the Gate 0 anchor packet of the original part 5a",
            "grids": list(GRIDS),
            "top_mode": [anchor_top_mode(n) for n in GRIDS],
        },
        "global_spectral_interpolant": spectral_cost_model(SCALED_GRIDS),
        "predicted_resolution_scaled": {
            route.name: {str(n): predicted_errors(route, n, x, scaled_modes[n])
                         for n in SCALED_GRIDS}
            for route in routes},
        "predicted_well_resolved": {
            route.name: {str(n): predicted_errors(route, n, x) for n in GRIDS}
            for route in routes},
    }


def amendment_rows(routes, realized, prediction_block, grids, tag) -> list:
    """Return runs.csv rows of one amendment regime, in the original schema."""
    key = ("predicted_resolution_scaled" if tag == "resolution_scaled"
           else "predicted_well_resolved")
    rows = []
    for route in routes:
        for npts in grids:
            got = realized[(route.name, npts)]
            pred = prediction_block[key][route.name][str(npts)]
            for q in range(3):
                predicted = pred["modal_exact"][f"channel_{q}"]
                got_value = got[f"channel_{q}_rel_sup"]
                ratio = got_value / predicted if predicted > 0.0 else float("nan")
                rows.append({
                    "route": f"{route.name}|{tag}", "N": npts,
                    "h": 2.0 * np.pi / npts,
                    "channel": CHANNEL_NAMES[q],
                    "max_abs_error": got[f"channel_{q}_max_abs"],
                    "rel_sup_error": got_value,
                    "rel_l2_error": got[f"channel_{q}_rel_l2"],
                    "predicted_rel_sup_modal_exact": predicted,
                    "predicted_rel_sup_modal_bound": pred["modal_bound"][f"channel_{q}"],
                    "realized_over_predicted": ratio,
                    "taps_per_query": route.taps_per_query,
                    "solves_per_field": route.solves_per_field,
                })
    return rows


def agreement_summary(rows) -> dict:
    """Return the worst realized over predicted deviation of a row block."""
    worst = 0.0
    worst_row = None
    worst_above = 0.0
    worst_above_row = None
    exceeding = []
    for row in rows:
        ratio = row["realized_over_predicted"]
        if not np.isfinite(ratio) or ratio <= 0.0:
            continue
        deviation = max(ratio, 1.0 / ratio)
        if deviation > worst:
            worst = deviation
            worst_row = row
        if row["predicted_rel_sup_modal_exact"] >= PREDICTION_FLOOR:
            if deviation > worst_above:
                worst_above = deviation
                worst_above_row = row
        if deviation > AGREEMENT_FACTOR:
            exceeding.append({k: row[k] for k in ("route", "N", "channel",
                                                  "rel_sup_error",
                                                  "predicted_rel_sup_modal_exact",
                                                  "realized_over_predicted")})
    return {
        "factor": AGREEMENT_FACTOR,
        "worst_realized_over_predicted_deviation": worst,
        "worst_row": worst_row,
        "all_within_factor": bool(worst <= AGREEMENT_FACTOR),
        "prediction_floor": PREDICTION_FLOOR,
        "worst_deviation_above_floor": worst_above,
        "worst_row_above_floor": worst_above_row,
        "all_within_factor_above_floor": bool(worst_above <= AGREEMENT_FACTOR),
        "rows_beyond_the_factor": exceeding,
    }


def order_block(routes, realized, grids, triple=None) -> dict:
    """Return the all-grid, triple and pairwise order fits of one regime."""
    out = {}
    for route in routes:
        entry = {}
        for q, channel in enumerate(CHANNEL_NAMES):
            errors = [realized[(route.name, n)][f"channel_{q}_rel_sup"] for n in grids]
            item = {"all_grids": fitted_order(grids, errors),
                    "pairwise": pairwise_orders(list(grids), errors),
                    "pairs": [f"{grids[i]}->{grids[i + 1]}"
                              for i in range(len(grids) - 1)]}
            if triple is not None:
                sub = [realized[(route.name, n)][f"channel_{q}_rel_sup"] for n in triple]
                item["triple"] = fitted_order(triple, sub)
                item["triple_grids"] = list(triple)
            entry[channel] = item
        out[route.name] = entry
    return out


def ratio_block(routes, realized, grids) -> dict:
    """Return the ratio of each route's relative sup error to the champion's."""
    out = {}
    for route in routes:
        entry = {}
        for q, channel in enumerate(CHANNEL_NAMES):
            cells = {}
            for npts in grids:
                mine = realized[(route.name, npts)][f"channel_{q}_rel_sup"]
                his = realized[("champion", npts)][f"channel_{q}_rel_sup"]
                cells[str(npts)] = float(mine / his) if his > 0.0 else float("nan")
            entry[channel] = cells
        out[route.name] = entry
    return out


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------

ROUTE_COLORS = {
    "champion": "#1f77b4", "BMC9": "#2ca02c", "BMC11": "#9467bd",
    "slc_operator": "#d62728", "slc_field": "#ff7f0e",
    "slc_operator_L8": "#d62728", "slc_field_L8": "#ff7f0e",
    "slc_operator_L12": "#8c564b", "slc_field_L12": "#e377c2",
    "slc_operator_L16": "#7f7f7f", "slc_field_L16": "#17becf",
}
TOP_PANEL_CURVES = ("champion", "BMC9", "BMC11", "slc_operator_L8",
                    "slc_field_L8", "slc_operator_L12", "slc_field_L12")
WELL_PANEL_ROUTES = ("champion", "BMC9", "BMC11", "slc_operator_L8",
                     "slc_field_L8", "slc_operator_L12", "slc_operator_L16")


def make_figure(amendment_curves, well_realized, scaled_realized) -> None:
    """Draw the symbol panel, the well-resolved panels and the scaled panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    FIGURES.mkdir(parents=True, exist_ok=True)
    curve_index = {name: (theta, ec) for name, theta, ec in amendment_curves}
    styles = {"slc_operator_L12": {"dashes": (2.2, 1.0)},
              "slc_field_L12": {"dashes": (2.2, 1.0)},
              "slc_operator_L16": {"dashes": (5.0, 1.6, 1.0, 1.6)}}
    labels = ("value", "first derivative", "literal second derivative")

    fig = plt.figure(figsize=(7.4, 10.2))
    grid = gridspec.GridSpec(3, 3, height_ratios=[1.6, 1.0, 1.0],
                             hspace=0.88, wspace=0.36)

    top = fig.add_subplot(grid[0, :])
    for name in TOP_PANEL_CURVES:
        theta, ec = curve_index[name]
        top.semilogy(theta / np.pi, np.maximum(ec, 1e-17), color=ROUTE_COLORS[name],
                     lw=1.4, label=name, **styles.get(name, {}))
    top.axvline(BANDS["B0"] / np.pi, color="0.35", ls="--", lw=1.0)
    top.annotate("B0 edge", xy=(BANDS["B0"] / np.pi, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(-4, -12), textcoords="offset points", ha="right", va="top",
                 fontsize=8, color="0.35")
    top.set_xlim(0.0, 1.0)
    top.set_ylim(1e-14, 10.0)
    top.set_xlabel(r"$\theta / \pi$")
    top.set_ylabel(r"$e_c(\theta)$")
    top.set_title("Coupled off-grid error, worst over the cell and over the three channels")
    top.legend(fontsize=7.5, loc="upper left", ncol=3, framealpha=0.9)
    top.grid(True, which="major", alpha=0.25)

    def panel_row(row, realized, grids, names, title):
        axes = []
        for q in range(3):
            ax = fig.add_subplot(grid[row, q])
            axes.append(ax)
            for name in names:
                errs = [realized[(name, n)][f"channel_{q}_rel_sup"] for n in grids]
                ax.loglog(grids, errs, marker="o", ms=3.5, lw=1.2,
                          color=ROUTE_COLORS[name], label=name,
                          **styles.get(name, {}))
            ax.set_xlabel("N")
            if q == 0:
                ax.set_ylabel("relative sup error")
            ax.set_title(labels[q], fontsize=9)
            ax.grid(True, which="both", alpha=0.2)
            ax.set_xticks(list(grids))
            ax.set_xticklabels([str(n) for n in grids], fontsize=7)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(axis="both", labelsize=8)
        top_edge = max(ax.get_position().y1 for ax in axes)
        bottom_edge = min(ax.get_position().y0 for ax in axes)
        fig.text(0.5, top_edge + 0.028, title, ha="center", va="bottom", fontsize=10)
        handles, tags = axes[0].get_legend_handles_labels()
        fig.legend(handles, tags, loc="upper center",
                   bbox_to_anchor=(0.5, bottom_edge - 0.036), ncol=4,
                   fontsize=7.5, frameon=False, handslcngth=1.8,
                   columnspacing=1.4)

    panel_row(1, well_realized, GRIDS, WELL_PANEL_ROUTES,
              "Well-resolved packet, top mode 0.23 pi at N = 128")
    panel_row(2, scaled_realized, SCALED_GRIDS, HEADLINE_NAMES,
              "Resolution-scaled packet, top mode at 2 pi / 3 on every grid")
    fig.savefig(FIGURES / "fig_coupled_offgrid.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "predict", "realize"), default="all")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    routes = build_routes()
    contract_hash = sha256(CONTRACT) if CONTRACT.exists() else None

    symbol_rows = []
    curves = []
    for route in routes:
        row, theta, ec = symbol_level(route)
        symbol_rows.append(row)
        curves.append((route.name, theta, ec))

    x = query_positions(2.0 * np.pi / GRIDS[0], GRIDS[0])
    prediction = {
        "producer": "code/analysis/exponential_family/gate5a_offgrid.py",
        "written_utc": now(),
        "part": "5a",
        "contract": {"path": "results/exponential_family/gate5/contract.md",
                     "sha256": contract_hash},
        "definition": {
            "e_c": "max over 129 xi in [-1/2, 1/2] of max(|T_0-1|, |T_1-1|, |T_2-1|)",
            "theta_grid": f"{THETA_POINTS} uniform angles on (0, pi]",
            "coherence": "C_1 = hat T_1 - (1/(i theta)) d hat T_0/dxi, "
                         "C_2 = hat T_2 + (1/theta^2) d^2 hat T_0/dxi^2, carrier kept",
            "modal_exact": "max over the 4,096 query points of the exact modal error "
                           "sum, divided by the sup of the analytic channel on the "
                           "same points",
            "modal_bound": "mode by mode triangle inequality with the worst xi",
            "agreement_factor": AGREEMENT_FACTOR,
        },
        "field": {"packet": "sin(3x+0.17) + 0.35 cos(7x-0.31) + 0.15 sin(11x+0.28) "
                            "+ 0.07 cos(15x-0.41)",
                  "grids": list(GRIDS), "queries": QUERIES, "seed": SEED},
        "symbol_level": symbol_rows,
        "predicted": {},
    }
    for route in routes:
        prediction["predicted"][route.name] = {
            str(npts): predicted_errors(route, npts, x) for npts in GRIDS}

    # The symbol level depends on the route alone and not on the packet, so the
    # rows of champion, BMC9, BMC11 and the two eight-node augmentations are the
    # rows already computed above, carried over under their explicit names.
    amendment_routes = build_amendment_routes()
    base_rows = {row["route"]: row for row in symbol_rows}
    base_curves = {name: (theta, ec) for name, theta, ec in curves}
    amend_symbol, amend_curves = amendment_symbol_rows(amendment_routes, base_rows,
                                                       base_curves)
    scaled_modes = {npts: scaled_packet_modes(npts) for npts in SCALED_GRIDS}
    amendment = amendment_prediction(amendment_routes, x, scaled_modes)
    amendment["symbol_level"] = amend_symbol
    prediction["amendment_1"] = amendment
    (OUT / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
    print(f"prediction written {OUT / 'prediction.json'}")
    if args.stage == "predict":
        return 0

    started = now()
    realized = {}
    for route in routes:
        for npts in GRIDS:
            realized[(route.name, npts)] = realized_errors(route, npts, x)

    scaled_realized = {}
    well_realized = {}
    for route in amendment_routes:
        for npts in SCALED_GRIDS:
            scaled_realized[(route.name, npts)] = realized_errors(
                route, npts, x, scaled_modes[npts])
        for npts in GRIDS:
            well_realized[(route.name, npts)] = realized_errors(route, npts, x)

    rows = []
    for route in routes:
        for npts in GRIDS:
            got = realized[(route.name, npts)]
            pred = prediction["predicted"][route.name][str(npts)]
            for q in range(3):
                predicted = pred["modal_exact"][f"channel_{q}"]
                got_value = got[f"channel_{q}_rel_sup"]
                ratio = got_value / predicted if predicted > 0.0 else float("nan")
                rows.append({
                    "route": route.name, "N": npts, "h": 2.0 * np.pi / npts,
                    "channel": ("value", "d1", "d2")[q],
                    "max_abs_error": got[f"channel_{q}_max_abs"],
                    "rel_sup_error": got_value,
                    "rel_l2_error": got[f"channel_{q}_rel_l2"],
                    "predicted_rel_sup_modal_exact": predicted,
                    "predicted_rel_sup_modal_bound": pred["modal_bound"][f"channel_{q}"],
                    "realized_over_predicted": ratio,
                    "taps_per_query": route.taps_per_query,
                    "solves_per_field": route.solves_per_field,
                })
    # The two regimes differ only in the packet, so they share the row schema and
    # are separated by the regime tag carried in the route field.
    new_routes = [r for r in amendment_routes if r.name not in base_rows
                  and r.name not in ORIGINAL_ALIAS]
    scaled_rows = amendment_rows(amendment_routes, scaled_realized, amendment,
                                 SCALED_GRIDS, "resolution_scaled")
    well_rows = amendment_rows(new_routes, well_realized, amendment,
                               GRIDS, "well_resolved")
    with (OUT / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(scaled_rows)
        writer.writerows(well_rows)

    orders = {}
    coarse = {}
    for route in routes:
        orders[route.name] = {
            ("value", "d1", "d2")[q]: fitted_order(
                GRIDS, [realized[(route.name, n)][f"channel_{q}_rel_sup"] for n in GRIDS])
            for q in range(3)}
        coarse[route.name] = {
            ("value", "d1", "d2")[q]: fitted_order(
                GRIDS[:2], [realized[(route.name, n)][f"channel_{q}_rel_sup"]
                            for n in GRIDS[:2]])
            for q in range(3)}

    worst_ratio = 0.0
    worst_row = None
    worst_above = 0.0
    worst_above_row = None
    exceeding = []
    for row in rows:
        ratio = row["realized_over_predicted"]
        if not np.isfinite(ratio) or ratio <= 0.0:
            continue
        deviation = max(ratio, 1.0 / ratio)
        if deviation > worst_ratio:
            worst_ratio = deviation
            worst_row = row
        if row["predicted_rel_sup_modal_exact"] >= PREDICTION_FLOOR:
            if deviation > worst_above:
                worst_above = deviation
                worst_above_row = row
        if deviation > AGREEMENT_FACTOR:
            exceeding.append({k: row[k] for k in ("route", "N", "channel",
                                                  "rel_sup_error",
                                                  "predicted_rel_sup_modal_exact",
                                                  "realized_over_predicted")})
    champion_row = next(r for r in symbol_rows if r["route"] == "champion")
    decision = {
        "producer": "code/analysis/exponential_family/gate5a_offgrid.py",
        "written_utc": now(),
        "realized_started_utc": started,
        "part": "5a",
        "contract": {"path": "results/exponential_family/gate5/contract.md",
                     "sha256": contract_hash},
        "rule": "Section 5a of the contract states quantities, not a decision rule. "
                "The recorded status is the completeness of those quantities and the "
                "predicted against realized agreement the contract asks for.",
        "status": "recorded",
        "quantities_complete": True,
        "prediction_before_runs": True,
        "agreement": {
            "factor": AGREEMENT_FACTOR,
            "worst_realized_over_predicted_deviation": worst_ratio,
            "worst_row": worst_row,
            "all_within_factor": bool(worst_ratio <= AGREEMENT_FACTOR),
            "prediction_floor": PREDICTION_FLOOR,
            "worst_deviation_above_floor": worst_above,
            "worst_row_above_floor": worst_above_row,
            "all_within_factor_above_floor": bool(worst_above <= AGREEMENT_FACTOR),
            "rows_beyond_the_factor": exceeding,
        },
        "symbol_level": symbol_rows,
        "ec_max_B0_ratio_to_champion": {
            row["route"]: row["ec_max_B0"] / champion_row["ec_max_B0"]
            for row in symbol_rows},
        "fitted_orders": orders,
        "fitted_orders_64_to_128": coarse,
        "realized": {f"{name}_N{npts}": value for (name, npts), value in realized.items()},
    }

    scaled_orders = order_block(amendment_routes, scaled_realized, SCALED_GRIDS,
                                triple=SCALED_TRIPLE)
    well_orders = order_block(amendment_routes, well_realized, GRIDS)
    amendment_champion = next(r for r in amend_symbol if r["route"] == "champion")
    decision["amendment_1"] = {
        "written_utc": now(),
        "status": "recorded",
        "scope": amendment["scope"],
        "quantities_complete": True,
        "prediction_before_runs": True,
        "defects_repaired": [
            ("The realized packet of the original part 5a had its top mode at "
             "0.469 pi on N = 64 and at 0.234 pi on N = 128, so the realized "
             "test sat in the low band where formal order dominates rather than "
             "in the design regime. The resolution-scaled packet places the top "
             "mode at the dealiased band edge on every grid, and the polynomial "
             "controls BMC9 and BMC11 are carried at every grid of both "
             "regimes."),
            ("The augmentation used an eight-node Lagrange layer while the wall "
             "experiment of the paper uses the twelve-node evaluator. Widths 8, "
             "12 and 16 are carried at the symbol level and realized."),
        ],
        "lagrange_widths": list(LAGRANGE_WIDTHS),
        "resolution_scaled_packet": amendment["resolution_scaled_packet"],
        "well_resolved_packet": amendment["well_resolved_packet"],
        "global_spectral_interpolant": amendment["global_spectral_interpolant"],
        "symbol_level": amend_symbol,
        "ec_max_B0_ratio_to_champion": {
            row["route"]: row["ec_max_B0"] / amendment_champion["ec_max_B0"]
            for row in amend_symbol},
        "headline_routes": list(HEADLINE_NAMES),
        "faithfulness_convention": (
            "The six headline realized routes take the operator-faithful form "
            "of the augmentation, which is the Milestone 10 and wall-experiment "
            "construction and the stronger of the two variants in the two "
            "derivative channels. The field-faithful variant is carried at "
            "every width as well and is in runs.csv and in the tables."),
        "agreement_resolution_scaled": agreement_summary(scaled_rows),
        "agreement_well_resolved": agreement_summary(well_rows),
        "fitted_orders_resolution_scaled": scaled_orders,
        "fitted_orders_well_resolved": well_orders,
        "ratio_to_champion_resolution_scaled": ratio_block(
            amendment_routes, scaled_realized, SCALED_GRIDS),
        "ratio_to_champion_well_resolved": ratio_block(
            amendment_routes, well_realized, GRIDS),
        "realized_resolution_scaled": {
            f"{name}_N{npts}": value
            for (name, npts), value in scaled_realized.items()},
        "realized_well_resolved": {
            f"{name}_N{npts}": value for (name, npts), value in well_realized.items()},
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    make_figure(amend_curves, well_realized, scaled_realized)
    write_readme(symbol_rows, rows, orders, coarse, decision,
                 amend_symbol, scaled_rows, well_rows, decision["amendment_1"])
    print(f"receipts written under {OUT}")
    return 0


FLOOR_ERROR = 1e-13


def _order_table(lines, block, grids, title, with_triple) -> None:
    pairs = [f"{grids[i]}->{grids[i + 1]}" for i in range(len(grids) - 1)]
    header = "| route | channel | all grids |"
    rule = "| --- | --- | --- |"
    if with_triple:
        header += f" {SCALED_TRIPLE[0]}, {SCALED_TRIPLE[1]}, {SCALED_TRIPLE[2]} |"
        rule += " --- |"
    header += " " + " | ".join(pairs) + " |"
    rule += " " + " | ".join(["---"] * len(pairs)) + " |"
    lines.append(title)
    lines.append("")
    lines.append(header)
    lines.append(rule)
    for name, entry in block.items():
        for channel in CHANNEL_NAMES:
            item = entry[channel]
            cells = f"| {name} | {channel} | {item['all_grids']:.2f} |"
            if with_triple:
                cells += f" {item['triple']:.2f} |"
            cells += " " + " | ".join(f"{v:.2f}" for v in item["pairwise"]) + " |"
            lines.append(cells)
    lines.append("")


def order_readme(amendment) -> list:
    """Return the realized order section, with the pairwise fits and the bias note."""
    lines = ["## Realized order"]
    lines.append("")
    lines.append("The all-grid fit is the log-log slope of the relative sup error against")
    lines.append("N over every grid of the regime; the pairwise fits are the successive")
    lines.append("two-grid slopes log(e_i / e_{i+1}) / log(N_{i+1} / N_i). A pairwise fit")
    lines.append("whose two errors are not both above the double precision floor of the")
    lines.append(f"field evaluation, taken as {FLOOR_ERROR:g}, is not a truncation slope.")
    lines.append("")
    scaled = amendment["fitted_orders_resolution_scaled"]
    well = amendment["fitted_orders_well_resolved"]
    _order_table(lines, scaled, SCALED_GRIDS,
                 "Resolution-scaled packet, every channel:", True)
    _order_table(lines, well, GRIDS, "Well-resolved packet, every channel:", False)

    scaled_err = amendment["realized_resolution_scaled"]
    well_err = amendment["realized_well_resolved"]
    lines.append("Second derivative channel, the channel the coupled contract binds on:")
    lines.append("")
    lines.append("| route | regime | all grids | three-grid fit | pairwise fits | "
                 "pairs with an endpoint at the floor |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    scaled_spread = []
    well_spread = []
    for name in scaled:
        for label, block, grids, errs in (
                ("resolution-scaled", scaled, SCALED_GRIDS, scaled_err),
                ("well-resolved", well, GRIDS, well_err)):
            item = block[name]["d2"]
            dirty = []
            clean = []
            for i, value in enumerate(item["pairwise"]):
                lo = errs[f"{name}_N{grids[i]}"]["channel_2_rel_sup"]
                hi = errs[f"{name}_N{grids[i + 1]}"]["channel_2_rel_sup"]
                (dirty if min(lo, hi) <= FLOOR_ERROR else clean).append(item["pairs"][i])
            values = [v for i, v in enumerate(item["pairwise"])
                      if item["pairs"][i] in clean]
            spread = max(values) - min(values) if len(values) > 1 else float("nan")
            triple = (f"{item['triple']:.2f}" if "triple" in item
                      else f"{item['all_grids']:.2f}")
            pairs = " ".join(f"{v:.2f}" for v in item["pairwise"])
            lines.append(f"| {name} | {label} | {item['all_grids']:.2f} | {triple} | "
                         f"{pairs} | {', '.join(dirty) if dirty else 'none'} |")
            (scaled_spread if label == "resolution-scaled" else well_spread).append(
                (name, spread))
    lines.append("")
    lines.append("The three-grid fit of the resolution-scaled regime is the fit over "
                 f"N = {SCALED_TRIPLE[0]}, {SCALED_TRIPLE[1]} and {SCALED_TRIPLE[2]},")
    lines.append("the doubling triple inside the six grids; in the well-resolved regime")
    lines.append("the three grids are all the grids, so the two columns agree there.")
    lines.append("")
    angles = [entry["theta_top_over_pi"] for entry in amendment
              ["resolution_scaled_packet"]["top_mode"]]
    lines.append("Where the fit is biased.")
    lines.append("")
    lines.append("* Resolution-scaled packet. The points per wavelength of the top mode")
    lines.append("  are held fixed as N grows, so the hardest mode of the packet is never")
    lines.append("  refined and there is no asymptotic order to measure: every all-grid")
    lines.append("  fit in that regime lies between "
                 f"{min(v['d2']['all_grids'] for v in scaled.values()):.2f} and "
                 f"{max(v['d2']['all_grids'] for v in scaled.values()):.2f}, that is,")
    lines.append("  at zero to the scatter. The scatter itself is not pre-asymptotic")
    lines.append("  behaviour either: floor(N/3) puts the top mode between "
                 f"{min(angles):.4f} pi")
    lines.append(f"  and {max(angles):.4f} pi depending on whether 3 divides N, and e_c")
    lines.append("  rises steeply there, so the pairwise slopes alternate in sign with")
    lines.append("  that alternation. A fit of any kind is the")
    lines.append("  wrong summary of this regime; the level of the error at each N is the")
    lines.append("  quantity, and it is in the tables above.")
    lines.append("* Well-resolved packet. Here the fit does measure an order, and the")
    lines.append("  pre-asymptotic range shows up as a spread between the two pairwise")
    lines.append("  fits. The second derivative spread is")
    for name, spread in well_spread:
        if np.isfinite(spread):
            lines.append(f"  {name} {spread:.2f},")
    lines.append("  so the routes with the largest spread are the ones whose three-grid")
    lines.append("  fit is least trustworthy. Two mechanisms are visible. The high order")
    lines.append("  routes reach the double precision floor of the field evaluation")
    lines.append("  inside the window, which flattens the second interval and pulls the")
    lines.append("  three-grid fit off the truncation slope; the cells at the floor are")
    floored = []
    for name in well:
        for q, channel in enumerate(CHANNEL_NAMES):
            for npts in GRIDS:
                value = well_err[f"{name}_N{npts}"][f"channel_{q}_rel_sup"]
                if value <= FLOOR_ERROR:
                    floored.append(f"{name} {channel} N = {npts} at {value:.2e}")
    lines.append("  " + ("; ".join(floored) if floored else "none") + ".")
    lines.append("  The operator-faithful augmentations at L12 and L16 go the other way:")
    lines.append("  once the interpolation layer is strong enough, the two derivative")
    lines.append("  channels are limited by the SLC nodal operators rather than by the")
    lines.append("  layer, so their second derivative fit falls to about 3.8 while their")
    lines.append("  value channel keeps the order of the layer.")
    lines.append("")
    return lines


def agreement_readme(amendment, all_rows) -> list:
    """Return the predicted against realized section of the amendment."""
    lines = ["## Predicted against realized, amendment 1"]
    lines.append("")
    lines.append("The amendment block of `prediction.json` is written before any")
    lines.append("amendment realized run, and carries the exact modal prediction of both")
    lines.append("regimes for all nine routes at every grid.")
    lines.append("")
    lines.append("| regime | worst deviation, all rows | worst deviation above the "
                 f"prediction floor {PREDICTION_FLOOR:g} | rows beyond the factor "
                 f"{AGREEMENT_FACTOR:g} |")
    lines.append("| --- | --- | --- | --- |")
    for label, key in (("resolution-scaled", "agreement_resolution_scaled"),
                       ("well-resolved, new rows", "agreement_well_resolved")):
        block = amendment[key]
        lines.append(f"| {label} | "
                     f"{block['worst_realized_over_predicted_deviation']:.4f} | "
                     f"{block['worst_deviation_above_floor']:.4f} | "
                     f"{len(block['rows_beyond_the_factor'])} |")
    lines.append("")
    beyond = (amendment["agreement_resolution_scaled"]["rows_beyond_the_factor"]
              + amendment["agreement_well_resolved"]["rows_beyond_the_factor"])
    if beyond:
        lines.append("Rows beyond the factor:")
        lines.append("")
        lines.append("| route | channel | N | realized | predicted | ratio |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in beyond:
            lines.append(f"| {row['route']} | {row['channel']} | {row['N']} | "
                         f"{row['rel_sup_error']:.3e} | "
                         f"{row['predicted_rel_sup_modal_exact']:.3e} | "
                         f"{row['realized_over_predicted']:.4f} |")
        lines.append("")
    else:
        lines.append("No amendment row is beyond the factor.")
        lines.append("")
    lines.append("## Convention decisions of amendment 1")
    lines.append("")
    lines.append("* The six headline realized routes take the operator-faithful form of")
    lines.append("  the augmentation, which is the Milestone 10 and wall-experiment")
    lines.append("  construction and the stronger of the two variants in the two")
    lines.append("  derivative channels. The field-faithful variant is carried at every")
    lines.append("  width as well, at the symbol level and realized, and is in")
    lines.append("  `runs.csv`.")
    lines.append("* The wider layers keep the stencil convention of the eight-node layer:")
    lines.append("  the base node is the floor of x/h and the n_L nodes start n_L/2 - 1 to")
    lines.append("  its left, so against the cell coordinate xi in [-1/2, 1/2] the offsets")
    lines.append("  are -(n_L/2 - 1)..+n_L/2 for xi >= 0 and -n_L/2..+n_L/2 - 1 for")
    lines.append("  xi < 0, and the stencil switches at xi = 0. That is the convention of")
    lines.append("  the twelve-point wall evaluator of the certified harness.")
    lines.append("* The cardinal functions are carried in the power basis of the local")
    lines.append("  coordinate. The nodes are small integers and the evaluation point")
    lines.append("  lies in [0, 1), so the value and the first two derivatives agree with")
    lines.append("  exact rational arithmetic to a relative 7e-16 at n_L = 8, 12 and 16.")
    lines.append("* `runs.csv` keeps its original schema and its original rows in their")
    lines.append("  original order. The amendment rows carry the regime in the route")
    lines.append("  field as `<route>|resolution_scaled` and `<route>|well_resolved`.")
    lines.append("  The `|well_resolved` block holds only the four rows that are new, the")
    lines.append("  L12 and L16 augmentations; the other five routes are the original")
    lines.append("  untagged rows.")
    lines.append("* The amplitude law of the resolution-scaled packet is a_k = k^{-1}.")
    lines.append("")
    return lines


def _cells(rows, route, channel, grids, tag, key="rel_sup_error"):
    """Return the per-grid cells of one route and channel from a row block."""
    name = f"{route}|{tag}" if tag else route
    index = {(r["route"], r["N"], r["channel"]): r for r in rows}
    out = []
    for npts in grids:
        row = index.get((name, npts, channel))
        out.append("n/a" if row is None else f"{row[key]:.3e}")
    return out


def amendment_readme(amend_symbol, rows, scaled_rows, well_rows, amendment) -> list:
    """Return the amendment section of README.md."""
    lines = []
    scaled = amendment["resolution_scaled_packet"]
    spectral = amendment["global_spectral_interpolant"]
    all_rows = list(rows) + list(scaled_rows) + list(well_rows)
    alias_rows = {"slc_operator_L8": "slc_operator", "slc_field_L8": "slc_field"}

    lines.append("")
    lines.append("# Amendment 1")
    lines.append("")
    lines.append("Status `recorded`, as in `decision.json` under `amendment_1`. Two")
    lines.append("defects of the tables above are repaired here; nothing above is")
    lines.append("removed or changed, and every number above is reproduced by the same")
    lines.append("code path.")
    lines.append("")
    lines.append("1. The realized packet above has its top mode at k = 15, that is at")
    lines.append("   0.469 pi on N = 64 and 0.234 pi on N = 128, deep in the low band")
    lines.append("   where formal order dominates. The realized test therefore measured")
    lines.append("   formal order rather than the design regime. The resolution-scaled")
    lines.append("   packet below carries every integer mode up to floor(N/3), so its top")
    lines.append("   mode sits at the dealiased band edge on every grid, and the")
    lines.append("   polynomial controls BMC9 and BMC11 are carried at every grid of both")
    lines.append("   regimes.")
    lines.append("2. The augmentation above uses an eight-node Lagrange layer, while the")
    lines.append("   wall experiment of the paper uses the twelve-node evaluator L12. The")
    lines.append("   layer is carried here at widths 8, 12 and 16, so the strongest local")
    lines.append("   evaluator is among the competitors at the symbol level and realized.")
    lines.append("")
    lines.append("`slc_operator_L8` is the route the tables above call `slc_operator`,")
    lines.append("and `slc_field_L8` is the route they call `slc_field`; their symbol")
    lines.append("rows are carried over unchanged. The shared evaluation layer is")
    lines.append("`code/analysis/exponential_family/lagrange_layers.py`.")
    lines.append("")
    lines.append("## Amendment routes")
    lines.append("")
    lines.append("| route | construction | fields | taps per query | cyclic solves per field |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in amend_symbol:
        lines.append(f"| {row['route']} | {row['description']} | {row['fields']} | "
                     f"{row['taps_per_query']} | {row['solves_per_field']} |")
    lines.append("")
    lines.append("## Symbol level with L8, L12 and L16")
    lines.append("")
    lines.append("Same definitions as above: `e_c(theta)` is the worst over the 129 cell")
    lines.append("positions and over the three channels of the relative transfer error,")
    lines.append("the reach is the first contiguous crossing of 1e-2 in units of pi, and")
    lines.append("the coherence residuals are the carrier-kept C_1 and C_2.")
    lines.append("")
    lines.append("| route | taps per query | e_c max B0 | e_c max B1 | e_c max B2 | "
                 "reach 1e-2 / pi | max C_1 on B0 | max C_2 on B0 | "
                 "max C_1 on (0.01 pi, pi] | max C_2 on (0.01 pi, pi] |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in amend_symbol:
        lines.append(f"| {row['route']} | {row['taps_per_query']} | "
                     f"{row['ec_max_B0']:.4e} | {row['ec_max_B1']:.4e} | "
                     f"{row['ec_max_B2']:.4e} | {row['reach_ec_1e-2_over_pi']:.4f} | "
                     f"{row['c1_max_B0']:.3e} | {row['c2_max_B0']:.3e} | "
                     f"{row['c1_max_theta_above_floor']:.3e} | "
                     f"{row['c2_max_theta_above_floor']:.3e} |")
    lines.append("")
    lines.append("Per channel band maxima on B0, and the ratio of `e_c` on B0 to the")
    lines.append("champion:")
    lines.append("")
    lines.append("| route | value | first derivative | literal second derivative | "
                 "e_c max B0 over champion |")
    lines.append("| --- | --- | --- | --- | --- |")
    ratios = amendment["ec_max_B0_ratio_to_champion"]
    for row in amend_symbol:
        lines.append(f"| {row['route']} | {row['e0_max_B0']:.4e} | "
                     f"{row['e1_max_B0']:.4e} | {row['e2_max_B0']:.4e} | "
                     f"{ratios[row['route']]:.4g} |")
    lines.append("")
    lines.append("## Why the global spectral interpolant is not a competitor")
    lines.append("")
    for chunk in spectral["reason"].split(". "):
        text = chunk.strip()
        if not text:
            continue
        lines.append(text if text.endswith(".") else text + ".")
    lines.append("")
    lines.append("Its cost model, for the record:")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(f"| taps per query | {spectral['taps_per_query_law']} |")
    taps = ", ".join(f"N = {n}: {v}" for n, v in spectral["taps_per_query"].items())
    lines.append(f"| taps per query on the grids used here | {taps} |")
    lines.append(f"| setup | {spectral['setup']} |")
    lines.append(f"| per query | {spectral['per_query']} |")
    lines.append(f"| locality | {spectral['locality']} |")
    lines.append("")
    lines.append("The compact routes of this part read between 8 and 48 taps per query")
    lines.append("with a support half width that does not grow with N.")
    lines.append("")
    lines.append("## The resolution-scaled packet")
    lines.append("")
    lines.append(f"    {scaled['law']}")
    lines.append("")
    lines.append(f"Amplitude law {scaled['amplitude_law']}, so the band edge carries")
    lines.append("energy comparable to the design weight rather than the geometric tail")
    lines.append("of the anchor packet; with u = 2 Re sum_k c_k exp(i k x) the amplitude")
    lines.append("is c_k = exp(i varphi_k) / (2 k). The phases are one draw of")
    lines.append(f"floor(max N / 3) values with seed {scaled['phase_seed']}, whose leading")
    lines.append("entries are reused on every grid, so the packet of a coarse grid is the")
    lines.append("low-wavenumber part of the packet of every finer grid. The query")
    lines.append(f"positions are the same {QUERIES} drawn with seed {SEED} that the tables")
    lines.append("above use.")
    lines.append("")
    lines.append("| N | modes | k max | theta of the top mode / pi | points per wavelength at the top mode |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in scaled["top_mode"]:
        lines.append(f"| {entry['N']} | {entry['modes']} | {entry['k_max']} | "
                     f"{entry['theta_top_over_pi']:.4f} | "
                     f"{entry['points_per_wavelength_top']:.3f} |")
    lines.append("")
    lines.append("The top mode sits at 2 pi / 3 exactly when 3 divides N and otherwise")
    lines.append("one mode below it, which is the rounding of floor, and the points per")
    lines.append("wavelength of the band edge are fixed as N grows.")
    lines.append("")
    lines.append("## Realized, resolution-scaled packet")
    lines.append("")
    lines.append("The six headline routes. The augmentation is taken in its")
    lines.append("operator-faithful form, which is the Milestone 10 and wall-experiment")
    lines.append("construction and the stronger of the two variants in the two derivative")
    lines.append("channels; the field-faithful variant follows in its own table.")
    lines.append("")
    head = " | ".join(f"N={n}" for n in SCALED_GRIDS)
    lines.append(f"| route | channel | {head} |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(SCALED_GRIDS)) + " |")
    for name in HEADLINE_NAMES:
        for channel in CHANNEL_NAMES:
            cells = " | ".join(_cells(scaled_rows, name, channel, SCALED_GRIDS,
                                      "resolution_scaled"))
            lines.append(f"| {name} | {channel} | {cells} |")
    lines.append("")
    lines.append("The field-faithful augmentation at the same widths:")
    lines.append("")
    lines.append(f"| route | channel | {head} |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(SCALED_GRIDS)) + " |")
    for width in LAGRANGE_WIDTHS:
        name = f"slc_field_L{width}"
        for channel in CHANNEL_NAMES:
            cells = " | ".join(_cells(scaled_rows, name, channel, SCALED_GRIDS,
                                      "resolution_scaled"))
            lines.append(f"| {name} | {channel} | {cells} |")
    lines.append("")
    lines.append("Ratio to the champion, resolution-scaled packet, per channel:")
    lines.append("")
    lines.append(f"| route | channel | {head} |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(SCALED_GRIDS)) + " |")
    ratio_scaled = amendment["ratio_to_champion_resolution_scaled"]
    for name in ratio_scaled:
        for channel in CHANNEL_NAMES:
            cells = " | ".join(f"{ratio_scaled[name][channel][str(n)]:.4g}"
                               for n in SCALED_GRIDS)
            lines.append(f"| {name} | {channel} | {cells} |")
    lines.append("")
    lines.append("## Realized, well-resolved packet (top mode 0.23 pi at N = 128)")
    lines.append("")
    lines.append("The Gate 0 anchor packet of the tables above, so that both regimes are")
    lines.append("visible. The rows of champion, BMC9, BMC11, `slc_operator_L8` and")
    lines.append("`slc_field_L8` are the rows of the realized table above; the four")
    lines.append("L12 and L16 rows are new and are in `runs.csv` under the")
    lines.append("`|well_resolved` tag.")
    lines.append("")
    whead = " | ".join(f"N={n}" for n in GRIDS)
    lines.append(f"| route | channel | {whead} |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(GRIDS)) + " |")
    for row in amend_symbol:
        name = row["route"]
        for channel in CHANNEL_NAMES:
            if name in alias_rows:
                cells = " | ".join(_cells(rows, alias_rows[name], channel, GRIDS, None))
            elif row["original_5a_route"]:
                cells = " | ".join(_cells(rows, name, channel, GRIDS, None))
            else:
                cells = " | ".join(_cells(well_rows, name, channel, GRIDS,
                                          "well_resolved"))
            lines.append(f"| {name} | {channel} | {cells} |")
    lines.append("")
    lines.append("Ratio to the champion, well-resolved packet, per channel:")
    lines.append("")
    lines.append(f"| route | channel | {whead} |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(GRIDS)) + " |")
    ratio_well = amendment["ratio_to_champion_well_resolved"]
    for name in ratio_well:
        for channel in CHANNEL_NAMES:
            cells = " | ".join(f"{ratio_well[name][channel][str(n)]:.4g}" for n in GRIDS)
            lines.append(f"| {name} | {channel} | {cells} |")
    lines.append("")
    lines.extend(order_readme(amendment))
    lines.extend(agreement_readme(amendment, all_rows))
    return lines

def write_readme(symbol_rows, rows, orders, coarse, decision,
                 amend_symbol, scaled_rows, well_rows, amendment) -> None:
    lines = []
    lines.append("# Gate 5 part 5a: coupled off-grid capability")
    lines.append("")
    lines.append("Producer `code/analysis/exponential_family/gate5a_offgrid.py`, which")
    lines.append("writes `prediction.json` before any realized run, then `runs.csv`,")
    lines.append("`decision.json`, this file and `figures/`.")
    lines.append("")
    lines.append("## Routes")
    lines.append("")
    lines.append("| route | construction | fields | taps per query | cyclic solves per field |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in symbol_rows:
        lines.append(f"| {row['route']} | {row['description']} | {row['fields']} | "
                     f"{row['taps_per_query']} | {row['solves_per_field']} |")
    lines.append("")
    lines.append("`taps per query` counts the nodal or coefficient reads needed to")
    lines.append("deliver all three channels at one query point. A one-field route reads")
    lines.append("the same support once; the operator-faithful augmentation reads eight")
    lines.append("nodes of each of three nodal arrays. `cyclic solves per field` counts")
    lines.append("the banded solves the route needs before any query can be answered.")
    lines.append("")
    lines.append("## Symbol level")
    lines.append("")
    lines.append("`e_c(theta)` is the worst over the 129 cell positions and over the three")
    lines.append("channels of the relative transfer error. The reach is the first")
    lines.append("contiguous crossing of 1e-2, in units of pi.")
    lines.append("")
    lines.append("| route | e_c max B0 | e_c max B1 | e_c max B2 | reach 1e-2 / pi | "
                 "max C_1 on B0 | max C_2 on B0 | max C_1 on (0.01 pi, pi] | "
                 "max C_2 on (0.01 pi, pi] |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in symbol_rows:
        lines.append(f"| {row['route']} | {row['ec_max_B0']:.4e} | {row['ec_max_B1']:.4e} | "
                     f"{row['ec_max_B2']:.4e} | {row['reach_ec_1e-2_over_pi']:.4f} | "
                     f"{row['c1_max_B0']:.3e} | {row['c2_max_B0']:.3e} | "
                     f"{row['c1_max_theta_above_floor']:.3e} | "
                     f"{row['c2_max_theta_above_floor']:.3e} |")
    lines.append("")
    lines.append("Per channel band maxima on B0:")
    lines.append("")
    lines.append("| route | value | first derivative | literal second derivative |")
    lines.append("| --- | --- | --- | --- |")
    for row in symbol_rows:
        lines.append(f"| {row['route']} | {row['e0_max_B0']:.4e} | {row['e1_max_B0']:.4e} | "
                     f"{row['e2_max_B0']:.4e} |")
    lines.append("")
    lines.append("The coherence residuals of the three one-field routes and of the")
    lines.append("field-faithful augmentation are the floating point floor of an identity")
    lines.append("that holds exactly, amplified by 1/theta and 1/theta^2 at the bottom of")
    lines.append("the angle grid. Only the operator-faithful augmentation carries a")
    lines.append("residual that is a property of the scheme.")
    lines.append("")
    lines.append("## Realized")
    lines.append("")
    lines.append("The Gate 0 anchor packet on N in {64, 128, 256}, 4,096 query positions")
    lines.append(f"drawn once with seed {SEED} and shared by every grid and route, errors")
    lines.append("relative to the sup of the analytic channel on the same points.")
    lines.append("")
    lines.append("| route | channel | N=64 | N=128 | N=256 | fitted order, three grids | "
                 "fitted order, 64 to 128 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for route_name in [r["route"] for r in symbol_rows]:
        for channel in ("value", "d1", "d2"):
            picked = [r for r in rows if r["route"] == route_name and r["channel"] == channel]
            picked.sort(key=lambda r: r["N"])
            cells = " | ".join(f"{r['rel_sup_error']:.3e}" for r in picked)
            lines.append(f"| {route_name} | {channel} | {cells} | "
                         f"{orders[route_name][channel]:.2f} | "
                         f"{coarse[route_name][channel]:.2f} |")
    lines.append("")
    lines.append("The three-grid fit of the two polynomial routes is pulled up by the")
    lines.append("N = 256 column, where their errors sit at the double precision floor of")
    lines.append("the field evaluation; the 64 to 128 fit is the one free of that floor.")
    lines.append("")
    lines.append("## Predicted against realized")
    lines.append("")
    lines.append("The prediction is the exact modal sum built from the transfers alone,")
    lines.append("evaluated at the same query points, written before the realized runs.")
    lines.append("The worst deviation of the ratio realized over predicted from one is")
    lines.append(f"{decision['agreement']['worst_realized_over_predicted_deviation']:.4f} "
                 f"over all rows and "
                 f"{decision['agreement']['worst_deviation_above_floor']:.4f} over the rows "
                 f"whose predicted error is at least {PREDICTION_FLOOR:g}, against the "
                 f"recorded factor {AGREEMENT_FACTOR:g}.")
    beyond = decision["agreement"]["rows_beyond_the_factor"]
    if beyond:
        lines.append("")
        lines.append("Rows beyond the factor, all of them at the double precision floor")
        lines.append("of both numbers:")
        lines.append("")
        lines.append("| route | channel | N | realized | predicted | ratio |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in beyond:
            lines.append(f"| {row['route']} | {row['channel']} | {row['N']} | "
                         f"{row['rel_sup_error']:.3e} | "
                         f"{row['predicted_rel_sup_modal_exact']:.3e} | "
                         f"{row['realized_over_predicted']:.4f} |")
    lines.append("")
    lines.append("Every row:")
    lines.append("")
    lines.append("| route | channel | N | realized | predicted | ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(f"| {row['route']} | {row['channel']} | {row['N']} | "
                     f"{row['rel_sup_error']:.3e} | "
                     f"{row['predicted_rel_sup_modal_exact']:.3e} | "
                     f"{row['realized_over_predicted']:.4f} |")
    lines.append("")
    lines.append("## Conventions resolved here")
    lines.append("")
    lines.append("* Lagrange stencil centering. The eight nodes are the ones floor")
    lines.append("  selects, so against the cell coordinate xi in [-1/2, 1/2] measured")
    lines.append("  from the nearest node the offsets are -3..+4 for xi >= 0 and -4..+3")
    lines.append("  for xi < 0. The stencil switches at xi = 0.")
    lines.append("* Transfer assembly. With the carrier exp(i theta xi) divided out, the")
    lines.append("  A6 identity reads T_1 = T_0 + (1/(i theta)) dT_0/dxi, so the coherence")
    lines.append("  residuals are written with the carrier kept, which is the form that is")
    lines.append("  identically zero for a one-field route.")
    lines.append("* The SLC circulants are applied through their symbols on the periodic")
    lines.append("  grid, which is the same operator as the banded solve.")
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append("`figures/fig_coupled_offgrid.pdf`, described in `figures/captions.md`.")
    lines.extend(amendment_readme(amend_symbol, rows, scaled_rows, well_rows, amendment))
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    caption = [
        "# Captions",
        "",
        "## fig_coupled_offgrid.pdf",
        "",
        "Top panel: the coupled off-grid error e_c(theta) of the five routes of",
        "part 5a together with the two twelve-node augmentations of amendment 1, on",
        "(0, pi], logarithmic in the error, with the dealiased band edge B0 = 2 pi / 3",
        "marked. e_c is the worst over 129 cell positions xi in [-1/2, 1/2] and over the",
        "three channels of |T_q - 1|, the transfers of the Gate 0 contract. The three",
        "one-field spline routes read value, first derivative and literal second",
        "derivative from one coefficient vector; the two augmentations bolt a degree",
        "seven Lagrange interpolant on the nearest eight nodes onto SLC 1992, either",
        "interpolating the two nodal derivative fields (operator faithful) or",
        "differentiating the value interpolant (field faithful).",
        "",
        "The L12 curves of amendment 1 are dashed: the twelve-node layer is the wall",
        "evaluator of the paper, and it is the strongest local evaluator carried at",
        "the symbol level together with L8 and L16.",
        "",
        "Middle row: the realized relative sup errors of the three channels against the",
        "analytic Gate 0 anchor packet, at 4,096 random query positions on N = 64, 128",
        "and 256, for the five routes of part 5a together with the operator-faithful",
        "augmentation at widths 12 and 16. That packet has its top mode at k = 15, that",
        "is at 0.234 pi on N = 128, so this row is the well-resolved regime in which",
        "formal order dominates.",
        "",
        "Bottom row: the same three channels on the resolution-scaled packet of",
        "amendment 1, u(x) = sum_{k=1}^{floor(N/3)} k^{-1} cos(k x + varphi_k) with",
        "fixed phases, on N = 48, 64, 96, 128, 192 and 256. Every grid carries its own",
        "packet whose top mode sits at the dealiased band edge 2 pi / 3, so the points",
        "per wavelength at the band edge are fixed as N grows and the row is the design",
        "regime. The six routes are the headline routes of amendment 1, with the",
        "augmentation in its operator-faithful form at widths 8, 12 and 16.",
        "",
        "All three rows are produced by",
        "code/analysis/exponential_family/gate5a_offgrid.py; the numbers are in",
        "runs.csv and decision.json.",
    ]
    (FIGURES / "captions.md").write_text("\n".join(caption) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
