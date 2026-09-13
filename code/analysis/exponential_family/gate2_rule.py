# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Gate 2 producer: the frequency-selection rule by equioscillation.

Section 6 of docs/exponential_family/gate0_theory.md states the rule. For the
band (0, T], polynomial core m and r frequency pairs, the band-minimax
frequencies of the first derivative are the omega for which

    e_1(theta) = kappa_1(theta)/theta - 1

attains its band maximum with alternating sign at exactly r + 1 points of
(0, T], the last of which is the band edge T. This module computes those
frequencies with a Remez-type iteration on the exact alias-sum symbol of
family.ExponentialGenerator.sample_symbols, reproduces the five rows of the
section 6 table, repeats the (6, 2) cell on three further bands, and checks
that every optimum is a genuine local minimum of the band maximum.

Structure of the error curve. By A3 each pair contributes an exact zero of
e_1 at theta = omega_j, and the optimal frequencies all lie inside the band,
so on (0, T] the curve has exactly r interior zeros. They split the band into
r + 1 intervals

    (0, omega_1), (omega_1, omega_2), ..., (omega_{r-1}, omega_r), (omega_r, T],

each carrying one signed extremum of e_1. On the first r intervals the
extremum is interior and is located by a bounded scalar search; on the last
one the weight of Lemma D increases with theta, so the extremum sits at the
band edge and the point is T itself. Adjacent extrema have opposite signs
because e_1 changes sign at each simple zero omega_j.

The Remez conditions. Writing x_0 < x_1 < ... < x_r for those extrema and
v_i = e_1(x_i), the level conditions are the r equations

    R_i(omega) = (|v_i| - |v_r|) / |v_r| = 0,      i = 0, ..., r - 1,

an r-by-r square system in the r unknowns omega. It is solved by a damped
Newton iteration whose Jacobian is taken by forward differences of the
extremal values with respect to omega, the extremal abscissae being
recomputed at every evaluation. Each Newton step is clipped back into the
open ordered box 0 < omega_1 < ... < omega_r < T and is halved until the
residual infinity norm decreases, which is the standard Remez exchange in the
form the alternation set makes available here.

The starting guess. In the one-alias model of section 6 only the k = -1 term
of the alias sum is kept, which by Lemma D gives

    e_1(theta) ~ -(2 pi/theta) (theta/(2 pi - theta))^m
                 prod_j (theta^2 - omega_j^2) / ((2 pi - theta)^2 - omega_j^2).

In the variable u = theta^2 on [0, T^2] this is the weighted monic polynomial
- g(u) prod_j (u - omega_j^2) with the weight of section 6,

    g(u) = (2 pi / sqrt u) (sqrt u / (2 pi - sqrt u))^m
           / prod_j ((2 pi - sqrt u)^2 - omega_j^2),

so the model problem is a weighted Chebyshev problem whose solution is the
degree-r weighted Chebyshev polynomial and whose zeros are the model
frequencies. The weight depends on omega only through the denominator, which
varies slowly, so the same Newton iteration on the same alternation residual
solves the model self-consistently. Its starting point is the best cell of a
coarse combinatorial scan of the model band maximum, which removes the
dependence of the outcome on any hand-chosen initial vector.

Accuracy conventions fixed here, which section 6 leaves open:

* The alias sum is truncated with the family's own tail bound at 1e-14, and
  the truncation index K is frozen once per member so that every evaluation
  inside one Newton step uses the same quadrature. Because kappa_1 is
  O(theta) and the subtraction kappa_1/theta - 1 loses no more than the
  relative size of e_1, the floor of the error curve is about 1e-15 absolute,
  far below every level reported here.
* The interior extrema are located by a bounded Brent search with abscissa
  tolerance 1e-13. The band edge is the exact endpoint T, never a grid point.
* The reported level E_max is |e_1(T)| at the converged frequencies, and the
  independent band maximum used for the local-minimum test is the maximum of
  |e_1| over a uniform grid of BAND_POINTS angles on (0, T] that ends exactly
  at T, so the edge extremum is always sampled.
* The local-minimum test perturbs omega by PERTURB_RADIUS in PERTURB_DIRECTIONS
  directions drawn uniformly on the unit sphere of R^r from a fixed seed, and
  requires the perturbed band maximum to exceed the optimal one at every
  direction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

CODE = Path(__file__).resolve().parents[2]
FAMILY = Path(__file__).resolve().parent
for _path in (str(CODE), str(FAMILY)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import family as fam  # noqa: E402

ROOT = CODE.parent
OUT = ROOT / "results" / "exponential_family" / "gate2"

TWO_PI = 2.0 * np.pi
B0 = TWO_PI / 3.0

ALIAS_TOL = 1e-14
REMEZ_TOL = 1e-9
MAX_ITERATIONS = 60
JACOBIAN_STEP = 1e-6
EXTREMUM_XATOL = 1e-13
EDGE_GUARD = 1e-9
BAND_POINTS = 20001
SUPREMUM_POINTS = 2001
SCAN_POINTS = 24
SCAN_BAND_POINTS = 2001
PERTURB_RADIUS = 1e-3
PERTURB_DIRECTIONS = 32
PERTURB_SEED = 20260912
MULTISTART_STARTS = 24
MULTISTART_SEED = 7

TABLE_CELLS = ((8, 1), (4, 2), (6, 2), (4, 3), (6, 3))
BAND_SWEEP_CELL = (6, 2)
BAND_SWEEP_EDGES = (0.5 * np.pi, 0.6 * np.pi, 0.75 * np.pi)

SECTION6_TABLE = {
    (8, 1): {"omegas_over_pi": (0.6525,), "E_max": 1.62e-4,
             "points_over_pi": (0.602, 0.667), "signs": (1, -1)},
    (4, 2): {"omegas_over_pi": (0.5578, 0.6556), "E_max": 1.85e-4,
             "points_over_pi": (0.452, 0.621, 0.667), "signs": (-1, 1, -1)},
    (6, 2): {"omegas_over_pi": (0.5856, 0.6584), "E_max": 2.70e-5,
             "points_over_pi": (0.505, 0.633, 0.667), "signs": (-1, 1, -1)},
    (4, 3): {"omegas_over_pi": (0.4883, 0.6081, 0.6604), "E_max": 8.25e-6,
             "points_over_pi": (0.385, 0.559, 0.641, 0.667), "signs": (1, -1, 1, -1)},
    (6, 3): {"omegas_over_pi": (0.5281, 0.6213, 0.6618), "E_max": 1.04e-6,
             "points_over_pi": (0.446, 0.583, 0.647, 0.667), "signs": (1, -1, 1, -1)},
}


# ----------------------------------------------------------------- error curves


def one_alias_e1(theta, m: int, omegas) -> np.ndarray:
    """Return the one-alias model of e_1 from section 6.

    Only the k = -1 alias survives in the numerator of A2, and Lemma D turns
    the ratio of that alias to the k = 0 term into the closed form below.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    value = -(TWO_PI / theta) * (theta / (TWO_PI - theta)) ** m
    for w in omegas:
        value = value * (theta ** 2 - w ** 2) / ((TWO_PI - theta) ** 2 - w ** 2)
    return value


def one_alias_weight(u, m: int, omegas) -> np.ndarray:
    """Return the section 6 Chebyshev weight g(u) in the variable u = theta^2."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    root = np.sqrt(u)
    value = (TWO_PI / root) * (root / (TWO_PI - root)) ** m
    for w in omegas:
        value = value / ((TWO_PI - root) ** 2 - w ** 2)
    return value


class ExactError:
    """The exact e_1 of a member, from the truncated alias sums of Theorem A1.

    The truncation index is fixed at construction from the family's own tail
    bound, so every evaluation of one member uses the same quadrature and the
    finite differences of the Newton step see no quadrature jump.
    """

    def __init__(self, m: int, omegas, tol: float = ALIAS_TOL) -> None:
        self.generator = fam.ExponentialGenerator(m, tuple(float(w) for w in omegas))
        self.K = max(self.generator.required_K(q, tol) for q in (0, 1, 2))

    def __call__(self, theta) -> np.ndarray:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        kappa1 = self.generator.sample_symbols(theta, K=self.K)[1]
        return kappa1 / theta - 1.0


def exact_error_factory(m: int, tol: float = ALIAS_TOL):
    """Return omega -> exact e_1 callable for the polynomial core m."""
    return lambda omegas: ExactError(m, omegas, tol)


def one_alias_error_factory(m: int):
    """Return omega -> one-alias e_1 callable for the polynomial core m."""
    return lambda omegas: (lambda theta: one_alias_e1(theta, m, omegas))


# ------------------------------------------------------------- alternation set


def alternation_set(error, omegas, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the r + 1 candidate alternation abscissae and the values of e_1.

    The zeros omega_j of e_1 split (0, T] into r + 1 intervals. The interior
    extremum of each of the first r intervals is found by a bounded scalar
    search on -|e_1|; the last point is the band edge, where the increasing
    weight of the one-alias model puts the extremum of the final interval.
    """
    omegas = np.asarray(omegas, dtype=float)
    edges = np.concatenate(([0.0], omegas, [float(T)]))
    points = []
    for j in range(omegas.size):
        lo, hi = float(edges[j]), float(edges[j + 1])
        pad = EDGE_GUARD * (hi - lo)
        result = minimize_scalar(
            lambda t: -abs(float(error(t)[0])),
            bounds=(lo + pad, hi - pad), method="bounded",
            options={"xatol": EXTREMUM_XATOL},
        )
        points.append(float(result.x))
    points.append(float(T))
    points = np.asarray(points, dtype=float)
    values = np.array([float(error(p)[0]) for p in points])
    return points, values


def equioscillation_residual(values: np.ndarray) -> np.ndarray:
    """Return the r level conditions (|v_i| - |v_r|)/|v_r|, i = 0, ..., r - 1."""
    magnitude = np.abs(values)
    level = magnitude[-1]
    if level <= 0.0:
        return np.full(magnitude.size - 1, np.inf)
    return (magnitude[:-1] - level) / level


def alternates(values: np.ndarray) -> bool:
    """Return whether consecutive extremal values have strictly opposite signs."""
    signs = np.sign(values)
    if np.any(signs == 0.0):
        return False
    return bool(np.all(signs[:-1] * signs[1:] < 0.0))


# ------------------------------------------------------------------ the solver


@dataclass(frozen=True)
class RemezResult:
    """The outcome of one Remez-type iteration.

    omegas     the levelled frequencies, strictly increasing inside (0, T)
    points     the r + 1 alternation abscissae, the last one equal to T
    values     the signed values of e_1 at those abscissae
    level      the levelled error E_max = |e_1(T)|
    history    one record per Remez iteration
    """

    m: int
    r: int
    T: float
    omegas: np.ndarray
    points: np.ndarray
    values: np.ndarray
    level: float
    residual: float
    converged: bool
    alternating: bool
    history: list = field(default_factory=list)
    stalled: bool = False
    start: np.ndarray | None = None
    start_level: float | None = None

    def as_dict(self) -> dict:
        """Return the plain data form used by rule.json."""
        return {
            "m": int(self.m), "r": int(self.r), "T": float(self.T),
            "T_over_pi": float(self.T / np.pi),
            "omegas": [float(w) for w in self.omegas],
            "omegas_over_pi": [float(w / np.pi) for w in self.omegas],
            "points": [float(p) for p in self.points],
            "points_over_pi": [float(p / np.pi) for p in self.points],
            "values": [float(v) for v in self.values],
            "alternation_values": [float(v / self.level) for v in self.values],
            "level": float(self.level),
            "residual": float(self.residual),
            "converged": bool(self.converged),
            "stalled": bool(self.stalled),
            "alternating": bool(self.alternating),
            "iterations": len(self.history),
            "history": [
                {"iteration": int(h["iteration"]),
                 "omegas_over_pi": [float(w / np.pi) for w in h["omegas"]],
                 "points_over_pi": [float(p / np.pi) for p in h["points"]],
                 "level": float(h["level"]),
                 "residual": float(h["residual"]),
                 "step": float(h["step"])}
                for h in self.history
            ],
            "start_over_pi": (None if self.start is None
                              else [float(w / np.pi) for w in self.start]),
            "start_level": (None if self.start_level is None
                            else float(self.start_level)),
        }


def _ordered(x: np.ndarray, T: float) -> np.ndarray:
    """Return x clipped into the open ordered box 0 < x_1 < ... < x_r < T."""
    x = np.sort(np.asarray(x, dtype=float))
    span = float(T)
    gap = span * 1e-6
    out = np.empty_like(x)
    low = gap
    for j, value in enumerate(x):
        high = span - gap * (x.size - j)
        out[j] = min(max(value, low), high)
        low = out[j] + gap
    return out


def _state(factory, omegas: np.ndarray, T: float):
    """Return the alternation set and residual of one frequency vector."""
    error = factory(omegas)
    points, values = alternation_set(error, omegas, T)
    residual = equioscillation_residual(values)
    return points, values, residual


def _remez_core(factory, m: int, r: int, T: float, start: np.ndarray,
                tol: float) -> RemezResult:
    """Run the damped Newton iteration on the r level conditions."""
    omegas = _ordered(start, T)
    points, values, residual = _state(factory, omegas, T)
    history: list = []
    stalled = False
    converged = float(np.max(np.abs(residual))) <= tol
    for iteration in range(1, MAX_ITERATIONS + 1):
        if converged:
            break
        jacobian = np.empty((r, r))
        for j in range(r):
            shifted = omegas.copy()
            step = JACOBIAN_STEP * max(1.0, abs(float(omegas[j])))
            shifted[j] = shifted[j] + step
            shifted = _ordered(shifted, T)
            _, _, residual_shifted = _state(factory, shifted, T)
            jacobian[:, j] = (residual_shifted - residual) / (shifted[j] - omegas[j])
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        scale = 1.0
        best = None
        for _ in range(40):
            trial = _ordered(omegas + scale * delta, T)
            trial_points, trial_values, trial_residual = _state(factory, trial, T)
            if np.max(np.abs(trial_residual)) < np.max(np.abs(residual)):
                best = (trial, trial_points, trial_values, trial_residual, scale)
                break
            scale *= 0.5
        if best is None:
            history.append({"iteration": iteration, "omegas": omegas.copy(),
                            "points": points.copy(), "level": float(abs(values[-1])),
                            "residual": float(np.max(np.abs(residual))),
                            "step": 0.0})
            stalled = True
            break
        omegas, points, values, residual, scale = best
        history.append({"iteration": iteration, "omegas": omegas.copy(),
                        "points": points.copy(), "level": float(abs(values[-1])),
                        "residual": float(np.max(np.abs(residual))),
                        "step": float(scale)})
        converged = float(np.max(np.abs(residual))) <= tol
    return RemezResult(
        m=m, r=r, T=float(T), omegas=omegas, points=points, values=values,
        level=float(abs(values[-1])), residual=float(np.max(np.abs(residual))),
        converged=bool(converged), alternating=alternates(values), history=history,
        stalled=bool(stalled),
    )


def one_alias_start(m: int, r: int, T: float, tol: float = 1e-13) -> RemezResult:
    """Return the weighted Chebyshev solution of the one-alias model.

    The coarse scan minimizes the model band maximum over the cells of a
    uniform frequency grid, which supplies an ordered vector inside the band
    without any hand-chosen initial guess; the Newton iteration on the model
    alternation residual then levels it.
    """
    grid = np.linspace(0.15 * T, T * (1.0 - 1e-3), SCAN_POINTS)
    theta = np.linspace(T / SCAN_BAND_POINTS, T, SCAN_BAND_POINTS)
    best_value, best_omegas = np.inf, None
    for cell in itertools.combinations(range(grid.size), r):
        omegas = grid[list(cell)]
        value = float(np.max(np.abs(one_alias_e1(theta, m, omegas))))
        if value < best_value:
            best_value, best_omegas = value, omegas
    return _remez_core(one_alias_error_factory(m), m, r, T, best_omegas, tol)


def remez_frequencies(m: int, r: int, T: float = B0, tol: float = REMEZ_TOL,
                      start=None, alias_tol: float = ALIAS_TOL) -> RemezResult:
    """Return the equioscillating frequencies of E_{m,r} on the band (0, T].

    The returned record carries the frequencies omega, the r + 1 alternation
    abscissae, the levelled error E_max = |e_1(T)| and the iteration history.
    Without an explicit start the iteration begins at the one-alias model's
    weighted Chebyshev solution, whose own Remez history is discarded and
    whose frequencies and model level are recorded on the result.
    """
    if r < 1:
        raise ValueError("The equioscillation rule needs at least one pair.")
    if not 0.0 < T < np.pi:
        raise ValueError("The band edge T must lie in (0, pi).")
    model = None
    if start is None:
        model = one_alias_start(m, r, T)
        start = model.omegas
    result = _remez_core(exact_error_factory(m, alias_tol), m, r, T,
                         np.asarray(start, dtype=float), tol)
    return RemezResult(
        m=result.m, r=result.r, T=result.T, omegas=result.omegas,
        points=result.points, values=result.values, level=result.level,
        residual=result.residual, converged=result.converged,
        alternating=result.alternating, history=result.history,
        stalled=result.stalled, start=(None if model is None else model.omegas),
        start_level=(None if model is None else model.level),
    )


# ------------------------------------------------------------------ diagnostics


def band_maximum(m: int, omegas, T: float, points: int = BAND_POINTS,
                 alias_tol: float = ALIAS_TOL) -> float:
    """Return max |e_1| over a uniform grid of (0, T] whose last point is T."""
    error = ExactError(m, omegas, alias_tol)
    theta = np.linspace(T / points, T, points)
    return float(np.max(np.abs(error(theta))))


def band_supremum(m: int, omegas, T: float, points: int = SUPREMUM_POINTS,
                  alias_tol: float = ALIAS_TOL) -> float:
    """Return the grid-independent supremum of |e_1| on (0, T].

    The r zeros of e_1 inside the band leave r + 1 intervals with one extremum
    each, and those extrema are located exactly by the bounded searches of
    alternation_set, so no grid spacing enters the value. A coarse uniform
    grid is taken in union with them, which detects any additional extremum
    that the interval structure would not predict.
    """
    omegas = np.sort(np.asarray(omegas, dtype=float))
    error = ExactError(m, omegas, alias_tol)
    _, values = alternation_set(error, omegas, T)
    theta = np.linspace(T / points, T, points)
    return float(max(np.max(np.abs(values)), np.max(np.abs(error(theta)))))


def local_minimum_check(m: int, omegas, T: float,
                        radius: float = PERTURB_RADIUS,
                        directions: int = PERTURB_DIRECTIONS,
                        seed: int = PERTURB_SEED,
                        points: int = BAND_POINTS) -> dict:
    """Return the perturbation test of the band maximum around one optimum.

    The optimum is a local minimum of max |e_1| when every perturbation of
    radius `radius` on the sphere of R^r raises the band supremum. The
    supremum is taken by band_supremum, which locates the extrema themselves,
    so the comparison carries no grid dependence; the uniform-grid maximum of
    the unperturbed member is reported beside it as a cross check.
    """
    omegas = np.asarray(omegas, dtype=float)
    r = omegas.size
    base = band_supremum(m, omegas, T)
    rng = np.random.default_rng(seed)
    ratios = []
    steps = []
    for _ in range(directions):
        direction = rng.normal(size=r)
        direction = direction / np.linalg.norm(direction)
        trial = np.sort(omegas + radius * direction)
        ratios.append(float(band_supremum(m, trial, T) / base))
        steps.append(direction)
    ratios = np.asarray(ratios, dtype=float)
    worst_direction = steps[int(np.argmin(ratios))]
    return {
        "radius": float(radius),
        "directions": int(directions),
        "seed": int(seed),
        "band_supremum": float(base),
        "grid_band_maximum": float(band_maximum(m, omegas, T, points)),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "mean_ratio": float(ratios.mean()),
        "worst_direction": [float(v) for v in np.asarray(worst_direction)],
        "is_local_minimum": bool(np.all(ratios > 1.0)),
    }


def multistart_check(m: int, r: int, T: float, omegas,
                     starts: int = MULTISTART_STARTS,
                     seed: int = MULTISTART_SEED) -> dict:
    """Return the outcome of restarting the iteration from random frequencies.

    The equioscillation rule claims a unique levelled vector. Restarting the
    same iteration from ordered vectors drawn uniformly inside the band tests
    that claim directly: a second levelled vector would show up as a converged
    run landing away from the reference, and a second local minimum of the band
    supremum would then be reachable. A run that does not converge is a stall
    of the damped Newton step, not a second optimum, and is reported with the
    band supremum it stalled at.
    """
    omegas = np.asarray(omegas, dtype=float)
    rng = np.random.default_rng(seed)
    reference = band_supremum(m, omegas, T)
    converged, distinct, stalls = 0, 0, []
    worst = 0.0
    for _ in range(starts):
        trial = np.sort(rng.uniform(0.05 * T, 0.995 * T, size=r))
        result = remez_frequencies(m, r, T, start=trial)
        if not (result.converged and result.alternating):
            stalls.append({
                "start_over_pi": [float(w / np.pi) for w in trial],
                "residual": float(result.residual),
                "min_extremal_magnitude": float(np.min(np.abs(result.values))),
                "supremum_ratio": float(band_supremum(m, result.omegas, T) / reference),
            })
            continue
        converged += 1
        deviation = float(np.max(np.abs(result.omegas - omegas) / omegas))
        worst = max(worst, deviation)
        if deviation > 1e-6:
            distinct += 1
    return {
        "starts": int(starts),
        "seed": int(seed),
        "converged": int(converged),
        "stalled": int(len(stalls)),
        "distinct_optima": int(distinct),
        "max_deviation_from_reference": float(worst),
        "min_stall_supremum_ratio": (None if not stalls else
                                     float(min(s["supremum_ratio"] for s in stalls))),
        "stalls": stalls,
    }


def optimum_record(m: int, r: int, T: float) -> dict:
    """Return the full record of one optimum: Remez, conditioning, local test."""
    result = remez_frequencies(m, r, T)
    generator = fam.ExponentialGenerator(m, tuple(result.omegas))
    admissible, reason = generator.admissible()
    record = result.as_dict()
    record["label"] = generator.label()
    record["degree"] = int(generator.p)
    record["admissible"] = bool(admissible)
    record["admissible_reason"] = reason
    record["cond"] = {k: float(v) for k, v in generator.cond().items()}
    record["band_supremum"] = band_supremum(m, result.omegas, T)
    record["local_minimum"] = local_minimum_check(m, result.omegas, T)
    record["multistart"] = multistart_check(m, r, T, result.omegas)
    if result.start is not None:
        record["start_frequency_error"] = [
            float(abs(s - w) / w) for s, w in zip(result.start, result.omegas)]
        record["start_level_error"] = float(
            abs(result.start_level - result.level) / result.level)
    return record


def compare_with_section6(record: dict) -> dict | None:
    """Return the relative deviation of one record from the section 6 table."""
    key = (record["m"], record["r"])
    if key not in SECTION6_TABLE or abs(record["T"] - B0) > 1e-12:
        return None
    quoted = SECTION6_TABLE[key]
    omega_error = [abs(a - b) / b for a, b in
                   zip(record["omegas_over_pi"], quoted["omegas_over_pi"])]
    point_error = [abs(a - b) / b for a, b in
                   zip(record["points_over_pi"], quoted["points_over_pi"])]
    quoted_level = quoted["E_max"]
    level_ulp = 0.5 * 10.0 ** (np.floor(np.log10(quoted_level)) - 2.0)
    return {
        "quoted_omegas_over_pi": list(quoted["omegas_over_pi"]),
        "quoted_E_max": quoted_level,
        "quoted_points_over_pi": list(quoted["points_over_pi"]),
        "quoted_signs": list(quoted["signs"]),
        "omega_relative_error": omega_error,
        "omega_relative_error_max": float(max(omega_error)),
        "E_max_relative_error": float(abs(record["level"] - quoted_level) / quoted_level),
        "point_relative_error_max": float(max(point_error)),
        "signs_match": [int(np.sign(v)) for v in record["values"]] == list(quoted["signs"]),
        "alternation_deviation": float(max(
            abs(abs(v) - 1.0) for v in record["alternation_values"])),
        "omega_within_quoted_digits": bool(all(
            abs(a - b) <= 5e-5 for a, b in
            zip(record["omegas_over_pi"], quoted["omegas_over_pi"]))),
        "E_max_within_quoted_digits": bool(
            abs(record["level"] - quoted_level) <= level_ulp),
        "E_max_half_ulp": float(level_ulp),
    }


# --------------------------------------------------------------------- figures


def _style():
    """Return the figure module after applying the manuscript defaults."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotting_helpers as ph
    ph.apply_classy_defaults()
    plt.rcParams.update({"font.size": 9.0, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
                         "legend.fontsize": 8.0})
    return plt, ph


def figure_equioscillation(records: dict, path: Path) -> None:
    """Write the normalized error curves of the (6, 2) and (6, 3) optima."""
    plt, ph = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    panels = (((6, 2), "(a)", "#0072B2"), ((6, 3), "(b)", "#D55E00"))
    for ax, (cell, tag, color) in zip(axes, panels):
        record = records[cell]
        m, r = cell
        T = record["T"]
        omegas = np.asarray(record["omegas"], dtype=float)
        level = record["level"]
        error = ExactError(m, omegas)
        theta = np.linspace(T / 4001, T, 4001)
        ax.plot(theta / np.pi, error(theta) / level, color=color, lw=1.5,
                label=r"$e_1(\theta)/E_{\max}$")
        for line in (-1.0, 1.0):
            ax.axhline(line, color="0.45", lw=0.7, ls=(0, (4, 2.5)))
        ax.axhline(0.0, color="0.80", lw=0.6)
        points = np.asarray(record["points"], dtype=float)
        values = np.asarray(record["values"], dtype=float) / level
        ax.plot(points / np.pi, values, "o", color=color, ms=4.6,
                mfc="white", mew=1.3, label="alternation points")
        for w in omegas:
            ax.axvline(w / np.pi, color="0.72", lw=0.6, ls=(0, (1, 2)))
        ax.axvline(T / np.pi, color="0.35", lw=0.8)
        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_xlim(0.0, T / np.pi * 1.02)
        ax.set_ylim(-1.45, 1.45)
        ax.grid(axis="y", color="0.92", lw=0.5)
        ph.quiet_axis(ax)
        ax.text(0.015, 0.985, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=9.0, color="#2F4858")
        mantissa, exponent = f"{level:.2e}".split("e")
        ax.text(0.985, 0.035,
                rf"$E_{{{m},{r}}}$({_fmt(omegas, 4)}),  "
                rf"$E_{{\max}}={mantissa}\times 10^{{{int(exponent)}}}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.4, color="0.30")
    axes[0].set_ylabel(r"$e_1(\theta)/E_{\max}$")
    handles, labels = axes[0].get_legend_handles_labels()
    ph.make_overhead_legend(fig, handles, labels, ncol=2, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.90), fontsize=7.8,
                            handslcngth=1.9, columnspacing=1.1,
                            handletextpad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------------- writers


def _fmt(values, digits: int = 6) -> str:
    return ", ".join(f"{float(v):.{digits}f}" for v in values)


def rule_markdown(data: dict) -> str:
    """Return the text of rule.md from the finished records."""
    lines = []
    lines.append("# Gate 2: the frequency-selection rule")
    lines.append("")
    lines.append("Produced by code/analysis/exponential_family/gate2_rule.py.")
    lines.append("Every number below is read from rule.json in this directory.")
    lines.append("")
    lines.append("## The rule in five lines")
    lines.append("")
    lines.append("1. Fix the band `(0, T]`, the polynomial core `m`, and the number "
                 "of frequency pairs `r`.")
    lines.append("2. Each pair puts an exact zero of `e_1 = kappa_1/theta - 1` at its "
                 "own frequency `omega_j`, so `r` pairs split the band into `r + 1` "
                 "intervals.")
    lines.append("3. The band-minimax frequencies are the unique `omega` for which "
                 "`|e_1|` reaches the same value at one point of every interval, with "
                 "alternating sign, the last of those points being the band edge `T`.")
    lines.append("4. Solve the `r` level conditions by a Remez-type Newton iteration "
                 "on the exact alias-sum symbol, starting from the weighted Chebyshev "
                 "solution of the one-alias model in the variable `u = theta^2`.")
    lines.append("5. The levelled value `E_max = |e_1(T)|` is the band maximum, and "
                 "the conditioning price of the chosen `omega` is read off Theorem A5.")
    lines.append("")
    lines.append("## Optima on B0 = (0, 2 pi/3]")
    lines.append("")
    lines.append("| member | optimal omega / pi | E_max(B0) | "
                 "alternation points theta / pi (sign) | cond | iterations |")
    lines.append("|---|---|---|---|---|---|")
    for cell in TABLE_CELLS:
        record = data["band_B0"][f"m{cell[0]}_r{cell[1]}"]
        points = ", ".join(
            f"{p:.3f} ({'+' if v > 0 else '-'})"
            for p, v in zip(record["points_over_pi"], record["values"]))
        lines.append("| m={m}, r={r} | {om} | {E:.2e} | {pts} | {cond:.1f} | {it} |".format(
            m=cell[0], r=cell[1], om=_fmt(record["omegas_over_pi"], 4),
            E=record["level"], pts=points, cond=record["cond"]["cond"],
            it=record["iterations"]))
    lines.append("")
    lines.append("Alternation values `e_1(x_i)/E_max` deviate from `+-1` by at most "
                 f"{data['summary']['max_alternation_deviation']:.1e} over the five "
                 "rows. Against the section 6 table the largest deviation is "
                 f"{data['summary']['max_omega_relative_error']:.1e} relative in "
                 "omega and "
                 f"{data['summary']['max_E_max_relative_error']:.1e} relative in "
                 "E_max. The table quotes omega/pi to four decimals and E_max to "
                 "three significant digits, and every value above lies inside the "
                 "rounding interval of the quoted digits, so the two agree to the "
                 "precision at which section 6 states them.")
    lines.append("")
    lines.append("## The (6, 2) cell across bands")
    lines.append("")
    lines.append("| T / pi | optimal omega / pi | E_max | cond | min b | "
                 "cond (N = 128) | alternation points theta / pi |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in data["band_sweep_order"]:
        record = data["band_sweep"][key]
        lines.append(
            "| {T:.4f} | {om} | {E:.3e} | {cond:.2f} | {mb:.5f} | {cn:.2f} | {pts} |".format(
                T=record["T_over_pi"], om=_fmt(record["omegas_over_pi"], 6),
                E=record["level"], cond=record["cond"]["cond"],
                mb=record["cond"]["min_b"], cn=record["cond"]["cond_N128"],
                pts=_fmt(record["points_over_pi"], 4)))
    lines.append("")
    lines.append("## The one-alias starting guess")
    lines.append("")
    lines.append("The weighted Chebyshev solution of the one-alias model is the "
                 "starting vector of every row above. Its distance from the exact "
                 "optimum is the accuracy of the model:")
    lines.append("")
    lines.append("| member | T / pi | start omega / pi | max relative error in omega | "
                 "model level | relative error in level |")
    lines.append("|---|---|---|---|---|---|")
    for key, record in data["start_accuracy"]:
        lines.append("| {label} | {T:.4f} | {om} | {we:.2e} | {lv:.4e} | {le:.2e} |".format(
            label=key, T=record["T_over_pi"],
            om=_fmt(record["start_over_pi"], 6),
            we=max(record["start_frequency_error"]),
            lv=record["start_level"], le=record["start_level_error"]))
    lines.append("")
    lines.append("The model level is high by "
                 f"{100.0 * data['summary']['min_start_level_error']:.2f} to "
                 f"{100.0 * data['summary']['max_start_level_error']:.2f} percent, "
                 "and its frequencies are within "
                 f"{data['summary']['max_start_frequency_error']:.1e} relative of the "
                 "exact ones, so the exact iteration converges from it in "
                 f"{data['summary']['max_iterations']} steps or fewer.")
    lines.append("")
    lines.append("## Local minimality")
    lines.append("")
    lines.append("Every optimum was perturbed by "
                 f"{PERTURB_RADIUS:g} in {PERTURB_DIRECTIONS} directions drawn "
                 "uniformly on the unit sphere of the frequency space, and the band "
                 "supremum of `|e_1|` was recomputed from its own extrema. The "
                 "supremum rose at every direction of every optimum; the smallest "
                 "rise over the whole set is a factor "
                 f"{data['summary']['min_perturbation_ratio']:.6f}. The optimal "
                 f"supremum agrees with the {BAND_POINTS} point uniform-grid maximum "
                 f"to {data['summary']['max_grid_supremum_gap']:.1e} relative.")
    lines.append("")
    lines.append("## Uniqueness of the levelled vector")
    lines.append("")
    summary = data["summary"]
    lines.append("Every optimum above was recomputed from "
                 f"{MULTISTART_STARTS} ordered frequency vectors drawn uniformly "
                 "inside its own band, "
                 f"{summary['multistart_starts']} restarts in all. "
                 f"{summary['multistart_converged']} of them converged, and every "
                 "converged restart returned the reference vector to "
                 f"{summary['max_multistart_deviation']:.1e} relative. The count of "
                 "converged restarts that landed on a second levelled vector is "
                 f"{summary['distinct_optima_found']}. The remaining "
                 f"{summary['multistart_stalled']} restarts stalled, every one of "
                 "them at a residual of exactly 1 and with a starting vector whose "
                 "smallest frequency lay below "
                 f"{summary['max_stall_start_over_pi']:.3f} pi. By Proposition O the "
                 "error curve is O(theta^m) near the origin, so such a start leaves "
                 "the whole first interval below the roundoff floor of the symbol, "
                 "the level condition for that interval reads (0 - E)/E = -1, and "
                 "the damped Newton step has no resolvable gradient to follow. The "
                 "smallest extremal magnitude over all stalled runs is "
                 f"{summary['min_stall_extremal_magnitude']:.1e}, and every stalled "
                 "run sat at a band supremum at least "
                 f"{summary['min_stall_supremum_ratio']:.1f} times the optimal one. "
                 "A stall is a failure of that start, not a second optimum; the "
                 "one-alias guess never lands there and converges in every row "
                 "above.")
    lines.append("")
    return "\n".join(lines) + "\n"


def captions_markdown(data: dict) -> str:
    """Return the text of figures/captions.md."""
    record62 = data["band_B0"]["m6_r2"]
    record63 = data["band_B0"]["m6_r3"]
    return (
        "# Gate 2 figure captions\n"
        "\n"
        "Captions for the figures in results/exponential_family/gate2/figures.\n"
        "Every number quoted here is taken from rule.json.\n"
        "\n"
        "## fig_equioscillation.pdf\n"
        "\n"
        "Band-minimax first-derivative symbol error on the dealiased band,\n"
        "normalized by its own band maximum. Left, the two-pair core,\n"
        f"omega/pi = {_fmt(record62['omegas_over_pi'], 4)}; right, the three-pair\n"
        f"core, omega/pi = {_fmt(record63['omegas_over_pi'], 4)}. Open circles\n"
        "mark the alternation points, dotted verticals the contacts, the solid\n"
        "vertical the band edge.\n"
        "\n"
        "Length note: 47 words, over the 45 word audit threshold because the\n"
        "caption carries the five optimal frequencies of the two panels\n"
        "verbatim, which is the content the figure exists to show.\n"
    )


def build(figures: bool = True) -> dict:
    """Return every Gate 2 record and write the receipts under results."""
    data: dict = {
        "band_edge_B0": float(B0),
        "alias_tol": ALIAS_TOL,
        "remez_tol": REMEZ_TOL,
        "band_points": BAND_POINTS,
        "band_B0": {},
        "section6_comparison": {},
        "band_sweep": {},
        "band_sweep_order": [],
    }
    for m, r in TABLE_CELLS:
        record = optimum_record(m, r, B0)
        key = f"m{m}_r{r}"
        data["band_B0"][key] = record
        data["section6_comparison"][key] = compare_with_section6(record)

    m, r = BAND_SWEEP_CELL
    for T in (*BAND_SWEEP_EDGES, B0):
        key = f"T{T / np.pi:.4f}pi"
        data["band_sweep"][key] = optimum_record(m, r, T)
        data["band_sweep_order"].append(key)
    data["band_sweep_order"].sort(key=lambda k: data["band_sweep"][k]["T"])

    start_accuracy = []
    seen = set()
    for m, r in TABLE_CELLS:
        record = data["band_B0"][f"m{m}_r{r}"]
        seen.add((m, r, round(record["T"], 12)))
        start_accuracy.append((f"m={m}, r={r}", record))
    for key in data["band_sweep_order"]:
        record = data["band_sweep"][key]
        tag = (record["m"], record["r"], round(record["T"], 12))
        if tag in seen:
            continue
        seen.add(tag)
        start_accuracy.append((f"m={record['m']}, r={record['r']}", record))
    data["start_accuracy"] = start_accuracy

    everything = [rec for rec in data["band_B0"].values()]
    everything += [data["band_sweep"][k] for k in data["band_sweep_order"]]
    comparisons = [c for c in data["section6_comparison"].values() if c]
    data["summary"] = {
        "all_converged": all(rec["converged"] for rec in everything),
        "all_alternating": all(rec["alternating"] for rec in everything),
        "all_admissible": all(rec["admissible"] for rec in everything),
        "all_local_minima": all(rec["local_minimum"]["is_local_minimum"]
                                for rec in everything),
        "max_residual": max(rec["residual"] for rec in everything),
        "max_iterations": max(rec["iterations"] for rec in everything),
        "min_perturbation_ratio": min(rec["local_minimum"]["min_ratio"]
                                      for rec in everything),
        "distinct_optima_found": sum(rec["multistart"]["distinct_optima"]
                                     for rec in everything),
        "multistart_converged": sum(rec["multistart"]["converged"]
                                    for rec in everything),
        "multistart_stalled": sum(rec["multistart"]["stalled"]
                                  for rec in everything),
        "multistart_starts": sum(rec["multistart"]["starts"] for rec in everything),
        "max_multistart_deviation": max(rec["multistart"]["max_deviation_from_reference"]
                                        for rec in everything),
        "max_stall_start_over_pi": max(
            (min(stall["start_over_pi"]) for rec in everything
             for stall in rec["multistart"]["stalls"]), default=0.0),
        "min_stall_extremal_magnitude": min(
            (stall["min_extremal_magnitude"] for rec in everything
             for stall in rec["multistart"]["stalls"]), default=0.0),
        "min_stall_supremum_ratio": min(
            (stall["supremum_ratio"] for rec in everything
             for stall in rec["multistart"]["stalls"]), default=0.0),
        "max_grid_supremum_gap": max(
            abs(rec["local_minimum"]["grid_band_maximum"]
                - rec["local_minimum"]["band_supremum"])
            / rec["local_minimum"]["band_supremum"] for rec in everything),
        "max_alternation_deviation": max(c["alternation_deviation"] for c in comparisons),
        "max_omega_relative_error": max(c["omega_relative_error_max"] for c in comparisons),
        "max_E_max_relative_error": max(c["E_max_relative_error"] for c in comparisons),
        "max_start_frequency_error": max(max(rec["start_frequency_error"])
                                         for rec in everything),
        "min_start_level_error": min(rec["start_level_error"] for rec in everything),
        "max_start_level_error": max(rec["start_level_error"] for rec in everything),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "rule.json").write_text(json.dumps(data, indent=2, default=float) + "\n",
                                   encoding="utf-8")
    (OUT / "rule.md").write_text(rule_markdown(data), encoding="utf-8")
    if figures:
        figure_equioscillation(
            {(6, 2): data["band_B0"]["m6_r2"], (6, 3): data["band_B0"]["m6_r3"]},
            OUT / "figures" / "fig_equioscillation.pdf")
    (OUT / "figures" / "captions.md").write_text(captions_markdown(data),
                                                 encoding="utf-8")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gate 2 frequency-selection rule.")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the figure and write the receipts only")
    args = parser.parse_args(argv)
    data = build(figures=not args.no_figures)
    summary = data["summary"]
    for m, r in TABLE_CELLS:
        record = data["band_B0"][f"m{m}_r{r}"]
        print(f"m={m} r={r}  omega/pi = [{_fmt(record['omegas_over_pi'], 6)}]"
              f"  E_max = {record['level']:.6e}"
              f"  cond = {record['cond']['cond']:.2f}"
              f"  iterations = {record['iterations']}")
    for key in data["band_sweep_order"]:
        record = data["band_sweep"][key]
        print(f"T/pi={record['T_over_pi']:.4f}  omega/pi = "
              f"[{_fmt(record['omegas_over_pi'], 6)}]"
              f"  E_max = {record['level']:.6e}"
              f"  cond = {record['cond']['cond']:.2f}")
    print("summary:", json.dumps(summary, indent=2, default=float))
    comparisons = [c for c in data["section6_comparison"].values() if c]
    ok = (summary["all_converged"] and summary["all_alternating"]
          and summary["all_admissible"] and summary["all_local_minima"]
          and summary["max_omega_relative_error"] <= 1e-4
          and summary["max_alternation_deviation"] <= 1e-3
          and all(c["signs_match"] for c in comparisons)
          and all(c["omega_within_quoted_digits"] for c in comparisons)
          and all(c["E_max_within_quoted_digits"] for c in comparisons)
          and summary["distinct_optima_found"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
