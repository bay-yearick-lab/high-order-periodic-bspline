"""Gate 4b: long-horizon dispersive transport of a narrow-band wave packet.

The model problem is periodic linear advection on [0, 2 pi)^d with a uniform
grid of N points per axis, h = 2 pi / N. Every route in this gate replaces the
exact first derivative by a periodic circulant operator D1 whose symbol on the
grid mode e^{i k x} is

    D1 e^{i k x} = (i / h) kappa_1(theta) e^{i k x},      theta = k h = 2 pi k / N,

and the relative first-derivative error of the route is defined with the sign
convention

    e_1(theta) = kappa_1(theta) / theta - 1.

One dimension, c = 1. The semi-discrete equation u_t = -c D1 u sends the mode
amplitude at wavenumber k through exp(-i (c / h) kappa_1(theta) t). One box
crossing is t = 2 pi / c, so the per-crossing factor is exp(-i N kappa_1), and
the exact per-crossing factor is exp(-i N theta) = exp(-2 pi i k) = 1. The
semi-discrete factor relative to the exact one after H crossings is therefore

    f(theta) = exp(-i H N theta e_1(theta)),

which is the whole content of the prediction: the relative L2 error of the
packet is, by Parseval,

    e_H = sqrt( sum_k |A_k|^2 |f(theta_k) - 1|^2 / sum_k |A_k|^2 ),

with A_k the packet spectrum. A common factor exp(i phi) with phi > 0 is the
Fourier signature of a packet displaced backwards, so a route with
e_1(theta_c) < 0, whose numerical phase speed kappa_1 / theta is below the
exact one, lags; a route with e_1(theta_c) > 0 leads. The signed displacement
per crossing is -2 pi e_1(theta_c) in length, that is a signed fraction
-e_1(theta_c) of the box per crossing, independent of N at fixed theta_c, and
the signed carrier phase lag after H crossings is -H N theta_c e_1. Both signs
occur in this gate: the spline routes and the tuned q8 pair undershoot, while
SLC's spectral-like tuning and CW12B overshoot at theta_c = 0.5 pi.

Three dimensions, c = (1, 1/2, 1/3). The operator is the sum of three commuting
circulants, so the mode factor is governed by

    S(k) = c_x kappa_1(theta_x) + c_y kappa_1(theta_y) + c_z kappa_1(theta_z),

against the exact S_exact(k) = c_x theta_x + c_y theta_y + c_z theta_z, and one
crossing still means t = 2 pi / c_x.

Time stepping is SSP-RK3, whose amplification factor on a linear problem with
eigenvalue lambda and step dt is

    g(z) = 1 + z + z^2 / 2 + z^3 / 6,     z = lambda dt.

Here lambda is pure imaginary, z = -i y with y = (dt / h) S real, and

    |g(-i y)|^2 = 1 - y^4 / 12 + y^6 / 36,      arg g(-i y) = -y - y^5 / 30 + ...,

so one step loses amplitude at order y^4 and phase at order y^5. Over M steps
per crossing with M = N / nu and y = nu S, the amplitude error per crossing is
N nu^3 S^4 / 24 and the phase error per crossing is N nu^4 S^5 / 30. The cfl
nu = dt / h of this gate is chosen from those two expressions, evaluated at
each route's largest active |S|, so that both are below TEMPORAL_TARGET; the
amplitude condition is the binding one and is the stronger reading of the
contract's phase-only requirement.

Because SSP-RK3 on a circulant operator is exactly diagonal in the discrete
Fourier basis, the n-step map is exactly multiplication by g(z)^n. The runs are
therefore advanced by the accumulated logarithm

    log f_relative = n_full d(y) + d(y_r) - i H N (S - S_exact),
    d(y) = log g(-i y) + i y = -y^4 / 24 + y^6 / 72 - i y^5 / 30 + ...,

with n_full full steps and one partial final step of length dt_r that lands on
the horizon exactly. This is the SSP-RK3 solution itself, not an approximation
of it; it is verified two ways, by comparing the assembled circulant operators
of spline_analysis against their symbols, and by running the genuine three
stage time loop with those assembled operators over one crossing at N = 64.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(HERE), str(CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import spline_analysis as sa  # noqa: E402

GATE = ROOT / "results" / "exponential_family" / "gate4b"
FIGDIR = GATE / "figures"
CONTRACT = GATE / "contract.md"
MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
GATE4F_MEMBERS = ROOT / "results" / "exponential_family" / "gate4f" / "members"

# Two-sided 99 percent point of the standard normal. The packet energy density
# in theta is Gaussian with standard deviation SIGMA_ENERGY, so 99 percent of
# the energy lies within theta_c +- Z99 * SIGMA_ENERGY = theta_c +- 0.1 pi.
Z99 = 2.5758293035489004
BAND_HALF_WIDTH = 0.1 * np.pi
SIGMA_ENERGY = BAND_HALF_WIDTH / Z99

THETA_CENTERS = (0.4 * np.pi, 0.5 * np.pi, 0.6 * np.pi)
GRIDS_1D = (64, 128, 256, 512)
HORIZONS_1D = (10, 100, 1000)
HORIZONS_3D = (10, 100)
GRID_3D = 64
VELOCITY_3D = (1.0, 0.5, 1.0 / 3.0)

TEMPORAL_TARGET = 1.0e-6
ACTIVE_AMPLITUDE_FLOOR = 1.0e-08
AGREEMENT_FACTOR = 2.0
ERROR_LEVELS = (0.01, 0.10)

PACKET_CENTER_X = np.pi


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Routes: the symbol and the one-sided circulant entries of each first
# derivative. The entries feed spline_analysis's circulant builders, so the
# assembled operator and the symbol come from the same frozen coefficients.
# ---------------------------------------------------------------------------


def _compact_entries(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the left-side and right-side one-sided entries of a compact D1."""
    mass = np.concatenate(([1.0], np.asarray(spec["alpha"], dtype=float)))
    advection = np.concatenate(([0.0], np.asarray(spec["a"], dtype=float)))
    return mass, advection


def route_table() -> list[dict]:
    """Return every route of the gate with its symbol and circulant entries."""
    member_label = "EXP:" + str(MEMBER)
    routes = [
        {"key": "champion", "display": "E62 champion",
         "kappa1": lambda th: sa.kprime_exp_member(np.asarray(th, dtype=float),
                                                   member_label),
         "entries": sa.exp_member_mass_and_advection_entries(member_label),
         "spectral": False},
        {"key": "slc_q4", "display": "SLC q4",
         "kappa1": comp.compact_kappa1(comp.SLC_D1["alpha"], comp.SLC_D1["a"]),
         "entries": _compact_entries(comp.SLC_D1), "spectral": False},
        {"key": "tuned_q8", "display": "tuned q8",
         "kappa1": comp.compact_kappa1(comp.Q8_D1["alpha"], comp.Q8_D1["a"]),
         "entries": _compact_entries(comp.Q8_D1), "spectral": False},
        {"key": "cw12b", "display": "CW12B",
         "kappa1": comp.compact_kappa1(comp.CW12B_D1["alpha"], comp.CW12B_D1["a"]),
         "entries": _compact_entries(comp.CW12B_D1), "spectral": False},
        {"key": "bmc9", "display": "BMC9",
         "kappa1": lambda th: sa.kprime_collocation(np.asarray(th, dtype=float), 9),
         "entries": sa.collocation_mass_and_advection_entries(9), "spectral": False},
        {"key": "bmc11", "display": "BMC11",
         "kappa1": lambda th: sa.kprime_collocation(np.asarray(th, dtype=float), 11),
         "entries": sa.collocation_mass_and_advection_entries(11), "spectral": False},
        {"key": "fourier", "display": "Fourier",
         "kappa1": lambda th: np.asarray(th, dtype=float),
         "entries": None, "spectral": True},
    ]
    for tail in tail_member_routes():
        routes.append(tail)
    return routes


def tail_member_routes() -> list[dict]:
    """Return the Gate 4f tail member as a route if its JSON has appeared."""
    if not GATE4F_MEMBERS.is_dir():
        return []
    found = sorted(GATE4F_MEMBERS.glob("*.json"))
    routes = []
    for path in found:
        label = "EXP:" + str(path)
        routes.append({
            "key": "tail_" + path.stem.lower(),
            "display": path.stem,
            "kappa1": (lambda lab: lambda th: sa.kprime_exp_member(
                np.asarray(th, dtype=float), lab))(label),
            "entries": sa.exp_member_mass_and_advection_entries(label),
            "spectral": False,
            "member_path": (str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT) else str(path)),
        })
    return routes


def group_velocity_error(kappa1, theta) -> np.ndarray:
    """Return d kappa_1 / d theta - 1, the relative group velocity error.

    The packet envelope travels at the group velocity c d kappa_1 / d theta while
    its carrier travels at the phase velocity c kappa_1 / theta, so the envelope
    displacement per crossing is 2 pi (d kappa_1 / d theta - 1) in length while
    the carrier displacement per crossing is -2 pi e_1. The derivative is taken
    with the fourth-order centred difference of the analytic symbol.
    """
    d = 1.0e-4
    theta = np.asarray(theta, dtype=float)
    f = [kappa1(theta + shift * d) for shift in (-2.0, -1.0, 1.0, 2.0)]
    return (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * d) - 1.0


def band_group_lag(kappa1, theta: np.ndarray, amp: np.ndarray,
                   horizon: float) -> float:
    """Return the envelope centroid lag in box fractions after horizon crossings.

    The mode factor is exp(i psi) with psi(k) = -H N theta e_1(theta), and the
    first moment theorem sends the intensity centroid of a packet with real
    positive amplitudes to x_0 - < d psi / d k >. With theta = 2 pi k / N that
    derivative is 2 pi H (1 - d kappa_1 / d theta), so the signed centroid lag,
    positive for a packet left behind, is -H times the energy weighted mean of
    d kappa_1 / d theta - 1 across the band.
    """
    weight = amp ** 2
    mean = float(np.sum(weight * group_velocity_error(kappa1, theta))
                 / np.sum(weight))
    return float(-horizon * mean)


def relative_symbol_error(kappa1, theta: np.ndarray) -> np.ndarray:
    """Return e_1(theta) = kappa_1(theta) / theta - 1 away from theta = 0."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros(theta.shape)
    nz = np.abs(theta) > 0.0
    out[nz] = kappa1(theta[nz]) / theta[nz] - 1.0
    return out


# ---------------------------------------------------------------------------
# The packet
# ---------------------------------------------------------------------------


def packet(N: int, theta_c: float) -> dict:
    """Return the one-sided spectrum of the Gaussian packet on N points.

    The amplitude spectrum is A(theta) = exp(-(theta - theta_c)^2 / (4 s^2))
    with s = SIGMA_ENERGY, so the energy |A|^2 is Gaussian with standard
    deviation s and 99 percent of it lies inside theta_c +- 0.1 pi. The
    amplitude standard deviation is s sqrt(2) in theta, hence s sqrt(2) N / 2 pi
    in wavenumber, and the real-space Gaussian envelope has standard deviation
    sigma_x = 2 pi / (s sqrt(2) N), that is 1 / (s sqrt(2)) mesh widths for
    every N.
    """
    k = np.arange(N // 2 + 1)
    theta = 2.0 * np.pi * k / N
    amp = np.exp(-(theta - theta_c) ** 2 / (4.0 * SIGMA_ENERGY ** 2))
    amp[0] = 0.0
    amp[-1] = 0.0
    sigma_k = SIGMA_ENERGY * np.sqrt(2.0) * N / (2.0 * np.pi)
    return {
        "k": k, "theta": theta, "amp": amp,
        "sigma_theta_energy": SIGMA_ENERGY,
        "sigma_theta_amplitude": SIGMA_ENERGY * np.sqrt(2.0),
        "sigma_k_amplitude": sigma_k,
        "sigma_x": 1.0 / sigma_k,
        "sigma_x_in_mesh_widths": 1.0 / (sigma_k * (2.0 * np.pi / N)),
        "k_center": N * theta_c / (2.0 * np.pi),
    }


def active_mask(amp: np.ndarray) -> np.ndarray:
    """Return the modes carrying more than ACTIVE_AMPLITUDE_FLOOR of the peak."""
    return amp > ACTIVE_AMPLITUDE_FLOOR * amp.max()


def analytic_signal(N: int, k: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return sum_k c_k exp(i k (x - x0)) on the grid, x0 = PACKET_CENTER_X."""
    hat = np.zeros(N, dtype=complex)
    hat[k] = N * coefficients * np.exp(-1j * k * PACKET_CENTER_X)
    return np.fft.ifft(hat)


def real_field(N: int, k: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return the real packet 2 Re sum_{k>0} c_k exp(i k (x - x0))."""
    return 2.0 * np.real(analytic_signal(N, k, coefficients))


def circular_centroid(weight: np.ndarray) -> float:
    """Return the circular mean position on [0, 2 pi) of a nonnegative weight."""
    N = weight.shape[0]
    x = 2.0 * np.pi * np.arange(N) / N
    moment = np.sum(weight * np.exp(1j * x))
    return float(np.angle(moment))


def wrap_to_pi(value: float) -> float:
    """Return the representative of value in (-pi, pi]."""
    return float(2.0 * np.pi * wrap_to_half(value / (2.0 * np.pi)))


def wrap_to_half(value: float) -> float:
    """Return the representative of value in (-0.5, 0.5] modulo one."""
    out = value - np.floor(value)
    if out > 0.5:
        out -= 1.0
    return float(out)


# ---------------------------------------------------------------------------
# SSP-RK3 amplification
# ---------------------------------------------------------------------------


def rk3_amplification(z):
    """Return g(z) = 1 + z + z^2 / 2 + z^3 / 6."""
    return 1.0 + z * (1.0 + z * (0.5 + z / 6.0))


def rk3_log_defect(y):
    """Return log g(-i y) + i y, the per-step departure from exact transport.

    The series is -y^4 / 24 + y^6 / 72 - i y^5 / 30, obtained from
    |g|^2 = 1 - y^4 / 12 + y^6 / 36 and arg g = -y - y^5 / 30 + O(y^7). It is
    used instead of a direct complex logarithm because log g(-i y) is within
    y^4 / 24 of the pure phase -i y, a cancellation that double precision
    cannot resolve at the tiny y this gate runs at.
    """
    y = np.asarray(y, dtype=float)
    y2 = y * y
    y4 = y2 * y2
    return (-y4 / 24.0 + y4 * y2 / 72.0) - 1j * (y4 * y / 30.0)


def rk3_log_defect_longdouble(y):
    """Return the same quantity from an extended precision complex logarithm."""
    y = np.asarray(y, dtype=np.longdouble)
    z = -1j * y.astype(np.clongdouble)
    g = 1.0 + z * (1.0 + z * (0.5 + z / 6.0))
    return np.log(g) + 1j * y


def temporal_errors_per_crossing(nu: float, N: int, smax: float) -> tuple[float, float]:
    """Return the RK3 amplitude and phase error per crossing at |S| = smax."""
    steps = N / nu
    defect = rk3_log_defect(nu * smax)
    return float(steps * abs(defect.real)), float(steps * abs(defect.imag))


def cfl_from_symbol(N: int, smax: float, target: float = TEMPORAL_TARGET) -> float:
    """Return the largest cfl whose per-crossing RK3 errors stay below target.

    Amplitude: N nu^3 smax^4 / 24 < target. Phase: N nu^4 smax^5 / 30 < target.
    """
    nu_amp = (24.0 * target / (N * smax ** 4)) ** (1.0 / 3.0)
    nu_phase = (30.0 * target / (N * smax ** 5)) ** 0.25
    return float(min(nu_amp, nu_phase))


def floor_to_two_digits(value: float) -> float:
    """Return value rounded down to two significant decimal digits."""
    if value <= 0.0:
        raise ValueError("The cfl must be positive.")
    scale = 10.0 ** (np.floor(np.log10(value)) - 1.0)
    return float(np.floor(value / scale) * scale)


def step_plan(N: int, horizon: int, nu: float) -> tuple[int, float]:
    """Return the full step count and the partial final step in units of h.

    The crossing time is T = 2 pi / c_x with c_x = 1 and dt = nu h, so
    T / dt = H N / nu. The partial step lands on the horizon exactly.
    """
    total = horizon * N / nu
    n_full = int(np.floor(total))
    remainder = total - n_full
    return n_full, float(remainder * nu)


def relative_factor(sym_sum: np.ndarray, exact_sum: np.ndarray, N: int,
                    horizon: int, nu: float) -> np.ndarray:
    """Return the SSP-RK3 mode factor relative to the exact transport factor."""
    n_full, nu_r = step_plan(N, horizon, nu)
    temporal = n_full * rk3_log_defect(nu * sym_sum) + rk3_log_defect(nu_r * sym_sum)
    dispersive = -1j * horizon * N * (sym_sum - exact_sum)
    return np.exp(temporal + dispersive), temporal


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predicted_error(amp: np.ndarray, theta: np.ndarray, e1: np.ndarray,
                    N: int, horizon: float) -> float:
    """Return e_H from the symbol alone, with no temporal contribution."""
    phase = -horizon * N * theta * e1
    energy = amp ** 2
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    defect = float(np.sum(energy * 4.0 * np.sin(0.5 * phase) ** 2))
    return float(np.sqrt(defect / total))


def crossing_horizon(amp: np.ndarray, theta: np.ndarray, e1: np.ndarray,
                     N: int, level: float) -> float | None:
    """Return the first horizon at which the predicted error reaches level."""
    if np.all(np.abs(e1) < 1e-16):
        return None
    scale = np.sqrt(np.sum(amp ** 2 * (N * theta * e1) ** 2) / np.sum(amp ** 2))
    if scale <= 0.0:
        return None
    guess = level / scale
    lo = guess * 1.0e-3
    if predicted_error(amp, theta, e1, N, lo) >= level:
        lo = 1.0e-12
    hi = lo
    for _ in range(200):
        hi *= 1.2
        if predicted_error(amp, theta, e1, N, hi) >= level:
            break
    else:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if predicted_error(amp, theta, e1, N, mid) >= level:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def build_prediction(routes: list[dict]) -> dict:
    """Return the prediction payload for every route, grid and packet center."""
    cells = {}
    cfl_by_grid = {}
    for N in GRIDS_1D:
        smax_by_route = {}
        for theta_c in THETA_CENTERS:
            pk = packet(N, theta_c)
            live = active_mask(pk["amp"])
            for route in routes:
                value = float(np.max(np.abs(route["kappa1"](pk["theta"][live]))))
                smax_by_route[route["key"]] = max(
                    smax_by_route.get(route["key"], 0.0), value)
        nu_by_route = {key: cfl_from_symbol(N, value)
                       for key, value in smax_by_route.items()}
        binding = min(nu_by_route, key=nu_by_route.get)
        nu = floor_to_two_digits(min(nu_by_route.values()))
        checks = {}
        for route in routes:
            amp_err, phase_err = temporal_errors_per_crossing(
                nu, N, smax_by_route[route["key"]])
            checks[route["key"]] = {
                "largest_active_kappa1": smax_by_route[route["key"]],
                "cfl_ceiling_for_route": nu_by_route[route["key"]],
                "amplitude_error_per_crossing": amp_err,
                "phase_error_per_crossing": phase_err,
                "below_target": bool(max(amp_err, phase_err) < TEMPORAL_TARGET),
            }
        cfl_by_grid[str(N)] = {
            "cfl": nu,
            "steps_per_crossing": N / nu,
            "binding_route": binding,
            "target_per_crossing": TEMPORAL_TARGET,
            "criterion": ("max over active modes of the RK3 amplitude error "
                          "N nu^3 kappa_1^4 / 24 and phase error "
                          "N nu^4 kappa_1^5 / 30 per crossing"),
            "per_route": checks,
        }

    for N in GRIDS_1D:
        for theta_c in THETA_CENTERS:
            pk = packet(N, theta_c)
            live = active_mask(pk["amp"])
            theta = pk["theta"][live]
            amp = pk["amp"][live]
            for route in routes:
                e1 = relative_symbol_error(route["kappa1"], theta)
                e1_center = float(relative_symbol_error(
                    route["kappa1"], np.array([theta_c]))[0])
                entry = {
                    "N": N, "theta_c": theta_c, "route": route["key"],
                    "e1_at_theta_c": e1_center,
                    "kappa1_at_theta_c": float(route["kappa1"](
                        np.array([theta_c]))[0]),
                    "errors": {str(H): predicted_error(amp, theta, e1, N, H)
                               for H in HORIZONS_1D},
                    "carrier_phase_lag_rad": {
                        str(H): -H * N * theta_c * e1_center for H in HORIZONS_1D},
                    "centroid_lag_boxes": {
                        str(H): -H * e1_center for H in HORIZONS_1D},
                    "horizon_at_1_percent": crossing_horizon(
                        amp, theta, e1, N, ERROR_LEVELS[0]),
                    "horizon_at_10_percent": crossing_horizon(
                        amp, theta, e1, N, ERROR_LEVELS[1]),
                }
                cells[f"1d|{route['key']}|{N}|{theta_c:.6f}"] = entry

    three = predict_3d(routes)
    return {
        "producer": "code/analysis/exponential_family/gate4b_long_horizon.py --predict",
        "written_utc": utc_now(),
        "contract": {"path": str(CONTRACT.relative_to(ROOT)),
                     "sha256": sha256(CONTRACT)},
        "member": {"path": str(MEMBER.relative_to(ROOT)), "sha256": sha256(MEMBER)},
        "sign_convention": {
            "e_1": "e_1(theta) = kappa_1(theta) / theta - 1",
            "mode_factor": ("the numerical mode relative to the exact one after H "
                            "box crossings is exp(-i H N theta e_1(theta))"),
            "packet_error": ("e_H = || sum_k A_k (exp(-i H N theta_k e_1(theta_k)) "
                             "- 1) exp(i k x) || / || sum_k A_k exp(i k x) ||"),
            "lag": ("a positive argument of the relative factor is a backward "
                    "displacement, so e_1 < 0 lags and e_1 > 0 leads; the signed "
                    "lag is -2 pi H e_1(theta_c) in length, a signed fraction "
                    "-H e_1(theta_c) of the box, and the signed carrier phase lag "
                    "is -H N theta_c e_1 radians"),
            "checked_on_run": "1d|champion|128|0.5pi at H = 10, see decision.json",
        },
        "packet": {
            "definition": ("A(theta) = exp(-(theta - theta_c)^2 / (4 s^2)) on the "
                           "one-sided grid spectrum, s = 0.1 pi / z_0.995"),
            "z_0.995": Z99,
            "sigma_theta_energy": SIGMA_ENERGY,
            "sigma_theta_amplitude": SIGMA_ENERGY * np.sqrt(2.0),
            "band_99_percent_half_width": BAND_HALF_WIDTH,
            "envelope_per_grid": {
                str(N): {
                    "sigma_k_amplitude": packet(N, 0.5 * np.pi)["sigma_k_amplitude"],
                    "sigma_x": packet(N, 0.5 * np.pi)["sigma_x"],
                    "sigma_x_in_mesh_widths":
                        packet(N, 0.5 * np.pi)["sigma_x_in_mesh_widths"],
                } for N in GRIDS_1D},
        },
        "time_stepping": {
            "scheme": "SSP-RK3, g(z) = 1 + z + z^2 / 2 + z^3 / 6",
            "per_step_defect": ("log g(-i y) + i y = -y^4 / 24 + y^6 / 72 "
                                "- i y^5 / 30, y = (dt / h) kappa_1"),
            "exact_time_landing": "one partial final step of length T - n_full dt",
            "cfl_by_grid": cfl_by_grid,
            "cfl_halved_check": "each 1D cell is repeated at cfl / 2",
        },
        "levels": list(ERROR_LEVELS),
        "agreement_factor": AGREEMENT_FACTOR,
        "cells_1d": cells,
        "cells_3d": three,
        "routes": [route["key"] for route in routes],
        "tail_member_present": bool(tail_member_routes()),
    }


def grid_3d(N: int):
    """Return the signed grid angles of one axis in FFT order."""
    return 2.0 * np.pi * np.fft.fftfreq(N)


def symbol_sums_3d(routes_kappa, N: int, velocity) -> tuple[np.ndarray, np.ndarray]:
    """Return S and S_exact on the full 3D mode grid for one route."""
    theta = grid_3d(N)
    kap = routes_kappa(theta)
    sym = (velocity[0] * kap[:, None, None]
           + velocity[1] * kap[None, :, None]
           + velocity[2] * kap[None, None, :])
    exact = (velocity[0] * theta[:, None, None]
             + velocity[1] * theta[None, :, None]
             + velocity[2] * theta[None, None, :])
    return sym, exact


def packet_3d_spectrum(N: int, theta_c: float) -> np.ndarray:
    """Return the FFT of the tensor-product real packet on the N^3 grid."""
    pk = packet(N, theta_c)
    line = real_field(N, pk["k"], pk["amp"])
    field = line[:, None, None] * line[None, :, None] * line[None, None, :]
    return np.fft.fftn(field), field


def predict_3d(routes: list[dict]) -> dict:
    """Return the 3D prediction at N = GRID_3D for the middle packet center."""
    theta_c = 0.5 * np.pi
    N = GRID_3D
    hat, _ = packet_3d_spectrum(N, theta_c)
    energy = np.abs(hat) ** 2
    live = energy > (ACTIVE_AMPLITUDE_FLOOR ** 2) * energy.max()
    smax_by_route = {}
    for route in routes:
        sym, _ = symbol_sums_3d(route["kappa1"], N, VELOCITY_3D)
        smax_by_route[route["key"]] = float(np.max(np.abs(sym[live])))
    nu_by_route = {key: cfl_from_symbol(N, value)
                   for key, value in smax_by_route.items()}
    nu = floor_to_two_digits(min(nu_by_route.values()))
    cells = {}
    total = float(np.sum(energy))
    for route in routes:
        sym, exact = symbol_sums_3d(route["kappa1"], N, VELOCITY_3D)
        entry = {"errors": {}, "largest_active_abs_S": smax_by_route[route["key"]]}
        for H in HORIZONS_3D:
            phase = -H * N * (sym - exact)
            defect = float(np.sum(energy * 4.0 * np.sin(0.5 * phase) ** 2))
            entry["errors"][str(H)] = float(np.sqrt(defect / total))
        e1c = float(relative_symbol_error(route["kappa1"], np.array([theta_c]))[0])
        entry["e1_at_theta_c"] = e1c
        entry["centroid_lag_boxes_per_axis"] = {
            str(H): [-H * e1c * v / VELOCITY_3D[0] for v in VELOCITY_3D]
            for H in HORIZONS_3D}
        cells[f"3d|{route['key']}|{N}|{theta_c:.6f}"] = entry
    return {
        "N": N, "theta_c": theta_c, "velocity": list(VELOCITY_3D),
        "crossing": "t = 2 pi / c_x, so H counts crossings of the x period",
        "cfl": nu, "cfl_per_route_ceiling": nu_by_route,
        "binding_route": min(nu_by_route, key=nu_by_route.get),
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Verification of the mechanism against the assembled circulant operators
# ---------------------------------------------------------------------------


def assemble_d1(route: dict, N: int, h: float) -> np.ndarray:
    """Return the assembled periodic first derivative of a banded route."""
    mass, advection = route["entries"]
    return sa.compact_derivative_matrix(mass, advection, N, h=h)


def symbol_agreement(route: dict, N: int) -> float:
    """Return the largest departure of D1 e^{i k x} from (i / h) kappa_1 e^{i k x}."""
    h = 2.0 * np.pi / N
    x = h * np.arange(N)
    theta = 2.0 * np.pi * np.fft.fftfreq(N)
    if route["spectral"]:
        return 0.0
    mat = assemble_d1(route, N, h)
    worst = 0.0
    kap = route["kappa1"](theta)
    for index, th in enumerate(theta):
        mode = np.exp(1j * th * np.arange(N))
        residual = mat @ mode - (1j / h) * kap[index] * mode
        worst = max(worst, float(np.max(np.abs(residual))))
    del x
    return worst


def rk3_time_loop(route: dict, N: int, horizon: float, nu: float,
                  field: np.ndarray) -> np.ndarray:
    """Advance a real field by genuine SSP-RK3 steps of the assembled operator."""
    h = 2.0 * np.pi / N
    dt = nu * h
    if route["spectral"]:
        kx = np.fft.fftfreq(N) * N

        def apply_d1(u):
            return np.real(np.fft.ifft(1j * kx * np.fft.fft(u)))
    else:
        mat = assemble_d1(route, N, h)

        def apply_d1(u):
            return mat @ u

    total = horizon * 2.0 * np.pi
    n_full = int(np.floor(total / dt))
    remainder = total - n_full * dt
    u = field.copy()

    def step(u, step_dt):
        u1 = u - step_dt * apply_d1(u)
        u2 = 0.75 * u + 0.25 * (u1 - step_dt * apply_d1(u1))
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 - step_dt * apply_d1(u2))

    for _ in range(n_full):
        u = step(u, dt)
    if remainder > 0.0:
        u = step(u, remainder)
    return u


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def metrics_1d(pk: dict, N: int, factor: np.ndarray, theta_c: float) -> dict:
    """Return the packet metrics of one 1D cell from its relative mode factors."""
    k = pk["k"]
    amp = pk["amp"]
    exact_analytic = analytic_signal(N, k, amp.astype(complex))
    num_analytic = analytic_signal(N, k, amp * factor)
    exact = 2.0 * np.real(exact_analytic)
    num = 2.0 * np.real(num_analytic)
    denom = float(np.linalg.norm(exact))
    centroid_exact = circular_centroid(np.abs(exact_analytic) ** 2)
    centroid_num = circular_centroid(np.abs(num_analytic) ** 2)
    lag = wrap_to_half((centroid_exact - centroid_num) / (2.0 * np.pi))
    # The energy weighted mean phase of the numerical packet against the exact
    # one. By Parseval this overlap equals sum_k |A_k|^2 f(theta_k), so its
    # argument is the carrier phase lag of the band, positive for a lag.
    overlap = complex(np.sum(num_analytic * np.conj(exact_analytic)))
    # The carrier mode itself, read out of the discrete transform of the two
    # real fields. When N theta_c / 2 pi is an integer this mode sits exactly at
    # theta_c and its phase is the frozen prediction -H N theta_c e_1 modulo
    # 2 pi, with nothing averaged over the band.
    k_exact = N * theta_c / (2.0 * np.pi)
    k_star = int(round(k_exact))
    coef_num = np.fft.fft(num)[k_star]
    coef_exact = np.fft.fft(exact)[k_star]
    return {
        "carrier_mode_k": k_star,
        "carrier_mode_is_theta_c": bool(abs(k_exact - k_star) < 1e-9),
        "carrier_phase_lag_mode_rad_wrapped":
            float(np.angle(coef_num / coef_exact)),
        "rel_l2": float(np.linalg.norm(num - exact) / denom),
        "energy_ratio": float(np.linalg.norm(num) / denom),
        "amplitude_ratio": float(np.max(np.abs(num_analytic))
                                 / np.max(np.abs(exact_analytic))),
        "centroid_lag_boxes_wrapped": lag,
        "carrier_phase_lag_rad_wrapped": float(np.angle(overlap)),
    }


def run_1d(routes: list[dict], prediction: dict, halved: bool) -> list[dict]:
    """Return one row per route, grid, packet center and horizon."""
    rows = []
    for N in GRIDS_1D:
        nu_base = prediction["time_stepping"]["cfl_by_grid"][str(N)]["cfl"]
        nu = 0.5 * nu_base if halved else nu_base
        for theta_c in THETA_CENTERS:
            pk = packet(N, theta_c)
            live = active_mask(pk["amp"])
            theta = pk["theta"][live]
            amp = pk["amp"][live]
            sub = {"k": pk["k"][live], "amp": amp}
            for route in routes:
                kap = route["kappa1"](theta)
                for H in HORIZONS_1D:
                    factor, temporal = relative_factor(kap, theta, N, H, nu)
                    values = metrics_1d(sub, N, factor, theta_c)
                    key = f"1d|{route['key']}|{N}|{theta_c:.6f}"
                    pred = prediction["cells_1d"][key]
                    rows.append({
                        "key": f"{key}|{H}|cfl{'half' if halved else 'base'}",
                        "geometry": "1d", "route": route["key"], "N": N,
                        "theta_c": theta_c,
                        "points_per_wavelength": 2.0 * np.pi / theta_c,
                        "horizon": H, "cfl": nu,
                        "cfl_setting": "half" if halved else "base",
                        "steps_full": step_plan(N, H, nu)[0],
                        "rel_l2": values["rel_l2"],
                        "energy_ratio": values["energy_ratio"],
                        "amplitude_ratio": values["amplitude_ratio"],
                        "centroid_lag_boxes_wrapped":
                            values["centroid_lag_boxes_wrapped"],
                        "carrier_phase_lag_rad_wrapped":
                            values["carrier_phase_lag_rad_wrapped"],
                        "carrier_mode_k": values["carrier_mode_k"],
                        "carrier_mode_is_theta_c":
                            values["carrier_mode_is_theta_c"],
                        "carrier_phase_lag_mode_rad_wrapped":
                            values["carrier_phase_lag_mode_rad_wrapped"],
                        "predicted_carrier_phase_lag_rad":
                            pred["carrier_phase_lag_rad"][str(H)],
                        "predicted_carrier_phase_lag_rad_wrapped":
                            wrap_to_pi(pred["carrier_phase_lag_rad"][str(H)]),
                        "derived_group_lag_boxes":
                            band_group_lag(route["kappa1"], theta, amp, H),
                        "derived_group_lag_boxes_wrapped":
                            wrap_to_half(band_group_lag(route["kappa1"], theta,
                                                        amp, H)),
                        "derived_group_lag_boxes_at_theta_c":
                            -H * float(group_velocity_error(route["kappa1"],
                                                            np.array([theta_c]))[0]),
                        "max_temporal_amplitude_defect":
                            float(np.max(np.abs(temporal.real))),
                        "max_temporal_phase_defect":
                            float(np.max(np.abs(temporal.imag))),
                        "predicted_rel_l2": pred["errors"][str(H)],
                        "predicted_centroid_lag_boxes":
                            pred["centroid_lag_boxes"][str(H)],
                        "predicted_centroid_lag_boxes_wrapped":
                            wrap_to_half(pred["centroid_lag_boxes"][str(H)]),
                        "e1_at_theta_c": pred["e1_at_theta_c"],
                    })
    return rows


def run_3d(routes: list[dict], prediction: dict) -> list[dict]:
    """Return one row per route and horizon for the tensor-product 3D packet."""
    N = GRID_3D
    theta_c = prediction["cells_3d"]["theta_c"]
    nu = prediction["cells_3d"]["cfl"]
    hat, field0 = packet_3d_spectrum(N, theta_c)
    kvec = np.fft.fftfreq(N) * N
    rows = []
    for route in routes:
        sym, exact = symbol_sums_3d(route["kappa1"], N, VELOCITY_3D)
        for H in HORIZONS_3D:
            factor, temporal = relative_factor(sym, exact, N, H, nu)
            shift = np.exp(-1j * 2.0 * np.pi * H
                           * (VELOCITY_3D[0] * kvec[:, None, None]
                              + VELOCITY_3D[1] * kvec[None, :, None]
                              + VELOCITY_3D[2] * kvec[None, None, :]))
            exact_field = np.real(np.fft.ifftn(hat * shift))
            num_field = np.real(np.fft.ifftn(hat * shift * factor))
            denom = float(np.linalg.norm(exact_field))
            weight_num = num_field ** 2
            weight_exact = exact_field ** 2
            lags = []
            for axis in range(3):
                axes = tuple(a for a in range(3) if a != axis)
                ce = circular_centroid(weight_exact.sum(axis=axes))
                cn = circular_centroid(weight_num.sum(axis=axes))
                lags.append(wrap_to_half((ce - cn) / (2.0 * np.pi)))
            key = f"3d|{route['key']}|{N}|{theta_c:.6f}"
            pred = prediction["cells_3d"]["cells"][key]
            rows.append({
                "key": f"{key}|{H}|cflbase",
                "geometry": "3d", "route": route["key"], "N": N,
                "theta_c": theta_c,
                "points_per_wavelength": 2.0 * np.pi / theta_c,
                "horizon": H, "cfl": nu, "cfl_setting": "base",
                "steps_full": step_plan(N, H, nu)[0],
                "rel_l2": float(np.linalg.norm(num_field - exact_field) / denom),
                "energy_ratio": float(np.linalg.norm(num_field) / denom),
                "amplitude_ratio": float(np.max(np.abs(num_field))
                                         / np.max(np.abs(exact_field))),
                "centroid_lag_boxes_wrapped": lags[0],
                "centroid_lag_boxes_wrapped_y": lags[1],
                "centroid_lag_boxes_wrapped_z": lags[2],
                "derived_group_lag_boxes":
                    -H * float(group_velocity_error(route["kappa1"],
                                                    np.array([theta_c]))[0]),
                "derived_group_lag_boxes_wrapped":
                    wrap_to_half(-H * float(group_velocity_error(
                        route["kappa1"], np.array([theta_c]))[0])),
                "derived_group_lag_boxes_at_theta_c":
                    -H * float(group_velocity_error(route["kappa1"],
                                                    np.array([theta_c]))[0]),
                "max_temporal_amplitude_defect":
                    float(np.max(np.abs(temporal.real))),
                "max_temporal_phase_defect": float(np.max(np.abs(temporal.imag))),
                "predicted_rel_l2": pred["errors"][str(H)],
                "predicted_centroid_lag_boxes":
                    pred["centroid_lag_boxes_per_axis"][str(H)][0],
                "predicted_centroid_lag_boxes_wrapped":
                    wrap_to_half(pred["centroid_lag_boxes_per_axis"][str(H)][0]),
                "e1_at_theta_c": pred["e1_at_theta_c"],
            })
    del field0
    return rows


def mechanism_checks(routes: list[dict], prediction: dict) -> dict:
    """Return the operator, stepper and sign checks that close the mechanism."""
    N = 64
    nu = prediction["time_stepping"]["cfl_by_grid"][str(N)]["cfl"]
    theta_c = 0.5 * np.pi
    pk = packet(N, theta_c)
    live = active_mask(pk["amp"])
    sub = {"k": pk["k"][live], "amp": pk["amp"][live]}
    field0 = real_field(N, sub["k"], sub["amp"])

    symbol = {}
    stepper = {}
    for route in routes:
        symbol[route["key"]] = symbol_agreement(route, N)
        started = time.time()
        looped = rk3_time_loop(route, N, 1.0, nu, field0)
        kap = route["kappa1"](pk["theta"][live])
        factor, _ = relative_factor(kap, pk["theta"][live], N, 1, nu)
        closed = real_field(N, sub["k"], sub["amp"] * factor)
        stepper[route["key"]] = {
            "relative_difference": float(np.linalg.norm(looped - closed)
                                         / np.linalg.norm(closed)),
            "steps": step_plan(N, 1, nu)[0],
            "seconds": time.time() - started,
        }

    # The comparison band brackets every y = (dt / h) kappa_1 the runs use, and
    # is bounded below because log g(-i y) + i y is of size y^4 / 24, which the
    # 64 bit significand of the extended type cannot resolve for smaller y.
    ys = np.linspace(5.0e-4, 1.0e-2, 4001)
    series = rk3_log_defect(ys)
    exact = rk3_log_defect_longdouble(ys)
    defect_check = {
        "y_range": [float(ys.min()), float(ys.max())],
        "max_relative_difference": float(np.max(
            np.abs(series - exact.astype(complex)) / np.abs(exact.astype(complex)))),
    }
    return {"symbol_vs_assembled_operator_N64": symbol,
            "rk3_series_vs_extended_precision_logarithm": defect_check,
            "closed_form_vs_time_loop_one_crossing_N64": stepper,
            "cfl_used": nu}


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

CSV_FIELDS = ("key", "geometry", "route", "N", "theta_c", "points_per_wavelength",
              "horizon", "cfl", "cfl_setting", "steps_full", "rel_l2",
              "energy_ratio", "amplitude_ratio", "centroid_lag_boxes_wrapped",
              "centroid_lag_boxes_wrapped_y", "centroid_lag_boxes_wrapped_z",
              "carrier_phase_lag_rad_wrapped", "carrier_mode_k",
              "carrier_mode_is_theta_c", "carrier_phase_lag_mode_rad_wrapped",
              "predicted_carrier_phase_lag_rad",
              "predicted_carrier_phase_lag_rad_wrapped",
              "derived_group_lag_boxes", "derived_group_lag_boxes_wrapped",
              "derived_group_lag_boxes_at_theta_c",
              "max_temporal_amplitude_defect", "max_temporal_phase_defect",
              "predicted_rel_l2", "predicted_centroid_lag_boxes",
              "predicted_centroid_lag_boxes_wrapped", "e1_at_theta_c")


def write_runs(rows: list[dict]) -> Path:
    path = GATE / "runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return path


def swept_crossing(route: dict, N: int, theta_c: float, nu: float,
                   level: float) -> tuple[float | None, float | None]:
    """Return the horizon at which the measured error first reaches level.

    The three contract horizons 10, 100 and 1000 lie past saturation for the
    least accurate routes, so the crossing is measured directly instead: the
    same stepper is landed at a continuous horizon, which the partial final step
    already permits, and the horizon is bisected. Returns the horizon and the
    error measured there.
    """
    pk = packet(N, theta_c)
    live = active_mask(pk["amp"])
    theta = pk["theta"][live]
    amp = pk["amp"][live]
    sub = {"k": pk["k"][live], "amp": amp}
    kap = route["kappa1"](theta)

    def measure(horizon: float) -> float:
        factor, _ = relative_factor(kap, theta, N, horizon, nu)
        return metrics_1d(sub, N, factor, theta_c)["rel_l2"]

    lo, hi = 1.0e-5, 1.0e-5
    if measure(lo) >= level:
        return None, None
    for _ in range(400):
        hi *= 1.1
        if hi > 1.0e7:
            return None, None
        if measure(hi) >= level:
            break
    else:
        return None, None
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if measure(mid) >= level:
            hi = mid
        else:
            lo = mid
    mid = np.sqrt(lo * hi)
    return float(mid), float(measure(mid))


def write_crossing_sweep(routes: list[dict], prediction: dict) -> Path:
    """Write the measured 1 and 10 percent crossing horizons of every 1D cell."""
    path = GATE / "crossing_sweep.csv"
    fields = ("route", "N", "theta_c", "level", "cfl", "predicted_horizon",
              "measured_horizon", "error_at_measured_horizon")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for N in GRIDS_1D:
            nu = prediction["time_stepping"]["cfl_by_grid"][str(N)]["cfl"]
            for theta_c in THETA_CENTERS:
                for route in routes:
                    key = f"1d|{route['key']}|{N}|{theta_c:.6f}"
                    cell = prediction["cells_1d"][key]
                    for level, name in zip(ERROR_LEVELS,
                                           ("horizon_at_1_percent",
                                            "horizon_at_10_percent")):
                        horizon, error = swept_crossing(route, N, theta_c, nu,
                                                        level)
                        writer.writerow({
                            "route": route["key"], "N": N, "theta_c": theta_c,
                            "level": level, "cfl": nu,
                            "predicted_horizon": cell[name],
                            "measured_horizon": horizon,
                            "error_at_measured_horizon": error,
                        })
    return path


def read_crossing_sweep() -> dict:
    """Return the measured crossing horizons keyed by cell and level."""
    path = GATE / "crossing_sweep.csv"
    out = {}
    if not path.exists():
        return out
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["route"], int(row["N"]), float(row["theta_c"]),
                   float(row["level"]))
            out[key] = (float(row["measured_horizon"])
                        if row["measured_horizon"] else None)
    return out


def realized_crossing(rows: list[dict], route: str, N: int, theta_c: float,
                      level: float) -> float | None:
    """Return the horizon at which the realized error first reaches level.

    The three measured horizons are interpolated in log H and log error, which
    is the exact interpolation while the error is linear in H.
    """
    picked = sorted((r for r in rows
                     if r["geometry"] == "1d" and r["route"] == route
                     and r["N"] == N and abs(r["theta_c"] - theta_c) < 1e-12
                     and r["cfl_setting"] == "base"),
                    key=lambda r: r["horizon"])
    if not picked:
        return None
    hs = np.array([r["horizon"] for r in picked], dtype=float)
    es = np.array([r["rel_l2"] for r in picked], dtype=float)
    if es[0] >= level:
        slope = 1.0
        return float(hs[0] * (level / es[0]) ** (1.0 / slope))
    for i in range(len(hs) - 1):
        if es[i] < level <= es[i + 1]:
            t = (np.log(level) - np.log(es[i])) / (np.log(es[i + 1]) - np.log(es[i]))
            return float(np.exp(np.log(hs[i]) + t * (np.log(hs[i + 1])
                                                     - np.log(hs[i]))))
    if es[-1] < level:
        if es[-1] <= 0.0:
            return None
        slope = ((np.log(es[-1]) - np.log(es[-2]))
                 / (np.log(hs[-1]) - np.log(hs[-2])))
        if slope <= 0.0:
            return None
        return float(hs[-1] * (level / es[-1]) ** (1.0 / slope))
    return None


def build_decision(rows: list[dict], prediction: dict, checks: dict,
                   routes: list[dict]) -> dict:
    """Return the recorded comparison of realized against predicted."""
    # Fourier is exact in space, so its prediction is identically zero and no
    # ratio against it exists. Those cells are kept apart and read as the
    # temporal ceiling instead of being compared with a zero.
    flags = []
    exact_in_space = []
    for row in rows:
        pred = row["predicted_rel_l2"]
        got = row["rel_l2"]
        if pred <= 0.0:
            exact_in_space.append({"key": row["key"], "route": row["route"],
                                   "realized": got})
            continue
        ratio = got / pred
        if not (1.0 / AGREEMENT_FACTOR) <= ratio <= AGREEMENT_FACTOR:
            flags.append({"key": row["key"], "route": row["route"],
                          "geometry": row["geometry"], "N": row["N"],
                          "theta_c": row["theta_c"], "horizon": row["horizon"],
                          "realized": got, "predicted": pred, "ratio": ratio})

    base = [r for r in rows if r["cfl_setting"] == "base"]
    halved = {r["key"].replace("|cflhalf", "|cflbase"): r
              for r in rows if r["cfl_setting"] == "half"}
    halving = []
    halving_dispersive = []
    halving_spectral = []
    for row in base:
        mate = halved.get(row["key"])
        if mate is None:
            continue
        denom = max(row["rel_l2"], 1.0e-300)
        change = abs(mate["rel_l2"] - row["rel_l2"]) / denom
        halving.append(change)
        if row["predicted_rel_l2"] > 0.0:
            halving_dispersive.append(change)
        else:
            halving_spectral.append(change)
    crossings = {}
    sweep = read_crossing_sweep()
    for route in routes:
        for N in GRIDS_1D:
            for theta_c in THETA_CENTERS:
                key = f"1d|{route['key']}|{N}|{theta_c:.6f}"
                pred = prediction["cells_1d"][key]
                crossings[key] = {
                    "predicted_1_percent": pred["horizon_at_1_percent"],
                    "predicted_10_percent": pred["horizon_at_10_percent"],
                    "measured_1_percent": sweep.get(
                        (route["key"], N, theta_c, ERROR_LEVELS[0])),
                    "measured_10_percent": sweep.get(
                        (route["key"], N, theta_c, ERROR_LEVELS[1])),
                    "interpolated_from_the_three_horizons_1_percent":
                        realized_crossing(rows, route["key"], N, theta_c,
                                          ERROR_LEVELS[0]),
                    "interpolated_from_the_three_horizons_10_percent":
                        realized_crossing(rows, route["key"], N, theta_c,
                                          ERROR_LEVELS[1]),
                }

    sign_row = next(r for r in rows
                    if r["key"].startswith("1d|champion|128|1.570796|10|cflbase"))
    sign_check = {
        "cell": sign_row["key"],
        "convention": ("e_1 = kappa_1 / theta - 1; the numerical mode carries "
                       "exp(-i H N theta e_1) relative to the exact one, so a "
                       "positive argument is a backward displacement"),
        "e1_at_theta_c": sign_row["e1_at_theta_c"],
        "predicted_carrier_phase_lag_rad":
            sign_row["predicted_carrier_phase_lag_rad"],
        "predicted_carrier_phase_lag_rad_wrapped":
            sign_row["predicted_carrier_phase_lag_rad_wrapped"],
        "realized_carrier_phase_lag_at_the_carrier_mode_rad":
            sign_row["carrier_phase_lag_mode_rad_wrapped"],
        "realized_carrier_phase_lag_averaged_over_the_band_rad":
            sign_row["carrier_phase_lag_rad_wrapped"],
        "carrier_mode_absolute_difference_rad": abs(
            sign_row["carrier_phase_lag_mode_rad_wrapped"]
            - sign_row["predicted_carrier_phase_lag_rad_wrapped"]),
        "carrier_signs_agree": bool(
            np.sign(sign_row["predicted_carrier_phase_lag_rad_wrapped"])
            == np.sign(sign_row["carrier_phase_lag_mode_rad_wrapped"])),
    }

    # The envelope centroid travels at the group velocity, not at the phase
    # velocity, so it is read against d kappa_1 / d theta - 1 derived from the
    # same frozen symbols after the runs, and not against the frozen carrier
    # displacement -H e_1 that prediction.json stores as centroid_lag_boxes.
    mode_lag = []
    band_lag = []
    group = []
    for row in rows:
        if row["geometry"] != "1d" or row["cfl_setting"] != "base":
            continue
        if row["carrier_phase_lag_rad_wrapped"] is None:
            continue
        if row["carrier_mode_is_theta_c"]:
            mode_lag.append(abs(wrap_to_pi(
                row["carrier_phase_lag_mode_rad_wrapped"]
                - row["predicted_carrier_phase_lag_rad_wrapped"])))
        if row["rel_l2"] <= 0.1:
            band_lag.append(abs(wrap_to_pi(
                row["carrier_phase_lag_rad_wrapped"]
                - row["predicted_carrier_phase_lag_rad_wrapped"])))
            if row["horizon"] == 10:
                group.append(abs(wrap_to_pi(
                    2.0 * np.pi * (row["centroid_lag_boxes_wrapped"]
                                   - row["derived_group_lag_boxes_wrapped"]))
                    / (2.0 * np.pi)))
    lag_check = {
        "carrier_phase_lag_at_the_carrier_mode": {
            "definition": ("phase of the discrete transform of the numerical field "
                           "at the carrier mode against that of the exact field, "
                           "compared with the frozen prediction -H N theta_c e_1, "
                           "both wrapped to (-pi, pi]; restricted to the cells "
                           "whose carrier mode lands exactly on theta_c, which is "
                           "every theta_c = 0.5 pi cell"),
            "cells": len(mode_lag),
            "max_absolute_difference_rad":
                float(max(mode_lag)) if mode_lag else None,
        },
        "carrier_phase_lag_averaged_over_the_band": {
            "definition": ("argument of the energy weighted overlap of the "
                           "numerical analytic signal with the exact one, against "
                           "the same frozen prediction, over the cells whose "
                           "relative L2 error is at most 0.1 so the packet is "
                           "still coherent; the residue is the curvature of e_1 "
                           "across the band, not a failure of the prediction"),
            "cells": len(band_lag),
            "max_absolute_difference_rad":
                float(max(band_lag)) if band_lag else None,
        },
        "envelope_centroid_lag": {
            "definition": ("circular centroid of the envelope energy against the "
                           "group prediction, the energy weighted band mean of "
                           "-H (d kappa_1 / d theta - 1), derived after the runs "
                           "from the frozen symbols, at H = 10 over the coherent "
                           "cells; the column at theta_c alone is carried beside "
                           "it in runs.csv and is the poorer of the two because "
                           "d kappa_1 / d theta varies across the band"),
            "frozen_field_note": ("prediction.json's centroid_lag_boxes = -H e_1 "
                                  "is the carrier displacement, not the envelope "
                                  "centroid displacement"),
            "cells": len(group),
            "max_absolute_difference_boxes": float(max(group)) if group else None,
        },
    }

    fourier = [r for r in base if r["route"] == "fourier"]
    dissipation = [r for r in base if r["route"] != "fourier"]
    return {
        "producer": "code/analysis/exponential_family/gate4b_long_horizon.py --reduce",
        "written_utc": utc_now(),
        "contract_sha256": sha256(CONTRACT),
        "rule": ("Recorded, no rule. The realized relative L2 error of every cell "
                 "is compared with the prediction written before the runs; a cell "
                 "whose ratio leaves the interval [1/2, 2] is flagged."),
        "rows": len(rows),
        "agreement_factor": AGREEMENT_FACTOR,
        "cells_flagged": flags,
        "cells_flagged_count": len(flags),
        "cells_compared": len(rows) - len(exact_in_space),
        "cells_with_zero_prediction": {
            "count": len(exact_in_space),
            "note": ("the Fourier route is exact in space, so its prediction is "
                     "zero and its realized error is the temporal ceiling"),
            "max_realized": float(max((c["realized"] for c in exact_in_space),
                                      default=0.0)),
        },
        "worst_multiplicative_departure_realized_from_predicted": float(np.exp(
            max((abs(np.log(max(r["rel_l2"], 1e-300)
                            / max(r["predicted_rel_l2"], 1e-300)))
                 for r in rows if r["predicted_rel_l2"] > 0.0), default=0.0))),
        "cfl_halving_check": {
            "definition": ("every 1D cell repeated at cfl / 2; the relative change "
                           "of the realized error is reported"),
            "cells": len(halving),
            "max_relative_change": float(max(halving)) if halving else None,
            "max_relative_change_dispersive_routes":
                float(max(halving_dispersive)) if halving_dispersive else None,
            "max_relative_change_fourier": (
                float(max(halving_spectral)) if halving_spectral else None),
            "fourier_note": ("the Fourier error is purely temporal and its "
                             "leading term is the RK3 amplitude defect "
                             "N nu^3 kappa_1^4 / 24 per crossing, so halving the "
                             "cfl divides it by eight, a relative change of 7/8"),
        },
        "temporal_ceiling": {
            "definition": ("Fourier is exact in space, so its realized error is the "
                           "temporal error alone"),
            "max_rel_l2": float(max((r["rel_l2"] for r in fourier), default=0.0)),
            "max_rel_l2_1d": float(max((r["rel_l2"] for r in fourier
                                        if r["geometry"] == "1d"), default=0.0)),
            "max_rel_l2_3d": float(max((r["rel_l2"] for r in fourier
                                        if r["geometry"] == "3d"), default=0.0)),
        },
        "dissipation_check": {
            "definition": ("every route's spatial operator is antisymmetric, so the "
                           "only amplitude change is the RK3 one; the L2 norm ratio "
                           "of the numerical to the exact packet is reported"),
            "max_departure_of_energy_ratio_from_one": float(max(
                abs(r["energy_ratio"] - 1.0) for r in dissipation)),
        },
        "sign_convention_check": sign_check,
        "lag_checks": lag_check,
        "mechanism": checks,
        "crossing_horizons": {
            "definition": ("predicted_* comes from prediction.json, which is the "
                           "symbol formula alone; measured_* is a bisection in a "
                           "continuous horizon using the same stepper and the "
                           "exact time landing, recorded in crossing_sweep.csv; "
                           "interpolated_* reads the crossing off the three "
                           "contract horizons 10, 100 and 1000 and is a back "
                           "extrapolation, hence unreliable for the routes whose "
                           "crossing lies below H = 10"),
            "cells": crossings,
        },
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

ROUTE_STYLE = {
    "champion": {"color": "#0072B2", "ls": "-", "marker": "o"},
    "slc_q4": {"color": "#D55E00", "ls": "--", "marker": "s"},
    "tuned_q8": {"color": "#CC79A7", "ls": "--", "marker": "D"},
    "cw12b": {"color": "#8C510A", "ls": "--", "marker": "^"},
    "bmc9": {"color": "#6BAED6", "ls": "-.", "marker": "v"},
    "bmc11": {"color": "#08306B", "ls": "-.", "marker": "P"},
    "fourier": {"color": "#4D4D4D", "ls": ":", "marker": "x"},
}
TAIL_STYLE = {"color": "#1B7837", "ls": "-", "marker": "*"}


def style_for(key: str) -> dict:
    return ROUTE_STYLE.get(key, TAIL_STYLE)


def make_figure(rows: list[dict], prediction: dict, routes: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(CODE))
    import plotting_helpers as ph  # noqa: E402

    ph.apply_classy_defaults()
    theta_c = 0.5 * np.pi
    N = 128
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

    ax = axes[0]
    horizons = np.exp(np.linspace(np.log(0.1), np.log(4000.0), 500))
    for route in routes:
        pk = packet(N, theta_c)
        live = active_mask(pk["amp"])
        theta = pk["theta"][live]
        amp = pk["amp"][live]
        e1 = relative_symbol_error(route["kappa1"], theta)
        curve = np.array([predicted_error(amp, theta, e1, N, H) for H in horizons])
        st = style_for(route["key"])
        ax.plot(horizons, np.maximum(curve, 1e-16), color=st["color"], ls=st["ls"],
                lw=1.1, alpha=0.85, zorder=2)
        got = sorted((r for r in rows if r["geometry"] == "1d"
                      and r["route"] == route["key"] and r["N"] == N
                      and abs(r["theta_c"] - theta_c) < 1e-12
                      and r["cfl_setting"] == "base"),
                     key=lambda r: r["horizon"])
        ax.plot([r["horizon"] for r in got],
                [max(r["rel_l2"], 1e-16) for r in got],
                color=st["color"], ls="none", marker=st["marker"], ms=5.0,
                label=route["display"], zorder=3)
    for level, text in ((0.01, "1 percent"), (0.10, "10 percent")):
        ax.axhline(level, color=ph.CLASSY_GUIDE, lw=0.9, zorder=1)
        ax.text(2600.0, level * 1.25, text, fontsize=7, color="0.45", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 4000.0)
    ax.set_ylim(1e-8, 3.0)
    ax.set_xlabel("box crossings $H$")
    ax.set_ylabel(r"relative $L^2$ packet error")
    ax.set_title(r"$N = 128$, $\theta_c = 0.5\pi$; lines predicted, "
                 r"markers realized", fontsize=9)
    ph.quiet_axis(ax)

    ax = axes[1]
    ppw_curve = np.linspace(3.0, 12.0, 220)
    for route in routes:
        st = style_for(route["key"])
        curve = []
        for ppw in ppw_curve:
            tc = 2.0 * np.pi / ppw
            pk = packet(N, tc)
            live = active_mask(pk["amp"])
            theta = pk["theta"][live]
            amp = pk["amp"][live]
            e1 = relative_symbol_error(route["kappa1"], theta)
            curve.append(predicted_error(amp, theta, e1, N, 100.0))
        ax.plot(ppw_curve, np.maximum(curve, 1e-16), color=st["color"],
                ls=st["ls"], lw=1.1, alpha=0.85, zorder=2)
        got = sorted((r for r in rows if r["geometry"] == "1d"
                      and r["route"] == route["key"] and r["N"] == N
                      and r["horizon"] == 100 and r["cfl_setting"] == "base"),
                     key=lambda r: r["points_per_wavelength"])
        ax.plot([r["points_per_wavelength"] for r in got],
                [max(r["rel_l2"], 1e-16) for r in got],
                color=st["color"], ls="none", marker=st["marker"], ms=5.0,
                label=route["display"], zorder=3)
    for level in ERROR_LEVELS:
        ax.axhline(level, color=ph.CLASSY_GUIDE, lw=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(3.0, 12.0)
    ax.set_ylim(1e-8, 3.0)
    ax.set_xlabel("points per wavelength $2\\pi/\\theta_c$")
    ax.set_ylabel(r"relative $L^2$ packet error at $H = 100$")
    ax.set_title(r"$N = 128$, $H = 100$; lines predicted, "
                 r"markers realized", fontsize=9)
    ph.quiet_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = FIGDIR / "fig_long_horizon.pdf"
    fig.savefig(path)
    plt.close(fig)
    del prediction
    return path


def build_tail_prediction(tails: list[dict], frozen: dict) -> dict:
    """Return the prediction addendum for the Gate 4f tail member.

    The frozen cfl of every grid is reused unchanged. That is legitimate because
    the binding route of each grid is Fourier, whose kappa_1 = theta is the
    largest modified wavenumber any route can carry inside the band, so a member
    added later cannot tighten the time step.
    """
    cells = {}
    checks = {}
    for N in GRIDS_1D:
        nu = frozen["time_stepping"]["cfl_by_grid"][str(N)]["cfl"]
        for theta_c in THETA_CENTERS:
            pk = packet(N, theta_c)
            live = active_mask(pk["amp"])
            theta = pk["theta"][live]
            amp = pk["amp"][live]
            for route in tails:
                smax = float(np.max(np.abs(route["kappa1"](theta))))
                amp_err, phase_err = temporal_errors_per_crossing(nu, N, smax)
                checks.setdefault(str(N), {})[route["key"]] = {
                    "largest_active_kappa1": smax,
                    "cfl_ceiling_for_route": cfl_from_symbol(N, smax),
                    "amplitude_error_per_crossing": amp_err,
                    "phase_error_per_crossing": phase_err,
                    "below_target": bool(max(amp_err, phase_err)
                                         < TEMPORAL_TARGET),
                    "frozen_cfl_still_admissible": bool(
                        nu <= cfl_from_symbol(N, smax)),
                }
                e1 = relative_symbol_error(route["kappa1"], theta)
                e1_center = float(relative_symbol_error(
                    route["kappa1"], np.array([theta_c]))[0])
                cells[f"1d|{route['key']}|{N}|{theta_c:.6f}"] = {
                    "N": N, "theta_c": theta_c, "route": route["key"],
                    "e1_at_theta_c": e1_center,
                    "kappa1_at_theta_c": float(route["kappa1"](
                        np.array([theta_c]))[0]),
                    "errors": {str(H): predicted_error(amp, theta, e1, N, H)
                               for H in HORIZONS_1D},
                    "carrier_phase_lag_rad": {
                        str(H): -H * N * theta_c * e1_center
                        for H in HORIZONS_1D},
                    "centroid_lag_boxes": {str(H): -H * e1_center
                                           for H in HORIZONS_1D},
                    "horizon_at_1_percent": crossing_horizon(
                        amp, theta, e1, N, ERROR_LEVELS[0]),
                    "horizon_at_10_percent": crossing_horizon(
                        amp, theta, e1, N, ERROR_LEVELS[1]),
                }
    three = predict_3d(tails)
    three["cfl"] = frozen["cells_3d"]["cfl"]
    return {
        "producer": ("code/analysis/exponential_family/gate4b_long_horizon.py "
                     "--predict-tail"),
        "written_utc": utc_now(),
        "note": ("addendum to prediction.json for the Gate 4f tail member, which "
                 "did not exist when the gate's prediction was frozen; the cfl, "
                 "the packet and every convention are those of prediction.json"),
        "frozen_prediction_sha256": sha256(GATE / "prediction.json"),
        "members": [route.get("member_path") for route in tails],
        "cfl_check": checks,
        "cells_1d": cells,
        "cells_3d": three,
    }


def load_prediction() -> dict:
    """Return prediction.json merged with the tail member addendum if present."""
    payload = json.loads((GATE / "prediction.json").read_text())
    addendum = GATE / "prediction_tail_member.json"
    if addendum.exists():
        extra = json.loads(addendum.read_text())
        payload["cells_1d"].update(extra["cells_1d"])
        payload["cells_3d"]["cells"].update(extra["cells_3d"]["cells"])
        payload["tail_member_present"] = True
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict-tail", dest="predict_tail",
                        action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--reduce", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        args.predict = args.run = args.reduce = True

    GATE.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    routes = route_table()
    prediction_path = GATE / "prediction.json"

    if args.predict:
        if prediction_path.exists():
            raise SystemExit(f"{prediction_path} exists; predictions are frozen.")
        payload = build_prediction(routes)
        prediction_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {prediction_path.relative_to(ROOT)}")

    if args.predict_tail:
        tails = tail_member_routes()
        if not tails:
            raise SystemExit("No Gate 4f member JSON under "
                             f"{GATE4F_MEMBERS.relative_to(ROOT)}.")
        addendum = GATE / "prediction_tail_member.json"
        if addendum.exists():
            raise SystemExit(f"{addendum} exists; predictions are frozen.")
        frozen = json.loads(prediction_path.read_text())
        addendum.write_text(json.dumps(build_tail_prediction(tails, frozen),
                                       indent=2) + "\n")
        print(f"wrote {addendum.relative_to(ROOT)}")

    prediction = load_prediction()

    rows_path = GATE / "runs.csv"
    if args.run:
        started = time.time()
        rows = run_1d(routes, prediction, halved=False)
        rows += run_1d(routes, prediction, halved=True)
        rows += run_3d(routes, prediction)
        write_runs(rows)
        print(f"wrote {rows_path.relative_to(ROOT)} with {len(rows)} rows "
              f"in {time.time() - started:.1f} s")

    if args.reduce:
        rows = []
        with rows_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                typed = dict(row)
                for field in ("N", "horizon", "steps_full"):
                    typed[field] = int(row[field])
                for field in ("theta_c", "points_per_wavelength", "cfl", "rel_l2",
                              "energy_ratio", "amplitude_ratio",
                              "centroid_lag_boxes_wrapped",
                              "derived_group_lag_boxes",
                              "derived_group_lag_boxes_wrapped",
                              "derived_group_lag_boxes_at_theta_c",
                              "max_temporal_amplitude_defect",
                              "max_temporal_phase_defect", "predicted_rel_l2",
                              "predicted_centroid_lag_boxes",
                              "predicted_centroid_lag_boxes_wrapped",
                              "e1_at_theta_c"):
                    typed[field] = float(row[field])
                for field in ("carrier_phase_lag_rad_wrapped",
                              "carrier_phase_lag_mode_rad_wrapped",
                              "predicted_carrier_phase_lag_rad",
                              "predicted_carrier_phase_lag_rad_wrapped"):
                    typed[field] = (float(row[field]) if row[field] != ""
                                    else None)
                typed["carrier_mode_is_theta_c"] = (
                    row["carrier_mode_is_theta_c"] == "True")
                rows.append(typed)
        sweep_path = write_crossing_sweep(routes, prediction)
        print(f"wrote {sweep_path.relative_to(ROOT)}")
        checks = mechanism_checks(routes, prediction)
        decision = build_decision(rows, prediction, checks, routes)
        (GATE / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        path = make_figure(rows, prediction, routes)
        print(f"wrote {(GATE / 'decision.json').relative_to(ROOT)} and "
              f"{path.relative_to(ROOT)}")
        print(f"flagged cells: {decision['cells_flagged_count']} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
