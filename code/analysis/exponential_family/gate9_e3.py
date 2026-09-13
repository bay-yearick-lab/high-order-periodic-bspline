"""Gate 9, E3: long-horizon transport of the everything member in closed form.

The gate adds one route to the Gate 4b table, keyed "everything", whose
symbol is read from the lattice samples of the Gate 8 member
X8_n10_C_SLC1_pert5. Every other object of the run, the packet, the horizons,
the cfl, the SSP-RK3 amplification, the crossing bisection and the three
dimensional velocity, is imported from gate4b_long_horizon.py unchanged, so
the numbers of the stored routes re-reduce exactly.

Sign convention. code/solver/EXP_ROUTE.md defines the applied first derivative
of a member by its lattice samples,

    b(theta) = beta(0) + 2 sum_{n >= 1} beta(n) cos(n theta),
    a(theta) =           2 sum_{n >= 1} beta'(n) sin(n theta),
    D1 eigenvalue on exp(i n theta) = -(i / h) a(theta) / b(theta),

while Gate 4b defines kappa_1 by D1 exp(i k x) = (i / h) kappa_1 exp(i k x).
The modified wavenumber of a member is therefore

    kappa_1(theta) = -a(theta) / b(theta),

which is the form gate8_family.symbols returns (its kappa_1 b is
-2 sum beta'(n) sin(n theta)) and the form spline_analysis builds when it
negates the one-sided derivative samples into the advection entries. The
opposite sign would put the packet at rest at leading order, because
sum_n n beta'(n) is about -1/2 for a normalized member. The prediction step
records the agreement of the three routes to 1e-13.

Steps.

--predict     writes e3_prediction.json before any other step: per contract
              carrier the ratio |e_1^{SLC}(theta_c)| / |e_1^{pert5}(theta_c)|
              and the sign of -e_1^{pert5}(theta_c), which is the sign of the
              signed displacement per crossing. The file is frozen once
              written.
--closed-form writes e3_transport.{json,md}: the horizon at which the closed
              form packet error first reaches one percent, at N = 128 in one
              dimension and on the Gate 4b three dimensional cell, for pert5,
              SLC q4, tuned q8, BMC11, CW12B and the band champion, with the
              signed displacement per crossing of each, the realized ratio
              against SLC and its departure from the frozen prediction.
--run-check   advances the pert5 packet at N = 128, theta_c = 0.5 pi and
              H = 10 by genuine SSP-RK3 steps of the assembled circulant
              built from the member file, at the frozen cfl of that grid, and
              compares it with the closed form.
--regression  recomputes the stored Gate 4b champion numbers, the closed form
              cells of prediction.json and the champion rows of runs.csv,
              through the imported functions.

Two options carry the same procedure to another member. --member <path>
fields that member file instead of pert5, and --tag <name> puts _<name>
before the extension of every output name (e3_prediction_<name>.json,
e3_transport_<name>.json, e3_transport_<name>.md) and makes <name> the route
label inside the receipts. Without them the bindings are the frozen pert5
ones, so the stored pert5 receipts re-reduce bit for bit.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate4b_long_horizon as g4b  # noqa: E402
import gate8_family as g8  # noqa: E402
import spline_analysis as sa  # noqa: E402

GATE9 = ROOT / "results" / "exponential_family" / "gate9"
GATE4B = ROOT / "results" / "exponential_family" / "gate4b"
DEFAULT_MEMBER = (ROOT / "results" / "exponential_family" / "gate8" / "members"
                  / "X8_n10_C_SLC1_pert5.json")
DEFAULT_KEY = "everything"
DEFAULT_LABEL = "pert5"
DEFAULT_DISPLAY = "pert5 everything member"

MEMBER = DEFAULT_MEMBER
MEMBER_LABEL = "EXP:" + str(DEFAULT_MEMBER)
MEMBER_KEY = DEFAULT_KEY
MEMBER_DISPLAY = DEFAULT_DISPLAY
LABEL = DEFAULT_LABEL
TAG = ""
PREDICTION_PATH = GATE9 / "e3_prediction.json"
TRANSPORT_PATH = GATE9 / "e3_transport.json"
TRANSPORT_MD = GATE9 / "e3_transport.md"

COMPARATORS = ("slc_q4", "tuned_q8", "bmc11", "cw12b", "champion")
REFERENCE = "slc_q4"
N_1D = 128
HORIZON_CHECK = 10
LEVEL = g4b.ERROR_LEVELS[0]
SYMBOL_TOL = 1.0e-13
RUN_CHECK_TOL = 1.0e-6
REGRESSION_TOL = 1.0e-12
RATIO_TOL = 0.10


def configure(member=None, tag=None) -> None:
    """Bind the member file, the route label and every output name.

    Called once by main. Without arguments the bindings are the frozen pert5
    ones of the E3 receipt, so that the stored files re-reduce bit for bit.
    With a tag every output name carries _<tag> before its extension and the
    route key and label inside the receipts are the tag.
    """
    global MEMBER, MEMBER_LABEL, MEMBER_KEY, MEMBER_DISPLAY, LABEL, TAG
    global PREDICTION_PATH, TRANSPORT_PATH, TRANSPORT_MD
    MEMBER = Path(member).resolve() if member else DEFAULT_MEMBER
    MEMBER_LABEL = "EXP:" + str(MEMBER)
    TAG = str(tag) if tag else ""
    LABEL = TAG if TAG else DEFAULT_LABEL
    MEMBER_KEY = TAG if TAG else DEFAULT_KEY
    MEMBER_DISPLAY = f"{TAG} member" if TAG else DEFAULT_DISPLAY
    suffix = f"_{TAG}" if TAG else ""
    PREDICTION_PATH = GATE9 / f"e3_prediction{suffix}.json"
    TRANSPORT_PATH = GATE9 / f"e3_transport{suffix}.json"
    TRANSPORT_MD = GATE9 / f"e3_transport{suffix}.md"
    member_samples.cache_clear()


# ---------------------------------------------------------------------------
# The member route
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def member_samples() -> tuple:
    """Return beta and beta' of the member, one sided, at full precision."""
    record = json.loads(MEMBER.read_text())
    return (np.asarray(record["beta"], dtype=float),
            np.asarray(record["dbeta"], dtype=float))


def member_kappa1(theta) -> np.ndarray:
    """Return kappa_1(theta) = -a(theta) / b(theta) of the member samples."""
    beta, dbeta = member_samples()
    theta = np.asarray(theta, dtype=float)
    n = np.arange(1, beta.size, dtype=float)
    arg = np.multiply.outer(theta, n)
    b = beta[0] + 2.0 * (np.cos(arg) @ beta[1:])
    a = 2.0 * (np.sin(arg) @ dbeta[1:])
    return -a / b


def member_route() -> dict:
    """Return the Gate 4b route record of the everything member."""
    return {"key": MEMBER_KEY, "display": MEMBER_DISPLAY,
            "kappa1": member_kappa1,
            "entries": sa.exp_member_mass_and_advection_entries(MEMBER_LABEL),
            "spectral": False,
            "member_path": str(MEMBER.relative_to(ROOT))}


def routes_by_key() -> dict:
    """Return the Gate 4b routes keyed by name, with the member route added."""
    table = {route["key"]: route for route in g4b.route_table()}
    table[MEMBER_KEY] = member_route()
    return table


def symbol_agreement_block() -> dict:
    """Return the agreement of the three readings of the member symbol.

    The three are the samples of the member file read by this producer, the
    same samples read through spline_analysis's member loader, and the
    generator rebuilt from the member record by gate8_family.from_record.
    """
    theta = np.linspace(1.0e-6, np.pi, 4001)
    mine = member_kappa1(theta)
    through_sa = sa.kprime_exp_member(theta, MEMBER_LABEL)
    gen = g8.from_record(json.loads(MEMBER.read_text()))
    from_generator = gen.symbols(theta)[1]
    scale = float(np.max(np.abs(mine)))
    dev_sa = float(np.max(np.abs(mine - through_sa)))
    dev_gen = float(np.max(np.abs(mine - from_generator)))
    wrong = float(np.max(np.abs(-mine - through_sa)))
    return {
        "convention": ("kappa_1 = -a / b with a = 2 sum beta'(n) sin(n theta), "
                       "the applied-operator convention of EXP_ROUTE.md and "
                       "the kappa_1 of gate8_family.symbols"),
        "theta_points": int(theta.size),
        "max_abs_kappa1": scale,
        "versus_spline_analysis_kprime_exp_member": dev_sa,
        "versus_gate8_family_from_record_symbols": dev_gen,
        "versus_the_opposite_sign": wrong,
        "tolerance": SYMBOL_TOL,
        "ok": bool(max(dev_sa, dev_gen) <= SYMBOL_TOL),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def e1_at(route: dict, theta_c: float) -> float:
    """Return e_1(theta_c) of one route."""
    return float(g4b.relative_symbol_error(route["kappa1"],
                                           np.array([theta_c]))[0])


def build_prediction() -> dict:
    """Return the E3 prediction payload, written before any other step."""
    table = routes_by_key()
    carriers = {}
    for theta_c in g4b.THETA_CENTERS:
        member = e1_at(table[MEMBER_KEY], theta_c)
        slc = e1_at(table[REFERENCE], theta_c)
        carriers[f"{theta_c:.6f}"] = {
            "theta_c": float(theta_c),
            "theta_c_over_pi": float(theta_c / np.pi),
            "points_per_wavelength": float(2.0 * np.pi / theta_c),
            f"e1_{LABEL}": member,
            "e1_slc_q4": slc,
            f"abs_ratio_slc_over_{LABEL}": float(abs(slc) / abs(member)),
            f"predicted_crossing_ratio_{LABEL}_over_slc_q4":
                float(abs(slc) / abs(member)),
            f"displacement_per_crossing_boxes_{LABEL}": float(-member),
            f"sign_of_minus_e1_{LABEL}": int(np.sign(-member)),
            "reading": ("a negative signed displacement is a forward "
                        "displacement, the packet leading the exact one"),
        }
    return {
        "producer": "code/analysis/exponential_family/gate9_e3.py --predict",
        "written_utc": g4b.utc_now(),
        "contract": "results/exponential_family/gate9/contract.md, section E3",
        "gate4b_contract_sha256": g4b.sha256(g4b.CONTRACT),
        "member": {"path": str(MEMBER.relative_to(ROOT)),
                   "sha256": g4b.sha256(MEMBER)},
        "sign_convention": symbol_agreement_block(),
        "packet": {"definition": ("the Gate 4b Gaussian packet, 99 percent of "
                                  "the energy inside theta_c +- 0.1 pi"),
                   "sigma_theta_energy": float(g4b.SIGMA_ENERGY)},
        "level": float(LEVEL),
        "N_1d": N_1D,
        "tolerance_on_the_ratio": RATIO_TOL,
        "prediction": (f"the crossings to one percent of {LABEL} over those "
                       "of SLC q4 equal |e_1^{SLC}(theta_c)| / "
                       f"|e_1^{{{LABEL}}}(theta_c)| to within the packet width "
                       f"correction, under 10 percent; the sign of {LABEL}'s "
                       "displacement per crossing is the sign of "
                       f"-e_1^{{{LABEL}}}(theta_c)"),
        "carriers": carriers,
    }


# ---------------------------------------------------------------------------
# Closed form crossings
# ---------------------------------------------------------------------------

def band_factor(amp: np.ndarray, theta: np.ndarray, e1: np.ndarray,
                N: int, theta_c: float, e1c: float) -> float:
    """Return the packet width correction of one cell.

    In the linear regime the closed form error at horizon H is
    H sqrt(<(N theta e_1)^2>) with the energy weighted mean over the packet,
    so the crossing is level over that scale. The factor returned here is that
    scale divided by the carrier value N theta_c |e_1(theta_c)| the frozen
    prediction uses, and the ratio of two routes' crossings is the ratio of
    their carrier values corrected by the reciprocal ratio of these factors.
    A route whose e_1 has a zero close to the carrier carries a factor well
    above one, because the carrier value then understates the band. The band
    is the active mode band of Gate 4b, every mode whose amplitude exceeds
    1e-8 of the peak.
    """
    energy = amp ** 2
    scale = float(np.sqrt(np.sum(energy * (N * theta * e1) ** 2)
                          / np.sum(energy)))
    carrier = abs(N * theta_c * e1c)
    return float(scale / carrier) if carrier > 0.0 else float("inf")


def crossing_1d(route: dict, N: int, theta_c: float, level: float) -> dict:
    """Return the closed form one percent crossing of one 1D cell."""
    pk = g4b.packet(N, theta_c)
    live = g4b.active_mask(pk["amp"])
    theta = pk["theta"][live]
    amp = pk["amp"][live]
    e1 = g4b.relative_symbol_error(route["kappa1"], theta)
    horizon = g4b.crossing_horizon(amp, theta, e1, N, level)
    e1c = e1_at(route, theta_c)
    sign_changes = int(np.count_nonzero(np.sign(e1[:-1]) * np.sign(e1[1:]) < 0))
    return {"horizon_at_1_percent": horizon,
            "e1_at_theta_c": e1c,
            "kappa1_at_theta_c": float(route["kappa1"](np.array([theta_c]))[0]),
            "displacement_per_crossing_boxes": float(-e1c),
            "displacement_sign": int(np.sign(-e1c)),
            "e1_min_on_the_packet_band": float(np.min(e1)),
            "e1_max_on_the_packet_band": float(np.max(e1)),
            "e1_sign_changes_on_the_packet_band": sign_changes,
            "packet_width_factor": band_factor(amp, theta, e1, N, theta_c, e1c),
            "error_at_H10": g4b.predicted_error(amp, theta, e1, N, 10.0),
            "error_at_H100": g4b.predicted_error(amp, theta, e1, N, 100.0)}


def crossing_3d(route: dict, energy: np.ndarray, total_amp: np.ndarray,
                N: int, level: float, theta_c: float) -> dict:
    """Return the closed form one percent crossing of the 3D cell.

    The 3D mode factor is exp(-i H N (S - S_exact)), so the packet error is the
    Gate 4b one with theta e_1 replaced by S - S_exact. Passing that difference
    as the theta argument and a unit e_1 to the frozen bisection keeps the
    crossing definition identical to the one dimensional one.
    """
    sym, exact = g4b.symbol_sums_3d(route["kappa1"], N, g4b.VELOCITY_3D)
    delta = (sym - exact).ravel()
    ones = np.ones_like(delta)
    horizon = g4b.crossing_horizon(total_amp, delta, ones, N, level)
    e1c = e1_at(route, theta_c)
    errors = {}
    for H in g4b.HORIZONS_3D:
        errors[str(H)] = g4b.predicted_error(total_amp, delta, ones, N, float(H))
    return {"horizon_at_1_percent": horizon,
            "e1_at_theta_c": e1c,
            "largest_abs_S": float(np.max(np.abs(sym))),
            "displacement_per_crossing_boxes_per_axis":
                [float(-e1c * v / g4b.VELOCITY_3D[0]) for v in g4b.VELOCITY_3D],
            "displacement_sign": int(np.sign(-e1c)),
            "errors": errors,
            "energy_checksum": float(np.sum(energy))}


def build_transport(prediction: dict) -> dict:
    """Return the closed form crossing payload of every route and geometry."""
    table = routes_by_key()
    keys = (MEMBER_KEY,) + COMPARATORS
    theta_c_3d = 0.5 * np.pi
    hat, _ = g4b.packet_3d_spectrum(g4b.GRID_3D, theta_c_3d)
    energy = np.abs(hat) ** 2
    total_amp = np.abs(hat).ravel()

    one_d = {}
    for theta_c in g4b.THETA_CENTERS:
        cell = {}
        for key in keys:
            cell[key] = crossing_1d(table[key], N_1D, theta_c, LEVEL)
        reference = cell[REFERENCE]["horizon_at_1_percent"]
        for key in keys:
            horizon = cell[key]["horizon_at_1_percent"]
            cell[key]["crossing_ratio_over_slc_q4"] = (
                float(horizon / reference)
                if horizon is not None and reference else None)
        one_d[f"{theta_c:.6f}"] = cell

    three_d = {}
    for key in keys:
        three_d[key] = crossing_3d(table[key], energy, total_amp,
                                   g4b.GRID_3D, LEVEL, theta_c_3d)
    reference = three_d[REFERENCE]["horizon_at_1_percent"]
    for key in keys:
        horizon = three_d[key]["horizon_at_1_percent"]
        three_d[key]["crossing_ratio_over_slc_q4"] = (
            float(horizon / reference)
            if horizon is not None and reference else None)

    agreement = {}
    for theta_c in g4b.THETA_CENTERS:
        tag = f"{theta_c:.6f}"
        predicted = prediction["carriers"][tag][
            f"predicted_crossing_ratio_{LABEL}_over_slc_q4"]
        realized_1d = one_d[tag][MEMBER_KEY]["crossing_ratio_over_slc_q4"]
        factor_member = one_d[tag][MEMBER_KEY]["packet_width_factor"]
        factor_reference = one_d[tag][REFERENCE]["packet_width_factor"]
        corrected = predicted * factor_reference / factor_member
        entry = {"predicted_ratio": predicted,
                 "realized_ratio_1d": realized_1d,
                 "relative_departure_1d": float(abs(realized_1d - predicted)
                                                / predicted),
                 f"packet_width_factor_{LABEL}": factor_member,
                 "packet_width_factor_slc_q4": factor_reference,
                 "packet_width_corrected_ratio": float(corrected),
                 "relative_departure_of_the_corrected_ratio_1d":
                     float(abs(realized_1d - corrected) / corrected),
                 f"e1_sign_changes_on_the_packet_band_{LABEL}":
                     one_d[tag][MEMBER_KEY]["e1_sign_changes_on_the_packet_band"],
                 "e1_sign_changes_on_the_packet_band_slc_q4":
                     one_d[tag][REFERENCE]["e1_sign_changes_on_the_packet_band"],
                 "tolerance": RATIO_TOL}
        entry["within_tolerance_1d"] = bool(entry["relative_departure_1d"]
                                            <= RATIO_TOL)
        if abs(theta_c - theta_c_3d) < 1.0e-12:
            realized_3d = three_d[MEMBER_KEY]["crossing_ratio_over_slc_q4"]
            entry["realized_ratio_3d"] = realized_3d
            entry["relative_departure_3d"] = float(
                abs(realized_3d - predicted) / predicted)
            entry["within_tolerance_3d"] = bool(
                entry["relative_departure_3d"] <= RATIO_TOL)
        agreement[tag] = entry

    return {
        "producer": "code/analysis/exponential_family/gate9_e3.py --closed-form",
        "written_utc": g4b.utc_now(),
        "prediction": str(PREDICTION_PATH.relative_to(ROOT)),
        "prediction_written_utc": prediction["written_utc"],
        "level": float(LEVEL),
        "definitions": {
            "crossing": ("the horizon at which the closed form relative L2 "
                         "packet error first reaches one percent, by "
                         "gate4b_long_horizon.crossing_horizon on the frozen "
                         "packet spectrum"),
            "displacement": ("the signed carrier displacement per crossing in "
                             "box fractions, -e_1(theta_c); negative is a "
                             "forward displacement"),
            "3d": (f"the Gate 4b three dimensional cell, N = {g4b.GRID_3D}, "
                   "theta_c = 0.5 pi, velocity (1, 1/2, 1/3), one crossing "
                   "counted on the x period"),
        },
        "N_1d": N_1D, "N_3d": g4b.GRID_3D,
        "routes": list(keys),
        "cells_1d": one_d,
        "cells_3d": three_d,
        "ratio_agreement": agreement,
    }


def transport_markdown(payload: dict) -> str:
    """Return the crossing table of the closed form step."""
    lines = ["# Gate 9 E3: long-horizon transport in closed form", "",
             "Crossings are the horizon in box crossings at which the closed",
             "form relative L2 packet error first reaches one percent. The",
             "displacement is the signed carrier displacement per crossing in",
             "box fractions, negative for a packet that leads.", ""]
    for tag, cell in payload["cells_1d"].items():
        theta_c = float(tag)
        lines += [f"## One dimension, N = {payload['N_1d']}, "
                  f"theta_c = {theta_c / np.pi:.2f} pi "
                  f"({2.0 * np.pi / theta_c:.1f} points per wavelength)", "",
                  "| route | e_1(theta_c) | crossing to 1 percent | "
                  "ratio over SLC q4 | displacement per crossing |",
                  "| --- | --- | --- | --- | --- |"]
        for key in payload["routes"]:
            row = cell[key]
            horizon = row["horizon_at_1_percent"]
            ratio = row["crossing_ratio_over_slc_q4"]
            lines.append(
                f"| {key} | {row['e1_at_theta_c']:+.4e} | "
                f"{'none' if horizon is None else f'{horizon:.4g}'} | "
                f"{'none' if ratio is None else f'{ratio:.4g}'} | "
                f"{row['displacement_per_crossing_boxes']:+.4e} |")
        lines.append("")
    lines += [f"## Three dimensions, N = {payload['N_3d']}, theta_c = 0.5 pi, "
              "velocity (1, 1/2, 1/3)", "",
              "| route | e_1(theta_c) | crossing to 1 percent | "
              "ratio over SLC q4 | displacement per crossing, x axis |",
              "| --- | --- | --- | --- | --- |"]
    for key in payload["routes"]:
        row = payload["cells_3d"][key]
        horizon = row["horizon_at_1_percent"]
        ratio = row["crossing_ratio_over_slc_q4"]
        lines.append(
            f"| {key} | {row['e1_at_theta_c']:+.4e} | "
            f"{'none' if horizon is None else f'{horizon:.4g}'} | "
            f"{'none' if ratio is None else f'{ratio:.4g}'} | "
            f"{row['displacement_per_crossing_boxes_per_axis'][0]:+.4e} |")
    lines += ["", "## The frozen ratio prediction", "",
              "| theta_c | predicted | realized 1d | departure | "
              "realized 3d | departure |", "| --- | --- | --- | --- | --- | --- |"]
    for tag, entry in payload["ratio_agreement"].items():
        theta_c = float(tag)
        three = entry.get("realized_ratio_3d")
        dep3 = entry.get("relative_departure_3d")
        lines.append(
            f"| {theta_c / np.pi:.2f} pi | {entry['predicted_ratio']:.4f} | "
            f"{entry['realized_ratio_1d']:.4f} | "
            f"{100.0 * entry['relative_departure_1d']:.2f} percent | "
            f"{'none' if three is None else f'{three:.4f}'} | "
            f"{'none' if dep3 is None else f'{100.0 * dep3:.2f} percent'} |")
    lines += ["", "The packet width correction is the root of the energy",
              "weighted band mean of (N theta e_1)^2 divided by its value at",
              "the carrier, over the active mode band of Gate 4b. It explains",
              "the departure in full: a carrier that sits near a zero of e_1",
              "understates the band the packet occupies. The sign change count",
              "is taken on the same active band, whose tails reach far beyond",
              "the 99 percent energy width.", "",
              f"| theta_c | factor {LABEL} | factor SLC q4 | sign changes "
              f"{LABEL} | corrected ratio | departure |",
              "| --- | --- | --- | --- | --- | --- |"]
    for tag, entry in payload["ratio_agreement"].items():
        theta_c = float(tag)
        lines.append(
            f"| {theta_c / np.pi:.2f} pi | "
            f"{entry['packet_width_factor_' + LABEL]:.4f} | "
            f"{entry['packet_width_factor_slc_q4']:.4f} | "
            f"{entry['e1_sign_changes_on_the_packet_band_' + LABEL]} | "
            f"{entry['packet_width_corrected_ratio']:.4f} | "
            f"{100.0 * entry['relative_departure_of_the_corrected_ratio_1d']:.2f}"
            " percent |")
    if "run_check" in payload:
        run = payload["run_check"]
        lines += ["", "## The time stepped check", "",
                  f"{LABEL} at N = {run['N']}, theta_c = 0.5 pi, "
                  f"H = {run['horizon']}, cfl {run['cfl']:g}, "
                  f"{run['steps_full']} full SSP-RK3 steps of the assembled "
                  f"circulant: relative difference from the closed form "
                  f"{run['relative_difference']:.3e} against the tolerance "
                  f"{run['tolerance']:g}."]
    if "regression" in payload:
        reg = payload["regression"]
        lines += ["", "## Gate 4b regression", "",
                  f"worst relative deviation of the stored champion numbers "
                  f"{reg['worst']:.3e} over {reg['compared']} values against "
                  f"the tolerance {reg['tolerance']:g}."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The time stepped check
# ---------------------------------------------------------------------------

def run_check(N: int = N_1D, horizon: int = HORIZON_CHECK) -> dict:
    """Return the departure of a genuine SSP-RK3 run from the closed form."""
    prediction = g4b.load_prediction()
    nu = prediction["time_stepping"]["cfl_by_grid"][str(N)]["cfl"]
    theta_c = 0.5 * np.pi
    route = member_route()
    pk = g4b.packet(N, theta_c)
    live = g4b.active_mask(pk["amp"])
    k = pk["k"][live]
    amp = pk["amp"][live]
    theta = pk["theta"][live]
    field0 = g4b.real_field(N, k, amp)
    started = time.time()
    looped = g4b.rk3_time_loop(route, N, float(horizon), nu, field0)
    seconds = time.time() - started
    kap = route["kappa1"](theta)
    factor, _ = g4b.relative_factor(kap, theta, N, horizon, nu)
    closed = g4b.real_field(N, k, amp * factor)
    n_full, nu_r = g4b.step_plan(N, horizon, nu)
    difference = float(np.linalg.norm(looped - closed)
                       / np.linalg.norm(closed))
    return {
        "route": MEMBER_KEY, "N": N, "theta_c": float(theta_c),
        "horizon": int(horizon), "cfl": float(nu),
        "cfl_source": ("results/exponential_family/gate4b/prediction.json, "
                       f"time_stepping.cfl_by_grid.{N}.cfl"),
        "stepper": ("gate4b_long_horizon.rk3_time_loop, SSP-RK3 on the "
                    "assembled periodic circulant of the member, with one "
                    "partial final step landing on the horizon"),
        "operator": ("spline_analysis.compact_derivative_matrix on the member "
                     "left side beta and the advection entries -beta'"),
        "steps_full": int(n_full), "partial_step_in_h": float(nu_r),
        "relative_difference": difference,
        "tolerance": RUN_CHECK_TOL,
        "ok": bool(difference <= RUN_CHECK_TOL),
        "symbol_versus_assembled_operator_N64":
            g4b.symbol_agreement(route, 64),
        "seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Regression on the stored Gate 4b champion numbers
# ---------------------------------------------------------------------------

def _deviation(stored, fresh) -> float:
    scale = max(abs(float(stored)), 1.0e-300)
    return abs(float(stored) - float(fresh)) / scale


def regression() -> dict:
    """Return the reproduction of the stored Gate 4b champion closed form."""
    table = routes_by_key()
    champion = table["champion"]
    stored = json.loads((GATE4B / "prediction.json").read_text())
    worst, worst_key, compared, failures = 0.0, None, 0, []

    def record(key, value, fresh):
        nonlocal worst, worst_key, compared
        if value is None or fresh is None:
            return
        deviation = _deviation(value, fresh)
        compared += 1
        if deviation > worst:
            worst, worst_key = deviation, key
        if deviation > REGRESSION_TOL:
            failures.append({"key": key, "stored": float(value),
                             "fresh": float(fresh), "relative": deviation})

    for N in g4b.GRIDS_1D:
        for theta_c in g4b.THETA_CENTERS:
            tag = f"1d|champion|{N}|{theta_c:.6f}"
            cell = stored["cells_1d"][tag]
            pk = g4b.packet(N, theta_c)
            live = g4b.active_mask(pk["amp"])
            theta = pk["theta"][live]
            amp = pk["amp"][live]
            e1 = g4b.relative_symbol_error(champion["kappa1"], theta)
            e1c = e1_at(champion, theta_c)
            record(tag + "|e1_at_theta_c", cell["e1_at_theta_c"], e1c)
            record(tag + "|kappa1_at_theta_c", cell["kappa1_at_theta_c"],
                   float(champion["kappa1"](np.array([theta_c]))[0]))
            for H in g4b.HORIZONS_1D:
                record(f"{tag}|errors|{H}", cell["errors"][str(H)],
                       g4b.predicted_error(amp, theta, e1, N, H))
                record(f"{tag}|carrier_phase_lag_rad|{H}",
                       cell["carrier_phase_lag_rad"][str(H)],
                       -H * N * theta_c * e1c)
                record(f"{tag}|centroid_lag_boxes|{H}",
                       cell["centroid_lag_boxes"][str(H)], -H * e1c)
            record(tag + "|horizon_at_1_percent", cell["horizon_at_1_percent"],
                   g4b.crossing_horizon(amp, theta, e1, N,
                                        g4b.ERROR_LEVELS[0]))
            record(tag + "|horizon_at_10_percent",
                   cell["horizon_at_10_percent"],
                   g4b.crossing_horizon(amp, theta, e1, N,
                                        g4b.ERROR_LEVELS[1]))

    tag3 = f"3d|champion|{g4b.GRID_3D}|{stored['cells_3d']['theta_c']:.6f}"
    cell3 = stored["cells_3d"]["cells"][tag3]
    hat, _ = g4b.packet_3d_spectrum(g4b.GRID_3D, stored["cells_3d"]["theta_c"])
    energy = np.abs(hat) ** 2
    total = float(np.sum(energy))
    sym, exact = g4b.symbol_sums_3d(champion["kappa1"], g4b.GRID_3D,
                                    g4b.VELOCITY_3D)
    for H in g4b.HORIZONS_3D:
        phase = -H * g4b.GRID_3D * (sym - exact)
        defect = float(np.sum(energy * 4.0 * np.sin(0.5 * phase) ** 2))
        record(f"{tag3}|errors|{H}", cell3["errors"][str(H)],
               float(np.sqrt(defect / total)))
    e1c3 = e1_at(champion, stored["cells_3d"]["theta_c"])
    record(tag3 + "|e1_at_theta_c", cell3["e1_at_theta_c"], e1c3)
    for H in g4b.HORIZONS_3D:
        for axis, v in enumerate(g4b.VELOCITY_3D):
            record(f"{tag3}|centroid_lag_boxes_per_axis|{H}|{axis}",
                   cell3["centroid_lag_boxes_per_axis"][str(H)][axis],
                   -H * e1c3 * v / g4b.VELOCITY_3D[0])

    rows = 0
    with (GATE4B / "runs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["route"] != "champion" or row["geometry"] != "1d":
                continue
            if row["cfl_setting"] != "base":
                continue
            N = int(row["N"])
            theta_c = float(row["theta_c"])
            H = int(row["horizon"])
            nu = float(row["cfl"])
            pk = g4b.packet(N, theta_c)
            live = g4b.active_mask(pk["amp"])
            sub = {"k": pk["k"][live], "amp": pk["amp"][live]}
            kap = champion["kappa1"](pk["theta"][live])
            factor, _ = g4b.relative_factor(kap, pk["theta"][live], N, H, nu)
            values = g4b.metrics_1d(sub, N, factor, theta_c)
            for field in ("rel_l2", "energy_ratio", "amplitude_ratio",
                          "centroid_lag_boxes_wrapped",
                          "carrier_phase_lag_rad_wrapped"):
                record(f"runs.csv|{row['key']}|{field}", float(row[field]),
                       values[field])
            rows += 1

    return {"tolerance": REGRESSION_TOL, "compared": compared,
            "runs_csv_rows": rows, "worst": worst, "worst_key": worst_key,
            "failures": failures, "ok": not failures,
            "sources": ["results/exponential_family/gate4b/prediction.json",
                        "results/exponential_family/gate4b/runs.csv"]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--closed-form", dest="closed_form",
                        action="store_true")
    parser.add_argument("--run-check", dest="run_check", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--member", default=None,
                        help="member file to field; default the pert5 "
                             "everything member of Gate 8")
    parser.add_argument("--tag", default=None,
                        help="suffix of every output name and the route label "
                             "inside the receipts; default none, the frozen "
                             "pert5 names")
    args = parser.parse_args(argv)
    if not any((args.predict, args.closed_form, args.run_check,
                args.regression)):
        parser.error("give --predict, --closed-form, --run-check or "
                     "--regression")

    configure(args.member, args.tag)
    GATE9.mkdir(parents=True, exist_ok=True)
    prediction_path = PREDICTION_PATH
    transport_path = TRANSPORT_PATH

    if args.predict:
        if prediction_path.exists():
            raise SystemExit(f"{prediction_path} exists; predictions are frozen.")
        payload = build_prediction()
        prediction_path.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {prediction_path.relative_to(ROOT)}")

    if not prediction_path.exists():
        raise SystemExit(f"{prediction_path} is absent; run --predict first.")
    prediction = json.loads(prediction_path.read_text())

    payload = (json.loads(transport_path.read_text())
               if transport_path.exists() else None)

    if args.closed_form:
        payload = build_transport(prediction)
        transport_path.write_text(json.dumps(payload, indent=1) + "\n")
        TRANSPORT_MD.write_text(transport_markdown(payload))
        print(f"wrote {transport_path.relative_to(ROOT)} and "
              f"{TRANSPORT_MD.relative_to(ROOT)}")

    if args.run_check or args.regression:
        if payload is None:
            raise SystemExit(f"{transport_path} is absent; run --closed-form "
                             "first.")
        if args.run_check:
            payload["run_check"] = run_check()
            print(json.dumps(payload["run_check"], indent=1))
        if args.regression:
            payload["regression"] = regression()
            print(json.dumps({k: v for k, v in payload["regression"].items()
                              if k != "failures"}, indent=1))
        transport_path.write_text(json.dumps(payload, indent=1) + "\n")
        TRANSPORT_MD.write_text(transport_markdown(payload))
        print(f"wrote {transport_path.relative_to(ROOT)} and "
              f"{TRANSPORT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
