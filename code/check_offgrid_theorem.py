# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "scipy>=1.11",
# ]
# ///
"""Numerical verification for Theorem thm:offgrid (off-grid value/derivative orders).

Three independent checks, each targeting one repaired step of the proof:

  (1) Finite-grid inverse behavior supporting the uniform-in-N proof.
      The periodic collocation matrix B is a banded symmetric circulant whose
      symbol f(omega) = sum_s beta_p(s) e^{-i s omega} is a strictly positive
      trigonometric polynomial (min m_p > 0, max M_p = 1 by partition of unity,
      both independent of N). The circulant inverse entries are the
      periodization of the Fourier coefficients c_k of 1/f, so the analytic
      proof gives
          ||B^{-1}||_inf = sum_s |sum_k c_{s+kN}| <= sum_k |c_k| = ||1/f||_Wiener,
      a bound independent of N. The checker does not compute the infinite
      Wiener sum. It records finite-grid inverse row sums for N spanning
      8..8192, requires stabilization across the last three distinct grids,
      and reports the measured geometric decay rate of c_k against the
      classical Demko-Moss-Smith rate q = (sqrt(kappa)-1)/(sqrt(kappa)+1),
      kappa = M_p/m_p. The analytic proof, not the empirical plateau, certifies
      the uniform theorem.

  (2) Convergence on a SMOOTH, NON-band-limited field (hypothesis satisfied).
      u(x) = exp(sin x) is analytic and periodic but carries every Fourier mode
      (Bessel-function coefficients), so it lies outside the finite Fourier-mode
      family used in the manuscript plot. The off-grid value error tracks order
      p+1 and the off-grid first-derivative error tracks order p.

  (3) Convergence when the regularity hypothesis is VIOLATED (rate must degrade).
      u(x) = (1 - cos x)^{5/4} behaves like c*|x|^{5/2} near x = 0, an algebraic
      (non-polynomial) singularity: u in C^2(T) with u'' Holder-1/2 and u''' not
      in L^inf, so u is not in C^3 and lies in the Sobolev/Besov smoothness class
      s = 5/2. The theorem hypothesis u in C^{p+1} then fails for every p >= 2,
      and the observed value order collapses to the smoothness index 5/2 and the
      derivative order to 3/2, independent of the spline degree p. The rate
      tracks the regularity of u, not p+1, so the numerical test fails exactly
      when the hypothesis is removed. (A polynomial-type kink such as |x|^3 is
      avoided on purpose: a degree-p spline reproduces it whenever p >= 3 and no
      degradation would show.)

Run:  uv run code/check_offgrid_theorem.py
Writes: results/evidence/offgrid_theorem_verification.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from spline_analysis import (  # noqa: E402
    centered_cardinal_bspline,
    coefficients_from_nodal_values,
    evaluate_periodic_spline,
)

DEGREES = [2, 3, 4, 5, 6, 7]
TWO_PI = 2.0 * np.pi

SYMBOL_POSITIVITY_FLOOR = 1.0e-12
INVERSE_MONOTONICITY_ABS_TOL = 1.0e-9
EMPIRICAL_STABILIZATION_ABS_TOL = 1.0e-10
EMPIRICAL_STABILIZATION_GRID_COUNT = 3
DECAY_ENVELOPE_ABS_TOL = 5.0e-3
INTERPOLATION_RESIDUAL_TOL = 1.0e-12
SMOOTH_ROUNDOFF_FLOOR = 5.0e-13
SMOOTH_ORDER_LOWER_TOL = 0.35
SMOOTH_ORDER_UPPER_TOL = 0.50
LIMITED_VALUE_TARGET = 2.5
LIMITED_DERIV_TARGET = 1.5
LIMITED_ORDER_ABS_TOL = 0.10


# --------------------------------------------------------------------------
# (1) Finite-grid inverse behavior and reciprocal-symbol decay
# --------------------------------------------------------------------------
def band_generator(degree: int) -> np.ndarray:
    """First column of the circulant B for one period, as a bi-infinite band.

    Returns the symmetric coefficient list [beta_p(0), beta_p(1), ..., beta_p(R)]
    with R = floor(p/2) the support radius of the cardinal B-spline at integers.
    """
    radius = degree // 2
    offsets = np.arange(0, radius + 1, dtype=float)
    return centered_cardinal_bspline(degree, offsets)


def circulant_first_column(coeffs: np.ndarray, npts: int) -> np.ndarray:
    col = np.zeros(npts, dtype=float)
    col[0] = coeffs[0]
    for s in range(1, len(coeffs)):
        col[s % npts] += coeffs[s]
        col[(-s) % npts] += coeffs[s]
    return col


def inverse_infinity_norm(coeffs: np.ndarray, npts: int) -> float:
    """||B^{-1}||_inf for the banded symmetric circulant B via FFT.

    B^{-1} is circulant, so its inf-norm equals the absolute sum of its first
    column, obtained as the inverse DFT of 1 / (DFT of B's first column).
    """
    col = circulant_first_column(coeffs, npts)
    eig = np.fft.fft(col).real  # symbol samples f(omega_m), real by symmetry
    inv_col = np.fft.ifft(1.0 / eig).real
    return float(np.sum(np.abs(inv_col)))


def symbol_extremes(coeffs: np.ndarray) -> tuple[float, float]:
    w = np.linspace(0.0, np.pi, 20001)
    f = np.full_like(w, coeffs[0])
    for s in range(1, len(coeffs)):
        f += 2.0 * coeffs[s] * np.cos(s * w)
    return float(f.min()), float(f.max())


def reciprocal_decay_rate(coeffs: np.ndarray, npts: int = 8192) -> float:
    """Measure the geometric decay rate of sampled reciprocal coefficients.

    At large npts the low-index inverse DFT entries approximate the Laurent
    coefficients c_k of 1/f before circular aliasing becomes visible. The decay
    rate rho is the fit exp(slope) of log|c_k| versus k over the clean
    pre-roundoff range. No finite inverse row sum is identified with the
    infinite Wiener norm.
    """
    col = circulant_first_column(coeffs, npts)
    eig = np.fft.fft(col).real
    inv_col = np.fft.ifft(1.0 / eig).real
    ck = np.abs(inv_col[: npts // 2])
    ks = np.arange(len(ck))
    mask = (ks >= 1) & (ck > 1e-13) & (ks <= 60)
    slope = np.polyfit(ks[mask], np.log(ck[mask]), 1)[0]
    return float(np.exp(slope))


def finite_grid_inverse_report() -> dict:
    out = {}
    n_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for p in DEGREES:
        coeffs = band_generator(p)
        radius = p // 2
        m_p, M_p = symbol_extremes(coeffs)
        kappa = M_p / m_p
        q_dms = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
        # Demko-Moss-Smith per-index decay rate for a band of radius R:
        # |(B^{-1})_{jk}| <= C lambda^{|j-k|} with lambda = q^{1/R}.
        lambda_dms = q_dms ** (1.0 / radius)
        rho_meas = reciprocal_decay_rate(coeffs)
        norms = [(n, inverse_infinity_norm(coeffs, n)) for n in n_list]
        finite_positive = all(np.isfinite(val) and val > 0.0 for _, val in norms)
        monotone = all(
            norms[i][1]
            <= norms[i + 1][1] + INVERSE_MONOTONICITY_ABS_TOL
            for i in range(len(norms) - 1)
        )
        stabilization_rows = norms[-EMPIRICAL_STABILIZATION_GRID_COUNT:]
        stabilization_changes = [
            {
                "lower_N": stabilization_rows[i][0],
                "upper_N": stabilization_rows[i + 1][0],
                "absolute_difference": abs(
                    stabilization_rows[i + 1][1] - stabilization_rows[i][1]
                ),
            }
            for i in range(len(stabilization_rows) - 1)
        ]
        max_stabilization_change = max(
            change["absolute_difference"] for change in stabilization_changes
        )
        out[f"p{p}"] = {
            "band_radius_R": p // 2,
            "symbol_min_m_p": m_p,
            "symbol_max_M_p": M_p,
            "l2_condition_kappa": kappa,
            "dms_aggregate_rate_q": q_dms,
            "dms_per_index_rate_lambda": lambda_dms,
            "measured_fourier_decay_rho": rho_meas,
            # DMS is a valid decay envelope: sharp (equality) for R=1, strict
            # for R>1. Tolerance absorbs log-fit noise on the sharp case.
            "measured_rho_below_dms_lambda": bool(
                rho_meas <= lambda_dms + DECAY_ENVELOPE_ABS_TOL
            ),
            "inverse_inf_norm_vs_N": [{"N": n, "norm": v} for n, v in norms],
            "finite_grid_inverse_norms_finite_positive": bool(finite_positive),
            "inf_norm_monotone_in_N": bool(monotone),
            "high_N_empirical_inverse_row_sum_reference": {
                "N": norms[-1][0],
                "value": norms[-1][1],
                "role": "finite high-N empirical reference, not an analytic Wiener norm",
            },
            "empirical_stabilization_witness": {
                "grid_count": EMPIRICAL_STABILIZATION_GRID_COUNT,
                "rows": [
                    {"N": n, "norm": value} for n, value in stabilization_rows
                ],
                "consecutive_absolute_differences": stabilization_changes,
                "maximum_absolute_difference": max_stabilization_change,
                "maximum_allowed": EMPIRICAL_STABILIZATION_ABS_TOL,
                "status": (
                    "PASS"
                    if max_stabilization_change <= EMPIRICAL_STABILIZATION_ABS_TOL
                    else "FAIL"
                ),
                "interpretation": (
                    "finite-grid regression witness only; the analytic proof "
                    "establishes the uniform inverse bound"
                ),
            },
        }
    return out


# --------------------------------------------------------------------------
# (2) + (3) Off-grid convergence on functions outside the finite Fourier family
# --------------------------------------------------------------------------
def field_smooth(x: np.ndarray, r: int) -> np.ndarray:
    """u(x) = exp(sin x); analytic, non-band-limited. r in {0,1}."""
    s, c = np.sin(x), np.cos(x)
    if r == 0:
        return np.exp(s)
    if r == 1:
        return c * np.exp(s)
    raise ValueError(r)


def field_limited(x: np.ndarray, r: int) -> np.ndarray:
    """u(x) = (1 - cos x)^{5/4} ~ c|x|^{5/2} near 0; smoothness class s = 5/2.

    C^2(T) with an algebraic (non-polynomial) singularity: u''' not in L^inf,
    u not in C^3, so the C^{p+1} hypothesis fails for every p >= 2.
    """
    alpha = 1.25
    base = 1.0 - np.cos(x)
    if r == 0:
        return base ** alpha
    if r == 1:
        return alpha * base ** (alpha - 1.0) * np.sin(x)
    raise ValueError(r)


# Dense, node-avoiding, N-independent evaluation set on [0, 2pi):
# cell-interior sample points so the sup captures the worst (near-singular) cell.
_M_EVAL = 20000
X_EVAL = (np.arange(_M_EVAL) + 0.5) / _M_EVAL * TWO_PI


def convergence_report(field, name: str, n_list: list[int]) -> dict:
    out = {}
    for p in DEGREES:
        val_err, der_err, hs = [], [], []
        for n in n_list:
            h = TWO_PI / n
            x_nodes = np.arange(n) * h
            u_nodes = field(x_nodes, 0)
            alpha = coefficients_from_nodal_values(u_nodes, p)
            uh = evaluate_periodic_spline(alpha, X_EVAL, h, p, derivative=0)
            uhp = evaluate_periodic_spline(alpha, X_EVAL, h, p, derivative=1)
            u_true = field(X_EVAL, 0)
            up_true = field(X_EVAL, 1)
            val_err.append(float(np.max(np.abs(uh - u_true))))
            der_err.append(float(np.max(np.abs(uhp - up_true))))
            hs.append(h)
        val_order = fit_order(hs, val_err)
        der_order = fit_order(hs, der_err)
        out[f"p{p}"] = {
            "N": n_list,
            "h": hs,
            "value_error": val_err,
            "deriv_error": der_err,
            "value_order_fit": val_order,
            "deriv_order_fit": der_order,
            "value_order_pairwise": pairwise_orders(hs, val_err),
            "deriv_order_pairwise": pairwise_orders(hs, der_err),
            "expected_value_order": p + 1,
            "expected_deriv_order": p,
        }
    out["_field"] = name
    return out


def pairwise_orders(hs: list[float], err: list[float]) -> list[float]:
    hs = np.asarray(hs)
    err = np.asarray(err)
    orders = []
    for i in range(len(hs) - 1):
        if err[i] > 0 and err[i + 1] > 0:
            orders.append(
                float(np.log(err[i] / err[i + 1]) / np.log(hs[i] / hs[i + 1]))
            )
        else:
            orders.append(float("nan"))
    return orders


def fit_order(hs: list[float], err: list[float]) -> float:
    """Least-squares slope over the clean range: error above the round-off floor
    and below the pre-asymptotic ceiling."""
    hs = np.asarray(hs, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = (err > 1e-12) & (err < 1e-1)
    if np.count_nonzero(mask) < 2:
        mask = err > 0
    return float(np.polyfit(np.log(hs[mask]), np.log(err[mask]), 1)[0])


def interpolation_check() -> dict:
    """Confirm the spline interpolates the nodal data (B alpha = u recovered)."""
    out = {}
    for p in DEGREES:
        n = 64
        h = TWO_PI / n
        x_nodes = np.arange(n) * h
        u_nodes = field_smooth(x_nodes, 0)
        alpha = coefficients_from_nodal_values(u_nodes, p)
        uh_nodes = evaluate_periodic_spline(alpha, x_nodes, h, p, derivative=0)
        out[f"p{p}"] = float(np.max(np.abs(uh_nodes - u_nodes)))
    return out


def last_clean_pairwise_witness(report: dict, quantity: str) -> dict:
    """Return the finest adjacent-grid order above the roundoff error floor."""
    errors = report[f"{quantity}_error"]
    orders = report[f"{quantity}_order_pairwise"]
    grids = report["N"]
    candidates = [
        i
        for i, order in enumerate(orders)
        if np.isfinite(order)
        and errors[i] > SMOOTH_ROUNDOFF_FLOOR
        and errors[i + 1] > SMOOTH_ROUNDOFF_FLOOR
    ]
    if not candidates:
        return {"available": False, "status": "FAIL"}
    i = candidates[-1]
    return {
        "available": True,
        "coarse_N": grids[i],
        "fine_N": grids[i + 1],
        "coarse_error": errors[i],
        "fine_error": errors[i + 1],
        "observed_order": orders[i],
    }


def acceptance_criteria() -> dict:
    """Finite numerical tolerances used to accept the executable certificate."""
    return {
        "finite_grid_inverse_behavior": {
            "sampled_symbol_min_strictly_above": SYMBOL_POSITIVITY_FLOOR,
            "inverse_norm_monotonicity_abs_tolerance": (
                INVERSE_MONOTONICITY_ABS_TOL
            ),
            "empirical_stabilization_grid_count": (
                EMPIRICAL_STABILIZATION_GRID_COUNT
            ),
            "empirical_stabilization_absolute_tolerance": (
                EMPIRICAL_STABILIZATION_ABS_TOL
            ),
            "measured_decay_below_dms_abs_tolerance": DECAY_ENVELOPE_ABS_TOL,
            "required_conditions": [
                "finite positive sampled symbol",
                "measured Fourier decay does not exceed the DMS envelope",
                "all finite-grid inverse infinity norms are finite and positive",
                "inverse infinity norms are nondecreasing within tolerance",
                "the last three distinct grids stabilize within absolute tolerance",
            ],
            "scope": (
                "empirical finite-grid regression only; the analytic proof "
                "establishes the uniform theorem"
            ),
        },
        "nodal_interpolation": {
            "max_abs_residual_at_N64": INTERPOLATION_RESIDUAL_TOL,
        },
        "smooth_nonbandlimited_orders": {
            "expected_value_order": "p+1",
            "expected_first_derivative_order": "p",
            "least_squares_order_minimum": "expected minus 0.35",
            "last_clean_pairwise_order_interval": (
                "[expected minus 0.35, expected plus 0.50]"
            ),
            "roundoff_error_floor": SMOOTH_ROUNDOFF_FLOOR,
            "interpretation": (
                "The least-squares fit has no upper cutoff, allowing "
                "pre-asymptotic superconvergence. The last adjacent pair with "
                "both errors above the roundoff floor must recover the expected rate."
            ),
        },
        "limited_regularity_orders": {
            "value_order_target": LIMITED_VALUE_TARGET,
            "first_derivative_order_target": LIMITED_DERIV_TARGET,
            "absolute_tolerance": LIMITED_ORDER_ABS_TOL,
        },
    }


def acceptance_decisions(result: dict) -> dict:
    """Apply every required criterion and return auditable per-check decisions."""
    inverse_by_degree = {}
    for p in DEGREES:
        report = result["check1_finite_grid_inverse_behavior"][f"p{p}"]
        conditions = {
            "sampled_symbol_positive": bool(
                np.isfinite(report["symbol_min_m_p"])
                and report["symbol_min_m_p"] > SYMBOL_POSITIVITY_FLOOR
            ),
            "measured_decay_within_dms_envelope": report[
                "measured_rho_below_dms_lambda"
            ],
            "finite_grid_inverse_norms_finite_positive": report[
                "finite_grid_inverse_norms_finite_positive"
            ],
            "inverse_norms_nondecreasing": report["inf_norm_monotone_in_N"],
            "last_three_grids_empirically_stabilized": (
                report["empirical_stabilization_witness"]["status"] == "PASS"
            ),
        }
        passed = all(conditions.values())
        inverse_by_degree[f"p{p}"] = {
            "status": "PASS" if passed else "FAIL",
            "conditions": conditions,
        }

    interpolation_by_degree = {}
    for p in DEGREES:
        residual = result["interpolation_residual_at_nodes"][f"p{p}"]
        passed = bool(
            np.isfinite(residual) and residual <= INTERPOLATION_RESIDUAL_TOL
        )
        interpolation_by_degree[f"p{p}"] = {
            "status": "PASS" if passed else "FAIL",
            "observed_max_abs_residual": residual,
            "maximum_allowed": INTERPOLATION_RESIDUAL_TOL,
        }

    smooth_by_degree = {}
    smooth = result["check2_smooth_nonbandlimited_exp_sin"]
    for p in DEGREES:
        report = smooth[f"p{p}"]
        quantity_decisions = {}
        for quantity, expected in (("value", p + 1), ("deriv", p)):
            fitted = report[f"{quantity}_order_fit"]
            witness = last_clean_pairwise_witness(report, quantity)
            fit_pass = bool(
                np.isfinite(fitted) and fitted >= expected - SMOOTH_ORDER_LOWER_TOL
            )
            witness_pass = bool(
                witness.get("available", False)
                and expected - SMOOTH_ORDER_LOWER_TOL
                <= witness["observed_order"]
                <= expected + SMOOTH_ORDER_UPPER_TOL
            )
            quantity_pass = fit_pass and witness_pass
            witness["status"] = "PASS" if witness_pass else "FAIL"
            quantity_decisions[quantity] = {
                "status": "PASS" if quantity_pass else "FAIL",
                "expected_order": expected,
                "least_squares_order": fitted,
                "least_squares_minimum": expected - SMOOTH_ORDER_LOWER_TOL,
                "least_squares_status": "PASS" if fit_pass else "FAIL",
                "last_clean_pairwise_witness": witness,
                "pairwise_allowed_interval": [
                    expected - SMOOTH_ORDER_LOWER_TOL,
                    expected + SMOOTH_ORDER_UPPER_TOL,
                ],
            }
        passed = all(v["status"] == "PASS" for v in quantity_decisions.values())
        smooth_by_degree[f"p{p}"] = {
            "status": "PASS" if passed else "FAIL",
            "quantities": quantity_decisions,
        }

    limited_by_degree = {}
    limited = result["check3_limited_regularity_1mcos_pow_1p25"]
    for p in DEGREES:
        report = limited[f"p{p}"]
        value_order = report["value_order_fit"]
        deriv_order = report["deriv_order_fit"]
        value_pass = bool(
            np.isfinite(value_order)
            and abs(value_order - LIMITED_VALUE_TARGET) <= LIMITED_ORDER_ABS_TOL
        )
        deriv_pass = bool(
            np.isfinite(deriv_order)
            and abs(deriv_order - LIMITED_DERIV_TARGET) <= LIMITED_ORDER_ABS_TOL
        )
        passed = value_pass and deriv_pass
        limited_by_degree[f"p{p}"] = {
            "status": "PASS" if passed else "FAIL",
            "value": {
                "status": "PASS" if value_pass else "FAIL",
                "observed_order": value_order,
                "target_order": LIMITED_VALUE_TARGET,
                "absolute_deviation": abs(value_order - LIMITED_VALUE_TARGET),
            },
            "first_derivative": {
                "status": "PASS" if deriv_pass else "FAIL",
                "observed_order": deriv_order,
                "target_order": LIMITED_DERIV_TARGET,
                "absolute_deviation": abs(deriv_order - LIMITED_DERIV_TARGET),
            },
        }

    categories = {
        "finite_grid_inverse_behavior": inverse_by_degree,
        "nodal_interpolation": interpolation_by_degree,
        "smooth_nonbandlimited_orders": smooth_by_degree,
        "limited_regularity_orders": limited_by_degree,
    }
    decisions = {}
    for name, per_degree in categories.items():
        passed = all(entry["status"] == "PASS" for entry in per_degree.values())
        decisions[name] = {
            "status": "PASS" if passed else "FAIL",
            "per_degree": per_degree,
        }
    return decisions


def main() -> int:
    n_conv = [16, 24, 32, 48, 64, 96, 128]
    result = {
        "meta": {
            "theorem": "thm:offgrid (off-grid value order p+1, derivative order p+1-r)",
            "domain": "[0, 2pi), periodic",
            "eval_set": "20000 cell-interior points, N-independent, sup-norm",
            "degrees": DEGREES,
            "grids_convergence": n_conv,
        },
        "check1_finite_grid_inverse_behavior": finite_grid_inverse_report(),
        "check2_smooth_nonbandlimited_exp_sin": convergence_report(
            field_smooth, "exp(sin x)  [analytic, all Fourier modes]", n_conv
        ),
        "check3_limited_regularity_1mcos_pow_1p25": convergence_report(
            field_limited,
            "(1-cos x)^{5/4} ~ |x|^{5/2}  [smoothness s=5/2, hypothesis violated]",
            n_conv,
        ),
        "interpolation_residual_at_nodes": interpolation_check(),
    }
    result["acceptance_criteria"] = acceptance_criteria()
    result["decisions"] = acceptance_decisions(result)
    result["overall_status"] = (
        "PASS"
        if all(entry["status"] == "PASS" for entry in result["decisions"].values())
        else "FAIL"
    )

    out_path = HERE.parent / "results" / "evidence" / "offgrid_theorem_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    # Console summary
    print("== (1) finite-grid inverse behavior and reciprocal decay ==")
    for p in DEGREES:
        r = result["check1_finite_grid_inverse_behavior"][f"p{p}"]
        print(
            f"  p={p}: kappa={r['l2_condition_kappa']:.3g}  "
            f"empirical ||B^-1||_inf(N=8192)="
            f"{r['inverse_inf_norm_vs_N'][-1]['norm']:.4f}  "
            f"tail_delta={r['empirical_stabilization_witness']['maximum_absolute_difference']:.3e}  "
            f"stabilized={r['empirical_stabilization_witness']['status']}  "
            f"lambda_DMS={r['dms_per_index_rate_lambda']:.3f} "
            f"rho_meas={r['measured_fourier_decay_rho']:.3f} "
            f"(rho<=lambda:{r['measured_rho_below_dms_lambda']})"
        )
    for key, tag in [
        ("check2_smooth_nonbandlimited_exp_sin", "(2) exp(sin x) smooth"),
        ("check3_limited_regularity_1mcos_pow_1p25", "(3) (1-cos x)^1.25 s=5/2"),
    ]:
        print(f"== {tag} ==")
        for p in DEGREES:
            r = result[key][f"p{p}"]
            print(
                f"  p={p}: value order={r['value_order_fit']:.2f} "
                f"(expect {r['expected_value_order']})   "
                f"deriv order={r['deriv_order_fit']:.2f} (expect {r['expected_deriv_order']})"
            )
    print("== acceptance decisions ==")
    for name, decision in result["decisions"].items():
        print(f"  {name}: {decision['status']}")
    print(f"  overall_status: {result['overall_status']}")
    print(f"\nwrote {out_path}")
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
