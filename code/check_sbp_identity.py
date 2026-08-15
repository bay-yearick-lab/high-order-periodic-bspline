"""Verify the diagonal-norm SBP identity for the periodic operators.

Checks Theorem~\\ref{thm:periodic-green} at the matrix level:

  H D + D^T H = 0,    H = h I,

for the BMC and BMG first-derivative operators on a periodic grid.
The check covers every admitted degree in the BMC2 through BMC11 catalog plus
the matched BMG operators; it also sanity-checks the self-adjoint
nonpositive direct second derivative

  H D^(2) = (H D^(2))^T,   <D^(2) u, u>_H <= 0   for u not equal to constants,

over a small set of random periodic test fields.
"""

from __future__ import annotations

import sys

import numpy as np

import spline_analysis as sa


def check_sbp_first_derivative(label: str, D: np.ndarray, h: float, tol: float) -> float:
    H = sa.periodic_sbp_norm(D.shape[0], h=h)
    residual = H @ D + D.T @ H
    err = float(np.max(np.abs(residual)))
    print(f"{label:<8s} ||HD + D^T H||_inf = {err:.3e}")
    if err > tol:
        raise SystemExit(
            f"SBP first-derivative identity failed for {label}: err={err:.3e}, tol={tol:.0e}"
        )
    return err


def check_self_adjoint_nonpositive(
    label: str, D2: np.ndarray, h: float, tol: float, n_tests: int = 4
) -> tuple[float, float]:
    H = sa.periodic_sbp_norm(D2.shape[0], h=h)
    sym_residual = H @ D2 - (H @ D2).T
    sym_err = float(np.max(np.abs(sym_residual)))
    rng = np.random.default_rng(0xC0DE_BEEF)
    worst_pos = -np.inf
    for _ in range(n_tests):
        u = rng.standard_normal(D2.shape[0])
        u -= u.mean()
        worst_pos = max(worst_pos, float(u @ H @ D2 @ u))
    print(
        f"{label:<8s} ||HD^(2) - (HD^(2))^T||_inf = {sym_err:.3e}  "
        f"max <D^(2) u, u>_H = {worst_pos:.3e}"
    )
    if sym_err > tol:
        raise SystemExit(
            f"D^(2) self-adjointness failed for {label}: err={sym_err:.3e}, tol={tol:.0e}"
        )
    if worst_pos > 1e-9:
        raise SystemExit(
            f"D^(2) nonpositivity failed for {label}: max u^T H D^(2) u = {worst_pos:.3e}"
        )
    return sym_err, worst_pos


def main() -> int:
    npts = 64
    h = 1.0 / npts
    tol = 1e-10
    for p in range(2, 12):
        D = sa.collocation_derivative_matrix(p, npts, h=h)
        check_sbp_first_derivative(f"BMC{p}", D, h, tol)
        D2 = sa.collocation_second_derivative_matrix(p, npts, h=h)
        check_self_adjoint_nonpositive(f"BMC{p}", D2, h, tol)
    for p in range(1, 6):
        Dg = sa.iga_derivative_matrix(p, npts, h=h)
        check_sbp_first_derivative(f"BMG{p}", Dg, h, tol)
        D2g = sa.iga_second_derivative_matrix(p, npts, h=h)
        check_self_adjoint_nonpositive(f"BMG{p}", D2g, h, tol)
    print("SBP first-derivative and self-adjoint nonpositive D^(2) identities pass at tol", tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
