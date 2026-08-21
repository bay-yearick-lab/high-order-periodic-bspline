"""Verify periodic discrete gradient-divergence adjointness.

Checks Corollary~\\ref{cor:periodic-gradient-divergence}:

  <u, D p>_H + <p, D u>_H = 0   for periodic u, p,

which is the diagonal-norm SBP discrete Green identity applied to the
pair ``(u, p)``. On the periodic uniform grid the right-hand side
vanishes for every pair of periodic grid functions because there is no
boundary, so the discrete pressure-gradient force does no net work
against the velocity field once the divergence-free constraint is
enforced.
"""

from __future__ import annotations

import sys

import numpy as np

import spline_analysis as sa


def check(label: str, D: np.ndarray, h: float, tol: float, n_tests: int = 6) -> float:
    rng = np.random.default_rng(0x12345)
    worst = 0.0
    for _ in range(n_tests):
        npts = D.shape[0]
        u = rng.standard_normal(npts)
        u -= np.mean(u)
        p = rng.standard_normal(npts)
        p -= np.mean(p)
        lhs = h * float(np.dot(u, D @ p)) + h * float(np.dot(p, D @ u))
        norm = h * float(np.dot(u, u) + np.dot(p, p)) + 1e-30
        rel = abs(lhs) / norm
        worst = max(worst, rel)
    print(f"{label:<8s} max relative |<u, Dp>_H + <p, Du>_H| = {worst:.3e}")
    if worst > tol:
        raise SystemExit(
            f"gradient-divergence adjointness failed for {label}: rel={worst:.3e}, tol={tol:.0e}"
        )
    return worst


def main() -> int:
    tol = 1e-10
    for npts in (32, 48, 64):
        h = 1.0 / npts
        for p in range(2, 9):
            D = sa.collocation_derivative_matrix(p, npts, h=h)
            check(f"BMC{p} N={npts}", D, h, tol)
        for p in range(1, 6):
            D = sa.iga_derivative_matrix(p, npts, h=h)
            check(f"BMG{p} N={npts}", D, h, tol)
    print("gradient-divergence adjointness passes at tol", tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
