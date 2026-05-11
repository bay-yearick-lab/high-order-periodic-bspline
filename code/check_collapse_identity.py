"""Verify the periodic flux/Galerkin collapse identities at machine precision.

Checks the two entrywise statements proved in
Section~\\ref{sec:identity}:

  BMC_{2 p + 1} = BMG_p   (first derivative)
  BMC_{2 p + 1}^{(2)} = BMG_p^{(2)}   (direct second derivative)

for spline degrees ``p in {1, 2, 3, 4, 5}`` on a periodic grid of size
``N = 64``. Exits with a non-zero status if any entrywise difference
exceeds ``tol = 1e-12``.
"""

from __future__ import annotations

import sys

import numpy as np

import spline_analysis as sa


def check(p: int, npts: int, tol: float) -> tuple[float, float]:
    h = 1.0
    bmc_first = sa.collocation_derivative_matrix(2 * p + 1, npts, h=h)
    bmg_first = sa.iga_derivative_matrix(p, npts, h=h)
    bmc_second = sa.collocation_second_derivative_matrix(2 * p + 1, npts, h=h)
    bmg_second = sa.iga_second_derivative_matrix(p, npts, h=h)
    err_first = float(np.max(np.abs(bmc_first - bmg_first)))
    err_second = float(np.max(np.abs(bmc_second - bmg_second)))
    print(
        f"p={p}  ||BMC_{2*p+1} - BMG_{p}||_inf = {err_first:.3e}  "
        f"||BMC_{2*p+1}^(2) - BMG_{p}^(2)||_inf = {err_second:.3e}"
    )
    if err_first > tol or err_second > tol:
        raise SystemExit(
            f"collapse identity failed for p={p}: first={err_first:.3e}, second={err_second:.3e}, tol={tol:.0e}"
        )
    return err_first, err_second


def main() -> int:
    tol = 1e-12
    for p in (1, 2, 3, 4, 5):
        check(p, 64, tol)
    print("collapse identities pass at tol", tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
