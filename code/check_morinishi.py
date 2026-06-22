"""Verify the periodic Morinishi skew-symmetric product identity in 2D.

Checks Theorem~\\ref{thm:periodic-morinishi} on the periodic torus
``T^2``: with a discretely divergence-free velocity ``(u, v)`` the
Morinishi convective operator

  C(u, v; phi) = 0.5 * ( D_x (u * phi) + u * D_x phi
                       + D_y (v * phi) + v * D_y phi )

satisfies the discrete skew-symmetric orthogonality

  <C(u, v; phi), phi>_H = 0.5 * <D_x u + D_y v, phi^2>_H = 0.

The test enforces the 2D divergence-free constraint by FFT-projecting
random 2D Fourier velocity fields in the operator's own modified-
wavenumber metric (the same spectral projection used by the vendored
Navier--Stokes solver in ``code/solver``). The orthogonality then must
hold to round-off.
"""

from __future__ import annotations

import sys

import numpy as np

import spline_analysis as sa


def first_derivative_symbol(label: str, w: np.ndarray) -> np.ndarray:
    if label.startswith("BMC"):
        return sa.kprime_collocation(w, int(label[3:]))
    if label.startswith("BMG"):
        return sa.kprime_iga(w, int(label[3:]))
    raise ValueError(f"Unknown label {label!r}")


def first_derivative_matrix(label: str, npts: int, h: float) -> np.ndarray:
    if label.startswith("BMC"):
        return sa.collocation_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("BMG"):
        return sa.iga_derivative_matrix(int(label[3:]), npts, h=h)
    raise ValueError(f"Unknown label {label!r}")


def divergence_free_random(label: str, npts: int, h: float, rng) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * np.fft.fftfreq(npts, d=1.0)
    kprime = first_derivative_symbol(label, omega)
    eig = (kprime / h) ** 2
    eig2d = eig[:, None] + eig[None, :]
    null_mask = eig2d <= 1e-14
    eig2d_safe = np.where(null_mask, 1.0, eig2d)
    inv_lap = np.where(null_mask, 0.0, -1.0 / eig2d_safe)
    u = rng.standard_normal((npts, npts))
    v = rng.standard_normal((npts, npts))

    D1 = first_derivative_matrix(label, npts, h)
    div = D1 @ u + v @ D1.T
    rhs_hat = np.fft.fft2(div)
    p_hat = inv_lap * rhs_hat
    p = np.real(np.fft.ifft2(p_hat))
    return u - D1 @ p, v - p @ D1.T


def check(label: str, npts: int, tol: float, n_tests: int = 4) -> float:
    h = 1.0 / npts
    D1 = first_derivative_matrix(label, npts, h)
    rng = np.random.default_rng(0xBEE_F00D)
    worst = 0.0
    for _ in range(n_tests):
        u, v = divergence_free_random(label, npts, h, rng)
        phi = rng.standard_normal((npts, npts))
        phi -= phi.mean()
        cx = 0.5 * (D1 @ (u * phi) + u * (D1 @ phi))
        cy = 0.5 * ((v * phi) @ D1.T + v * (phi @ D1.T))
        morinishi = cx + cy
        inner = (h * h) * float(np.sum(morinishi * phi))
        kinetic = (h * h) * float(np.sum(phi * phi)) + 1e-30
        worst = max(worst, abs(inner) / kinetic)
    print(f"{label:<8s} N={npts:>3d}  max relative |<C, phi>_H| / <phi, phi>_H = {worst:.3e}")
    if worst > tol:
        raise SystemExit(
            f"Morinishi orthogonality failed for {label} N={npts}: rel={worst:.3e}, tol={tol:.0e}"
        )
    return worst


def main() -> int:
    tol = 1e-10
    for npts in (32, 48, 64):
        for label in ("BMC6", "BMC7", "BMG2", "BMG3"):
            check(label, npts, tol)
    print("Morinishi 2D divergence-free orthogonality passes at tol", tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
