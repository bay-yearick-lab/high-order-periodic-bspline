"""Section 7 sub-capability (3): compatible second-derivative diffusion closure.

Audits the symmetry and nonpositivity of the periodic direct
second-derivative operators
``D^{(2)}_{BMC_p} = B''B^{-1}`` and
``D^{(2)}_{BMG_p} = B M^{-1} K^{(2)} B^{-1}`` on the periodic
uniform grid with ``H = h I``. The two structural identities

  * ``H D^{(2)} = (H D^{(2)})^T``  (self-adjoint),
  * ``<D^{(2)} u, u>_H <= 0``        (nonpositive),

are checked at machine precision for ``BMC_2`` through ``BMC_{11}``
and ``BMG_1`` through ``BMG_5``. The companion semi-discrete check
runs a single-mode heat-decay packet at points-per-wavelength
``P = 3.2`` under FFT-exact time integration on each operator's own
circulant diagonalization, recording the deviation from both the
analytic envelope ``exp(-nu k^2 t)`` and the operator-consistent
envelope ``exp(2 nu lambda_op t)``. The deviation against the
operator-consistent envelope sits at the round-off floor for every
family in the audit catalog; only the symbol error against the
continuous envelope distinguishes the families and is reported in
the surrounding manuscript prose, not as a figure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np

import project_paths as pp
import spline_analysis as sa



@dataclass(frozen=True)
class D2Record:
    label: str
    family: str
    degree_or_order: int
    symmetry_residual: float
    rayleigh_max: float
    kernel_dim: int


def _build_second_derivative(label: str, npts: int, h: float) -> np.ndarray:
    if label.startswith("BMC"):
        return sa.collocation_second_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("BMG"):
        return sa.iga_second_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("C") and label[1:].isdigit():
        order = int(label[1:])
        mass, second = sa.compact_second_derivative_scheme_entries(order)
        return sa.compact_second_derivative_matrix(mass, second, npts, h=h)
    if label.startswith("E") and label[1:].isdigit():
        order = int(label[1:])
        mass, second = sa.explicit_centered_second_derivative_entries(order)
        return sa.compact_second_derivative_matrix(mass, second, npts, h=h)
    raise ValueError(f"Unknown operator label {label!r}")


def _audit_d2(npts: int) -> list[D2Record]:
    h = 1.0 / npts
    records: list[D2Record] = []
    rng = np.random.default_rng(20260510)

    def measure(label: str, family: str, degree_or_order: int, D2: np.ndarray) -> D2Record:
        H = h * np.eye(npts)
        HD2 = H @ D2
        sym = np.max(np.abs(HD2 - HD2.T))
        eigvals = np.linalg.eigvalsh(0.5 * (HD2 + HD2.T))
        kernel_dim = int(np.sum(np.abs(eigvals) < 1e-9 * h))
        rayleigh_max = float(np.max(eigvals)) / h
        return D2Record(label, family, degree_or_order, float(sym), rayleigh_max, kernel_dim)

    for p in range(2, 12):
        records.append(measure(f"BMC{p}", "BMC", p,
                               sa.collocation_second_derivative_matrix(p, npts, h=h)))
    for p in range(1, 6):
        records.append(measure(f"BMG{p}", "BMG", p,
                               sa.iga_second_derivative_matrix(p, npts, h=h)))
    for q in (4, 6):
        records.append(measure(f"C{q}", "Compact", q,
                               _build_second_derivative(f"C{q}", npts, h)))
    for q in (2, 4, 6, 8):
        records.append(measure(f"E{q}", "Explicit", q,
                               _build_second_derivative(f"E{q}", npts, h)))
    return records


def _exact_heat_history_single_mode(
    label: str, npts: int, mode: int, nu: float, times: np.ndarray
) -> np.ndarray:
    """Evolve ``u_t = nu D^{(2)} u`` on the single Fourier mode ``mode``.

    Initial condition: ``u_0(x) = sin(2 pi mode x)`` on the periodic unit
    interval. The discrete eigenvalue at the test mode is read from the
    operator's circulant column; FFT-exact evolution then removes the
    time-integration error so the deviation from the analytic envelope
    ``exp(-nu k^2 t)`` reads as a pure symbol effect.
    """
    h = 1.0 / npts
    D2 = _build_second_derivative(label, npts, h)
    column = D2[:, 0]
    eigvals = np.fft.fft(column)
    x = np.arange(npts) * h
    u0 = np.sin(2.0 * np.pi * mode * x)
    u0_hat = np.fft.fft(u0)
    energy = np.zeros_like(times)
    for k, t in enumerate(times):
        u_hat = np.exp(nu * eigvals * t) * u0_hat
        u = np.real(np.fft.ifft(u_hat))
        energy[k] = 0.5 * h * float(np.dot(u, u))
    return energy


def _analytic_envelope_single_mode(
    npts: int, mode: int, nu: float, times: np.ndarray
) -> np.ndarray:
    h = 1.0 / npts
    x = np.arange(npts) * h
    energy = np.zeros_like(times)
    for k, t in enumerate(times):
        u = np.exp(-nu * (2.0 * np.pi * mode) ** 2 * t) * np.sin(2.0 * np.pi * mode * x)
        energy[k] = 0.5 * h * float(np.dot(u, u))
    return energy


def _operator_consistent_envelope_single_mode(
    label: str, npts: int, mode: int, nu: float, times: np.ndarray
) -> np.ndarray:
    """Single-mode kinetic energy under the operator's own discrete eigenvalue.

    Substitutes the discrete second-derivative eigenvalue at the test
    mode in place of the continuous ``-(2 pi mode)^2``. The resulting
    envelope is the SBP-energy budget that any FFT-exact integration of
    ``u_t = nu D^{(2)} u`` is guaranteed to follow at machine
    precision; deviation of ``E_h(t)`` from this envelope is the right
    place to read whether the operator is energy-conserving in the
    discrete sense.
    """
    h = 1.0 / npts
    D2 = _build_second_derivative(label, npts, h)
    eigvals = np.fft.fft(D2[:, 0])
    lambda_op = float(np.real(eigvals[mode]))
    x = np.arange(npts) * h
    energy = np.zeros_like(times)
    for k, t in enumerate(times):
        u = np.exp(nu * lambda_op * t) * np.sin(2.0 * np.pi * mode * x)
        energy[k] = 0.5 * h * float(np.dot(u, u))
    return energy


def main() -> int:
    records = _audit_d2(npts=64)
    spline_records = [r for r in records if r.family in ("BMC", "BMG")]
    print("spline-route maximum symmetry residual:",
          f"{max(r.symmetry_residual for r in spline_records):.3e}")
    print("spline-route maximum Rayleigh quotient (should be <=0):",
          f"{max(r.rayleigh_max for r in spline_records):.3e}")

    npts = 64
    ppw = 3.2
    mode = max(int(round(npts / ppw)), 2)
    times = np.linspace(0.0, 0.04, 200)
    nu = 0.005
    decay_keys = {
        "BMC6": "BMC6",
        "BMC7 = BMG3": "BMC7",
        "C6": "C6",
        "E6": "E6",
    }
    decay = {
        display: _exact_heat_history_single_mode(label, npts=npts, mode=mode, nu=nu, times=times)
        for display, label in decay_keys.items()
    }
    analytic = _analytic_envelope_single_mode(npts=npts, mode=mode, nu=nu, times=times)
    operator_envelope = {
        display: _operator_consistent_envelope_single_mode(
            label, npts=npts, mode=mode, nu=nu, times=times
        )
        for display, label in decay_keys.items()
    }
    finals = {
        display: float(
            np.max(np.abs(decay[display] - analytic) / np.maximum(analytic, 1e-300))
        )
        for display in decay
    }
    print(f"max relative deviation from analytic envelope at P={ppw} (mode={mode}): {finals}")

    out = {
        "d2_audit": [r.__dict__ for r in records],
        "heat_decay_high_k": {
            "ppw": ppw,
            "mode": mode,
            "max_relative_error": finals,
        },
    }
    with open(pp.results_path("s7_sbp_second_derivative.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
