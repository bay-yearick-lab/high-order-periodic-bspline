import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import experiment_ibm as ibm


class IbmHelmholtzNullspaceTests(unittest.TestCase):
    """Gate for the kappa=0 kernel handling of periodic_helmholtz_solver.

    The kappa=0 branch is latent (no current call site reaches it), so the
    solver-level suites never exercise it. This gate pins the fix directly:
    the composed |D_h|^2 Laplacian at kappa=0 has kernel {0, Nyquist}^2 for a
    scheme whose first-derivative symbol vanishes at omega=pi, not merely the
    mean. The pre-fix mean-only pin divided a Nyquist right-hand side by a
    round-off symbol; the fix pins the whole kernel by a relative cut.
    """

    LABEL = "E6"          # centered explicit: kp(pi) = 0, so Nyquist is kernel
    N = 32

    def _denom(self):
        h = 2.0 * np.pi / self.N
        w = np.fft.fftfreq(self.N, d=1.0) * 2.0 * np.pi
        w = np.where(w >= 0, w, w + 2.0 * np.pi)
        w = np.where(w <= np.pi, w, w - 2.0 * np.pi)
        w = np.abs(w)
        kp = ibm.first_derivative_symbol(self.LABEL, w)
        sym1 = (kp * kp) / (h * h)
        return sym1[:, None] + sym1[None, :], h

    def _mode(self, i, j):
        x = np.arange(self.N)
        return np.real(np.outer(np.exp(2j * np.pi * i * x / self.N),
                                np.exp(2j * np.pi * j * x / self.N)))

    def test_kernel_is_mean_plus_all_nyquist_pairs(self) -> None:
        denom, _ = self._denom()
        kern = np.abs(denom) <= 1e-10 * np.max(np.abs(denom))
        idx = sorted(map(tuple, np.argwhere(kern).tolist()))
        nyq = self.N // 2
        self.assertEqual(idx, [(0, 0), (0, nyq), (nyq, 0), (nyq, nyq)])

    def test_kernel_mode_rhs_is_projected_out(self) -> None:
        solve = ibm.periodic_helmholtz_solver(self.LABEL, self.N, 2.0 * np.pi / self.N, kappa=0.0)
        nyq = self.N // 2
        for (i, j) in [(0, nyq), (nyq, 0), (nyq, nyq)]:
            out = solve(self._mode(i, j))
            self.assertLess(float(np.max(np.abs(out))), 1e-12)

    def test_compatible_rhs_round_trips_through_operator(self) -> None:
        denom, _ = self._denom()
        kern = np.abs(denom) <= 1e-10 * np.max(np.abs(denom))
        solve = ibm.periodic_helmholtz_solver(self.LABEL, self.N, 2.0 * np.pi / self.N, kappa=0.0)
        f = (np.sin(np.arange(self.N) * 2 * np.pi / self.N)[:, None]
             * np.cos(2 * np.arange(self.N) * 2 * np.pi / self.N)[None, :])
        fhat = np.fft.fft2(f)
        fhat[kern] = 0.0
        f = np.real(np.fft.ifft2(fhat))
        u = solve(f)
        Lu = np.real(np.fft.ifft2(np.fft.fft2(u) * denom))
        self.assertLess(float(np.max(np.abs(Lu - f)) / np.max(np.abs(f))), 1e-10)


if __name__ == "__main__":
    unittest.main()
