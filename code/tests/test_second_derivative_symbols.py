import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import spline_analysis as sa


class SecondDerivativeSymbolTests(unittest.TestCase):
    def test_explicit_second_derivative_weights_match_classic_centered_stencils(self) -> None:
        np.testing.assert_allclose(sa.explicit_centered_second_derivative_weights(2), [1.0])
        np.testing.assert_allclose(
            sa.explicit_centered_second_derivative_weights(4),
            [4.0 / 3.0, -1.0 / 12.0],
        )
        np.testing.assert_allclose(
            sa.explicit_centered_second_derivative_weights(6),
            [3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0],
        )

    def test_compact_second_derivative_entries_match_slc_c4_and_c6_coefficients(self) -> None:
        mass4, second4 = sa.compact_second_derivative_scheme_entries(4)
        np.testing.assert_allclose(mass4, [1.0, 1.0 / 10.0])
        np.testing.assert_allclose(second4, [-12.0 / 5.0, 6.0 / 5.0])

        mass6, second6 = sa.compact_second_derivative_scheme_entries(6)
        np.testing.assert_allclose(mass6, [1.0, 2.0 / 11.0])
        np.testing.assert_allclose(second6, [-51.0 / 22.0, 12.0 / 11.0, 3.0 / 44.0])

    def test_bspline_direct_second_derivative_symbol_matches_bmg_bmc_identity(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 512)
        for degree in (1, 2, 3):
            with self.subTest(degree=degree):
                np.testing.assert_allclose(
                    sa.ksecond_iga(w, degree),
                    sa.ksecond_collocation(w, 2 * degree + 1),
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )

    def test_second_derivative_order_estimates_recover_known_formal_orders(self) -> None:
        w = np.linspace(1.0e-6, 0.8, 2000)

        e4_order = sa.estimate_second_derivative_order(w, sa.ksecond_centered_fd(w, 4))
        c4_order = sa.estimate_second_derivative_order(w, sa.ksecond_compact(w, 4))
        c6_order = sa.estimate_second_derivative_order(w, sa.ksecond_compact(w, 6))

        self.assertAlmostEqual(e4_order, 4.0, delta=0.2)
        self.assertAlmostEqual(c4_order, 4.0, delta=0.2)
        self.assertAlmostEqual(c6_order, 6.0, delta=0.2)

    def test_second_derivative_summary_reports_finite_resolution_and_stability_metrics(self) -> None:
        w = np.linspace(1.0e-6, np.pi, 4000)
        metrics = sa.summarize_second_derivative_curve(w, sa.ksecond_compact(w, 6))

        self.assertGreater(metrics.spectral_radius, 0.0)
        self.assertGreater(metrics.rk4_parabolic_cfl, 0.0)
        self.assertGreater(metrics.omega_pi_1pct, 0.0)
        self.assertGreater(metrics.omega_pi_5pct, metrics.omega_pi_1pct)
        self.assertGreater(metrics.ppw_1pct, 2.0)
        self.assertGreater(metrics.ppw_5pct, 2.0)


if __name__ == "__main__":
    unittest.main()