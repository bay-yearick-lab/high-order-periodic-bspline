"""Regression tests for the new periodic baselines and off-grid evaluator."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import experiment_ibm
import spline_analysis as sa


class NewBaselineSymbolTests(unittest.TestCase):
    def test_drp_tam_webb_is_antisymmetric_seven_point(self) -> None:
        mass, advection = sa.drp_tam_webb_entries()
        self.assertEqual(mass.shape, (1,))
        self.assertEqual(advection.shape, (4,))
        self.assertEqual(advection[0], 0.0)

    def test_drp_symbol_is_purely_imaginary_pattern(self) -> None:
        w = np.linspace(0.05, np.pi - 0.05, 100)
        kp = sa.kprime_drp_tam_webb(w)
        self.assertTrue(np.all(np.isfinite(kp)))
        # DRP is fourth-order: exact at small w, deviates at large w
        self.assertAlmostEqual(float(kp[0]) / float(w[0]), 1.0, places=2)

    def test_bogey_bailly_is_eleven_point(self) -> None:
        mass, advection = sa.bogey_bailly_entries()
        self.assertEqual(mass.shape, (1,))
        self.assertEqual(advection.shape, (6,))
        self.assertEqual(advection[0], 0.0)

    def test_bogey_bailly_resolves_more_than_drp(self) -> None:
        # At a moderately short wave, the optimized 11-point stencil should
        # carry less phase error than the 7-point DRP.
        w = 1.6
        err_drp = abs(sa.kprime_drp_tam_webb(np.array([w]))[0] - w)
        err_bb = abs(sa.kprime_bogey_bailly(np.array([w]))[0] - w)
        self.assertLess(err_bb, err_drp)

    def test_pseudospectral_symbol_equals_w(self) -> None:
        w = np.linspace(0.0, np.pi, 50)
        np.testing.assert_allclose(sa.kprime_pseudospectral(w), w)


class OffGridEvaluationTests(unittest.TestCase):
    def test_periodic_basis_matrix_recovers_identity_at_constants(self) -> None:
        for degree in (3, 5, 7):
            B = sa.cyclic_collocation_basis_matrix(degree, 32).toarray()
            ones = np.ones(32)
            np.testing.assert_allclose(B @ ones, ones, atol=1e-12)

    def test_off_grid_evaluation_is_exact_for_sampled_wave_at_grid(self) -> None:
        npts = 32
        h = 1.0 / npts
        x = np.arange(npts) * h
        u = np.sin(2.0 * np.pi * x)
        for degree in (3, 5, 7):
            alpha = sa.coefficients_from_nodal_values(u, degree)
            u_at_grid = sa.evaluate_periodic_spline(alpha, x, h, degree)
            np.testing.assert_allclose(u_at_grid, u, atol=1e-12)

    def test_off_grid_convergence_matches_predicted_order(self) -> None:
        rng = np.random.default_rng(7)
        x_test = rng.uniform(0.0, 1.0, 40)
        for degree in (3, 5, 7):
            errs = []
            for npts in (32, 64, 128):
                h = 1.0 / npts
                x = np.arange(npts) * h
                u = np.sin(2.0 * np.pi * x) + 0.4 * np.cos(6.0 * np.pi * x)
                alpha = sa.coefficients_from_nodal_values(u, degree)
                u_eval = sa.evaluate_periodic_spline(alpha, x_test, h, degree)
                u_true = np.sin(2.0 * np.pi * x_test) + 0.4 * np.cos(
                    6.0 * np.pi * x_test
                )
                errs.append(float(np.max(np.abs(u_eval - u_true))))
            # Empirical order from the last refinement
            empirical_order = np.log2(errs[-2] / errs[-1])
            expected = degree + 1
            # Allow some slack from finite mesh-resolution effects
            self.assertGreater(empirical_order, expected - 0.7)

    def test_off_grid_derivative_convergence(self) -> None:
        rng = np.random.default_rng(11)
        x_test = rng.uniform(0.0, 1.0, 40)
        degree = 5
        errs = []
        for npts in (32, 64, 128):
            h = 1.0 / npts
            x = np.arange(npts) * h
            u = np.sin(2.0 * np.pi * x) + 0.4 * np.cos(4.0 * np.pi * x)
            alpha = sa.coefficients_from_nodal_values(u, degree)
            ux_h = sa.evaluate_periodic_spline(alpha, x_test, h, degree, derivative=1)
            ux_true = (
                2.0 * np.pi * np.cos(2.0 * np.pi * x_test)
                - 0.4 * 4.0 * np.pi * np.sin(4.0 * np.pi * x_test)
            )
            errs.append(float(np.max(np.abs(ux_h - ux_true))))
        empirical_order = np.log2(errs[-2] / errs[-1])
        # Derivative loses one order: degree 5 -> first derivative tracks ~5
        self.assertGreater(empirical_order, 4.0)


class IbmManufacturedSolutionTests(unittest.TestCase):
    def test_manufactured_solution_is_periodic_on_torus(self) -> None:
        _, _, u_func, laplacian_u = experiment_ibm.manufactured_solution(16)
        box = 2.0 * np.pi
        points = np.array(
            [
                [0.13, 0.21],
                [1.70, 2.40],
                [5.80, 4.10],
            ]
        )

        for shift in (np.array([box, 0.0]), np.array([0.0, box]), np.array([box, box])):
            np.testing.assert_allclose(u_func(points + shift), u_func(points), atol=1e-13)
            np.testing.assert_allclose(
                laplacian_u(points + shift),
                laplacian_u(points),
                atol=1e-13,
            )


class PeriodicIdentityCheckTests(unittest.TestCase):
    def test_BMC2p1_equals_BMGp_first_derivative_entries(self) -> None:
        for p in (1, 2, 3, 4):
            bmc_mass, bmc_adv = sa.collocation_mass_and_advection_entries(2 * p + 1)
            bmg_mass, bmg_adv = sa.iga_mass_and_advection_entries(p)
            np.testing.assert_allclose(bmc_mass, bmg_mass, atol=1e-13)
            np.testing.assert_allclose(bmc_adv, bmg_adv, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
