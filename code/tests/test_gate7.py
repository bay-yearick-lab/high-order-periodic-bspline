"""Regression tests for Gate 7: the family with real parts, the compact
ceiling LP, and the dominance search machinery.

The theory is in results/exponential_family/gate7/theory.md. The tests check
the slot factor against the Gate 0 pair factor and its limits, the alias
factorization (Lemma D'), the sign structure of the error (Theorem Z), the
nonnegativity consequences (Lemma 4), the decimal route, the positivity
certificates, the parameter counts, the LP certificate inequality, the
consistency polish, and the grid-independent reach.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
FAMILY_DIR = CODE_DIR / "analysis" / "exponential_family"
for _path in (str(CODE_DIR), str(FAMILY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import family as fam  # noqa: E402
import gate7_ceiling as gc  # noqa: E402
import gate7_dominance as gd  # noqa: E402
import gate7_family as g7  # noqa: E402
import metrics as met  # noqa: E402

PI = np.pi
CHAMPION = [(0.0, 1.7701927977475826), (0.0, 2.0606855355641316)]


class SlotFactorTests(unittest.TestCase):
    def test_imaginary_slot_is_the_gate0_pair_factor(self) -> None:
        xi = np.linspace(-20.0, 20.0, 801)
        gen = g7.ExtendedGenerator(6, CHAMPION)
        ref = fam.ExponentialGenerator(6, [w for _, w in CHAMPION])
        np.testing.assert_allclose(gen.fourier(xi), ref.fourier(xi), rtol=0, atol=1e-15)

    def test_every_slot_kind_is_normalized_real_and_even(self) -> None:
        xi = np.linspace(-25.0, 25.0, 1001)
        for slots in ([(2.0, 0.0)], [(1.5, 2.0)], [(0.0, 1.0), (0.7, 0.0), (0.4, 2.5)]):
            gen = g7.ExtendedGenerator(4, slots)
            self.assertAlmostEqual(float(gen.fourier(np.array([0.0]))[0]), 1.0, places=14)
            np.testing.assert_allclose(gen.fourier(xi), gen.fourier(-xi), atol=1e-16)

    def test_real_and_quad_factors_have_no_real_zero(self) -> None:
        xi = np.linspace(-40.0, 40.0, 4001)
        self.assertGreater(float(g7.slot_factor(g7.Slot(1.3, 0.0), xi).min()), 0.0)
        self.assertGreater(float(g7.slot_factor(g7.Slot(0.8, 2.2), xi).min()), 0.0)
        imag = g7.slot_factor(g7.Slot(0.0, 2.2), xi)
        self.assertLess(float(imag.min()), 0.0)

    def test_quad_with_vanishing_real_part_is_the_double_pair(self) -> None:
        xi = np.linspace(-15.0, 15.0, 301)
        quad = g7.ExtendedGenerator(4, [(1e-9, 1.9)])
        double = g7.ExtendedGenerator(4, [(0.0, 1.9), (0.0, 1.9)])
        np.testing.assert_allclose(quad.fourier(xi), double.fourier(xi), atol=1e-12)

    def test_large_real_part_removes_the_slot(self) -> None:
        xi = np.linspace(-15.0, 15.0, 301)
        wide = g7.ExtendedGenerator(4, [(0.0, 1.9), (39.0, 0.0)])
        narrow = fam.ExponentialGenerator(4, (1.9,))
        self.assertLess(float(np.max(np.abs(wide.fourier(xi) - narrow.fourier(xi)))), 2e-3)
        self.assertEqual(wide.n, narrow.m + 2 * narrow.r + 2)

    def test_exponent_and_parameter_counts(self) -> None:
        gen = g7.ExtendedGenerator(4, [(0.0, 1.0), (2.0, 0.0), (1.0, 2.0)])
        self.assertEqual((gen.n, gen.p, gen.nmax, gen.q, gen.parameters), (12, 11, 6, 4, 4))
        counts = g7.parameter_counts(10)
        self.assertEqual(counts["compact_free_parameters"], 7)
        self.assertEqual(counts["family"]["4"]["parameters"], 3)
        self.assertEqual(counts["family"]["4"]["gap_to_compact"], 4)


class AliasStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = g7.ExtendedGenerator(4, [(0.0, 1.2), (1.0, 2.2), (0.0, 2.4)])
        self.theta = np.linspace(0.02, PI - 0.01, 400)

    def test_lemma_D_factorization(self) -> None:
        self.assertLess(g7.lemma_D_residual(self.gen, self.theta), 1e-10)

    def test_theorem_Z_zeros_are_the_imaginary_frequencies(self) -> None:
        structure = g7.error_sign_structure(self.gen, self.theta)
        self.assertTrue(structure["alias_sum_all_negative"])
        self.assertLess(structure["e1_factored_vs_direct"], 1e-13)
        e1 = structure["e1"]
        zeros = self.theta[np.flatnonzero(np.sign(e1[:-1]) * np.sign(e1[1:]) < 0)]
        self.assertEqual(zeros.size, 2)
        for z, w in zip(zeros, self.gen.omegas):
            self.assertLess(abs(z - w), self.theta[1] - self.theta[0])
        self.assertTrue(np.all(e1[self.theta > max(self.gen.omegas)] < 0.0))

    def test_real_pairs_only_give_negative_error(self) -> None:
        gen = g7.ExtendedGenerator(4, [(1.0, 0.0), (2.0, 0.0), (0.5, 2.0)])
        theta = np.linspace(0.05, PI - 1e-6, 2000)
        e1 = gen.symbols(theta)[1] / theta - 1.0
        self.assertTrue(np.all(e1 < 0.0))
        self.assertAlmostEqual(float(e1[-1]), -1.0, places=4)

    def test_lemma_4_unit_maximum_of_the_symbol(self) -> None:
        theta = np.linspace(0.0, PI, 2001)
        for slots in ([(2.0, 0.0), (0.0, 2.5)], [(1.0, 0.8 * PI)], [(0.3, 0.0), (0.7, 0.6 * PI)]):
            b = g7.ExtendedGenerator(4, slots).symbols(theta)[0]
            self.assertAlmostEqual(float(b[0]), 1.0, places=13)
            self.assertLessEqual(float(b.max()), 1.0 + 1e-13)

    def test_sufficient_conditions_and_certificates(self) -> None:
        conditions = g7.sufficient_conditions(self.gen)
        self.assertTrue(conditions["theorem_P"] and conditions["theorem_N"])
        cert = g7.certify_positivity(self.gen, points=50001)
        self.assertTrue(cert["b_certified"] and cert["kappa2b_certified"])
        self.assertAlmostEqual(cert["kappa2b_second_moment"], 1.0, places=12)
        no_imag = g7.sufficient_conditions(g7.ExtendedGenerator(4, [(1.0, 0.0), (2.0, 0.0)]))
        self.assertEqual(no_imag["S_m"], 0.0)
        self.assertTrue(no_imag["theorem_P"])


class RoutesTests(unittest.TestCase):
    def test_decimal_route_and_symbols_agree_for_a_mixed_member(self) -> None:
        gen = g7.ExtendedGenerator(4, [(0.0, 2.0), (1.5, 0.0), (0.7, 2.3)])
        report = g7.verify_routes(gen)
        self.assertLess(report["decimal_deviation"], 1e-13)
        self.assertLess(report["symbols_from_samples_versus_alias"], g7.SYMBOL_ROUTE_TOL)
        self.assertTrue(report["decimal_ok"] and report["symbols_ok"])

    def test_decimal_route_matches_gate1_for_the_champion(self) -> None:
        gen = g7.ExtendedGenerator(6, CHAMPION)
        report = g7.verify_routes(gen)
        self.assertLess(report["decimal_deviation"], 1e-13)
        self.assertLess(report["gate0_family_deviation"], 1e-14)

    def test_member_record_schema(self) -> None:
        gen = g7.ExtendedGenerator(4, [(0.0, 2.0), (1.5, 0.0), (0.7, 2.3)])
        rec = gen.member_record("test")
        for key in ("m", "r", "omegas", "real_pairs", "quads", "exponents", "parameters",
                    "p", "radius", "cond", "beta", "dbeta", "ddbeta"):
            self.assertIn(key, rec)
        self.assertEqual(len(rec["beta"]), rec["radius"] + 1)
        self.assertEqual(rec["exponents"], 12)
        back = g7.from_record(rec)
        self.assertEqual(back.label(), gen.label())


class CeilingTests(unittest.TestCase):
    def test_slc_is_feasible_below_its_own_reach_and_not_above_the_ceiling(self) -> None:
        C = met.symbol_cond(comp.compact_left(comp.SLC_D1["alpha"]))["cond"]
        low = gc.solve_lp(gc.build_lp(2, 3, C, 1e-3, 0.78 * PI, 1))
        self.assertGreaterEqual(low["delta"], 0.0)
        high = gc.solve_lp(gc.build_lp(2, 3, C, 1e-3, 0.9 * PI, 1))
        self.assertLess(high["delta_upper_bound"], -1e-6)
        self.assertGreaterEqual(high["delta_upper_bound"], high["delta"] - 1e-9)

    def test_dual_bound_never_undercuts_the_primal(self) -> None:
        for derivative in (1, 2):
            sol = gc.solve_lp(gc.build_lp(2, 3, 50.0, 1e-3, 0.6 * PI, derivative))
            self.assertGreaterEqual(sol["delta_upper_bound"] + 1e-9, sol["delta"])

    def test_polish_enforces_consistency_and_certification_is_within_budget(self) -> None:
        sol = gc.solve_lp(gc.build_lp(3, 3, 60.0, 1e-2, 0.7 * PI, 1))
        left, right = gc.polish_consistency(sol["left"], sol["right"], 1)
        L0 = left[0] + 2.0 * sum(left[1:])
        self.assertAlmostEqual(2.0 * sum(s * a for s, a in enumerate(right, start=1)), L0, places=12)
        cert = gc.certify_scheme(sol["left"], sol["right"], 1, 60.0)
        self.assertTrue(cert["cond_within_budget"])
        self.assertGreaterEqual(cert["reach_0.01"], 0.7 - 1e-6)
        sol2 = gc.solve_lp(gc.build_lp(3, 3, 60.0, 1e-4, gc.B0, 2))
        cert2 = gc.certify_scheme(sol2["left"], sol2["right"], 2, 60.0)
        self.assertLessEqual(cert2["band_max_B0"], 1e-4 * (1.0 + 1e-3))


class DominanceTests(unittest.TestCase):
    def test_robust_reach_detects_a_lobe_hidden_between_grid_points(self) -> None:
        theta = met.theta_grid(2001)
        tol = 1e-3
        # a lobe of height 1.02 tol centered between two grid points and two
        # grid spacings wide: the grid sees at most 0.958 tol, the continuum 1.02 tol
        h = theta[1] - theta[0]
        center = 0.5 * (theta[600] + theta[601])
        width = 2.0 * h
        err_fn = lambda t: 1.02 * tol * np.exp(-((np.asarray(t) - center) / width) ** 2) + 0.2 * tol * (np.asarray(t) / 2.5) ** 6
        err = err_fn(theta)
        grid_only = met.reach(theta, err, tol)["linear"]
        robust = gd.robust_reach(theta, err, tol, err_fn)
        self.assertGreater(grid_only, 2.0)
        self.assertLess(robust, center)
        self.assertGreater(robust, center - 2.0 * width)

    def test_robust_reach_agrees_with_metrics_reach_for_a_member(self) -> None:
        gen = g7.ExtendedGenerator(6, CHAMPION)
        theta = met.theta_grid()
        e1_fn = lambda t: gen.symbols(t)[1] / t - 1.0
        e1 = e1_fn(theta)
        for tol in met.REACH_TOLS:
            ref = met.reach(theta, e1, tol, err_fn=e1_fn)["refined"]
            self.assertAlmostEqual(gd.robust_reach(theta, e1, tol, e1_fn), ref, places=9)

    def test_parametrization_and_grid_ordering(self) -> None:
        slots = gd.unpack("IIR", [1.0, 2.0, np.log(3.0)])
        self.assertEqual([(a, w) for a, w in slots[:2]], [(0.0, 1.0), (0.0, 2.0)])
        self.assertAlmostEqual(slots[2][0], 3.0, places=12)
        self.assertEqual(slots[2][1], 0.0)
        self.assertEqual(gd.cell_dimension("QI"), 3)
        grid = gd.grid_of("II")
        self.assertTrue(np.all(grid[:, 0] < grid[:, 1]))
        self.assertTrue(gd.in_box("Q", [np.log(1.0), 2.0]))
        self.assertFalse(gd.in_box("I", [3.2]))

    def test_margins_sign_convention(self) -> None:
        target = {k: 1.0 for k in gd.YARDSTICKS}
        row = {k: (1.1 if gd.HIGHER_IS_BETTER[k] else 0.9) for k in gd.YARDSTICKS}
        self.assertTrue(all(v > 0 for v in gd.margins(row, target).values()))
        row = {k: (0.9 if gd.HIGHER_IS_BETTER[k] else 1.1) for k in gd.YARDSTICKS}
        self.assertTrue(all(v < 0 for v in gd.margins(row, target).values()))

    def test_evaluator_reproduces_gate0_metrics_for_the_champion(self) -> None:
        C = 93.36740899597527
        ev = gd.Evaluator(6, "II", C, safety=0.0)
        row = ev([w for _, w in CHAMPION], refine=True)
        self.assertTrue(row["feasible"])
        self.assertAlmostEqual(row["cond"], 88.71360211607256, places=6)
        self.assertAlmostEqual(row["reach_d1_0.001"], 0.720588, places=4)
        self.assertAlmostEqual(row["e1_max_B0"], 4.289235e-05, delta=2e-9)


if __name__ == "__main__":
    unittest.main()
