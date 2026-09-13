import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
FAMILY_DIR = CODE_DIR / "analysis" / "exponential_family"
for _path in (str(CODE_DIR), str(FAMILY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp
import family as fam
import gate1_champions as g1
import metrics as met
import spline_analysis as sa

FROZEN_BETA = (0.45570408509669036, 0.24238702597207298,
               0.029461339406865419, 0.00029959207271641975)
FROZEN_COND = 34.18391129180412
FROZEN_REACH_D1 = 0.783922599452434
FROZEN_REACH_COUPLED = 0.7776723532534208


class GeneratorTests(unittest.TestCase):
    def test_transform_is_normalized_even_and_real(self) -> None:
        generator = fam.ExponentialGenerator(4, (0.1, 2.36))
        xi = np.linspace(-12.0, 12.0, 401)
        values = generator.fourier(xi)
        self.assertAlmostEqual(float(generator.fourier(np.array([0.0]))[0]), 1.0, places=14)
        np.testing.assert_allclose(values, generator.fourier(-xi), atol=1e-15)

    def test_degree_and_stencil_half_width(self) -> None:
        self.assertEqual(fam.ExponentialGenerator(4, (0.1, 2.36)).p, 7)
        self.assertEqual(fam.ExponentialGenerator(4, (0.1, 2.36)).nmax, 4)
        self.assertEqual(fam.ExponentialGenerator(6, ()).p, 5)

    def test_tail_bound_decreases_and_meets_the_requested_tolerance(self) -> None:
        generator = fam.ExponentialGenerator(4, (2.0,))
        for q in (0, 1, 2):
            self.assertGreater(generator.tail_bound(q, 64), generator.tail_bound(q, 256))
            K = generator.required_K(q, 1e-14)
            self.assertLessEqual(generator.tail_bound(q, K), 1e-14)
            self.assertGreater(generator.tail_bound(q, K - 1), 1e-14)

    def test_admissibility_rejects_disordered_or_outside_frequencies(self) -> None:
        self.assertTrue(fam.ExponentialGenerator(4, (0.5, 2.0)).admissible()[0])
        self.assertFalse(fam.ExponentialGenerator(4, (2.0, 0.5)).admissible()[0])
        self.assertFalse(fam.ExponentialGenerator(4, (4.0,)).admissible()[0])
        self.assertFalse(fam.ExponentialGenerator(4, (0.0,)).admissible()[0])

    def test_alias_symbols_match_the_trigonometric_polynomial_symbols(self) -> None:
        generator = fam.ExponentialGenerator(4, (0.1, 2.36))
        theta = np.linspace(0.05, np.pi, 17)
        alias = generator.sample_symbols(theta, tol=1e-14)
        trig = generator.symbols(theta, tol=1e-14)
        for q in range(3):
            np.testing.assert_allclose(alias[q], trig[q], atol=1e-12)

    def test_alias_and_zero_factor_routes_agree_on_the_integer_samples(self) -> None:
        generator = fam.ExponentialGenerator(4, (0.7, 2.1))
        direct = generator._samples_from_alias(1e-14)
        reduced = generator._samples_from_differences(1e-14)
        np.testing.assert_allclose(direct.beta, reduced.beta, atol=1e-13)
        np.testing.assert_allclose(direct.dbeta, reduced.dbeta, atol=1e-13)
        np.testing.assert_allclose(direct.ddbeta, reduced.ddbeta, atol=1e-13)

    def test_samples_carry_the_partition_of_unity_and_derivative_sum_rules(self) -> None:
        samples = fam.ExponentialGenerator(6, (1.0, 2.0)).integer_samples()
        self.assertAlmostEqual(float(samples.beta.sum()), 1.0, places=13)
        self.assertAlmostEqual(float(samples.dbeta.sum()), 0.0, places=13)
        self.assertAlmostEqual(float(samples.ddbeta.sum()), 0.0, places=13)
        self.assertAlmostEqual(float((samples.n * samples.ddbeta).sum()), 0.0, places=13)

    def test_offgrid_transfer_routes_agree(self) -> None:
        generator = fam.ExponentialGenerator(4, (0.1, 2.36))
        theta = np.array([0.3, 1.1, 2.0])
        xi = np.linspace(-0.5, 0.5, 9)
        direct = generator.offgrid_transfers(theta, xi)
        n, s0, s1, s2 = generator.offgrid_samples(xi)
        arg = np.outer(theta, n.astype(float))
        expo = np.exp(-1j * arg)
        b = generator.symbols(theta)[0]
        shift = np.exp(-1j * np.outer(theta, xi))
        fast = (shift * (expo @ s0) / b[:, None],
                shift * (-1j) * ((expo - 1.0) @ s1) / (theta[:, None] * b[:, None]),
                shift * (-1.0) * ((expo - 1.0 + 1j * arg) @ s2)
                / (theta[:, None] ** 2 * b[:, None]))
        for q in range(3):
            np.testing.assert_allclose(direct[q], fast[q], atol=1e-12)


class PolynomialLimitTests(unittest.TestCase):
    def test_cubic_member_reproduces_the_closed_form_first_derivative_symbol(self) -> None:
        theta = np.linspace(1e-6, np.pi, 2001)
        kappa1 = fam.ExponentialGenerator(4, ()).symbols(theta)[1]
        closed = 3.0 * np.sin(theta) / (2.0 + np.cos(theta))
        self.assertLess(float(np.max(np.abs(kappa1 - closed))), 1e-13)

    def test_polynomial_members_match_the_collocation_symbols(self) -> None:
        theta = np.linspace(1e-6, np.pi, 2001)
        for p in (3, 5, 7, 9, 11):
            d1, d2 = comp.bmc_symbol_check(p, theta)
            self.assertLess(d1, 1e-13, msg=f"first derivative symbol at p={p}")
            self.assertLess(d2, 1e-13, msg=f"second derivative symbol at p={p}")

    def test_polynomial_members_match_the_collocation_integer_samples(self) -> None:
        for p in (3, 5, 7, 9, 11):
            samples = fam.ExponentialGenerator(p + 1, ()).integer_samples()
            values, derivatives = sa.collocation_integer_samples(p)
            for offset in range(min(len(values), samples.nmax if hasattr(samples, "nmax")
                                    else int(samples.n.max()) + 1)):
                index = int(np.flatnonzero(samples.n == offset)[0])
                self.assertAlmostEqual(float(samples.beta[index]), float(values[offset]),
                                       places=12, msg=f"beta at p={p}, n={offset}")
                self.assertAlmostEqual(float(samples.dbeta[index]),
                                       float(derivatives[offset]), places=12,
                                       msg=f"beta prime at p={p}, n={offset}")


class IndependentConstructionTests(unittest.TestCase):
    def test_pair_kernel_has_unit_mass(self) -> None:
        for w in (0.1, 1.0, 2.36, 3.0):
            self.assertAlmostEqual(fam.pair_kernel_mass(w), 1.0, places=12)

    def test_quadrature_convolution_reproduces_the_anchor_samples(self) -> None:
        generator = fam.anchor_generator()
        samples = generator.integer_samples()
        order = np.arange(0, int(samples.n.max()) + 1)
        for derivative, reference in ((0, samples.beta), (1, samples.dbeta),
                                      (2, samples.ddbeta)):
            quad = fam.convolution_samples(4, (0.1, 2.36), order, derivative=derivative)
            np.testing.assert_allclose(quad, reference[samples.n >= 0], atol=1e-12)


class FrozenAnchorTests(unittest.TestCase):
    def test_anchor_integer_samples_match_the_frozen_values(self) -> None:
        samples = fam.anchor_generator().integer_samples()
        got = samples.beta[samples.n >= 0][:4]
        np.testing.assert_allclose(got, FROZEN_BETA, rtol=0.0, atol=1e-12)

    def test_anchor_condition_number_matches_the_frozen_value(self) -> None:
        self.assertAlmostEqual(fam.anchor_generator().cond()["cond"] / FROZEN_COND,
                               1.0, places=12)

    def test_anchor_reaches_match_the_frozen_values(self) -> None:
        generator = fam.anchor_generator()
        theta = met.theta_grid()
        e1 = lambda t: generator.symbols(t)[1] / np.asarray(t, dtype=float) - 1.0
        res = met.reach(theta, e1(theta), 1e-2, err_fn=e1)
        self.assertAlmostEqual(res["refined"] / np.pi / FROZEN_REACH_D1, 1.0, places=9)
        xi = np.linspace(-0.5, 0.5, 129)
        ec = lambda t: generator.coupled_error(t, xi)
        res = met.reach(theta, ec(theta), 1e-2, err_fn=ec)
        self.assertAlmostEqual(res["refined"] / np.pi / FROZEN_REACH_COUPLED,
                               1.0, places=9)


class MetricTests(unittest.TestCase):
    def test_reach_interpolates_the_first_crossing_of_a_linear_curve(self) -> None:
        theta = met.theta_grid(2001)
        res = met.reach(theta, theta, 1e-2)
        self.assertAlmostEqual(res["linear"], 1e-2, places=12)

    def test_reach_takes_the_first_contiguous_crossing(self) -> None:
        theta = met.theta_grid(2001)
        err = np.where(theta < 0.5, 2e-2, 1e-4)
        res = met.reach(theta, err, 1e-2)
        self.assertEqual(res["linear"], 0.0)

    def test_reach_saturates_when_the_curve_never_crosses(self) -> None:
        theta = met.theta_grid(2001)
        res = met.reach(theta, np.full(theta.shape, 1e-6), 1e-2)
        self.assertTrue(res["saturated"])
        self.assertAlmostEqual(res["linear"], float(theta[-1]), places=14)

    def test_reach_refinement_finds_the_exact_crossing_of_a_smooth_curve(self) -> None:
        theta = met.theta_grid(201)
        err_fn = lambda t: np.asarray(t, dtype=float) ** 2
        res = met.reach(theta, err_fn(theta), 1e-2, err_fn=err_fn)
        self.assertAlmostEqual(res["refined"], 0.1, places=12)

    def test_condition_number_of_a_known_tridiagonal_symbol(self) -> None:
        left = comp.compact_left((0.25,))
        out = met.symbol_cond(left, 20001)
        self.assertAlmostEqual(out["cond"], 1.5 / 0.5, places=10)
        self.assertAlmostEqual(out["cond_N64"], 1.5 / 0.5, places=10)

    def test_band_statistics_on_a_constant_curve(self) -> None:
        theta = met.theta_grid(2001)
        err = np.full(theta.shape, 3e-3)
        self.assertAlmostEqual(met.band_max(theta, err, met.BANDS["B0"]), 3e-3, places=15)
        self.assertAlmostEqual(met.band_rms(theta, err, met.BANDS["B0"]), 3e-3, places=15)

    def test_fitted_order_recovers_a_synthetic_power_law(self) -> None:
        theta = met.theta_grid()
        fit = met.fitted_order(theta, 0.5 * theta ** 4)
        self.assertAlmostEqual(fit["order"], 4.0, places=8)

    def test_kappa2_sign_check_accepts_the_cubic_member(self) -> None:
        generator = fam.ExponentialGenerator(4, ())
        out = met.kappa2_sign_check(lambda t: generator.symbols(t)[2])
        for N in met.DISCRETE_N:
            self.assertTrue(out[f"sign_N{N}"])

    def test_group_velocity_difference_matches_the_analytic_derivative(self) -> None:
        generator = fam.ExponentialGenerator(6, (1.0, 2.0))
        theta = met.theta_grid(2001)
        _, error = met.group_velocity(lambda t: generator.symbols(t)[1], theta,
                                      analytic=lambda t: generator.symbol_derivative(t))
        self.assertLess(error, 1e-8)


class ComparatorTests(unittest.TestCase):
    def test_slc_symbols_match_the_contract_closed_forms(self) -> None:
        theta = np.linspace(1e-6, np.pi, 2001)
        al, be = 0.5771439, 0.0896406
        a, b, c = 1.3025166, 0.9935500, 0.03750245
        den = 1.0 + 2.0 * al * np.cos(theta) + 2.0 * be * np.cos(2.0 * theta)
        kappa1 = (a * np.sin(theta) + (b / 2.0) * np.sin(2.0 * theta)
                  + (c / 3.0) * np.sin(3.0 * theta)) / den
        np.testing.assert_allclose(
            comp.compact_kappa1(comp.SLC_D1["alpha"], comp.SLC_D1["a"])(theta),
            kappa1, atol=1e-14)
        al2, be2 = 0.50209266, 0.05569169
        a2, b2, c2 = 0.21564935, 1.7233220, 0.17659730
        den2 = 1.0 + 2.0 * al2 * np.cos(theta) + 2.0 * be2 * np.cos(2.0 * theta)
        kappa2 = (2.0 * a2 * (1.0 - np.cos(theta))
                  + (b2 / 2.0) * (1.0 - np.cos(2.0 * theta))
                  + (2.0 * c2 / 9.0) * (1.0 - np.cos(3.0 * theta))) / den2
        np.testing.assert_allclose(
            comp.compact_kappa2(comp.SLC_D2["alpha"], comp.SLC_D2["a0"],
                                comp.SLC_D2["a"])(theta), kappa2, atol=1e-14)

    def test_budget_condition_numbers_match_the_contract_statement(self) -> None:
        slc1 = met.symbol_cond(comp.compact_left(comp.SLC_D1["alpha"]))["cond"]
        slc2 = met.symbol_cond(comp.compact_left(comp.SLC_D2["alpha"]))["cond"]
        q8 = met.symbol_cond(comp.compact_left(comp.Q8_D1["alpha"]))["cond"]
        bmc11 = fam.ExponentialGenerator(12, ()).cond()["cond"]
        self.assertAlmostEqual(slc1, 93.4, delta=0.1)
        self.assertAlmostEqual(slc2, 19.7, delta=0.1)
        self.assertAlmostEqual(q8, 37.8, delta=0.1)
        self.assertAlmostEqual(bmc11, 112.8, delta=0.1)

    def test_detected_orders_match_the_published_orders(self) -> None:
        for scheme in comp.comparator_schemes():
            detected = scheme.extra.get("detected_order_d1")
            if detected is not None:
                self.assertEqual(detected, scheme.published_order1, msg=scheme.name)
            detected = scheme.extra.get("detected_order_d2")
            if detected is not None:
                self.assertEqual(detected, scheme.published_order2, msg=scheme.name)

    def test_drp_has_no_second_derivative_partner(self) -> None:
        scheme = [s for s in comp.comparator_schemes() if s.name == "DRP-TamWebb"][0]
        self.assertIsNone(scheme.kappa2)
        self.assertEqual(scheme.bandwidth_d2, "none")
        self.assertEqual(scheme.algebraic_order2, "none")


class SufficientConditionTests(unittest.TestCase):
    """The Gate 1 sufficient conditions S_m of the k = 0 dominance argument."""

    def test_closed_form_matches_the_truncated_series_within_its_tail_bound(self) -> None:
        for m in (2, 4, 6, 8):
            for omega in (0.3, 1.25, 2.06, 2.9):
                closed = g1.sufficient_condition_S(m, omega)
                for terms in (16, 64, 256):
                    partial, bound = g1.sufficient_condition_S_series(m, omega,
                                                                     terms=terms)
                    self.assertGreaterEqual(closed, partial * (1.0 - 1e-12))
                    self.assertLessEqual(closed - partial, bound + 1e-15,
                                         msg=f"m={m}, omega={omega}, "
                                             f"terms={terms}")

    def test_the_series_tail_bound_shrinks_with_the_truncation_index(self) -> None:
        coarse_sum, coarse = g1.sufficient_condition_S_series(2, 2.06, terms=64)
        fine_sum, fine = g1.sufficient_condition_S_series(2, 2.06, terms=4096)
        self.assertLess(fine, coarse)
        self.assertGreater(fine_sum, coarse_sum)

    def test_the_condition_increases_with_the_frequency_and_decreases_with_m(self) -> None:
        values = [g1.sufficient_condition_S(4, w) for w in (0.5, 1.0, 1.5, 2.0, 2.5)]
        self.assertEqual(values, sorted(values))
        self.assertLess(g1.sufficient_condition_S(6, 2.06),
                        g1.sufficient_condition_S(4, 2.06))

    def test_the_champion_frequencies_satisfy_both_conditions(self) -> None:
        self.assertLess(g1.sufficient_condition_S(6, 2.0606855355642524), 1.0)
        self.assertLess(g1.sufficient_condition_S(4, 2.0606855355642524), 1.0)

    def test_a_frequency_near_nyquist_breaks_the_second_order_condition(self) -> None:
        self.assertGreater(g1.sufficient_condition_S(2, 3.05), 1.0)

    def test_frequencies_outside_the_admissible_range_are_rejected(self) -> None:
        self.assertEqual(g1.sufficient_condition_S(4, 0.0), float("inf"))
        self.assertEqual(g1.sufficient_condition_S(4, 4.0), float("inf"))
        with self.assertRaises(ValueError):
            g1.sufficient_condition_S(1, 2.0)


class DecimalRouteTests(unittest.TestCase):
    """The 60 digit decimal alias route with series sine and cosine."""

    def setUp(self) -> None:
        self.trig = g1.DecimalTrig(g1.DECIMAL_PREC)

    def test_series_sine_and_cosine_match_the_double_library(self) -> None:
        import math
        from decimal import Decimal
        for value in (0.0, 0.25, 1.0, -2.5, 3.14159, 12.75, -1000.0, 1234.5678):
            got_sin = float(self.trig.sin(Decimal(value)))
            got_cos = float(self.trig.cos(Decimal(value)))
            self.assertAlmostEqual(got_sin, math.sin(value), places=14)
            self.assertAlmostEqual(got_cos, math.cos(value), places=14)

    def test_the_alternating_series_truncation_bound_is_below_the_working_floor(self) -> None:
        from decimal import Decimal
        trig = g1.DecimalTrig(g1.DECIMAL_PREC)
        trig.sin(Decimal("2.5"))
        trig.cos(Decimal("2.5"))
        self.assertGreater(float(trig.bound), 0.0)
        self.assertLess(float(trig.bound), 10.0 ** (-g1.DECIMAL_DIGITS))

    def test_argument_reduction_lands_inside_one_period(self) -> None:
        from decimal import Decimal
        reduced = self.trig.reduce(Decimal("1000.0"))
        self.assertLessEqual(abs(float(reduced)), float(self.trig.pi) + 1e-12)

    def test_sinc_agrees_with_the_ratio_away_from_zero_and_the_series_near_it(self) -> None:
        import math
        from decimal import Decimal
        for value in (1e-9, 1e-3, 0.4, 0.6, 2.36, 7.5):
            got = float(self.trig.sinc(Decimal(value)))
            self.assertAlmostEqual(got, math.sin(value) / value, places=14)
        self.assertEqual(float(self.trig.sinc(Decimal(0))), 1.0)

    def test_decimal_route_reproduces_the_anchor_integer_samples(self) -> None:
        generator = fam.anchor_generator()
        alias = generator.integer_samples(1e-14)
        route = g1.decimal_samples_of(generator)
        np.testing.assert_allclose(route["beta"], alias.beta, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(route["dbeta"], alias.dbeta, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(route["ddbeta"], alias.ddbeta, rtol=0.0, atol=1e-14)
        self.assertEqual(route["K"], max(generator.required_K(q, 1e-14)
                                         for q in (0, 1, 2)))
        self.assertLess(route["series_truncation_bound"],
                        10.0 ** (-g1.DECIMAL_DIGITS))
        for q in (0, 1, 2):
            self.assertLessEqual(route["alias_tail_bound"][str(q)], 1e-14)

    def test_decimal_route_reproduces_the_frozen_anchor_values(self) -> None:
        generator = fam.anchor_generator()
        route = g1.decimal_samples_of(generator)
        positive = [v for n, v in zip(route["n"], route["beta"]) if n >= 0]
        np.testing.assert_allclose(positive[:4], FROZEN_BETA, rtol=0.0, atol=1e-14)

    def test_decimal_route_is_carried_at_the_stated_working_precision(self) -> None:
        generator = fam.ExponentialGenerator(6, (1.0, 2.0))
        route = g1.decimal_samples_of(generator)
        self.assertEqual(route["working_digits"], g1.DECIMAL_PREC)
        self.assertEqual(route["reported_digits"], g1.DECIMAL_DIGITS)
        self.assertGreaterEqual(len(route["beta_digits"][route["n"].index(0)]),
                                g1.DECIMAL_DIGITS)


class Gate1ConstraintTests(unittest.TestCase):
    """The Gate 1 hard constraints and the flags they drive."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.budgets, cls.references, cls.named = g1.budgets_and_references()
        cls.omegas = (1.7701927977466627, 2.0606855355642524)

    def test_a_constraint_clearing_member_reports_no_violation(self) -> None:
        row = g1.member_record(6, self.omegas, "C_SLC1",
                               self.budgets["C_SLC1"], self.references)
        self.assertEqual(g1.constraints_failed(row, "B0"), [])
        self.assertTrue(row["confirmation_ok"])
        self.assertLessEqual(row["confirmation_relative_gap"], g1.CONFIRM_RTOL)

    def test_a_tighter_budget_breaks_the_conditioning_and_margin_constraints(self) -> None:
        row = g1.member_record(6, self.omegas, "C_tight", 50.0, self.references)
        failed = g1.constraints_failed(row, "B0")
        self.assertIn("1 conditioning budget", failed)
        self.assertIn("2 sample symbol margin", failed)

    def test_a_polynomial_core_of_four_is_the_threshold_class(self) -> None:
        row = g1.member_record(4, (1.609463, 2.043040), "C_SLC1",
                               self.budgets["C_SLC1"], self.references)
        self.assertEqual(g1.constraints_failed(row, "B0"), ["4 formal order"])

    def test_the_coupled_channel_constraint_is_read_on_its_own_band(self) -> None:
        row = g1.member_record(6, self.omegas, "C_SLC1",
                               self.budgets["C_SLC1"], self.references)
        for band in g1.BANDS:
            block = row["constraints"]["c5_coupled"][band]
            self.assertAlmostEqual(block["limit"],
                                   2.0 * max(row[f"e1_max_{band}"],
                                             row[f"e2_max_{band}"]), places=15)
            self.assertTrue(block["ok"])

    def test_the_cheap_constraint_gate_rejects_a_frequency_near_nyquist(self) -> None:
        self.assertTrue(g1._cheap_constraints_ok(6, np.array(self.omegas),
                                                 self.budgets["C_SLC1"], 88.7))
        self.assertGreater(g1.sufficient_condition_S(2, 3.12), 1.0)
        self.assertFalse(g1._cheap_constraints_ok(4, np.array([3.12]),
                                                  self.budgets["C_SLC1"], 10.0))
        self.assertFalse(g1._cheap_constraints_ok(6, np.array(self.omegas),
                                                  50.0, 88.7))

    def test_the_polynomial_limit_conditioning_is_the_cell_infimum(self) -> None:
        self.assertAlmostEqual(g1.polynomial_limit_cond(6, 2),
                               self.budgets["C_BMC9"], places=9)
        self.assertAlmostEqual(g1.polynomial_limit_cond(8, 2),
                               self.budgets["C_BMC11"], places=9)
        cond = fam.ExponentialGenerator(6, (0.4, 0.8)).cond()["cond"]
        self.assertGreater(cond, g1.polynomial_limit_cond(6, 2))

    def test_no_admissible_member_beats_the_polynomial_limit_conditioning(self) -> None:
        """The cell infimum used by the infeasible rows of the ladder."""
        rng = np.random.default_rng(20260912)
        for m, r in g1.CELLS:
            floor = g1.polynomial_limit_cond(m, r)
            for _ in range(4):
                omegas = np.sort(rng.uniform(1e-3, np.pi - 1e-3, size=r))
                cond = fam.ExponentialGenerator(m, omegas).cond(2001)["cond"]
                self.assertGreaterEqual(cond, floor * (1.0 - 1e-12),
                                        msg=f"m={m}, r={r}, omegas={omegas}")

    def test_the_two_term_theorem_a5_formula_tracks_the_condition_number(self) -> None:
        formula = g1.theory_A5(6, self.omegas)
        actual = fam.ExponentialGenerator(6, self.omegas).cond()["cond"]
        self.assertLess(abs(formula["cond_two_term"] - actual) / actual, 1e-4)


if __name__ == "__main__":
    unittest.main()
