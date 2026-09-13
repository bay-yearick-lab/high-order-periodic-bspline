"""Regression tests for the Gate 2 frequency-selection rule.

The rule is stated in section 6 of docs/exponential_family/gate0_theory.md and
implemented in code/analysis/exponential_family/gate2_rule.py. These tests
check the pieces of the iteration against the theory of that document and then
check the five rows of the section 6 table, the band sweep of the (6, 2) cell
and the local minimality of every optimum.
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

import family as fam  # noqa: E402
import gate2_rule as g2  # noqa: E402

B0 = g2.B0
PI = np.pi


class OneAliasModelTests(unittest.TestCase):
    """The closed form of section 6 against the alias transform it abbreviates."""

    def test_model_is_the_k_minus_one_alias_ratio_of_lemma_D(self) -> None:
        m, omegas = 6, (0.5856 * PI, 0.6584 * PI)
        generator = fam.ExponentialGenerator(m, omegas)
        theta = np.linspace(0.2, B0, 41)
        ratio = (-(g2.TWO_PI / theta) * generator.fourier(theta - g2.TWO_PI)
                 / generator.fourier(theta))
        np.testing.assert_allclose(g2.one_alias_e1(theta, m, omegas), ratio, rtol=1e-12)

    def test_model_factors_as_the_section_six_weight_times_a_monic_polynomial(self) -> None:
        m, omegas = 4, (0.49 * PI, 0.61 * PI, 0.66 * PI)
        theta = np.linspace(0.3, B0, 33)
        u = theta ** 2
        monic = np.ones_like(u)
        for w in omegas:
            monic = monic * (u - w ** 2)
        weighted = -g2.one_alias_weight(u, m, omegas) * monic
        np.testing.assert_allclose(g2.one_alias_e1(theta, m, omegas), weighted, rtol=1e-12)

    def test_model_vanishes_at_every_exact_contact(self) -> None:
        omegas = (0.5 * PI, 0.62 * PI)
        values = g2.one_alias_e1(np.asarray(omegas), 6, omegas)
        np.testing.assert_allclose(values, 0.0, atol=1e-15)


class ExactErrorTests(unittest.TestCase):
    """The exact e_1 used by the iteration against the family's own symbols."""

    def test_exact_error_matches_the_trigonometric_polynomial_route(self) -> None:
        m, omegas = 6, (0.5856 * PI, 0.6584 * PI)
        error = g2.ExactError(m, omegas)
        theta = np.linspace(0.05, B0, 37)
        trig = fam.ExponentialGenerator(m, omegas).symbols(theta)[1] / theta - 1.0
        np.testing.assert_allclose(error(theta), trig, atol=1e-12)

    def test_exact_error_vanishes_at_every_exact_contact(self) -> None:
        omegas = (0.5281 * PI, 0.6213 * PI, 0.6618 * PI)
        values = g2.ExactError(6, omegas)(np.asarray(omegas))
        np.testing.assert_allclose(values, 0.0, atol=1e-13)


class AlternationSetTests(unittest.TestCase):
    """The r + 1 candidate extrema and their ordering."""

    def setUp(self) -> None:
        self.omegas = np.array([0.585580 * PI, 0.658417 * PI])
        self.error = g2.ExactError(6, self.omegas)

    def test_the_set_has_r_plus_one_points_and_ends_at_the_band_edge(self) -> None:
        points, values = g2.alternation_set(self.error, self.omegas, B0)
        self.assertEqual(points.size, self.omegas.size + 1)
        self.assertEqual(values.size, points.size)
        self.assertAlmostEqual(float(points[-1]), B0, places=15)

    def test_the_points_interlace_the_exact_contacts(self) -> None:
        points, _ = g2.alternation_set(self.error, self.omegas, B0)
        self.assertTrue(0.0 < points[0] < self.omegas[0])
        self.assertTrue(self.omegas[0] < points[1] < self.omegas[1])
        self.assertTrue(self.omegas[1] < points[2] <= B0)

    def test_the_signs_alternate_because_every_contact_is_a_simple_zero(self) -> None:
        _, values = g2.alternation_set(self.error, self.omegas, B0)
        self.assertTrue(g2.alternates(values))
        self.assertFalse(g2.alternates(np.array([1.0, 1.0, -1.0])))
        self.assertFalse(g2.alternates(np.array([1.0, 0.0, -1.0])))

    def test_the_residual_is_zero_only_at_a_levelled_set(self) -> None:
        np.testing.assert_allclose(
            g2.equioscillation_residual(np.array([2.0, -2.0, 2.0])), 0.0, atol=0.0)
        np.testing.assert_allclose(
            g2.equioscillation_residual(np.array([3.0, -2.0])), np.array([0.5]))


class SectionSixTableTests(unittest.TestCase):
    """The five rows of the section 6 table, reproduced from the exact symbol."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = {cell: g2.remez_frequencies(cell[0], cell[1], B0)
                       for cell in g2.TABLE_CELLS}

    def test_every_cell_converges_with_an_alternating_levelled_set(self) -> None:
        for cell, result in self.results.items():
            with self.subTest(cell=cell):
                self.assertTrue(result.converged, msg=f"residual {result.residual:.2e}")
                self.assertFalse(result.stalled)
                self.assertTrue(result.alternating)
                self.assertEqual(result.points.size, cell[1] + 1)
                self.assertLessEqual(result.residual, g2.REMEZ_TOL)

    def test_frequencies_are_ordered_inside_the_band(self) -> None:
        for cell, result in self.results.items():
            with self.subTest(cell=cell):
                omegas = result.omegas
                self.assertTrue(np.all(np.diff(omegas) > 0.0))
                self.assertGreater(float(omegas[0]), 0.0)
                self.assertLess(float(omegas[-1]), B0)

    def test_frequencies_match_the_table_to_one_part_in_ten_thousand(self) -> None:
        for cell, result in self.results.items():
            quoted = np.asarray(g2.SECTION6_TABLE[cell]["omegas_over_pi"])
            found = result.omegas / PI
            with self.subTest(cell=cell):
                np.testing.assert_allclose(found, quoted, rtol=1e-4)

    def test_levels_match_the_table_to_the_quoted_three_digits(self) -> None:
        for cell, result in self.results.items():
            quoted = g2.SECTION6_TABLE[cell]["E_max"]
            half_ulp = 0.5 * 10.0 ** (np.floor(np.log10(quoted)) - 2.0)
            with self.subTest(cell=cell):
                self.assertLessEqual(abs(result.level - quoted), half_ulp)
                self.assertLess(abs(result.level - quoted) / quoted, 4e-3)

    def test_alternation_values_are_plus_or_minus_one_to_one_per_mille(self) -> None:
        for cell, result in self.results.items():
            normalized = result.values / result.level
            with self.subTest(cell=cell):
                np.testing.assert_allclose(np.abs(normalized), 1.0, atol=1e-3)
                signs = [int(np.sign(v)) for v in normalized]
                self.assertEqual(signs, list(g2.SECTION6_TABLE[cell]["signs"]))

    def test_alternation_points_match_the_table_to_its_three_decimals(self) -> None:
        for cell, result in self.results.items():
            quoted = np.asarray(g2.SECTION6_TABLE[cell]["points_over_pi"])
            with self.subTest(cell=cell):
                np.testing.assert_allclose(result.points / PI, quoted, atol=5e-4)

    def test_the_comparison_helper_agrees_with_the_stored_table(self) -> None:
        for cell, result in self.results.items():
            record = result.as_dict()
            record["m"], record["r"] = cell
            comparison = g2.compare_with_section6(record)
            with self.subTest(cell=cell):
                self.assertTrue(comparison["signs_match"])
                self.assertTrue(comparison["omega_within_quoted_digits"])
                self.assertTrue(comparison["E_max_within_quoted_digits"])

    def test_the_history_reduces_the_residual_monotonically(self) -> None:
        for cell, result in self.results.items():
            residuals = [h["residual"] for h in result.history]
            with self.subTest(cell=cell):
                self.assertGreaterEqual(len(residuals), 1)
                self.assertTrue(all(b < a for a, b in zip(residuals[:-1], residuals[1:])))
                self.assertLessEqual(residuals[-1], g2.REMEZ_TOL)


class OneAliasStartTests(unittest.TestCase):
    """The accuracy of the weighted Chebyshev starting guess."""

    def test_the_start_is_within_a_few_per_mille_of_the_exact_optimum(self) -> None:
        for cell in ((8, 1), (6, 2), (6, 3)):
            result = g2.remez_frequencies(cell[0], cell[1], B0)
            with self.subTest(cell=cell):
                self.assertIsNotNone(result.start)
                np.testing.assert_allclose(result.start, result.omegas, rtol=3e-3)
                self.assertLess(abs(result.start_level - result.level) / result.level,
                                3e-2)
                self.assertGreater(result.start_level, result.level)

    def test_the_model_solution_levels_the_model_curve(self) -> None:
        model = g2.one_alias_start(6, 2, B0)
        self.assertTrue(model.converged)
        self.assertTrue(model.alternating)
        np.testing.assert_allclose(np.abs(model.values / model.level), 1.0, atol=1e-8)

    def test_an_explicit_start_reaches_the_same_optimum(self) -> None:
        reference = g2.remez_frequencies(6, 2, B0)
        shifted = g2.remez_frequencies(6, 2, B0,
                                       start=reference.omegas + np.array([0.02, -0.01]))
        np.testing.assert_allclose(shifted.omegas, reference.omegas, rtol=1e-6)
        self.assertAlmostEqual(shifted.level / reference.level, 1.0, places=6)


class BandSweepTests(unittest.TestCase):
    """The (6, 2) cell on the three further bands of the Gate 2 task."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.edges = (0.5 * PI, 0.6 * PI, B0, 0.75 * PI)
        cls.results = {T: g2.remez_frequencies(6, 2, T) for T in cls.edges}

    def test_every_band_converges_with_the_edge_in_the_alternation_set(self) -> None:
        for T, result in self.results.items():
            with self.subTest(T=T / PI):
                self.assertTrue(result.converged)
                self.assertTrue(result.alternating)
                self.assertAlmostEqual(float(result.points[-1]), T, places=15)
                self.assertLess(float(result.omegas[-1]), T)

    def test_the_level_and_the_frequencies_increase_with_the_band(self) -> None:
        levels = [self.results[T].level for T in self.edges]
        self.assertTrue(all(b > a for a, b in zip(levels[:-1], levels[1:])))
        for j in range(2):
            column = [self.results[T].omegas[j] for T in self.edges]
            self.assertTrue(all(b > a for a, b in zip(column[:-1], column[1:])))

    def test_the_conditioning_of_the_optimum_rises_with_the_band(self) -> None:
        conds = [fam.ExponentialGenerator(6, tuple(self.results[T].omegas)).cond(2001)["cond"]
                 for T in self.edges]
        self.assertTrue(all(b > a for a, b in zip(conds[:-1], conds[1:])))

    def test_every_optimum_is_an_admissible_member(self) -> None:
        for T, result in self.results.items():
            ok, reason = fam.ExponentialGenerator(6, tuple(result.omegas)).admissible()
            with self.subTest(T=T / PI):
                self.assertTrue(ok, msg=reason)


class SupremumAndLocalMinimumTests(unittest.TestCase):
    """The band supremum and the perturbation test of local minimality."""

    def test_the_supremum_agrees_with_a_fine_uniform_grid_maximum(self) -> None:
        for cell in ((6, 2), (4, 3)):
            result = g2.remez_frequencies(cell[0], cell[1], B0)
            supremum = g2.band_supremum(cell[0], result.omegas, B0)
            grid = g2.band_maximum(cell[0], result.omegas, B0, points=20001)
            with self.subTest(cell=cell):
                self.assertAlmostEqual(supremum / result.level, 1.0, places=8)
                self.assertLess(abs(grid - supremum) / supremum, 1e-6)

    def test_every_optimum_is_a_local_minimum_of_the_band_supremum(self) -> None:
        for cell in g2.TABLE_CELLS:
            result = g2.remez_frequencies(cell[0], cell[1], B0)
            check = g2.local_minimum_check(cell[0], result.omegas, B0, directions=12)
            with self.subTest(cell=cell):
                self.assertTrue(check["is_local_minimum"])
                self.assertGreater(check["min_ratio"], 1.0)

    def test_the_band_sweep_optima_are_local_minima_as_well(self) -> None:
        for T in (0.5 * PI, 0.6 * PI, 0.75 * PI):
            result = g2.remez_frequencies(6, 2, T)
            check = g2.local_minimum_check(6, result.omegas, T, directions=12)
            with self.subTest(T=T / PI):
                self.assertTrue(check["is_local_minimum"])

    def test_a_deliberately_unlevelled_member_has_a_larger_supremum(self) -> None:
        result = g2.remez_frequencies(6, 2, B0)
        optimal = g2.band_supremum(6, result.omegas, B0)
        for shift in (np.array([0.01, 0.0]), np.array([0.0, 0.01]),
                      np.array([-0.01, -0.01])):
            with self.subTest(shift=tuple(shift)):
                self.assertGreater(g2.band_supremum(6, result.omegas + shift, B0),
                                   optimal)


class ArgumentTests(unittest.TestCase):
    """The guards on the public entry point."""

    def test_the_rule_needs_at_least_one_pair(self) -> None:
        with self.assertRaises(ValueError):
            g2.remez_frequencies(6, 0, B0)

    def test_the_band_edge_must_lie_inside_the_first_brillouin_zone(self) -> None:
        for T in (0.0, -1.0, PI, 4.0):
            with self.subTest(T=T):
                with self.assertRaises(ValueError):
                    g2.remez_frequencies(6, 2, T)

    def test_the_ordering_clip_keeps_the_frequencies_inside_the_open_box(self) -> None:
        clipped = g2._ordered(np.array([2.5, -1.0, 0.3]), B0)
        self.assertTrue(np.all(np.diff(clipped) > 0.0))
        self.assertGreater(float(clipped[0]), 0.0)
        self.assertLess(float(clipped[-1]), B0)


if __name__ == "__main__":
    unittest.main()
