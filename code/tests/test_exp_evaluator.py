"""Checks for the pointwise evaluator of the exponential-family generator.

The evaluator builds phi, phi' and phi'' from the localized causal Green
function of D^m prod_j (D^2 + omega_j^2). Every property tested here is a
property the construction must have for any admissible member and is
independent of the route family.py takes to the same generator: parity, the
compact support, the lattice samples, partition of unity, the smoothness class,
the periodic field evaluation against the polynomial route, the confluent pair,
and the polynomial limit.

The sibling module general_evaluator.py carries the same construction through
the Gate 7 multiset (real pairs and quads) and the Gate 8 polynomial dial
psi = P(-D^2) phi. Its tests sit in the same file because the two modules share
one contract: the last classes cover a quad member, a member with the
polynomial factor, the exactly C^2 witness of such a member, and the regression
that the Gate 0 path of evaluator.py is untouched and is reproduced by the
general route.
"""

import json
import sys
import unittest
from fractions import Fraction
from math import factorial
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
ANALYSIS_DIR = CODE_DIR / "analysis" / "exponential_family"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import spline_analysis as sa
from evaluator import (ExponentialEvaluator, as_evaluator, evaluate_periodic_field,
                       periodic_coefficients, remainder_C2, remainder_C4,
                       remainder_S3, remainder_S5)
from family import ExponentialGenerator
from general_evaluator import (GeneralEvaluator, evaluator_from_member,
                               remainder_R)

CHAMPION = (6, (1.7701927977475826, 2.0606855355641316))
ROOT = CODE_DIR.parent
PERT5 = (ROOT / "results" / "exponential_family" / "gate8" / "members"
         / "X8_n10_C_SLC1_pert5.json")


class RemainderTests(unittest.TestCase):
    def test_series_branch_matches_the_direct_formula(self) -> None:
        """Where the direct form is itself accurate the two branches agree.

        The window starts at z = 1 because below it the direct differences
        cancel, which is the reason the series branch exists.
        """
        z = np.linspace(1.0, 3.9, 501)
        for mine, direct in ((remainder_C2(z), 1.0 - np.cos(z)),
                             (remainder_S3(z), z - np.sin(z)),
                             (remainder_C4(z), np.cos(z) - 1.0 + 0.5 * z * z),
                             (remainder_S5(z), np.sin(z) - z + z ** 3 / 6.0)):
            self.assertLess(float(np.max(np.abs(mine - direct) / np.abs(direct))), 1e-14)

    def test_remainders_vanish_to_their_stated_order(self) -> None:
        """Near zero each remainder is its leading monomial to relative 1e-7."""
        z = np.array([1e-3])
        leading = ((remainder_C2(z)[0], 0.5 * z[0] ** 2),
                   (remainder_S3(z)[0], z[0] ** 3 / 6.0),
                   (remainder_C4(z)[0], z[0] ** 4 / 24.0),
                   (remainder_S5(z)[0], z[0] ** 5 / 120.0))
        for value, monomial in leading:
            self.assertLess(abs(float(value) / monomial - 1.0), 1e-7)


class EvaluatorStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ev = ExponentialEvaluator(*CHAMPION)

    def test_support_is_exactly_zero_outside(self) -> None:
        half = 0.5 * self.ev.n
        outside = np.array([-half - 1e-12, -half - 0.3, -7.0, half + 1e-12, half + 2.0])
        for order in (0, 1, 2):
            values = self.ev.derivative(outside, order)
            self.assertTrue(np.all(values == 0.0))

    def test_parity_of_the_generator_and_its_derivatives(self) -> None:
        x = np.linspace(0.0, 5.0, 733)
        self.assertLess(float(np.max(np.abs(self.ev.value(x) - self.ev.value(-x)))), 1e-15)
        self.assertLess(float(np.max(np.abs(self.ev.d1(x) + self.ev.d1(-x)))), 1e-15)
        self.assertLess(float(np.max(np.abs(self.ev.d2(x) - self.ev.d2(-x)))), 1e-15)

    def test_lattice_samples_match_the_alias_route(self) -> None:
        gen = ExponentialGenerator(*CHAMPION)
        samples = gen.integer_samples()
        lattice = samples.n.astype(float)
        self.assertLess(float(np.max(np.abs(self.ev.value(lattice) - samples.beta))), 1e-14)
        self.assertLess(float(np.max(np.abs(self.ev.d1(lattice) - samples.dbeta))), 1e-14)
        self.assertLess(float(np.max(np.abs(self.ev.d2(lattice) - samples.ddbeta))), 1e-14)

    def test_offgrid_samples_match_the_alias_route(self) -> None:
        gen = ExponentialGenerator(*CHAMPION)
        xi = np.array([0.125, 0.5, -0.3125, 0.4])
        lattice, s0, s1, s2 = gen.offgrid_samples(xi)
        for i, shift in enumerate(xi):
            nodes = lattice.astype(float) + float(shift)
            self.assertLess(float(np.max(np.abs(self.ev.value(nodes) - s0[:, i]))), 1e-14)
            self.assertLess(float(np.max(np.abs(self.ev.d1(nodes) - s1[:, i]))), 1e-14)
            self.assertLess(float(np.max(np.abs(self.ev.d2(nodes) - s2[:, i]))), 1e-14)

    def test_partition_of_unity_and_its_derivatives(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.uniform(-2.0, 2.0, 257)
        total = sum(self.ev.value(x - k) for k in range(-9, 10))
        first = sum(self.ev.d1(x - k) for k in range(-9, 10))
        second = sum(self.ev.d2(x - k) for k in range(-9, 10))
        self.assertLess(float(np.max(np.abs(total - 1.0))), 1e-14)
        self.assertLess(float(np.max(np.abs(first))), 1e-14)
        self.assertLess(float(np.max(np.abs(second))), 1e-13)

    def test_smoothness_class_is_exactly_p_minus_one(self) -> None:
        knots = range(-(self.ev.n // 2), self.ev.n // 2 + 1)
        for order in range(self.ev.p):
            jump = max(abs(self.ev.one_sided(k, order, "left")
                           - self.ev.one_sided(k, order, "right")) for k in knots)
            self.assertLess(jump, 1e-13)
        top = max(abs(self.ev.one_sided(k, self.ev.p, "left")
                      - self.ev.one_sided(k, self.ev.p, "right")) for k in knots)
        self.assertGreater(top, 1.0)

    def test_derivatives_match_a_centered_difference(self) -> None:
        x = np.linspace(-4.37, 4.37, 401)
        step = 1e-4
        fd1 = (self.ev.value(x + step) - self.ev.value(x - step)) / (2.0 * step)
        fd2 = (self.ev.value(x + step) - 2.0 * self.ev.value(x)
               + self.ev.value(x - step)) / (step * step)
        self.assertLess(float(np.max(np.abs(fd1 - self.ev.d1(x)))), 1e-7)
        self.assertLess(float(np.max(np.abs(fd2 - self.ev.d2(x)))), 1e-6)

    def test_exponential_reproduction_of_the_pair_frequencies(self) -> None:
        """The translates carry exp(i omega x) with the single factor phi_hat(omega).

        Poisson summation gives sum_k exp(i omega k) phi(x - k) =
        sum_j phi_hat(omega + 2 pi j) exp(i (omega + 2 pi j) x), and every alias
        with j nonzero vanishes at a pair frequency, so the sum is exactly
        phi_hat(omega) exp(i omega x).
        """
        gen = ExponentialGenerator(*CHAMPION)
        x = np.linspace(-0.5, 0.5, 51)
        lattice = np.arange(-12, 13)
        for omega in CHAMPION[1]:
            value = np.zeros(x.size, dtype=complex)
            for k in lattice:
                value += np.exp(1j * omega * k) * self.ev.value(x - k)
            scale = float(gen.fourier(np.array([omega]))[0])
            self.assertGreater(abs(scale), 1e-3)
            self.assertLess(float(np.max(np.abs(value - scale * np.exp(1j * omega * x)))),
                            1e-13)

    def test_cell_table_round_trips_through_the_flat_layout(self) -> None:
        table = self.ev.cell_table()
        self.assertEqual(len(table["eval_poly"]),
                         table["eval_cells"] * (table["eval_poly_degree"] + 1))
        self.assertEqual(len(table["eval_cos"]), table["eval_cells"] * table["eval_term_count"])
        self.assertEqual(len(table["eval_sin"]), table["eval_cells"] * table["eval_term_count"])
        self.assertEqual(int(sum(table["eval_group_multiplicity"])), table["eval_term_count"])


class PolynomialLimitTests(unittest.TestCase):
    def test_r_zero_members_are_the_cardinal_bsplines(self) -> None:
        for degree in (3, 5, 9):
            ev = ExponentialEvaluator(degree + 1, ())
            x = np.linspace(-0.5 * (degree + 2), 0.5 * (degree + 2), 977)
            self.assertLess(float(np.max(np.abs(
                ev.value(x) - sa.centered_cardinal_bspline(degree, x)))), 1e-14)
            self.assertLess(float(np.max(np.abs(
                ev.d1(x) - sa.centered_cardinal_bspline_derivative(degree, x)))), 1e-14)
            self.assertLess(float(np.max(np.abs(
                ev.d2(x) - sa.centered_cardinal_bspline_second_derivative(degree, x)))), 1e-14)

    def test_frequencies_below_the_tolerance_collapse_to_zero_exponents(self) -> None:
        tiny = 0.1 * ExponentialEvaluator.ZERO_FREQUENCY_TOLERANCE
        ev = ExponentialEvaluator(6, (tiny, tiny))
        self.assertEqual(ev.m, 10)
        self.assertEqual(ev.r, 0)
        x = np.linspace(-5.5, 5.5, 977)
        self.assertLess(float(np.max(np.abs(
            ev.value(x) - sa.centered_cardinal_bspline(9, x)))), 1e-14)


class ConfluentTests(unittest.TestCase):
    def test_repeated_pair_is_one_group_of_multiplicity_two(self) -> None:
        ev = ExponentialEvaluator(4, (0.73, 0.73))
        self.assertEqual(len(ev.groups), 1)
        self.assertEqual(ev.groups[0].multiplicity, 2)
        gen = ExponentialGenerator(4, (0.73, 0.73))
        samples = gen.integer_samples()
        lattice = samples.n.astype(float)
        self.assertLess(float(np.max(np.abs(ev.value(lattice) - samples.beta))), 1e-14)
        self.assertLess(float(np.max(np.abs(ev.d1(lattice) - samples.dbeta))), 1e-14)
        self.assertLess(float(np.max(np.abs(ev.d2(lattice) - samples.ddbeta))), 1e-14)

    def test_the_confluent_member_is_the_limit_of_separated_pairs(self) -> None:
        x = np.linspace(-4.0, 4.0, 401)
        exact = ExponentialEvaluator(4, (0.73, 0.73)).value(x)
        previous = None
        for delta in (1e-2, 1e-3):
            near = ExponentialEvaluator(4, (0.73 - delta, 0.73 + delta)).value(x)
            error = float(np.max(np.abs(near - exact)))
            if previous is not None:
                self.assertLess(error, previous)
            previous = error
        self.assertLess(previous, 1e-6)


class PeriodicFieldTests(unittest.TestCase):
    def test_polynomial_member_matches_the_bspline_field_evaluator(self) -> None:
        npts, degree = 48, 9
        h = 2.0 * np.pi / npts
        rng = np.random.default_rng(11)
        coefficients = rng.normal(size=npts)
        x = rng.uniform(-3.0 * np.pi, 5.0 * np.pi, 512)
        ev = ExponentialEvaluator(degree + 1, ())
        for derivative in (0, 1, 2):
            mine = evaluate_periodic_field(coefficients, x, h, ev, derivative=derivative)
            reference = sa.evaluate_periodic_spline(coefficients, x, h, degree,
                                                    derivative=derivative)
            self.assertLess(float(np.max(np.abs(mine - reference))), 1e-12)

    def test_collocation_reproduces_the_nodal_values(self) -> None:
        npts = 64
        h = 2.0 * np.pi / npts
        nodes = h * np.arange(npts)
        values = np.sin(3.0 * nodes + 0.17) + 0.35 * np.cos(7.0 * nodes - 0.31)
        ev = ExponentialEvaluator(*CHAMPION)
        coefficients = periodic_coefficients(values, ev)
        recovered = evaluate_periodic_field(coefficients, nodes, h, ev)
        self.assertLess(float(np.max(np.abs(recovered - values))), 1e-12)

    def test_constants_are_carried_exactly(self) -> None:
        npts = 64
        h = 2.0 * np.pi / npts
        ev = ExponentialEvaluator(*CHAMPION)
        coefficients = periodic_coefficients(np.ones(npts), ev)
        rng = np.random.default_rng(3)
        x = rng.uniform(-2.0 * np.pi, 4.0 * np.pi, 256)
        self.assertLess(float(np.max(np.abs(
            evaluate_periodic_field(coefficients, x, h, ev) - 1.0))), 1e-13)
        self.assertLess(float(np.max(np.abs(
            evaluate_periodic_field(coefficients, x, h, ev, derivative=1)))), 1e-12)
        self.assertLess(float(np.max(np.abs(
            evaluate_periodic_field(coefficients, x, h, ev, derivative=2)))), 1e-11)

    def test_nodal_derivatives_reproduce_the_family_symbols(self) -> None:
        """At the nodes the field derivatives are the collocation symbols."""
        npts = 64
        h = 2.0 * np.pi / npts
        ev = ExponentialEvaluator(*CHAMPION)
        gen = ExponentialGenerator(*CHAMPION)
        nodes = h * np.arange(npts)
        mode = 7
        theta = 2.0 * np.pi * mode / npts
        values = np.cos(mode * nodes)
        coefficients = periodic_coefficients(values, ev)
        _, kappa1, kappa2 = gen.symbols(np.array([theta]))
        first = evaluate_periodic_field(coefficients, nodes, h, ev, derivative=1)
        second = evaluate_periodic_field(coefficients, nodes, h, ev, derivative=2)
        want1 = -(float(kappa1[0]) / h) * np.sin(mode * nodes)
        want2 = -(float(kappa2[0]) / (h * h)) * np.cos(mode * nodes)
        self.assertLess(float(np.max(np.abs(first - want1))), 1e-11)
        self.assertLess(float(np.max(np.abs(second - want2))), 1e-10)


def _exact_R(j: int, u: Fraction, v: Fraction, terms: int = 140) -> complex:
    """Return R_j(u + i v) by summing the defining series in exact rationals."""
    tre, tim = Fraction(1), Fraction(0)
    for i in range(1, j + 1):
        tre, tim = (tre * u - tim * v) / i, (tre * v + tim * u) / i
    sre, sim = tre, tim
    for i in range(j + 1, j + terms):
        tre, tim = (tre * u - tim * v) / i, (tre * v + tim * u) / i
        sre, sim = sre + tre, sim + tim
    return complex(float(sre), float(sim))


class RemainderRTests(unittest.TestCase):
    """R_j(z) = sum_{i>=j} z^i/i! is the remainder the folded cell form leaves."""

    POINTS = ((Fraction(1, 4), Fraction(3, 2)), (Fraction(-1, 2), Fraction(5, 2)),
              (Fraction(5, 2), Fraction(7, 2)), (Fraction(9, 2), Fraction(0)),
              (Fraction(-3), Fraction(-4)))

    def test_R_matches_the_exact_rational_series(self) -> None:
        """Both branches reproduce the exact summation to 1e-14 relative.

        The last three points lie above SERIES_LIMIT in modulus and so exercise
        the branch that subtracts the truncated exponential series.
        """
        for u, v in self.POINTS:
            for j in (0, 1, 4, 7, 10):
                want = _exact_R(j, u, v)
                re, im = remainder_R(j, np.array([float(u)]), np.array([float(v)]))
                got = complex(float(re[0]), float(im[0]))
                self.assertLess(abs(got - want) / abs(want), 1e-14)

    def test_R_obeys_the_recurrence_the_derivatives_use(self) -> None:
        """R_{j-1}(z) - R_j(z) is the monomial z^{j-1}/(j-1)!."""
        for u, v in self.POINTS:
            z = complex(float(u), float(v))
            for j in (1, 3, 8):
                lower = remainder_R(j - 1, np.array([float(u)]), np.array([float(v)]))
                upper = remainder_R(j, np.array([float(u)]), np.array([float(v)]))
                gap = complex(float(lower[0][0] - upper[0][0]),
                              float(lower[1][0] - upper[1][0]))
                monomial = z ** (j - 1) / factorial(j - 1)
                self.assertLess(abs(gap - monomial) / abs(monomial), 1e-13)

    def test_R_vanishes_to_its_stated_order(self) -> None:
        """Near the origin R_j is its leading monomial up to the next term z/(j+1)."""
        z = 1e-3
        for j in (2, 5, 9):
            re, im = remainder_R(j, np.array([z]), np.array([0.0]))
            ratio = float(re[0]) / (z ** j / factorial(j))
            self.assertLess(abs(ratio - 1.0), 2.0 * z / (j + 1))
            self.assertGreater(abs(ratio - 1.0), 0.5 * z / (j + 1))
            self.assertEqual(float(im[0]), 0.0)


class QuadMemberTests(unittest.TestCase):
    """A member whose slot is a quad +-alpha +- i omega, four complex exponents."""

    @classmethod
    def setUpClass(cls) -> None:
        import gate7_family as g7
        cls.gen = g7.ExtendedGenerator(4, [(0.4, 1.7)])
        cls.ev = as_evaluator(cls.gen)

    def test_dispatch_gives_the_general_evaluator(self) -> None:
        self.assertIsInstance(self.ev, GeneralEvaluator)
        self.assertEqual(self.ev.n, 8)
        self.assertEqual(self.ev.quads, ((0.4, 1.7),))
        self.assertEqual(len(self.ev.channels), 2)

    def test_support_and_parity(self) -> None:
        half = 0.5 * self.ev.n
        outside = np.array([-half - 1e-12, -half - 0.7, half + 1e-12, half + 3.0])
        for order in (0, 1, 2):
            self.assertTrue(np.all(self.ev.derivative(outside, order) == 0.0))
        x = np.linspace(0.0, 4.5, 401)
        self.assertLess(float(np.max(np.abs(self.ev.value(x) - self.ev.value(-x)))), 1e-15)
        self.assertLess(float(np.max(np.abs(self.ev.d1(x) + self.ev.d1(-x)))), 1e-15)
        self.assertLess(float(np.max(np.abs(self.ev.d2(x) - self.ev.d2(-x)))), 1e-15)

    def test_lattice_samples_match_the_alias_route(self) -> None:
        samples = self.gen.integer_samples()
        lattice = samples.n.astype(float)
        self.assertLess(float(np.max(np.abs(self.ev.value(lattice) - samples.beta))), 1e-13)
        self.assertLess(float(np.max(np.abs(self.ev.d1(lattice) - samples.dbeta))), 1e-13)
        self.assertLess(float(np.max(np.abs(self.ev.d2(lattice) - samples.ddbeta))), 1e-13)

    def test_partition_of_unity(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(-2.0, 2.0, 257)
        total = sum(self.ev.value(x - k) for k in range(-8, 9))
        self.assertLess(float(np.max(np.abs(total - 1.0))), 1e-14)

    def test_double_agrees_with_the_decimal_reference(self) -> None:
        rng = np.random.default_rng(31)
        half = 0.5 * self.ev.n
        x = np.concatenate([rng.uniform(-half, half, 60),
                            rng.uniform(-half, -half + 1.0, 20)])
        for order in (0, 1, 2):
            dense = np.linspace(-half, half, 4001)
            scale = float(np.max(np.abs(self.ev.derivative(dense, order))))
            gap = float(np.max(np.abs(self.ev.derivative(x, order)
                                      - self.ev.reference(x, order))))
            self.assertLess(gap, 1e-13 * scale)


class RealPairMemberTests(unittest.TestCase):
    def test_real_pair_lattice_samples_match_the_alias_route(self) -> None:
        import gate7_family as g7
        gen = g7.ExtendedGenerator(4, [(0.9, 0.0), (0.0, 1.3)])
        ev = as_evaluator(gen)
        self.assertEqual(ev.real_pairs, (0.9,))
        samples = gen.integer_samples()
        lattice = samples.n.astype(float)
        self.assertLess(float(np.max(np.abs(ev.value(lattice) - samples.beta))), 1e-13)
        self.assertLess(float(np.max(np.abs(ev.d1(lattice) - samples.dbeta))), 1e-13)
        self.assertLess(float(np.max(np.abs(ev.d2(lattice) - samples.ddbeta))), 1e-13)

    def test_a_repeated_slot_is_one_group_of_multiplicity_two(self) -> None:
        import gate7_family as g7
        gen = g7.ExtendedGenerator(4, [(0.0, 0.73), (0.0, 0.73)])
        ev = as_evaluator(gen)
        self.assertEqual(len(ev.channels), 1)
        self.assertEqual(ev.channels[0][2], 2)
        samples = gen.integer_samples()
        lattice = samples.n.astype(float)
        self.assertLess(float(np.max(np.abs(ev.value(lattice) - samples.beta))), 1e-13)
        self.assertLess(float(np.max(np.abs(ev.d2(lattice) - samples.ddbeta))), 1e-13)


class PolynomialFactorMemberTests(unittest.TestCase):
    """The Gate 8 member X8_n10_C_SLC1_pert5: psi_hat = P(xi^2) phi_hat."""

    @classmethod
    def setUpClass(cls) -> None:
        import gate8_family as g8
        cls.record = json.loads(PERT5.read_text())
        cls.gen = g8.from_record(cls.record)
        cls.ev = evaluator_from_member(PERT5)

    def test_shape_of_the_member(self) -> None:
        self.assertEqual(self.ev.n, 10)
        self.assertEqual(self.ev.p, 9)
        self.assertEqual(self.ev.smoothness, 2)
        self.assertEqual(len(self.ev.poly), 3)
        self.assertEqual(self.ev.fold_order, 10)

    def test_dispatch_on_the_generator_type(self) -> None:
        self.assertIs(as_evaluator(self.gen).__class__, GeneralEvaluator)
        self.assertIs(as_evaluator(self.ev), self.ev)
        self.assertIsInstance(as_evaluator(CHAMPION), ExponentialEvaluator)

    def test_lattice_samples_match_the_stored_alias_route(self) -> None:
        lattice = np.arange(0, int(self.record["radius"]) + 1).astype(float)
        for order, key in enumerate(("beta", "dbeta", "ddbeta")):
            reference = np.asarray(self.record[key], dtype=float)
            self.assertLess(float(np.max(np.abs(self.ev.derivative(lattice, order)
                                                - reference))), 1e-13)

    def test_the_dial_is_the_transform_side_statement(self) -> None:
        """The transform of psi from the samples is P(xi^2) phi_hat at the aliases.

        Poisson summation at theta = 0 with the translates carrying the constant
        gives sum_n psi(n + xi) = 1 for every xi, which is psi_hat(0) = 1 and
        psi_hat(2 pi k) = 0; the dial cannot move either because P(0) = 1.
        """
        rng = np.random.default_rng(41)
        x = rng.uniform(-1.0, 1.0, 257)
        total = sum(self.ev.value(x - k) for k in range(-9, 10))
        first = sum(self.ev.d1(x - k) for k in range(-9, 10))
        self.assertLess(float(np.max(np.abs(total - 1.0))), 1e-14)
        self.assertLess(float(np.max(np.abs(first))), 1e-14)

    def test_the_smoothness_class_is_exactly_C2(self) -> None:
        knots = range(-(self.ev.n // 2), self.ev.n // 2 + 1)
        for order in (0, 1, 2):
            jump = max(abs(self.ev.one_sided(k, order, "left")
                           - self.ev.one_sided(k, order, "right")) for k in knots)
            self.assertLess(jump, 1e-13)
        third = max(abs(self.ev.one_sided(k, 3, "left")
                        - self.ev.one_sided(k, 3, "right")) for k in knots)
        self.assertGreater(third, 1e-4)

    def test_derivatives_match_a_centered_difference(self) -> None:
        x = np.linspace(-4.37, 4.37, 401)
        step = 1e-4
        fd1 = (self.ev.value(x + step) - self.ev.value(x - step)) / (2.0 * step)
        fd2 = (self.ev.value(x + step) - 2.0 * self.ev.value(x)
               + self.ev.value(x - step)) / (step * step)
        self.assertLess(float(np.max(np.abs(fd1 - self.ev.d1(x)))), 1e-7)
        self.assertLess(float(np.max(np.abs(fd2 - self.ev.d2(x)))), 1e-6)

    def test_double_agrees_with_the_decimal_reference_in_the_end_cells(self) -> None:
        rng = np.random.default_rng(47)
        half = 0.5 * self.ev.n
        x = np.concatenate([rng.uniform(-half, -half + 1.0, 40),
                            rng.uniform(half - 1.0, half, 40),
                            rng.uniform(-half, half, 40)])
        dense = np.linspace(-half, half, 4001)
        for order in (0, 1, 2):
            scale = float(np.max(np.abs(self.ev.derivative(dense, order))))
            gap = float(np.max(np.abs(self.ev.derivative(x, order)
                                      - self.ev.reference(x, order))))
            self.assertLess(gap, 1e-13 * scale)

    def test_cell_table_is_a_version_2_block_of_the_right_shape(self) -> None:
        table = self.ev.cell_table()
        self.assertEqual(table["eval_version"], 2)
        self.assertEqual(table["eval_fold_order"], table["eval_poly_degree"] + 1)
        self.assertEqual(len(table["eval_poly"]),
                         table["eval_cells"] * (table["eval_poly_degree"] + 1))
        self.assertEqual(len(table["eval_cos"]),
                         table["eval_cells"] * table["eval_term_count"])
        self.assertEqual(len(table["eval_sin"]),
                         table["eval_cells"] * table["eval_term_count"])
        self.assertEqual(len(table["eval_group_alpha"]), table["eval_group_count"])
        self.assertEqual(int(sum(table["eval_group_multiplicity"])), table["eval_term_count"])

    def test_the_member_file_carries_the_block_and_nothing_else_moved(self) -> None:
        stored = json.loads(PERT5.read_text())
        table = self.ev.cell_table()
        for key, value in table.items():
            self.assertIn(key, stored)
            self.assertEqual(stored[key], value)
        for key in ("beta", "dbeta", "ddbeta", "cond", "polynomial", "smoothness"):
            self.assertIn(key, stored)


class GeneralVersusGate0Tests(unittest.TestCase):
    """The general route reproduces the Gate 0 route, which is left unchanged."""

    def test_the_champion_is_reproduced_to_1e_14(self) -> None:
        legacy = ExponentialEvaluator(*CHAMPION)
        general = GeneralEvaluator(CHAMPION[0], [(0.0, w) for w in CHAMPION[1]])
        x = np.linspace(-6.5, 6.5, 2001)
        for order in (0, 1, 2):
            self.assertLess(float(np.max(np.abs(legacy.derivative(x, order)
                                                - general.derivative(x, order)))), 1e-14)

    def test_the_gate0_cell_table_is_still_version_1(self) -> None:
        table = ExponentialEvaluator(*CHAMPION).cell_table()
        self.assertNotIn("eval_version", table)
        self.assertNotIn("eval_group_alpha", table)
        self.assertEqual(table["eval_poly_degree"], 5)

    def test_the_polynomial_core_is_the_cardinal_bspline(self) -> None:
        ev = GeneralEvaluator(10, ())
        x = np.linspace(-6.0, 6.0, 977)
        self.assertLess(float(np.max(np.abs(
            ev.value(x) - sa.centered_cardinal_bspline(9, x)))), 1e-14)
        self.assertLess(float(np.max(np.abs(
            ev.d2(x) - sa.centered_cardinal_bspline_second_derivative(9, x)))), 1e-14)

    def test_the_periodic_field_accepts_a_general_evaluator(self) -> None:
        npts = 48
        h = 2.0 * np.pi / npts
        ev = GeneralEvaluator(10, ())
        rng = np.random.default_rng(53)
        coefficients = rng.normal(size=npts)
        x = rng.uniform(-3.0 * np.pi, 5.0 * np.pi, 256)
        for derivative in (0, 1, 2):
            mine = evaluate_periodic_field(coefficients, x, h, ev, derivative=derivative)
            reference = sa.evaluate_periodic_spline(coefficients, x, h, 9,
                                                    derivative=derivative)
            self.assertLess(float(np.max(np.abs(mine - reference))), 1e-12)


if __name__ == "__main__":
    unittest.main()
