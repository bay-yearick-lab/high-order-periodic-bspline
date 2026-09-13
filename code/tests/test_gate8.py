"""Regression tests for Gate 8: the polynomial-augmented generators.

The theory is in results/exponential_family/gate8/theory.md. The tests check
the transform and the x-side operator P(-D^2), the reduction to Gate 7 at
P = 1, the three sample routes (alias with the exact zeta tail, residue
closed form, 60-digit decimal) against one another, the endpoint vanishing
that fixes the bandwidth, the exact reproduction of the eighth-order
pentadiagonal scheme by E_{6,0} x (1 - t/28), the singular recovery at a
real root inside the band and the soft contact of a root in (pi, 2 pi), the
zero count beyond Theorem Z, the dimension count, the sufficient conditions,
the residue off-grid route against the Gate 7 alias route, and the search
and realization machinery.
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
import gate7_dominance as gd  # noqa: E402
import gate7_family as g7  # noqa: E402
import gate8_dominance as gdom  # noqa: E402
import gate8_family as g8  # noqa: E402
import gate8_realization as gr  # noqa: E402
import metrics as met  # noqa: E402

PI = np.pi


class TransformTests(unittest.TestCase):
    def test_transform_is_P_times_phi_hat_and_reduces_to_gate7(self) -> None:
        xi = np.linspace(-30.0, 30.0, 1201)
        base = g7.ExtendedGenerator(4, [(0.0, 2.0), (1.5, 0.0)])
        gen = g8.PolynomialGenerator(4, [(0.0, 2.0), (1.5, 0.0)], [-0.02, 0.0003])
        expected = (1.0 - 0.02 * xi ** 2 + 0.0003 * xi ** 4) * base.fourier(xi)
        np.testing.assert_allclose(gen.fourier(xi), expected, rtol=0, atol=1e-15)
        plain = g8.PolynomialGenerator(4, [(0.0, 2.0), (1.5, 0.0)], [])
        rep = g8.verify_routes(plain)
        self.assertLess(rep["gate7_deviation"], 1e-14)
        self.assertEqual((gen.n, gen.q, gen.smoothness, gen.parameters), (8, 2, 2, 4))

    def test_smoothness_limit_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.4)], [-0.01, 0.0, 1e-5])
        g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.4)], [-0.01, 1e-4])

    def test_x_side_operator_by_finite_differences_on_the_transform(self) -> None:
        # psi = phi + p_1 (-phi'') : compare psi_hat with the transform of the difference form
        # through the samples: psi(n) = phi(n) - p_1 phi''(n)
        p1 = -0.02
        base = g7.ExtendedGenerator(6, [(0.0, 1.9)])
        gen = g8.PolynomialGenerator(6, [(0.0, 1.9)], [p1])
        s7 = base.integer_samples()
        s8 = gen.integer_samples()
        np.testing.assert_allclose(s8.beta, s7.beta - p1 * s7.ddbeta, atol=1e-14)


class RouteTests(unittest.TestCase):
    def test_three_routes_agree_for_a_C2_member(self) -> None:
        gen = g8.PolynomialGenerator(4, [(0.0, 2.1), (0.8, 2.5)], [-0.03, 0.0004])
        rep = g8.verify_routes(gen)
        self.assertLess(rep["decimal_deviation"], 1e-13)
        self.assertLess(rep["residue_deviation"], 1e-12)
        self.assertLess(rep["endpoint_samples"], 1e-13)
        self.assertTrue(rep["decimal_ok"] and rep["residue_ok"] and rep["symbols_ok"])

    def test_zeta_tail_matches_a_long_plain_alias_sum_for_a_C4_member(self) -> None:
        gen = g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.4), (0.0, 1.2)], [-0.03, 0.0004])
        theta = np.array([0.3, 1.1, 2.4, 3.0])
        S = gen._alias_moments(theta, 64, gen.tail_plan()["A"])
        K = 20000
        ks = np.arange(-K, K + 1, dtype=float)
        X = theta[:, None] + 2.0 * PI * ks[None, :]
        P = gen.fourier(X)
        plain = [P.sum(axis=1), (X * P).sum(axis=1), (X * X * P).sum(axis=1)]
        for j in range(3):
            # the plain tail beyond K decays like K^{-(n - j - 2q - 1)} = K^{-(5 - j)}
            np.testing.assert_allclose(S[j], plain[j], rtol=0, atol=5e-12)

    def test_decimal_hurwitz_zeta(self) -> None:
        from decimal import Decimal, localcontext
        from scipy.special import zeta
        with localcontext() as ctx:
            ctx.prec = 40
            for w, a in ((2, "65.3"), (5, "70.1"), (12, "100.7")):
                val = float(g8.decimal_hurwitz_zeta(w, Decimal(a)))
                self.assertAlmostEqual(val, float(zeta(w, float(a))), delta=1e-15 * abs(val))

    def test_residue_offgrid_route_matches_gate7_at_P_equal_one(self) -> None:
        gen = g8.PolynomialGenerator(4, [(0.0, 2.0), (1.5, 0.0), (0.7, 2.3)], [])
        theta = np.linspace(0.05, PI, 150)
        xi = np.linspace(-0.5, 0.5, 9)
        ec8 = gen.coupled_error(theta, xi)
        ec7 = gen.base.coupled_error(theta, xi)
        self.assertLess(float(np.max(np.abs(ec8 - ec7))), 1e-12)
        T0 = gen.offgrid_transfers(np.array([0.0]), xi)[0]
        np.testing.assert_allclose(T0, 1.0, atol=1e-15)

    def test_residue_kernel_against_brute_force_with_phases(self) -> None:
        # a member whose phased alias sums converge fast enough for brute force
        gen = g8.PolynomialGenerator(6, [(0.0, 2.0), (0.0, 2.6)], [-0.02])
        theta = np.array([0.7, 2.0, 2.9])
        xi = np.array([-0.5, -0.2, 0.0, 0.31, 0.5])
        S = gen.offgrid_moments(theta, xi)
        K = 4000
        ks = np.arange(-K, K + 1, dtype=float)
        X = theta[:, None] + 2.0 * PI * ks[None, :]
        P = gen.fourier(X)
        E = np.exp(2j * PI * np.outer(ks, xi))
        brute = [P @ E, (X * P) @ E, (X * X * P) @ E]
        for j in range(3):
            np.testing.assert_allclose(S[j], brute[j], rtol=0, atol=2e-8)


class StructureTests(unittest.TestCase):
    def test_eighth_order_pentadiagonal_scheme_is_E60_times_1_minus_t_over_28(self) -> None:
        a = gr.anchor_check()
        self.assertLess(a["max_dev"], 1e-13)
        self.assertLess(a["kappa1_max_dev"], 1e-13)
        self.assertEqual(a["generator"]["smoothness"], 2)

    def test_real_root_inside_the_band_makes_the_recovery_singular(self) -> None:
        # P(t) = 1 - t / xi0^2 with xi0 = 0.6 pi: the k = 0 weight vanishes at theta = xi0
        xi0 = 0.6 * PI
        gen = g8.PolynomialGenerator(6, [(0.0, 2.0)], [-1.0 / xi0 ** 2])
        b, k1, _ = gen.sample_symbols(np.array([xi0]))
        e1 = float(k1[0] / xi0 - 1.0)
        self.assertGreater(abs(e1), 0.5)
        T0 = gen.offgrid_transfers(np.array([xi0]), np.linspace(-0.5, 0.5, 65))[0]
        self.assertGreater(float(np.max(np.abs(T0 - 1.0))), 0.9)

    def test_root_in_pi_2pi_is_a_soft_contact_with_an_extra_zero_of_e1(self) -> None:
        # Theorem Z gives one zero (the imaginary pair); the soft contact adds a second one
        xi0 = 1.4 * PI
        gen = g8.PolynomialGenerator(4, [(0.0, 0.75 * PI), (2.0, 0.0)], [-1.0 / xi0 ** 2])
        theta = met.theta_grid(4001)
        e1 = gen.symbols(theta)[1] / theta - 1.0
        # sign changes between lobes above the rounding floor (e_1 = O(theta^4) is 1e-15 near 0)
        keep = np.abs(e1) > 1e-10
        th, sg = theta[keep], np.sign(e1[keep])
        zeros = th[np.flatnonzero(sg[:-1] * sg[1:] < 0)] / PI
        self.assertEqual(zeros.size, 2)
        self.assertLess(abs(zeros.max() - 0.75), 2e-3)
        soft = g8.poly_roots_xi(gen.poly)["soft_contacts_theta"][0] / PI
        self.assertAlmostEqual(soft, 0.6, places=12)
        # the soft contact sits where the k = -1 alias vanishes, within the one-alias model
        self.assertLess(abs(zeros.min() - soft), 0.03)
        base = g7.ExtendedGenerator(4, [(0.0, 0.75 * PI), (2.0, 0.0)])
        e1b = base.symbols(theta)[1] / theta - 1.0
        self.assertEqual(np.count_nonzero(np.sign(e1b[:-1]) * np.sign(e1b[1:]) < 0), 1)

    def test_exact_contacts_survive_the_polynomial_factor(self) -> None:
        gen = g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.5)], [-0.03, 0.0005])
        for w in gen.omegas:
            b, k1, k2 = gen.sample_symbols(np.array([w]))
            self.assertAlmostEqual(float(k1[0]), w, places=13)
            self.assertAlmostEqual(float(k2[0]), w * w, places=12)
            self.assertAlmostEqual(float(b[0]), float(gen.fourier(np.array([w]))[0]), places=14)

    def test_orders_are_unchanged_by_P(self) -> None:
        gen = g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.5)], [-0.03, 0.0005])
        theta = met.theta_grid()
        e1 = gen.symbols(theta)[1] / theta - 1.0
        e2 = gen.symbols(theta)[2] / theta ** 2 - 1.0
        self.assertAlmostEqual(met.fitted_order(theta, e1)["order"], 4.0, delta=0.15)
        self.assertAlmostEqual(met.fitted_order(theta, e2)["order"], 2.0, delta=0.15)

    def test_dimension_count(self) -> None:
        for n in (8, 10, 12):
            counts = g8.parameter_counts(n)
            for m in range(4, n + 1, 2):
                cell = counts["family"][str(m)]
                self.assertEqual(cell["gap_C2"], 0)
                self.assertEqual(cell["parameters_C2"], n - 2 - m // 2)
            self.assertEqual(counts["family"]["4"]["gap_C4"], 1 if n >= 8 else 0)

    def test_sufficient_conditions_and_certificates(self) -> None:
        # complex-root P with m > 2q: Theorem P'' applies; certificate agrees
        gen = g8.PolynomialGenerator(8, [(0.0, 1.5)], [-0.01, 0.0002])
        cond = g8.sufficient_conditions(gen)
        self.assertTrue(cond["P_positive_on_line"])
        self.assertTrue(cond["theorem_P"] and cond["theorem_N"])
        cert = g8.certify_positivity(gen, points=50001)
        self.assertTrue(cert["b_certified"] and cert["kappa2b_certified"])
        # a soft-contact P: the theorem does not apply, the certificate still decides
        gen2 = g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.4)], [-1.0 / (1.5 * PI) ** 2])
        cond2 = g8.sufficient_conditions(gen2)
        self.assertFalse(cond2["P_positive_on_line"])
        self.assertFalse(cond2["theorem_P"])
        cert2 = g8.certify_positivity(gen2, points=50001)
        self.assertIn("b_certified", cert2)

    def test_conditioning_uses_both_extrema(self) -> None:
        gen = g8.PolynomialGenerator(4, [(0.0, 2.0), (0.0, 2.4)], [-1.5 / PI ** 2, 0.6 / PI ** 4])
        c = gen.cond()
        self.assertAlmostEqual(c["cond"], c["max_b"] / c["min_b"], places=12)
        theta = np.linspace(0.0, PI, 20001)
        b = gen.symbols(theta)[0]
        self.assertAlmostEqual(float(b[0]), 1.0, places=13)
        self.assertGreaterEqual(c["max_b"], 1.0 - 1e-13)


class MachineryTests(unittest.TestCase):
    def test_parametrization_round_trip(self) -> None:
        slots, poly = gdom.unpack("IR", 2, [2.0, np.log(3.0), -1.0, 0.25])
        self.assertEqual(slots[0], (0.0, 2.0))
        self.assertAlmostEqual(slots[1][0], 3.0, places=12)
        self.assertAlmostEqual(poly[0], -1.0 / PI ** 2, places=15)
        self.assertAlmostEqual(poly[1], 0.25 / PI ** 4, places=15)
        self.assertTrue(gdom.in_box("IR", 2, [2.0, np.log(3.0), -1.0, 0.25]))
        self.assertFalse(gdom.in_box("IR", 2, [2.0, np.log(3.0), -9.0, 0.25]))
        c = gdom.scaled_from_roots([1.25, 2.0])
        self.assertAlmostEqual(c[0], -(1.0 / 1.25 + 0.5), places=14)
        self.assertAlmostEqual(c[1], 1.0 / 2.5, places=14)
        seeds = gdom.seeds_of("II", 1)
        self.assertTrue(all(gdom.in_box("II", 1, s) for s in seeds))

    def test_evaluator_reduces_to_gate7_at_P_equal_one(self) -> None:
        C = 93.36740899597527
        x = [1.7701927977475826, 2.0606855355641316, 0.0, 0.0]
        row8 = gdom.Evaluator(6, "II", 2, C, theta_points=met.THETA_POINTS, coupled=False, safety=0.0)(x)
        row7 = gd.Evaluator(6, "II", C, safety=0.0)(x[:2], refine=True)
        for key in gd.YARDSTICKS + ("cond",):
            self.assertAlmostEqual(row8[key], row7[key], delta=1e-9 * max(1.0, abs(row7[key])))

    def test_realization_parametrization(self) -> None:
        slots = [g7.Slot(0.0, 2.0), g7.Slot(1.5, 0.0), g7.Slot(0.7, 2.3)]
        q = gr.Q_from_slots(slots)
        back = gr.slots_from_Q(q)
        self.assertEqual(sorted(s.kind for s in back), sorted(s.kind for s in slots))
        for s in slots:
            match = min(back, key=lambda t: abs(t.alpha - s.alpha) + abs(t.omega - s.omega))
            self.assertLess(abs(match.alpha - s.alpha) + abs(match.omega - s.omega), 1e-9)
        target = gr.target_vector(gr.TARGETS["SLC1992"])
        self.assertEqual(target.size, 6)
        self.assertAlmostEqual(target[0], comp.SLC_D1["alpha"][0])
        self.assertEqual(target[2], 0.0)
        self.assertAlmostEqual(target[3], -comp.SLC_D1["a"][0])

    def test_realized_generator_reproduces_its_target_symbol(self) -> None:
        # the anchor as a realization: E_{6,0} x P_1 against the eighth-order pentadiagonal scheme
        spec = {"n": 6, "d1": {"alpha": (4.0 / 9.0, 1.0 / 36.0), "a": (20.0 / 27.0, 25.0 / 216.0)}}
        target = gr.target_vector(spec)
        self.assertEqual(target.size, 4)
        res = gr.residual([-1.0 / 28.0 * PI ** 2], 6, 0, 1, target)
        self.assertLess(float(np.max(np.abs(res))), 1e-13)


if __name__ == "__main__":
    unittest.main()


class JointAndCeilingTests(unittest.TestCase):
    def test_order_rows_annihilate_slc_and_the_eighth_order_scheme(self) -> None:
        import gate8_ceiling as g8c
        import gate8_joint as gj
        # SLC's D1 is exactly fourth order (to its printed precision) and not sixth
        x = gj.slc_vector(4, 4)
        A4, _ = g8c.order_rows(4, 4, 4)
        A6, _ = g8c.order_rows(4, 4, 6)
        self.assertLess(float(np.max(np.abs(A4 @ x))), 1e-5)
        self.assertGreater(float(np.max(np.abs(A6 @ x))), 1e-2)
        # the eighth-order pentadiagonal scheme satisfies the order-8 rows exactly
        alpha = (4.0 / 9.0, 1.0 / 36.0)
        a = (20.0 / 27.0, 25.0 / 216.0)
        vec = np.array([1.0] + list(alpha) + list(a) + [0.0])
        A8, _ = g8c.order_rows(2, 2, 8)
        self.assertLess(float(np.max(np.abs(A8 @ vec))), 1e-14)

    def test_slc_vector_is_feasible_in_its_own_joint_lp(self) -> None:
        import gate7_ceiling as gc
        import gate8_joint as gj
        target = gd.comparator_targets()["SLC1992"]
        C = gc.budgets()["C_SLC1"]
        x = gj.slc_vector(3, 3)
        alpha = list(x[1:4] / x[0]); a = list(x[4:7] / x[0])
        row = gj.true_yardsticks(alpha, a)
        mg = gj.margins_d1(row, target)
        self.assertLess(max(abs(v) for v in mg.values()), 1e-9)
        # the LP rows at delta = 0 hold at SLC's point up to the printed defect
        lp = gj.build_joint_lp(3, 3, C, 0.0, target, G_state=gj.g_state(alpha, a, 0.0, target, scale=float(x[0])))
        viol = lp["A_ub"] @ x - lp["b_ub"]
        # the only active violation is the deliberate conditioning margin C * COND_MARGIN
        self.assertLess(float(viol.max()), C * (gj.COND_MARGIN + 1e-5))

    def test_written_member_reloads_and_reproduces_its_samples(self) -> None:
        import json
        from pathlib import Path
        members = Path(__file__).resolve().parents[2] / "results" / "exponential_family" / "gate8" / "members"
        files = sorted(members.glob("X8_*.json")) if members.exists() else []
        if not files:
            self.skipTest("no member file written yet")
        rec = json.loads(files[0].read_text())
        gen = g8.from_record(rec)
        s = gen.integer_samples()
        c = int(np.searchsorted(s.n, 0))
        for key, arr in (("beta", s.beta), ("dbeta", s.dbeta), ("ddbeta", s.ddbeta)):
            np.testing.assert_allclose(arr[c:c + rec["radius"] + 1], rec[key], rtol=0, atol=1e-13)
        self.assertEqual(rec["polynomial_degree"], gen.q)
        self.assertEqual(rec["smoothness"], gen.smoothness)
        self.assertGreaterEqual(rec["smoothness"], 2)
