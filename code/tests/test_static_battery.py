"""Regression tests for the static yardstick battery.

The battery is a dispatcher over three frozen back ends, so the tests check
the dispatch, the conventions that the reported numbers carry (band maxima,
reaches, conditioning, bandwidth, margins), the invariants a member must
satisfy against itself, and the reproduction of the frozen receipts: the
Gate 1 champions of results/exponential_family/gate1/champions.json to 1e-9
and the reported Gate 7 members of results/exponential_family/gate7/
dominance.json to 1e-6.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

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
import metrics as met  # noqa: E402
import static_battery as sb  # noqa: E402

PI = float(np.pi)
CHAMPION = {"m": 6, "omegas": [1.7701927977475826, 2.0606855355641316]}
CHEAP = sb.default_options(comparison=False, positivity=False, coupled=False)


class SpecificationTests(unittest.TestCase):
    def test_scaled_polynomial_is_divided_by_the_band_scale(self) -> None:
        spec = sb.normalise_spec({"m": 4, "omegas": [2.0],
                                  "polynomial_scaled": [-0.325, -0.0211]})
        self.assertAlmostEqual(spec["polynomial"][0], -0.325 / PI ** 2, places=15)
        self.assertAlmostEqual(spec["polynomial"][1], -0.0211 / PI ** 4, places=15)

    def test_slots_fix_the_order_and_split_into_kinds(self) -> None:
        spec = sb.normalise_spec({"m": 4, "slots": [[2.0, 0.0], [0.0, 1.5],
                                                    [1.0, 2.0]]})
        self.assertEqual(spec["real_pairs"], [2.0])
        self.assertEqual(spec["omegas"], [1.5])
        self.assertEqual(spec["quads"], [[1.0, 2.0]])
        self.assertEqual(spec["slots"][0], [2.0, 0.0])

    def test_the_label_names_the_exponent_multiset(self) -> None:
        label = sb.normalise_spec({"m": 6, "omegas": [1.0]})["label"]
        self.assertIn("E_{6}", label)
        self.assertIn("i1.000000", label)

    def test_back_end_dispatch_follows_the_slot_kinds(self) -> None:
        self.assertEqual(sb.backend_of(sb.normalise_spec(CHAMPION)), "gate0")
        self.assertEqual(sb.backend_of(sb.normalise_spec(
            {"m": 4, "real_pairs": [2.0]})), "gate7")
        self.assertEqual(sb.backend_of(sb.normalise_spec(
            {"m": 4, "omegas": [2.0], "polynomial": [-0.03]})), "gate8")

    def test_a_forced_back_end_is_honoured(self) -> None:
        gen, backend = sb.make_generator(sb.normalise_spec(
            dict(CHAMPION, backend="gate7")))
        self.assertEqual(backend, "gate7")
        self.assertEqual(gen.m, 6)


class BlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = sb.normalise_spec(CHAMPION)
        cls.gen, cls.backend = sb.make_generator(cls.spec)

    def test_conditioning_repeats_the_frozen_symbol_cond(self) -> None:
        block = sb.conditioning_block(self.gen)
        reference = met.symbol_cond(lambda t: self.gen.symbols(t, sb.FINE_TOL)[0])
        for key in ("cond", "min_left", "max_left", "cond_N64", "cond_N127",
                    "cond_N128"):
            self.assertAlmostEqual(block[key], reference[key], places=12)

    def test_the_minimum_location_is_where_the_symbol_is_smallest(self) -> None:
        block = sb.conditioning_block(self.gen)
        theta = block["argmin_b_over_pi"] * PI
        value = float(self.gen.symbols(np.array([theta]), sb.FINE_TOL)[0][0])
        self.assertAlmostEqual(value, block["min_b"], places=14)
        self.assertAlmostEqual(block["argmin_b_over_pi"], 1.0, places=12)

    def test_bandwidth_counts_the_nonzero_lattice_samples(self) -> None:
        block = sb.stencil_block(self.gen)
        self.assertEqual(block["bandwidth_left"], 2 * self.gen.nmax - 1)
        self.assertEqual(block["bandwidth_d1"], 2 * self.gen.nmax - 2)
        self.assertEqual(block["bandwidth_d2"], 2 * self.gen.nmax - 1)
        self.assertEqual(block["solves"], 1)

    def test_the_positivity_certificates_hold_for_the_champion(self) -> None:
        import gate7_family as g7

        block = g7.certify_positivity(self.gen)
        self.assertTrue(block["b_certified"])
        self.assertTrue(block["kappa2b_certified"])

    def test_the_two_routes_to_S_m_agree(self) -> None:
        block = sb.sufficient_block(self.gen, self.spec, self.backend)
        self.assertTrue(block["applies"])
        self.assertAlmostEqual(block["S_m"], block["S_m_zeta"], places=12)
        self.assertLess(block["S_m"], 1.0)
        self.assertLess(block["S_m_minus_2"], 1.0)

    def test_the_decimal_route_confirms_the_double_samples(self) -> None:
        block = sb.decimal_block(self.gen, self.backend)
        self.assertLess(block["decimal_deviation"], 1e-13)
        self.assertTrue(block["decimal_ok"])
        self.assertEqual(block["working_digits"], 80)


class ComparisonTests(unittest.TestCase):
    def test_a_member_against_itself_has_zero_margin_and_dominates(self) -> None:
        result = sb.evaluate_spec(CHAMPION, CHEAP)
        row = result["metrics"]
        target = {key: sb._value(row, key) for key in sb.RATIO_YARDSTICKS}
        block = sb.comparison_block(row, {"self": target})["self"]
        self.assertAlmostEqual(block["smallest_margin"], 0.0, places=14)
        self.assertTrue(block["dominates"])
        for value in block["ratios"].values():
            self.assertAlmostEqual(value, 1.0, places=14)

    def test_the_margin_signs_follow_the_direction_of_each_yardstick(self) -> None:
        row = {"reach_d1_0.01_refined": 0.8, "e1_max_B0": 1e-3}
        target = {"reach_d1_0.01": 0.4, "e1_max_B0": 2e-3}
        margins = sb.margins_against(row, target,
                                     ("reach_d1_0.01", "e1_max_B0"))
        self.assertAlmostEqual(margins["reach_d1_0.01"], 1.0, places=14)
        self.assertAlmostEqual(margins["e1_max_B0"], 0.5, places=14)

    def test_the_budget_verdict_adds_the_conditioning_constraint(self) -> None:
        row = {"reach_d1_0.01_refined": 0.9, "reach_d1_0.001_refined": 0.9,
               "reach_d1_0.0001_refined": 0.9, "gv_reach_1e-2": 0.9,
               "e1_max_B0": 1e-5, "e2_max_B0": 1e-5, "cond": 1000.0}
        target = {"reach_d1_0.01": 0.8, "reach_d1_0.001": 0.8,
                  "reach_d1_0.0001": 0.8, "gv_reach_1e-2": 0.8,
                  "e1_max_B0": 1e-4, "e2_max_B0": 1e-4, "cond": 93.0}
        block = sb.comparison_block(row, {"target": target})["target"]
        self.assertTrue(block["dominates"])
        self.assertFalse(block["cond_ok"])
        self.assertFalse(block["dominates_at_budget"])
        self.assertAlmostEqual(block["cond_ratio"], 1000.0 / 93.0, places=12)

    def test_a_member_inside_the_budget_keeps_its_verdict(self) -> None:
        result = sb.evaluate_spec(CHAMPION, sb.default_options(positivity=False))
        block = result["comparison"]["SLC1992"]
        self.assertTrue(block["cond_ok"])
        self.assertEqual(block["dominates"], block["dominates_at_budget"])

    def test_every_comparator_of_the_table_carries_the_core_yardsticks(self) -> None:
        targets = sb.comparator_targets()
        self.assertEqual(set(targets), set(sb.COMPARATORS))
        for target in targets.values():
            for key in sb.DOMINANCE_YARDSTICKS:
                self.assertIsNotNone(target[key])


class BatteryTests(unittest.TestCase):
    def test_the_battery_carries_every_block_the_contract_asks_for(self) -> None:
        result = sb.evaluate_spec(CHAMPION, sb.default_options())
        self.assertIsNone(result["error"])
        self.assertTrue(result["feasible"])
        for key in ("family", "metrics", "conditioning", "stencil", "positivity",
                    "sufficient_conditions", "comparison", "admissible"):
            self.assertIn(key, result)
        metrics = result["metrics"]
        for band in ("B0", "B1", "B2"):
            for tag in ("e1_max_", "e1_rms_", "e2_max_", "e2_rms_", "ec_max_"):
                self.assertIsNotNone(metrics[tag + band])
        for tol in ("0.01", "0.001", "0.0001"):
            self.assertIsNotNone(metrics[f"reach_d1_{tol}_refined"])
            self.assertIsNotNone(metrics[f"reach_d2_{tol}_refined"])
        self.assertIsNotNone(metrics["gv_reach_1e-2"])

    def test_an_inadmissible_member_is_reported_and_not_raised(self) -> None:
        result = sb.evaluate_spec({"m": 4, "omegas": [3.5]}, CHEAP)
        self.assertFalse(result["feasible"])
        self.assertIn("pi", result["reason"])

    def test_the_gate7_back_end_repeats_the_gate0_member(self) -> None:
        options = sb.default_options(comparison=False, positivity=False)
        base = sb.evaluate_spec(CHAMPION, options)
        extended = sb.evaluate_spec(dict(CHAMPION, backend="gate7"), options)
        for key in ("e1_max_B0", "e2_max_B0", "reach_d1_0.001_refined",
                    "gv_reach_1e-2", "ec_max_B0"):
            self.assertAlmostEqual(base["metrics"][key], extended["metrics"][key],
                                   delta=1e-9 * abs(base["metrics"][key]) + 1e-15)

    def test_the_polynomial_factor_reduces_to_the_gate7_member(self) -> None:
        options = sb.default_options(comparison=False, positivity=False,
                                     coupled=False)
        plain = sb.evaluate_spec({"m": 4, "omegas": [2.0, 2.4]}, options)
        augmented = sb.evaluate_spec({"m": 4, "omegas": [2.0, 2.4],
                                      "polynomial": [0.0]}, options)
        self.assertEqual(augmented["backend"], "gate8")
        self.assertAlmostEqual(plain["metrics"]["e1_max_B0"],
                               augmented["metrics"]["e1_max_B0"], places=12)

    def test_many_specifications_run_in_parallel_like_one_at_a_time(self) -> None:
        specs = [CHAMPION, {"m": 8, "omegas": [2.0]}]
        serial = sb.evaluate_specs(specs, CHEAP, processes=1)
        parallel = sb.evaluate_specs(specs, CHEAP, processes=2)
        for one, two in zip(serial, parallel):
            self.assertAlmostEqual(one["metrics"]["e1_max_B0"],
                                   two["metrics"]["e1_max_B0"], places=14)


class SweepTests(unittest.TestCase):
    def test_the_grid_drops_the_permutations_of_one_multiset(self) -> None:
        grid = {"m": [4, 6], "axes": [
            {"kind": "imaginary", "range": [1.0, 2.0], "count": 4},
            {"kind": "imaginary", "range": [1.0, 2.0], "count": 4}]}
        specs = sb.sweep_specs(grid)
        self.assertEqual(len(specs), 2 * 6)
        for spec in specs:
            self.assertLess(spec["omegas"][0], spec["omegas"][1])

    def test_a_logarithmic_axis_is_geometric(self) -> None:
        values = sb.axis_values({"range": [0.5, 8.0], "count": 5, "scale": "log"})
        np.testing.assert_allclose(values[1:] / values[:-1], 2.0, rtol=1e-12)

    def test_quad_and_polynomial_axes_build_the_right_slots(self) -> None:
        grid = {"m": 4, "axes": [
            {"kind": "quad", "alpha": {"range": [1.0, 2.0], "count": 2},
             "omega": {"range": [1.0, 2.0], "count": 2}},
            {"kind": "poly", "range": [-0.01, 0.01], "count": 3}]}
        specs = sb.sweep_specs(grid)
        self.assertEqual(len(specs), 12)
        self.assertEqual(specs[0]["quads"], [[1.0, 1.0]])
        self.assertEqual(len(specs[0]["polynomial"]), 1)

    def test_the_csv_row_carries_every_column(self) -> None:
        result = sb.evaluate_spec(CHAMPION, CHEAP)
        row = sb.csv_row(result)
        self.assertEqual(set(row), set(sb.CSV_COLUMNS))
        self.assertEqual(row["m"], 6)
        self.assertTrue(row["feasible"])


@unittest.skipUnless(sb.GATE1.exists(), "the Gate 1 receipt is not present")
class Gate1ReproductionTests(unittest.TestCase):
    def test_every_champion_metric_is_reproduced(self) -> None:
        cases = sb.gate1_cases()
        self.assertGreaterEqual(len(cases), 6)
        case = [c for c in cases if c["key"] == "B0|C_SLC1"][0]
        options = sb.default_options(comparison=False)
        result = sb.evaluate_spec(case["spec"], options)
        row = dict(result["metrics"])
        row.update(result["conditioning"])
        row.update(result["stencil"])
        report = sb._compare_floats(case["metrics"], row, sb.GATE1_RTOL)
        self.assertGreater(report["compared"], 50)
        self.assertTrue(report["ok"], report["failures"][:3])


@unittest.skipUnless(sb.GATE7.exists(), "the Gate 7 receipt is not present")
class Gate7ReproductionTests(unittest.TestCase):
    def test_a_reported_member_is_reproduced(self) -> None:
        cases = sb.gate7_cases()
        self.assertGreaterEqual(len(cases), 1)
        case = cases[0]
        options = sb.default_options(comparison=False)
        result = sb.evaluate_spec(case["spec"], options)
        row = dict(result["metrics"])
        row.update(result["conditioning"])
        row.update(result["stencil"])
        report = sb._compare_floats(case["metrics"], row, sb.GATE7_RTOL)
        self.assertGreater(report["compared"], 50)
        self.assertTrue(report["ok"], report["failures"][:3])
        positivity = sb._compare_floats(case["positivity"], result["positivity"],
                                        sb.GATE7_RTOL)
        self.assertTrue(positivity["ok"], positivity["failures"][:3])


class ContractTests(unittest.TestCase):
    def test_the_reach_convention_is_the_frozen_one(self) -> None:
        theta = met.theta_grid()
        self.assertEqual(theta.size, met.THETA_POINTS)
        self.assertAlmostEqual(float(theta[-1]), PI, places=14)

    def test_the_coupled_grid_spans_one_cell(self) -> None:
        xi = np.linspace(-0.5, 0.5, sb.XI_POINTS)
        self.assertEqual(xi.size, 129)
        self.assertAlmostEqual(float(xi[0]), -0.5, places=15)
        self.assertAlmostEqual(float(xi[-1]), 0.5, places=15)

    def test_the_polynomial_limit_member_is_the_collocation_scheme(self) -> None:
        result = sb.evaluate_spec({"m": 12, "label": "BMC11"}, CHEAP)
        reference = fam.ExponentialGenerator(12, ())
        self.assertAlmostEqual(result["conditioning"]["cond"],
                               reference.cond()["cond"], places=9)


if __name__ == "__main__":
    unittest.main()
