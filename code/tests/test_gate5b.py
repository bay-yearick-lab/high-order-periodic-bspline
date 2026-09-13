"""Checks for the semi-Lagrangian mission of Gate 5 part 5b.

The mission is a rigid periodic translation, so every departure point carries
the same cell coordinate and one step of every route is a tensor product of
three periodic one-dimensional operators. Everything tested here is that
identification and its consequences: the packet is real, divergence-free and
normalized; the half-spectrum bookkeeping reproduces the nodal field; the
periodic banded solve is the circulant inverse; the separable step is the
direct tensor-product evaluation at the departure points; the axis symbols are
the transfer functions of part 5a; the realized histories are the modal
prediction; and the gradient channel of a one-field route is the derivative of
its own interpolant while the operator-faithful augmentation's is not.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
ANALYSIS_DIR = CODE_DIR / "analysis" / "exponential_family"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import gate5a_offgrid as g5a
import gate5b_semilagrangian as g5b
from evaluator import periodic_coefficients


class PacketTests(unittest.TestCase):
    def test_amplitudes_are_divergence_free_and_conjugate(self) -> None:
        modes, amps = g5b.packet_modes(5)
        self.assertEqual(modes.shape[0], (2 * 5 + 1) ** 3 - 1)
        self.assertLess(float(np.max(np.abs(np.sum(modes * amps, axis=1)))), 1e-14)
        half = modes.shape[0] // 2
        self.assertTrue(np.array_equal(modes[half:], -modes[:half]))
        self.assertLess(float(np.max(np.abs(amps[half:] - np.conj(amps[:half])))),
                        0.0 + 1e-300)

    def test_mean_square_is_one(self) -> None:
        for K in (3, 7):
            _, amps = g5b.packet_modes(K)
            self.assertAlmostEqual(float(np.sum(np.abs(amps) ** 2)), 1.0, places=13)

    def test_synthesis_matches_the_analytic_sum(self) -> None:
        """The half-spectrum scatter is the nodal field, aliasing included."""
        modes, amps = g5b.packet_modes(4)
        for npts in (16, 6):
            keep, group, index, _ = g5b.grid_groups(modes, npts)
            count = index.shape[0]
            carried = g5b.half_sum(group, amps[keep][:, 0], count)
            got = g5b.synthesize(index, carried, npts)
            axis = 2.0 * np.pi * np.arange(npts) / npts
            X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
            want = np.zeros((npts, npts, npts))
            for k, c in zip(modes, amps[:, 0]):
                want += np.real(c * np.exp(1j * (k[0] * X + k[1] * Y + k[2] * Z)))
            self.assertLess(float(np.max(np.abs(got - want))), 1e-13)


class MemberTests(unittest.TestCase):
    def test_the_route_reads_the_frozen_champion(self) -> None:
        record = g5b.member_record()
        self.assertTrue(record["matches_route"])
        self.assertEqual((record["m"], record["p"]), (6, 9))
        route = g5b.build_routes()["champion"]
        self.assertEqual(route.evaluator.p, record["p"])


class OperatorTests(unittest.TestCase):
    def test_cyclic_banded_is_the_circulant_inverse(self) -> None:
        rng = np.random.default_rng(11)
        for coeffs in ([0.41, 0.245, 0.047, 0.002, 4.9e-6], [1.0, 0.5771439, 0.0896406]):
            npts = 37
            solver = g5b.CyclicBanded(np.asarray(coeffs), npts)
            rhs = rng.standard_normal((6, npts))
            got = solver.solve(rhs)
            theta = 2.0 * np.pi * np.arange(npts) / npts
            want = np.real(np.fft.ifft(np.fft.fft(rhs, axis=1)
                                       / solver.symbol(theta), axis=1))
            self.assertLess(float(np.max(np.abs(got - want))
                                  / np.max(np.abs(want))), 1e-12)

    def test_stencil_is_the_shift_sum(self) -> None:
        rng = np.random.default_rng(12)
        data = rng.standard_normal(13)
        offsets = np.arange(-4, 3)
        weights = rng.standard_normal(offsets.size)
        got = g5b.apply_stencil(data, 0, offsets, weights)
        want = np.zeros_like(data)
        for weight, offset in zip(weights, offsets):
            want += weight * np.roll(data, int(offset))
        self.assertLess(float(np.max(np.abs(got - want))), 1e-14)

    def test_lagrange_layer_matches_part_5a(self) -> None:
        """One axis of the SLC layer is the Milestone 10 Lagrange interpolant."""
        npts = 16
        step = 2.0 * np.pi / npts
        delta = g5b.shift_vector("B")
        route = g5b.build_routes()["slc_operator"]
        plan = route.plan(npts, delta)
        rng = np.random.default_rng(13)
        field = rng.standard_normal((npts, npts, npts))
        axis = plan.axes[2]
        got = g5b.apply_stencil(np.roll(field, plan.rolls[2], axis=2), 2,
                                axis.offsets, axis.weights[0])
        want = g5a.lagrange_apply(field[1, 2, :], step * np.arange(npts) - delta[2],
                                  step, 0)
        self.assertLess(float(np.max(np.abs(got[1, 2, :] - want))), 1e-13)


class SymbolTests(unittest.TestCase):
    def test_axis_symbols_are_the_part_5a_transfers(self) -> None:
        """Each axis operator is the 5a transfer at the cell the shift selects.

        The departure point x_j - delta sits at the cell coordinate -xi of the
        node j - round(s), so the axis symbol carries that integer shift times
        the carrier-kept transfer of 5a, with the exact channel factors
        (i theta / h)^q that 5a divides out.
        """
        npts = 24
        step = 2.0 * np.pi / npts
        theta = 2.0 * np.pi * np.arange(1, npts) / npts
        delta = g5b.shift_vector("A")
        reference = {route.name: route for route in g5a.build_routes()}
        for name in ("champion", "BMC9", "slc_operator"):
            route = g5b.build_routes()[name]
            plan = route.plan(npts, delta)
            for axis in range(3):
                s = delta[axis] / step
                shift = int(np.rint(s))
                xi = s - shift
                p0, p1, p2 = route.axis_symbols(plan, axis)
                hat0, hat1, hat2, _, _ = reference[name].hat_transfers(
                    theta, np.array([-xi]))
                carrier = np.exp(-1j * shift * theta)
                self.assertLess(float(np.max(np.abs(p0[1:] - carrier * hat0[:, 0]))),
                                1e-13)
                self.assertLess(float(np.max(np.abs(
                    p1[1:] - (1j * theta / step) * carrier * hat1[:, 0]))), 1e-11)
                self.assertLess(float(np.max(np.abs(
                    p2[1:] + (theta ** 2 / step ** 2) * carrier * hat2[:, 0]))), 1e-10)

    def test_step_reproduces_its_mode_transfers(self) -> None:
        npts = 24
        delta = g5b.shift_vector("A")
        axis = 2.0 * np.pi * np.arange(npts) / npts
        X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
        wave = np.array([3, -5, 7])
        field = np.real(np.exp(1j * (wave[0] * X + wave[1] * Y + wave[2] * Z)))
        for name, route in g5b.build_routes().items():
            plan = route.plan(npts, delta)
            got = {}

            def sink(key, array, store=got):
                store[key] = array.copy()

            route.step_channels(plan, field, sink)
            transfers = g5b.mode_transfers(route, plan, (wave % npts)[None, :])
            for key, value in transfers.items():
                want = np.real(value[0] * np.exp(1j * (wave[0] * X + wave[1] * Y
                                                       + wave[2] * Z)))
                scale = max(float(np.max(np.abs(want))), 1e-30)
                self.assertLess(float(np.max(np.abs(want - got[key]))) / scale, 1e-12)

    def test_constants_are_carried_exactly(self) -> None:
        """The zero mode of every step operator is one, so mass is conserved."""
        npts = 32
        delta = g5b.shift_vector("B")
        for route in g5b.build_routes().values():
            plan = route.plan(npts, delta)
            for axis in range(3):
                p0 = route.axis_symbols(plan, axis)[0]
                self.assertAlmostEqual(float(p0[0].real), 1.0, places=13)
                self.assertLess(abs(float(p0[0].imag)), 1e-13)


class SeparableTests(unittest.TestCase):
    def test_spline_step_matches_the_direct_tensor_product(self) -> None:
        """The rigid shift written as three axis stencils is the field read.

        The reference is the definition itself: recover the coefficient tensor
        one axis at a time with the routine of evaluator.py, then sum the
        tensor product of the generator and its derivatives over the support
        at the departure point.
        """
        npts = 16
        step = 2.0 * np.pi / npts
        delta = g5b.shift_vector("B")
        rng = np.random.default_rng(14)
        field = rng.standard_normal((npts, npts, npts))
        route = g5b.build_routes()["champion"]
        evaluator = route.evaluator
        got = {}

        def sink(key, array, store=got):
            store[key] = array.copy()

        route.step_channels(route.plan(npts, delta), field, sink)
        coeff = field.copy()
        for axis in range(3):
            coeff = np.apply_along_axis(
                lambda line: periodic_coefficients(line, evaluator), axis, coeff)
        radius = evaluator.n / 2.0
        for node in [(0, 0, 0), (3, 7, 11), (15, 2, 9)]:
            target = np.asarray(node) * step - delta
            local = target / step
            ranges = [np.arange(int(np.floor(local[d] - radius)),
                                int(np.ceil(local[d] + radius)) + 1) for d in range(3)]
            basis = [[evaluator.derivative(local[d] - ranges[d], q) for q in range(3)]
                     for d in range(3)]
            block = coeff[np.ix_(ranges[0] % npts, ranges[1] % npts,
                                 ranges[2] % npts)]

            def term(a, b, c):
                return float(np.einsum("ijk,i,j,k->", block, basis[0][a],
                                       basis[1][b], basis[2][c]))

            want = {"value": term(0, 0, 0),
                    "grad_x": term(1, 0, 0) / step,
                    "grad_y": term(0, 1, 0) / step,
                    "grad_z": term(0, 0, 1) / step,
                    "laplacian": (term(2, 0, 0) + term(0, 2, 0)
                                  + term(0, 0, 2)) / step ** 2}
            for key, value in want.items():
                self.assertLess(abs(value - float(got[key][node])),
                                1e-10 * max(abs(value), 1.0))


class AmendmentTests(unittest.TestCase):
    """The routes and the ladder amendment 1 adds."""

    def test_the_eight_point_augmentation_keeps_its_original_name(self) -> None:
        self.assertEqual(g5b.csv_route("slc_operator_L8"), "slc_operator")
        route = g5b.build_routes()["slc_operator"]
        self.assertEqual(route.n_nodes, 8)
        self.assertEqual(route.name, "slc_operator")

    def test_the_layer_width_sets_the_taps(self) -> None:
        for width in g5b.LAGRANGE_WIDTHS:
            route = g5b.build_routes()[g5b.csv_route(f"slc_operator_L{width}")]
            self.assertEqual(route.taps, width)
            self.assertEqual(route.n_nodes, width)

    def test_the_ladder_rungs_are_within_a_quarter_of_each_other(self) -> None:
        ladder = g5b.AMENDMENT_LADDER
        for lower, upper in zip(ladder, ladder[1:]):
            self.assertLessEqual(upper / lower, 1.25 + 1e-12)

    def test_amendment_axis_symbols_are_the_part_5a_transfers(self) -> None:
        """The added widths and BMC11 carry the amendment transfers of part 5a."""
        npts = 24
        step = 2.0 * np.pi / npts
        theta = 2.0 * np.pi * np.arange(1, npts) / npts
        delta = g5b.shift_vector("B")
        reference = {route.name: route for route in g5a.build_amendment_routes()}
        for name in ("BMC11", "slc_operator_L12", "slc_operator_L16"):
            route = g5b.build_routes()[g5b.csv_route(name)]
            plan = route.plan(npts, delta)
            for axis in range(3):
                s = delta[axis] / step
                shift = int(np.rint(s))
                xi = s - shift
                p0, p1, p2 = route.axis_symbols(plan, axis)
                hat0, hat1, hat2, _, _ = reference[name].hat_transfers(
                    theta, np.array([-xi]))
                carrier = np.exp(-1j * shift * theta)
                self.assertLess(float(np.max(np.abs(p0[1:] - carrier * hat0[:, 0]))),
                                1e-12)
                self.assertLess(float(np.max(np.abs(
                    p1[1:] - (1j * theta / step) * carrier * hat1[:, 0]))), 1e-10)
                self.assertLess(float(np.max(np.abs(
                    p2[1:] + (theta ** 2 / step ** 2) * carrier * hat2[:, 0]))), 1e-9)

    def test_amendment_histories_match_the_prediction(self) -> None:
        modes, amps = g5b.packet_modes(5)
        delta = g5b.shift_vector("B")
        for name in ("BMC11", "slc_operator_L12", "slc_operator_L16"):
            route = g5b.build_routes()[g5b.csv_route(name)]
            history, _ = g5b.run_mission(route, 16, delta, modes, amps, 3, True, 1)
            predicted = g5b.predicted_histories(route, 16, modes, amps, delta, 3)
            for key in g5b.CHANNELS:
                for step in range(3):
                    self.assertLess(abs(history[key][step] - predicted[key][step]),
                                    1e-8 * predicted[key][step] + 1e-14)


class MissionTests(unittest.TestCase):
    def test_realized_histories_match_the_prediction(self) -> None:
        """The realized nodal mission is the modal arithmetic of the transfers."""
        modes, amps = g5b.packet_modes(6)
        delta = g5b.shift_vector("A")
        for name in g5b.ROUTE_NAMES:
            route = g5b.build_routes()[name]
            for npts in (16, 32):
                history, _ = g5b.run_mission(route, npts, delta, modes, amps, 3,
                                             True, 1)
                predicted = g5b.predicted_histories(route, npts, modes, amps, delta, 3)
                for key in g5b.CHANNELS:
                    for step in range(3):
                        self.assertLess(
                            abs(history[key][step] - predicted[key][step]),
                            1e-8 * predicted[key][step] + 1e-14)

    def test_mass_is_conserved_to_rounding(self) -> None:
        modes, amps = g5b.packet_modes(6)
        route = g5b.build_routes()["slc_operator"]
        history, _ = g5b.run_mission(route, 32, g5b.shift_vector("B"), modes, amps,
                                     4, True, 1)
        self.assertLess(float(np.max(history["mass"])), g5b.MASS_TOL)

    def test_coherence_separates_the_routes(self) -> None:
        """A one-field route differentiates its own interpolant, SLC does not."""
        routes = g5b.build_routes()
        one_field = g5b.coherence_residual(routes["champion"], 32, 6, "A")
        augmented = g5b.coherence_residual(routes["slc_operator"], 32, 6, "A")
        for key in ("grad_x", "grad_y", "grad_z"):
            self.assertLess(one_field[key], 1e-8)
            self.assertGreater(augmented[key], 1e-4)


if __name__ == "__main__":
    unittest.main()
