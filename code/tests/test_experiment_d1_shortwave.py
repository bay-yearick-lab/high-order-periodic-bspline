import io
import math
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import experiment_d1
import numpy as np
import spline_analysis as sa
from experiment_d1 import (
    FORMAL_ORDER,
    SECOND_DERIVATIVE_UNIQUE_METHODS,
    UNIQUE_METHODS,
    CarrierRegimeSpec,
    FrontierPoint,
    MarchResult,
    PAPER_SHORTWAVE_REGIME,
    ShortwaveCandidateSummary,
    build_all_curves,
    build_unique_curves,
    collect_carrier_runtime_results,
    repeated_flop_model,
    select_detail_labels,
    collect_runtime_results,
    choose_headline_shortwave_regime,
    run_carrier_convergence_case,
    print_bspline_win_audit,
    build_all_second_derivative_curves,
    build_unique_second_derivative_curves,
    collect_gaussian_diffusion_convergence_results,
    plot_second_derivative_cost_tradeoff,
    plot_second_derivative_diffusion,
    plot_second_derivative_dispersion_landscape,
    second_derivative_work_units,
    summarize_shortwave_candidate,
    unique_scheme_entries,
    unique_second_derivative_scheme_entries,
)


class ShortwaveScreeningTests(unittest.TestCase):
    def test_paper_shortwave_regime_is_pinned_to_current_stress(self) -> None:
        self.assertEqual(PAPER_SHORTWAVE_REGIME.slug, "current_stress")

    def test_run_carrier_convergence_case_is_exact_at_zero_time(self) -> None:
        result = run_carrier_convergence_case(
            "E4",
            64,
            mode=8,
            time_final=0.0,
            sigma_env=0.1,
        )

        self.assertAlmostEqual(result.error, 0.0)
        self.assertGreater(result.derivative_flops, 0)

    def test_summarize_shortwave_candidate_counts_pairwise_advantages(self) -> None:
        regime = CarrierRegimeSpec(
            slug="candidate",
            title="Candidate regime",
            mode=68,
            sigma_env_scale=8.0,
            time_final=2.0,
            common_cfl=0.05,
            fairness_lens="common CFL",
        )
        results = {
            "E6": [
                MarchResult("E6", 128, 4.0, 0.01, 10, wall_time=1.0),
                MarchResult("E6", 256, 3.0, 0.01, 10, wall_time=1.0),
                MarchResult("E6", 512, 2.0, 0.01, 10, wall_time=1.0),
            ],
            "C6": [
                MarchResult("C6", 128, 2.0, 0.01, 10, wall_time=2.0),
                MarchResult("C6", 256, 1.0, 0.01, 10, wall_time=2.0),
                MarchResult("C6", 512, 1.0, 0.01, 10, wall_time=2.0),
            ],
            "BMC5 = BMG2": [
                MarchResult("BMC5 = BMG2", 128, 1.0, 0.01, 10, wall_time=2.1),
                MarchResult("BMC5 = BMG2", 256, 0.5, 0.01, 10, wall_time=2.1),
                MarchResult("BMC5 = BMG2", 512, 2.0, 0.01, 10, wall_time=2.1),
            ],
        }

        summary = summarize_shortwave_candidate(regime, results)

        self.assertEqual(summary.bspline_beats_compact, 2)
        self.assertEqual(summary.bspline_beats_explicit, 2)
        self.assertAlmostEqual(summary.compact_over_bspline_geomean, 2.0 ** (1.0 / 3.0))
        self.assertAlmostEqual(summary.explicit_over_bspline_geomean, 24.0 ** (1.0 / 3.0))
        self.assertAlmostEqual(summary.bspline_time_over_compact_geomean, 1.05)

    def test_choose_headline_shortwave_regime_prefers_consistent_c6_wins(self) -> None:
        common_fields = dict(
            bspline_beats_explicit=5,
            explicit_over_bspline_geomean=2.5,
            bspline_time_over_compact_geomean=1.03,
        )
        strong = ShortwaveCandidateSummary(
            regime=CarrierRegimeSpec(
                slug="strong",
                title="Strong",
                mode=68,
                sigma_env_scale=8.0,
                time_final=2.0,
                common_cfl=0.05,
                fairness_lens="common CFL",
            ),
            bspline_beats_compact=5,
            compact_over_bspline_geomean=1.8,
            **common_fields,
        )
        mixed = ShortwaveCandidateSummary(
            regime=CarrierRegimeSpec(
                slug="mixed",
                title="Mixed",
                mode=56,
                sigma_env_scale=6.0,
                time_final=8.0,
                common_cfl=0.05,
                fairness_lens="common CFL",
            ),
            bspline_beats_compact=3,
            compact_over_bspline_geomean=1.9,
            **common_fields,
        )

        chosen = choose_headline_shortwave_regime([mixed, strong])

        self.assertEqual(chosen.regime.slug, "strong")

    def test_spatial_cost_model_matches_manual_e4_c6_bmc5_counts(self) -> None:
        expected = {
            "E4": (2, 0, 7),
            "C6": (2, 1, 11),
            "C8": (3, 1, 14),
            "BMC5 = BMG2": (2, 2, 15),
        }

        for label, target in expected.items():
            mass_entries, advection_entries = unique_scheme_entries(label)
            model = repeated_flop_model(mass_entries, advection_entries)
            self.assertEqual(
                (model.rhs_pairs, model.solve_radius, model.derivative_flops_per_point),
                target,
            )

    def test_direct_second_derivative_cost_model_matches_manual_e4_c6_bmc5_counts(self) -> None:
        expected = {
            "E4": (2, 0, 7),
            "C6": (2, 1, 11),
            "BMC5 = BMG2": (2, 2, 15),
        }

        for label, target in expected.items():
            mass_entries, second_derivative_entries = unique_second_derivative_scheme_entries(label)
            model = repeated_flop_model(mass_entries, second_derivative_entries)
            self.assertEqual(
                (model.rhs_pairs, model.solve_radius, model.derivative_flops_per_point),
                target,
            )
            self.assertEqual(second_derivative_work_units(label), target[2])

    def test_unique_method_set_keeps_selected_even_degree_bspline_collocation_schemes(self) -> None:
        self.assertIn("C8", UNIQUE_METHODS)
        self.assertNotIn("BMC4", UNIQUE_METHODS)

        for label in ["BMC2", "BMC6", "BMC8", "BMC10"]:
            self.assertIn(label, UNIQUE_METHODS)

        w = np.linspace(1e-12, np.pi, 32)
        unique_curves = build_unique_curves(build_all_curves(w))
        self.assertIn("C8", unique_curves)
        self.assertNotIn("BMC4", unique_curves)
        for label in ["BMC2", "BMC6", "BMC8", "BMC10"]:
            self.assertIn(label, unique_curves)

    def test_unique_second_derivative_method_set_omits_unimplemented_c8_and_keeps_bspline_aliases(self) -> None:
        self.assertNotIn("C8", SECOND_DERIVATIVE_UNIQUE_METHODS)
        self.assertIn("BMC5 = BMG2", SECOND_DERIVATIVE_UNIQUE_METHODS)

        w = np.linspace(1e-12, np.pi, 32)
        unique_curves = build_unique_second_derivative_curves(build_all_second_derivative_curves(w))
        self.assertNotIn("C8", unique_curves)
        self.assertTrue(np.allclose(unique_curves["C4 = BMC3 = BMG1"], sa.ksecond_compact(w, 4)))
        self.assertTrue(np.allclose(unique_curves["BMC5 = BMG2"], sa.ksecond_collocation(w, 5)))

    def test_select_detail_labels_prefers_high_score_per_family(self) -> None:
        actual_frontier = [
            FrontierPoint("E8", 256, 1.0, 1.0e-3),
            FrontierPoint("E8", 512, 2.0, 1.0e-4),
            FrontierPoint("C6", 256, 1.5, 8.0e-4),
            FrontierPoint("BMC7 = BMG3", 256, 1.8, 7.0e-4),
            FrontierPoint("BMC7 = BMG3", 512, 3.6, 5.0e-4),
        ]
        resolution_frontier = [
            FrontierPoint("E8", 256, 1.0, 20.0),
            FrontierPoint("C6", 256, 1.5, 24.0),
            FrontierPoint("BMC7 = BMG3", 256, 1.8, 28.0),
            FrontierPoint("BMC7 = BMG3", 512, 3.6, 56.0),
        ]

        detail_labels = select_detail_labels(actual_frontier, resolution_frontier)

        self.assertEqual(detail_labels, ("E8", "C6", "BMC7 = BMG3"))

    def test_spline_labels_keep_degree_while_formal_orders_follow_operator_rule(self) -> None:
        self.assertEqual(sa.formal_order_collocation(2), 2)
        self.assertEqual(sa.formal_order_collocation(3), 4)
        self.assertEqual(sa.formal_order_collocation(5), 6)
        self.assertEqual(sa.formal_order_collocation(6), 6)
        self.assertEqual(sa.formal_order_iga(1), 4)
        self.assertEqual(sa.formal_order_iga(2), 6)

        self.assertEqual(FORMAL_ORDER["BMC3"], 4)
        self.assertEqual(FORMAL_ORDER["BMC5"], 6)
        self.assertEqual(FORMAL_ORDER["BMG2"], 6)
        self.assertEqual(FORMAL_ORDER["C4 = BMC3 = BMG1"], 4)
        self.assertEqual(FORMAL_ORDER["BMC5 = BMG2"], 6)

    def test_collocation_first_equality_labels_map_to_the_same_operator(self) -> None:
        combo_mass_entries, combo_advection_entries = unique_scheme_entries("BMC7 = BMG3")
        collocation_mass_entries, collocation_advection_entries = unique_scheme_entries("BMC7")

        self.assertTrue(np.allclose(combo_mass_entries, collocation_mass_entries))
        self.assertTrue(np.allclose(combo_advection_entries, collocation_advection_entries))
        self.assertEqual(FORMAL_ORDER["BMC7 = BMG3"], 8)

    def test_bspline_win_audit_prints_spaced_displayed_equality_labels(self) -> None:
        metrics_map = {
            "E6": SimpleNamespace(ppw_1pct=5.71, ppw_5pct=4.23, rk4_cfl=1.78),
            "C6": SimpleNamespace(ppw_1pct=3.98, ppw_5pct=3.15, rk4_cfl=1.42),
            "BMC5 = BMG2": SimpleNamespace(ppw_1pct=3.66, ppw_5pct=2.97, rk4_cfl=1.36),
        }

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            print_bspline_win_audit(metrics_map)

        output = stdout.getvalue()

        self.assertIn("PPW@1%: E6 5.71, C6 3.98, BMC5 = BMG2 3.66", output)
        self.assertIn("PPW@5%: E6 4.23, C6 3.15, BMC5 = BMG2 2.97", output)
        self.assertIn("RK4 CFL: E6 1.78, C6 1.42, BMC5 = BMG2 1.36", output)
        self.assertNotIn("BMC5=BMG2", output)

    def test_bm_banded_map_plot_data_exposes_periodic_banded_maps(self) -> None:
        data = experiment_d1.bm_banded_map_plot_data(npts=32)

        self.assertEqual(data["npts"], 32)
        self.assertEqual(data["bmc"]["B"].shape, (32, 32))
        self.assertEqual(data["bmc"]["B'"].shape, (32, 32))
        self.assertEqual(data["bmg"]["M"].shape, (32, 32))
        self.assertEqual(data["bmg"]["K"].shape, (32, 32))

        for family in ("bmc", "bmg"):
            for name in data[family]["matrix_order"]:
                mat = data[family][name]
                self.assertLess(np.count_nonzero(np.abs(mat) > 1.0e-14), mat.size)

        self.assertTrue(np.any(np.abs(data["bmc"]["B"][:4, -4:]) > 1.0e-14))
        self.assertTrue(np.any(np.abs(data["bmc"]["B'"][:4, -4:]) > 1.0e-14))
        self.assertTrue(np.any(np.abs(data["bmg"]["M"][:4, -4:]) > 1.0e-14))
        self.assertTrue(np.any(np.abs(data["bmg"]["K"][:4, -4:]) > 1.0e-14))
        self.assertEqual(data["structure"]["kind"], "imaginary_spectrum")
        self.assertTrue(
            np.allclose(data["structure"]["eigenvalues"].real, 0.0, atol=1.0e-10)
        )

    def test_bm_second_derivative_plot_data_exposes_maps_and_nonpositive_spectrum(
        self,
    ) -> None:
        data = experiment_d1.bm_second_derivative_plot_data(npts=32)

        self.assertEqual(data["npts"], 32)
        self.assertEqual(data["bmc"]["B"].shape, (32, 32))
        self.assertEqual(data["bmc"]["B''"].shape, (32, 32))
        self.assertEqual(data["bmg"]["M"].shape, (32, 32))
        self.assertEqual(data["bmg"]["S"].shape, (32, 32))
        self.assertEqual(data["structure"]["kind"], "nonpositive_spectrum")
        self.assertTrue(np.all(data["structure"]["eigenvalues"] <= 1.0e-10))
        self.assertTrue(np.any(np.abs(data["bmc"]["B''"][:4, -4:]) > 1.0e-14))
        self.assertTrue(np.any(np.abs(data["bmg"]["S"][:4, -4:]) > 1.0e-14))

    def test_plot_bm_banded_maps_writes_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    experiment_d1.plot_bm_banded_maps()

            output = outdir / "fig_bm_banded_maps.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_bm_second_derivative_structure_writes_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    experiment_d1.plot_bm_second_derivative_structure()

            output = outdir / "fig_bm_second_derivative_structure.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_same_degree_uses_overhead_legend_and_no_better_arrow(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 128)
        all_curves = build_all_curves(w)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with patch.object(experiment_d1, "add_better_arrow") as arrow:
                    with patch.object(experiment_d1, "_make_overhead_legend") as legend:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=DeprecationWarning)
                            experiment_d1.plot_same_degree(w, all_curves)

        self.assertFalse(arrow.called)
        legend.assert_called_once()
        self.assertEqual(legend.call_args.kwargs["ncol"], 3)
        self.assertEqual(legend.call_args.kwargs["fontsize"], 13.5)
        self.assertEqual(legend.call_args.kwargs["rect"], (0, 0, 1, 0.78))

    def test_plot_overlap_identity_writes_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    experiment_d1.plot_overlap_identity()

            output = outdir / "fig_overlap_identity.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_cost_tradeoff_writes_pdf(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 4000)
        detail_labels = ("E6", "C6", "BMC5 = BMG2")
        unique_curves = build_unique_curves(build_all_curves(w))
        metrics_map = {
            label: sa.summarize_curve(w, unique_curves[label])
            for label in detail_labels
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    experiment_d1.plot_cost_tradeoff({}, metrics_map, detail_labels)

            output = outdir / "fig_cost_tradeoff.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_second_derivative_dispersion_writes_pdf(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 512)
        unique_curves = build_unique_second_derivative_curves(build_all_second_derivative_curves(w))

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    plot_second_derivative_dispersion_landscape(w, unique_curves)

            output = outdir / "fig_second_derivative_dispersion.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_second_derivative_dispersion_uses_panel_legends(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 256)
        unique_curves = build_unique_second_derivative_curves(build_all_second_derivative_curves(w))
        legend_calls = []
        figure_legend_calls = []
        original_axes_legend = matplotlib.axes._axes.Axes.legend
        original_figure_legend = matplotlib.figure.Figure.legend

        def spy_axes_legend(ax, *args, **kwargs):
            legend_calls.append((kwargs.get("loc"), kwargs.get("ncol", 1)))
            return original_axes_legend(ax, *args, **kwargs)

        def spy_figure_legend(fig, *args, **kwargs):
            figure_legend_calls.append(kwargs.get("loc"))
            return original_figure_legend(fig, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with patch("matplotlib.axes._axes.Axes.legend", new=spy_axes_legend):
                    with patch("matplotlib.figure.Figure.legend", new=spy_figure_legend):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=DeprecationWarning)
                            plot_second_derivative_dispersion_landscape(w, unique_curves)

        self.assertEqual(figure_legend_calls, [])
        self.assertEqual(
            legend_calls,
            [("upper left", 1), ("upper left", 1), ("upper left", 1), ("upper left", 2)],
        )

    def test_plot_second_derivative_cost_tradeoff_writes_pdf(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 4000)
        detail_labels = ("E6", "C6", "BMC5 = BMG2")
        unique_curves = build_unique_second_derivative_curves(build_all_second_derivative_curves(w))
        metrics_map = {
            label: sa.summarize_second_derivative_curve(w, unique_curves[label])
            for label in detail_labels
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    plot_second_derivative_cost_tradeoff(metrics_map, detail_labels)

            output = outdir / "fig_second_derivative_cost_tradeoff.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_second_derivative_diffusion_writes_pdf(self) -> None:
        detail_labels = experiment_d1.DIRECT_SECOND_DERIVATIVE_TRIO_LABELS
        results = collect_gaussian_diffusion_convergence_results(
            npts_values=[48, 64, 96],
            labels=detail_labels,
            time_final=40.0,
            nu=2.0e-4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    plot_second_derivative_diffusion(results, detail_labels, time_final=40.0, nu=2.0e-4)

            output = outdir / "fig_second_derivative_diffusion.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_second_derivative_diffusion_uses_figure_legend(self) -> None:
        detail_labels = experiment_d1.DIRECT_SECOND_DERIVATIVE_TRIO_LABELS
        results = collect_gaussian_diffusion_convergence_results(
            npts_values=[48, 64, 96],
            labels=detail_labels,
            time_final=40.0,
            nu=2.0e-4,
        )
        axes_legend_calls = []
        figure_legend_calls = []
        original_axes_legend = matplotlib.axes._axes.Axes.legend
        original_figure_legend = matplotlib.figure.Figure.legend

        def spy_axes_legend(ax, *args, **kwargs):
            axes_legend_calls.append(kwargs.get("loc"))
            return original_axes_legend(ax, *args, **kwargs)

        def spy_figure_legend(fig, *args, **kwargs):
            figure_legend_calls.append((kwargs.get("loc"), kwargs.get("bbox_to_anchor")))
            return original_figure_legend(fig, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with patch("matplotlib.axes._axes.Axes.legend", new=spy_axes_legend):
                    with patch("matplotlib.figure.Figure.legend", new=spy_figure_legend):
                        plot_second_derivative_diffusion(results, detail_labels, time_final=40.0, nu=2.0e-4)

        self.assertEqual(axes_legend_calls, [])
        self.assertEqual(figure_legend_calls, [("upper center", (0.5, 0.985))])

    def test_diffusion_budget_ppw_requirement_grows_with_mission_length(self) -> None:
        ppw_grid = np.array([3.0, 4.0, 5.0, 6.0])
        relative_error = np.array([0.20, 0.08, 0.03, 0.01])
        missions = np.array([0.01, 0.03, 0.08])

        pmin = experiment_d1.minimum_ppw_curve_from_diffusion_budget(ppw_grid, relative_error, missions, tol=5.0e-2)

        self.assertTrue(np.all(np.diff(pmin) >= 0.0))
        self.assertTrue(np.all(np.isfinite(pmin)))

    def test_plot_second_derivative_mission_design_writes_pdf(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 2048)
        detail_labels = experiment_d1.DIRECT_SECOND_DERIVATIVE_TRIO_LABELS
        unique_curves = build_unique_second_derivative_curves(build_all_second_derivative_curves(w))

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    experiment_d1.plot_second_derivative_mission_design(w, unique_curves, detail_labels)

            output = outdir / "fig_second_derivative_mission_design.pdf"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_mission_design_uses_figure_legend(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 1024)
        detail_labels = experiment_d1.PDE_TRIO_LABELS
        unique_curves = build_unique_curves(build_all_curves(w))
        axes_legend_calls = []
        figure_legend_calls = []
        original_axes_legend = matplotlib.axes._axes.Axes.legend
        original_figure_legend = matplotlib.figure.Figure.legend

        def spy_axes_legend(ax, *args, **kwargs):
            axes_legend_calls.append(kwargs.get("loc"))
            return original_axes_legend(ax, *args, **kwargs)

        def spy_figure_legend(fig, *args, **kwargs):
            figure_legend_calls.append((kwargs.get("loc"), kwargs.get("bbox_to_anchor")))
            return original_figure_legend(fig, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with patch("matplotlib.axes._axes.Axes.legend", new=spy_axes_legend):
                    with patch("matplotlib.figure.Figure.legend", new=spy_figure_legend):
                        experiment_d1.plot_mission_design(w, unique_curves, detail_labels)

        self.assertEqual(axes_legend_calls, [])
        self.assertEqual(figure_legend_calls, [("upper center", (0.5, 0.995))])

    def test_plot_second_derivative_mission_design_uses_figure_legend(self) -> None:
        w = np.linspace(1.0e-12, np.pi, 1024)
        detail_labels = experiment_d1.DIRECT_SECOND_DERIVATIVE_TRIO_LABELS
        unique_curves = build_unique_second_derivative_curves(build_all_second_derivative_curves(w))
        axes_legend_calls = []
        figure_legend_calls = []
        original_axes_legend = matplotlib.axes._axes.Axes.legend
        original_figure_legend = matplotlib.figure.Figure.legend

        def spy_axes_legend(ax, *args, **kwargs):
            axes_legend_calls.append(kwargs.get("loc"))
            return original_axes_legend(ax, *args, **kwargs)

        def spy_figure_legend(fig, *args, **kwargs):
            figure_legend_calls.append((kwargs.get("loc"), kwargs.get("bbox_to_anchor")))
            return original_figure_legend(fig, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            with patch.object(experiment_d1, "FIGDIR", outdir):
                with patch("matplotlib.axes._axes.Axes.legend", new=spy_axes_legend):
                    with patch("matplotlib.figure.Figure.legend", new=spy_figure_legend):
                        experiment_d1.plot_second_derivative_mission_design(w, unique_curves, detail_labels)

        self.assertEqual(axes_legend_calls, [])
        self.assertEqual(figure_legend_calls, [("upper center", (0.5, 0.995))])

    def test_matched_order_blocks_follow_documented_phase_resolution_ordering(self) -> None:
        w = np.linspace(1e-12, np.pi, 4000)
        all_curves = build_all_curves(w)

        c4 = sa.summarize_curve(w, all_curves["C4"])
        e4 = sa.summarize_curve(w, all_curves["E4"])
        self.assertLess(c4.ppw_1pct, e4.ppw_1pct)
        self.assertGreater(e4.rk4_cfl, c4.rk4_cfl)
        self.assertTrue(np.allclose(all_curves["C4"], all_curves["BMC3"]))
        self.assertTrue(np.allclose(all_curves["C4"], all_curves["IGA1"]))

        q6_labels = ["E6", "C6", "BMC5 = BMG2"]
        q6_ppw = {label: sa.summarize_curve(w, all_curves[label if label != "BMC5 = BMG2" else "BMC5"]).ppw_1pct for label in q6_labels}
        q6_cfl = {label: sa.summarize_curve(w, all_curves[label if label != "BMC5 = BMG2" else "BMC5"]).rk4_cfl for label in q6_labels}
        self.assertLess(q6_ppw["BMC5 = BMG2"], q6_ppw["C6"])
        self.assertLess(q6_ppw["C6"], q6_ppw["E6"])
        self.assertGreater(q6_cfl["E6"], q6_cfl["C6"])
        self.assertGreater(q6_cfl["C6"], q6_cfl["BMC5 = BMG2"])

        q8_labels = ["E8", "C8", "BMC7 = BMG3"]
        q8_source = {"E8": "E8", "C8": "C8", "BMC7 = BMG3": "IGA3"}
        q8_ppw = {label: sa.summarize_curve(w, all_curves[q8_source[label]]).ppw_1pct for label in q8_labels}
        q8_cfl = {label: sa.summarize_curve(w, all_curves[q8_source[label]]).rk4_cfl for label in q8_labels}
        self.assertLess(q8_ppw["BMC7 = BMG3"], q8_ppw["C8"])
        self.assertLess(q8_ppw["C8"], q8_ppw["E8"])
        self.assertGreater(q8_cfl["E8"], q8_cfl["C8"])
        self.assertGreater(q8_cfl["C8"], q8_cfl["BMC7 = BMG3"])

    def test_collect_runtime_results_sorts_each_method_by_npts(self) -> None:
        metrics_map = {
            "E4": SimpleNamespace(rk4_cfl=0.1),
            "C6": SimpleNamespace(rk4_cfl=0.2),
        }

        def fake_measure(
            label: str,
            npts: int,
            *,
            time_final: float,
            cfl: float,
            repeats: int,
            length: float,
        ) -> MarchResult:
            return MarchResult(
                label,
                npts,
                error=0.0,
                dt=0.01,
                nsteps=1,
                wall_time=float(npts),
                setup_time=0.0,
            )

        with patch.object(experiment_d1, "UNIQUE_METHODS", ["E4", "C6"]):
            with patch.object(experiment_d1, "measure_marching_time", side_effect=fake_measure):
                grouped = collect_runtime_results(
                    metrics_map,
                    npts_values=[256, 64, 128],
                    repeats=1,
                )

        self.assertEqual([item.npts for item in grouped["E4"]], [64, 128, 256])
        self.assertEqual([item.npts for item in grouped["C6"]], [64, 128, 256])

    def test_collect_carrier_runtime_results_sorts_each_method_by_npts(self) -> None:
        regime = CarrierRegimeSpec(
            slug="candidate",
            title="Candidate regime",
            mode=68,
            sigma_env_scale=8.0,
            time_final=2.0,
            common_cfl=0.05,
            fairness_lens="common CFL",
        )

        def fake_measure(
            label: str,
            npts: int,
            *,
            mode: int,
            time_final: float,
            cfl: float,
            repeats: int,
            length: float,
            sigma_env: float,
        ) -> MarchResult:
            return MarchResult(
                label,
                npts,
                error=0.0,
                dt=0.01,
                nsteps=1,
                wall_time=float(npts),
                setup_time=0.0,
            )

        with patch.object(experiment_d1, "measure_carrier_marching_time", side_effect=fake_measure):
            grouped = collect_carrier_runtime_results(
                regime,
                npts_values=[384, 128, 256],
                labels=("E4", "C6"),
                repeats=1,
            )

        self.assertEqual([item.npts for item in grouped["E4"]], [128, 256, 384])
        self.assertEqual([item.npts for item in grouped["C6"]], [128, 256, 384])


if __name__ == "__main__":
    unittest.main()
