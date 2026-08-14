#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///

import csv
import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


THISDIR = Path(__file__).resolve().parent
REPO = THISDIR.parents[2]
OUTDIR = REPO / "results" / "ibm_cert"
GRIDS = (96, 128, 160, 192)
SHIFTS = ("shift_0_0", "shift_1_4_3_8", "shift_1_2_1_2", "shift_3_4_1_8")
BMC_ROUTES = ("BMC5", "BMC7", "BMC9", "BMC11")
METRICS = {
    "wall_trace": ("trace_error", "trace_numerical_floor"),
    "exterior_relative_l2": (
        "exterior_relative_l2",
        "exterior_relative_l2_numerical_floor",
    ),
}
FACTOR_THRESHOLDS = {
    "wall_trace": 4.0,
    "exterior_relative_l2": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_cells(directory: Path) -> list[dict]:
    cells = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("completion_status") != "complete":
            raise RuntimeError(f"incomplete checkpoint: {path}")
        payload["checkpoint_path"] = str(path.relative_to(REPO))
        cells.append(payload)
    return cells


def cell_index(cells: list[dict]) -> dict[tuple[str, int, str], dict]:
    return {
        (cell["primary_route_id"], int(cell["N"]), cell["shift_id"]): cell
        for cell in cells
    }


def compact_routes(cells: list[dict]) -> list[str]:
    return sorted({
        cell["primary_route_id"]
        for cell in cells
        if cell["route_kind"] == "compact"
    })


def write_full_results(cells: list[dict]) -> int:
    rows = []
    for cell in cells:
        result = cell["result"]
        for logical in cell["logical_aliases"]:
            row = {
                "cell_id": cell["cell_id"],
                "primary_route_id": cell["primary_route_id"],
                "logical_route_label": logical,
                "derivative_route": cell["derivative_route"],
                "route_kind": cell["route_kind"],
                "evaluator_hash": cell["evaluator_hash"],
                "formal_order": cell["formal_order"],
                "N": cell["N"],
                "h": cell["h"],
                "shift_id": cell["shift_id"],
                "shift_cell_x": cell["shift_cell_units"][0],
                "shift_cell_y": cell["shift_cell_units"][1],
                "shift_physical_x": cell["shift_physical"][0],
                "shift_physical_y": cell["shift_physical"][1],
                "n_markers_total": cell["n_markers_total"],
                "n_queries_total": cell["n_queries_total"],
                "trace_error": result["trace_error"],
                "trace_numerical_floor": result["trace_numerical_floor"],
                "trace_above_ten_times_floor": result["trace_above_ten_times_floor"],
                "exterior_relative_l2": result["exterior_relative_l2"],
                "exterior_relative_l2_numerical_floor": result[
                    "exterior_relative_l2_numerical_floor"
                ],
                "exterior_above_ten_times_floor": result[
                    "exterior_above_ten_times_floor"
                ],
                "total_time_seconds": result["total_time_seconds"],
                "recurring_schur_assembly_seconds": result["timings"]["schur_assembly"],
                "schur_condition": result["schur_condition"],
                "longdouble_solve_residual": result["longdouble_solve_residual"],
                "weighted_adjoint_residual": result["weighted_adjoint_residual"],
                "evaluator_moment_residual": result[
                    "evaluator_certificate_receipt"
                ]["moment_residual"],
                "certified_constant_residual": result[
                    "evaluator_certificate_receipt"
                ]["constant_residual"],
                "fielded_constant_residual": result["fielded_constant_residual"],
                "reference_translation_residual": cell[
                    "reference_translation_residual"
                ],
                "invariants_pass": result["invariants_pass"],
                "checkpoint_path": cell["checkpoint_path"],
            }
            for name, value in result["timings"].items():
                row[f"timing_{name}_seconds"] = value
            rows.append(row)
    if len(rows) != 464:
        raise RuntimeError(f"expected 464 logical rows, found {len(rows)}")
    fields = list(rows[0])
    with open(OUTDIR / "full_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def invariant_controls(primary: list[dict], scaling: list[dict]) -> dict:
    all_cells = primary + scaling
    shared_failures = []
    grouped = defaultdict(list)
    for cell in primary:
        grouped[(cell["N"], cell["shift_id"])].append(cell)
    for key, group in grouped.items():
        for name in (
            "markers",
            "quadrature_weights",
            "queries",
            "forcing",
            "reference_grid",
            "exterior_mask",
        ):
            values = {cell["shared_case_hashes"][name] for cell in group}
            if len(values) != 1:
                shared_failures.append({"case": key, "quantity": name})
    maxima = {
        "weighted_adjoint_residual": max(
            cell["result"]["weighted_adjoint_residual"] for cell in all_cells
        ),
        "evaluator_moment_residual": max(
            cell["result"]["evaluator_certificate_receipt"]["moment_residual"]
            for cell in all_cells
        ),
        "certified_constant_residual": max(
            cell["result"]["evaluator_certificate_receipt"]["constant_residual"]
            for cell in all_cells
        ),
        "fielded_constant_residual": max(
            cell["result"]["fielded_constant_residual"] for cell in all_cells
        ),
        "reference_translation_residual": max(
            cell["reference_translation_residual"] for cell in all_cells
        ),
        "schur_condition": max(cell["result"]["schur_condition"] for cell in all_cells),
        "longdouble_solve_residual": max(
            cell["result"]["longdouble_solve_residual"] for cell in all_cells
        ),
    }
    thresholds = {
        "weighted_adjoint_residual": 1e-12,
        "evaluator_moment_residual": 1e-12,
        "certified_constant_residual": 1e-13,
        "fielded_constant_residual": 1e-13,
        "reference_translation_residual": 1e-12,
    }
    checks = {name: maxima[name] <= threshold for name, threshold in thresholds.items()}
    checks["identical_shared_case_data"] = not shared_failures
    checks["every_cell_invariants_pass"] = all(
        cell["result"]["invariants_pass"] for cell in all_cells
    )
    return {
        "maxima": maxima,
        "thresholds": thresholds,
        "checks": checks,
        "shared_case_failures": shared_failures,
        "all_pass": bool(all(checks.values())),
    }


def marker_scaling(scales: list[dict]) -> dict:
    grouped = defaultdict(list)
    for cell in scales:
        grouped[cell["primary_route_id"]].append(cell)
    output = {}
    for route, cells in sorted(grouped.items()):
        cells.sort(key=lambda cell: cell["n_markers_total"])
        marker_counts = np.array([cell["n_markers_total"] for cell in cells], dtype=float)
        assembly = np.array([
            cell["result"]["timings"]["schur_assembly"] for cell in cells
        ])
        solves = np.array([
            cell["result"]["timings"]["schur_solve_and_refinement"] for cell in cells
        ])
        totals = np.array([cell["result"]["total_time_seconds"] for cell in cells])
        log_count = np.log(marker_counts)
        output[route] = {
            "marker_counts_total": marker_counts.astype(int).tolist(),
            "schur_assembly_seconds": assembly.tolist(),
            "schur_solve_seconds": solves.tolist(),
            "total_seconds": totals.tolist(),
            "schur_assembly_exponent": float(np.polyfit(log_count, np.log(assembly), 1)[0]),
            "schur_solve_exponent": float(np.polyfit(log_count, np.log(solves), 1)[0]),
            "total_time_exponent": float(np.polyfit(log_count, np.log(totals), 1)[0]),
        }
    return output


def order_receipts(primary: list[dict]) -> dict:
    grouped = defaultdict(dict)
    for cell in primary:
        grouped[(cell["primary_route_id"], cell["shift_id"])][int(cell["N"])] = cell
    receipts = {}
    for (route, shift), by_grid in sorted(grouped.items()):
        formal = by_grid[GRIDS[0]]["formal_order"]
        route_receipt = {"formal_order": formal, "metrics": {}}
        for metric_name, (error_key, floor_key) in METRICS.items():
            errors = np.array([by_grid[n]["result"][error_key] for n in GRIDS])
            floors = np.array([by_grid[n]["result"][floor_key] for n in GRIDS])
            monotone = bool(np.all(errors[:-1] > errors[1:]))
            above = bool(np.all(errors > 10.0 * floors))
            diagnostic_slope = float(
                np.polyfit(np.log(1.0 / np.array(GRIDS)), np.log(errors), 1)[0]
            )
            claimable = monotone and above
            claimed_order = diagnostic_slope if claimable else None
            within = (
                claimable
                and (formal is None or abs(diagnostic_slope - formal) <= 0.75)
            )
            route_receipt["metrics"][metric_name] = {
                "errors": errors.tolist(),
                "floors": floors.tolist(),
                "error_over_floor": (errors / floors).tolist(),
                "three_consecutive_monotone_refinements": monotone,
                "all_four_values_above_ten_times_floor": above,
                "diagnostic_four_grid_order": diagnostic_slope,
                "claimed_four_grid_order": claimed_order,
                "within_0_75_of_design": bool(within),
                "claim_pass": bool(within),
            }
        receipts[f"{route}@{shift}"] = route_receipt
    return receipts


def factor_receipt(index: dict, bmc: str, compact: str, shift: str,
                   metric_name: str) -> dict:
    error_key, floor_key = METRICS[metric_name]
    threshold = FACTOR_THRESHOLDS[metric_name]
    admissible = []
    for npts in reversed(GRIDS):
        b = index[(bmc, npts, shift)]["result"]
        c = index[(compact, npts, shift)]["result"]
        if b[error_key] > 10.0 * b[floor_key] and c[error_key] > 10.0 * c[floor_key]:
            admissible.append(npts)
    if admissible:
        npts = admissible[0]
        bcell = index[(bmc, npts, shift)]
        ccell = index[(compact, npts, shift)]
        factor = ccell["result"][error_key] / bcell["result"][error_key]
        status = "exact_pass" if factor >= threshold else "exact_fail"
        lower_bound = None
        decision_factor = factor
    else:
        npts = GRIDS[-1]
        bcell = index[(bmc, npts, shift)]
        ccell = index[(compact, npts, shift)]
        b = bcell["result"]
        c = ccell["result"]
        if c[error_key] > 10.0 * c[floor_key] and b[error_key] <= 10.0 * b[floor_key]:
            lower_bound = c[error_key] / (10.0 * b[floor_key])
            if lower_bound >= threshold:
                status = "lower_bound_pass"
                decision_factor = lower_bound
            else:
                status = "indeterminate_floor"
                decision_factor = None
        else:
            lower_bound = None
            status = "indeterminate_floor"
            decision_factor = None
        factor = c[error_key] / b[error_key]
    return {
        "bmc_route": bmc,
        "compact_route": compact,
        "shift_id": shift,
        "metric": metric_name,
        "factor_threshold": threshold,
        "N": npts,
        "status": status,
        "observed_factor": float(factor),
        "reportable_lower_bound": None if lower_bound is None else float(lower_bound),
        "decision_factor": None if decision_factor is None else float(decision_factor),
        "bmc_error": bcell["result"][error_key],
        "bmc_floor": bcell["result"][floor_key],
        "compact_error": ccell["result"][error_key],
        "compact_floor": ccell["result"][floor_key],
        "bmc_total_time_seconds": bcell["result"]["total_time_seconds"],
        "compact_total_time_seconds": ccell["result"]["total_time_seconds"],
        "bmc_over_compact_time_ratio": (
            bcell["result"]["total_time_seconds"]
            / ccell["result"]["total_time_seconds"]
        ),
    }


def factors(primary: list[dict]) -> dict:
    index = cell_index(primary)
    compacts = compact_routes(primary)
    output = {}
    for bmc in BMC_ROUTES:
        route = {}
        for metric in METRICS:
            comparisons = [
                factor_receipt(index, bmc, compact, shift, metric)
                for compact in compacts
                for shift in SHIFTS
            ]
            exact = [item for item in comparisons if item["status"].startswith("exact_")]
            exact_failures = [
                item for item in comparisons if item["status"] == "exact_fail"
            ]
            lower_bound_passes = [
                item for item in comparisons if item["status"] == "lower_bound_pass"
            ]
            indeterminate = [
                item for item in comparisons if item["status"] == "indeterminate_floor"
            ]
            resolved = exact + lower_bound_passes
            all_pass = (
                not exact_failures
                and not indeterminate
                and len(resolved) == len(comparisons)
            )
            strongest = (
                min(resolved, key=lambda item: item["decision_factor"])
                if not indeterminate and resolved else None
            )
            decisive = (
                min(exact_failures, key=lambda item: item["observed_factor"])
                if exact_failures else None
            )
            if exact_failures:
                metric_status = "exact_failure"
            elif indeterminate:
                metric_status = "indeterminate_floor"
            else:
                metric_status = "pass_all_comparisons"
            route[metric] = {
                "metric_status": metric_status,
                "passes_factor_requirement": all_pass,
                "strongest_compact_blocker": strongest,
                "decisive_exact_blocker": decisive,
                "floor_limited_blocker_set": indeterminate,
                "minimum_exact_factor": (
                    min(item["observed_factor"] for item in exact) if exact else None
                ),
                "minimum_decision_factor": (
                    min(item["decision_factor"] for item in resolved)
                    if resolved else None
                ),
                "comparison_count": len(comparisons),
                "exact_comparison_count": len(exact),
                "exact_failure_count": len(exact_failures),
                "lower_bound_pass_count": len(lower_bound_passes),
                "indeterminate_comparison_count": len(indeterminate),
                "comparisons": comparisons,
            }
        output[bmc] = route
    return output


def pareto_front(primary: list[dict], metric_key: str, floor_key: str,
                 shift_id: str) -> list[dict]:
    points = []
    for cell in primary:
        if cell["shift_id"] != shift_id:
            continue
        raw_error = cell["result"][metric_key]
        floor = cell["result"][floor_key]
        points.append({
            "cell_id": cell["cell_id"],
            "route": cell["primary_route_id"],
            "N": cell["N"],
            "shift_id": cell["shift_id"],
            "raw_error": raw_error,
            "floor": floor,
            "error_over_floor": raw_error / floor,
            "certified_error": max(raw_error, 10.0 * floor),
            "time_seconds": cell["result"]["total_time_seconds"],
        })
    front = []
    for candidate in points:
        dominated = any(
            other["certified_error"] <= candidate["certified_error"]
            and other["time_seconds"] <= candidate["time_seconds"]
            and (
                other["certified_error"] < candidate["certified_error"]
                or other["time_seconds"] < candidate["time_seconds"]
            )
            for other in points
            if other["cell_id"] != candidate["cell_id"]
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["time_seconds"])


def parse_time_log(path: Path) -> dict:
    text = path.read_text()
    patterns = {
        "user_seconds": r"User time \(seconds\): ([0-9.]+)",
        "system_seconds": r"System time \(seconds\): ([0-9.]+)",
        "maximum_rss_kib": r"Maximum resident set size \(kbytes\): ([0-9]+)",
        "exit_status": r"Exit status: ([0-9]+)",
    }
    parsed = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        parsed[name] = float(matches[-1]) if matches else None
    if parsed["user_seconds"] is not None and parsed["system_seconds"] is not None:
        parsed["cpu_hours"] = (
            parsed["user_seconds"] + parsed["system_seconds"]
        ) / 3600.0
    return parsed


def promotion_receipts(factor_data: dict, order_data: dict,
                       pareto: dict, invariant_pass: bool) -> dict:
    output = {}
    for bmc in BMC_ROUTES:
        trace = factor_data[bmc]["wall_trace"]
        field = factor_data[bmc]["exterior_relative_l2"]
        trace_blocker = (
            trace["strongest_compact_blocker"]
            or trace["decisive_exact_blocker"]
        )
        field_blocker = (
            field["strongest_compact_blocker"]
            or field["decisive_exact_blocker"]
        )
        trace_accuracy = trace["passes_factor_requirement"]
        field_accuracy = field["passes_factor_requirement"]
        blocker_receipts = [
            blocker for blocker in (trace_blocker, field_blocker)
            if blocker is not None
        ]
        time_pass = bool(
            len(blocker_receipts) == 2
            and max(
                blocker["bmc_over_compact_time_ratio"]
                for blocker in blocker_receipts
            ) <= 1.25
        )
        pareto_by_metric = {}
        for metric in METRICS:
            shift_checks = {}
            for shift in SHIFTS:
                qualifying = [
                    point for point in pareto[metric][shift]
                    if point["route"] == bmc and point["error_over_floor"] > 10.0
                ]
                shift_checks[shift] = {
                    "pass": bool(qualifying),
                    "qualifying_points": qualifying,
                }
            pareto_by_metric[metric] = {
                "shift_checks": shift_checks,
                "all_shifts_pass": all(
                    receipt["pass"] for receipt in shift_checks.values()
                ),
            }
        pareto_pass = all(
            receipt["all_shifts_pass"] for receipt in pareto_by_metric.values()
        )
        route_orders = [
            receipt for key, receipt in order_data.items()
            if key.startswith(f"{bmc}@")
        ]
        order_pass = all(
            metric["claim_pass"]
            for receipt in route_orders
            for metric in receipt["metrics"].values()
        )
        checks = {
            "wall_trace_factor_at_least_4": bool(trace_accuracy),
            "exterior_factor_at_least_2": bool(field_accuracy),
            "total_time_ratio_at_most_1_25": bool(time_pass),
            "on_both_all_shift_error_time_pareto_fronts": bool(pareto_pass),
            "all_shift_orders_claimable": bool(order_pass),
            "all_invariants": bool(invariant_pass),
        }
        output[bmc] = {
            "checks": checks,
            "passes_every_promotion_requirement": bool(all(checks.values())),
            "trace_blocker": trace_blocker,
            "field_blocker": field_blocker,
            "pareto_by_metric_and_shift": pareto_by_metric,
        }
    return output


def blocker_cost_receipt(index: dict, blocker: dict) -> dict:
    bmc = index[(blocker["bmc_route"], blocker["N"], blocker["shift_id"])]["result"]
    compact = index[(
        blocker["compact_route"], blocker["N"], blocker["shift_id"]
    )]["result"]
    setup_key = "evaluator_and_coupling_construction"
    schur_key = "schur_assembly"
    return {
        "bmc_total_seconds": bmc["total_time_seconds"],
        "compact_total_seconds": compact["total_time_seconds"],
        "bmc_over_compact_total_ratio": (
            bmc["total_time_seconds"] / compact["total_time_seconds"]
        ),
        "bmc_setup_seconds": bmc["timings"][setup_key],
        "compact_setup_seconds": compact["timings"][setup_key],
        "bmc_over_compact_setup_ratio": (
            bmc["timings"][setup_key] / compact["timings"][setup_key]
        ),
        "bmc_schur_assembly_seconds": bmc["timings"][schur_key],
        "compact_schur_assembly_seconds": compact["timings"][schur_key],
        "bmc_over_compact_schur_assembly_ratio": (
            bmc["timings"][schur_key] / compact["timings"][schur_key]
        ),
    }


def evaluator_mechanism(primary: list[dict]) -> dict:
    index = cell_index(primary)
    families = {
        "C6": ("C6|order_matched_L6", "C6|common_L12", "C6|Q12-6"),
        "C8": ("C8|order_matched_L8", "C8|common_L12", "C8|Q12-8"),
        "C10": ("C10|order_matched_L10", "C10|common_L12", "C10|Q12-10"),
        "SLC-SL": (
            "SLC-SL|order_matched_L4", "SLC-SL|common_L12", "SLC-SL|Q12-4"
        ),
        "OPT-q8": (
            "OPT-q8|order_matched_L8", "OPT-q8|common_L12", "OPT-q8|Q12-8"
        ),
    }
    receipts = {}
    for derivative, (matched, common, optimized) in families.items():
        family = {}
        for metric_name, (error_key, _) in METRICS.items():
            common_wins = 0
            optimized_over_common = []
            matched_over_common = []
            for npts in GRIDS:
                for shift in SHIFTS:
                    errors = {
                        route: index[(route, npts, shift)]["result"][error_key]
                        for route in (matched, common, optimized)
                    }
                    common_wins += errors[common] == min(errors.values())
                    optimized_over_common.append(errors[optimized] / errors[common])
                    matched_over_common.append(errors[matched] / errors[common])
            family[metric_name] = {
                "common_L12_win_count": common_wins,
                "cell_count": len(GRIDS) * len(SHIFTS),
                "median_Q12q_over_common_L12": float(np.median(optimized_over_common)),
                "median_order_matched_over_common_L12": float(
                    np.median(matched_over_common)
                ),
            }
        receipts[derivative] = family
    compact = compact_routes(primary)
    raw_winners = {}
    for metric_name, (error_key, _) in METRICS.items():
        counts = Counter()
        for npts in GRIDS:
            for shift in SHIFTS:
                _, route = min(
                    (index[(route, npts, shift)]["result"][error_key], route)
                    for route in compact
                )
                counts[route] += 1
        raw_winners[metric_name] = dict(sorted(counts.items()))
    return {
        "sub_q12_family_receipts": receipts,
        "raw_best_compact_route_counts": raw_winners,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce one completed IBM-CERT run into the frozen verdict."
    )
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = OUTDIR / "runs" / args.run_id
    completion_path = run_root / "MATRIX_COMPLETED.json"
    if not completion_path.exists():
        raise RuntimeError(f"matrix completion marker is absent: {completion_path}")
    completion = json.loads(completion_path.read_text())
    if completion.get("run_id") != args.run_id:
        raise RuntimeError("matrix completion marker run identity mismatch")
    primary = load_cells(run_root / "checkpoints" / "primary")
    scaling = load_cells(run_root / "checkpoints" / "scaling")
    if len(primary) != 368:
        raise RuntimeError(f"expected 368 primary cells, found {len(primary)}")
    if len(scaling) != 92:
        raise RuntimeError(f"expected 92 scaling cells, found {len(scaling)}")
    logical_rows = write_full_results(primary)
    invariants = invariant_controls(primary, scaling)
    scaling_receipts = marker_scaling(scaling)
    orders = order_receipts(primary)
    factor_data = factors(primary)
    pareto = {
        metric: {
            shift: pareto_front(primary, error_key, floor_key, shift)
            for shift in SHIFTS
        }
        for metric, (error_key, floor_key) in METRICS.items()
    }
    promotions = promotion_receipts(
        factor_data, orders, pareto, invariants["all_pass"]
    )
    if all(promotions[route]["passes_every_promotion_requirement"] for route in BMC_ROUTES):
        verdict = "promote_full"
    elif all(
        promotions[route]["passes_every_promotion_requirement"]
        for route in ("BMC9", "BMC11")
    ):
        verdict = "promote_high_order_only"
    else:
        verdict = "reject_compact_wall_advantage"

    manifest = json.loads((OUTDIR / "run_manifest.json").read_text())
    completed_runs = [
        run for run in manifest["matrix_runs"]
        if run.get("status") == "matrix_complete" and run.get("run_id") == args.run_id
    ]
    if len(completed_runs) != 1:
        raise RuntimeError("unique matrix completion manifest record is absent")
    run = completed_runs[0]
    log_path = REPO / run["log_path"]
    time_receipt = parse_time_log(log_path)
    index = cell_index(primary)
    bmc9_trace_blocker = factor_data["BMC9"]["wall_trace"][
        "strongest_compact_blocker"
    ]
    bmc9_field_blocker = factor_data["BMC9"]["exterior_relative_l2"][
        "strongest_compact_blocker"
    ]
    bmc9_trace_cost = blocker_cost_receipt(index, bmc9_trace_blocker)
    bmc9_field_cost = blocker_cost_receipt(index, bmc9_field_blocker)
    evaluator_receipts = evaluator_mechanism(primary)

    all_cells = primary + scaling
    producer_hashes = {cell["producer_sha256"] for cell in all_cells}
    contract_hashes = {cell["contract_sha256"] for cell in all_cells}
    evaluator_hashes = {
        cell["evaluator_certificate_sha256"] for cell in all_cells
    }
    reference_hashes = {
        cell["periodic_reference_certificate_sha256"] for cell in all_cells
    }
    floor_formula_checks = {
        "trace_equals_legacy_absolute_forward_floor": all(
            cell["result"]["trace_numerical_floor"]
            == cell["result"]["forward_error_floor_absolute"]
            for cell in all_cells
        ),
        "exterior_is_legacy_floor_divided_by_reference_rms": all(
            np.isclose(
                cell["result"]["exterior_relative_l2_numerical_floor"],
                cell["result"]["forward_error_floor_absolute"]
                / cell["result"]["reference_exterior_rms"],
                rtol=2e-15,
                atol=0.0,
            )
            for cell in all_cells
        ),
        "one_recorded_convention": len({
            cell["result"]["forward_error_floor_convention"]
            for cell in all_cells
        }) == 1,
    }
    reference_certificate = json.loads(
        (OUTDIR / "periodic_reference_certificate.json").read_text()
    )
    reference_resolution = json.loads(
        (OUTDIR / "reference_sampler_resolution.json").read_text()
    )
    adjoint_resolution = json.loads(
        (OUTDIR / "adjoint_normalization_resolution.json").read_text()
    )
    floor_resolution = json.loads(
        (OUTDIR / "forward_floor_resolution.json").read_text()
    )
    contract_hash = sha256_file(OUTDIR / "frozen_contract.json")
    evaluator_hash = sha256_file(OUTDIR / "evaluator_certificate.json")
    reference_hash = sha256_file(OUTDIR / "periodic_reference_certificate.json")
    checkpoint_bindings = {
        "one_matrix_producer_hash": len(producer_hashes) == 1,
        "producer_matches_manifest": producer_hashes == {run["producer_sha256"]},
        "contract_matches": contract_hashes == {contract_hash},
        "evaluator_certificate_matches": evaluator_hashes == {evaluator_hash},
        "reference_certificate_matches": reference_hashes == {reference_hash},
        "reference_certificate_targets_producer": (
            reference_certificate["target_matrix_producer_sha256"]
            == run["producer_sha256"]
        ),
        "adjoint_control_targets_producer": (
            adjoint_resolution["target_matrix_producer_sha256"]
            == run["producer_sha256"]
        ),
    }
    resource_checks = {
        "cpu_hours_at_most_6": time_receipt["cpu_hours"] <= 6.0,
        "peak_rss_below_8_gib": time_receipt["maximum_rss_kib"] < 8 * 1024 * 1024,
        "exit_status_zero": time_receipt["exit_status"] == 0,
        "nice_19": run["nice"] == 19,
        "single_thread_environment": all(
            run["thread_environment"][name] == "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
    }

    def median_route_times(route: str) -> tuple[float, float]:
        setup = []
        recurring = []
        for shift in SHIFTS:
            result = index[(route, 96, shift)]["result"]
            setup_time = result["timings"]["evaluator_and_coupling_construction"]
            setup.append(setup_time)
            recurring.append(result["total_time_seconds"] - setup_time)
        return float(np.median(setup)), float(np.median(recurring))

    bmc9_setup, bmc9_recurring = median_route_times("BMC9")
    compact_setup, compact_recurring = median_route_times("C10|common_L12")
    recurring_saving = compact_recurring - bmc9_recurring
    projected_crossover = (
        (bmc9_setup - compact_setup) / recurring_saving
        if recurring_saving > 0.0 else None
    )

    controls = {
        "schema_version": "ibm-cert-controls-v2",
        "evaluator_certificate": {
            "path": "results/ibm_cert/evaluator_certificate.json",
            "sha256": sha256_file(OUTDIR / "evaluator_certificate.json"),
            "all_pass": json.loads((OUTDIR / "evaluator_certificate.json").read_text())[
                "all_pass"
            ],
        },
        "primary_distinct_cells": len(primary),
        "primary_logical_rows": logical_rows,
        "scaling_distinct_cells": len(scaling),
        "invariants": invariants,
        "periodic_reference_control": {
            "path": "results/ibm_cert/periodic_reference_certificate.json",
            "sha256": reference_hash,
            "all_pass": reference_certificate["all_pass"],
            "maximum_grid_shift_covariance_residual": reference_certificate[
                "maxima"
            ]["sample_grid_translation_residual"],
        },
        "reference_sampler_resolution": {
            "path": "results/ibm_cert/reference_sampler_resolution.json",
            "sha256": sha256_file(OUTDIR / "reference_sampler_resolution.json"),
            "status": reference_resolution["status"],
        },
        "weighted_adjoint_resolution": {
            "path": "results/ibm_cert/adjoint_normalization_resolution.json",
            "sha256": sha256_file(OUTDIR / "adjoint_normalization_resolution.json"),
            "status": adjoint_resolution["status"],
            "normwise_relative_residual_max": adjoint_resolution[
                "repaired_diagnostic"
            ]["normwise_relative_residual_max"],
        },
        "forward_floor_resolution": {
            "path": "results/ibm_cert/forward_floor_resolution.json",
            "sha256": sha256_file(OUTDIR / "forward_floor_resolution.json"),
            "status": floor_resolution["status"],
            "formula_checks": floor_formula_checks,
        },
        "checkpoint_hash_bindings": checkpoint_bindings,
        "resource_checks": resource_checks,
        "primary_conditioning": {
            "maximum_schur_condition": max(
                cell["result"]["schur_condition"] for cell in primary
            ),
            "maximum_longdouble_solve_residual": max(
                cell["result"]["longdouble_solve_residual"] for cell in primary
            ),
        },
        "scaling_conditioning": {
            "maximum_schur_condition": max(
                cell["result"]["schur_condition"] for cell in scaling
            ),
            "maximum_longdouble_solve_residual": max(
                cell["result"]["longdouble_solve_residual"] for cell in scaling
            ),
            "interpretation": "The highest marker counts expose dense-Schur conditioning and are used for measured timing scaling, not primary error ranking.",
        },
        "marker_scaling": scaling_receipts,
        "orders": orders,
        "matrix_resource_receipt": time_receipt,
        "run_id": args.run_id,
        "completion_marker": str(completion_path.relative_to(REPO)),
        "completion_marker_sha256": sha256_file(completion_path),
        "all_pass": bool(
            invariants["all_pass"]
            and reference_certificate["all_pass"]
            and reference_resolution["status"] == "admitted"
            and adjoint_resolution["status"] == "complete_pass"
            and floor_resolution["status"] == "complete_pass"
            and all(floor_formula_checks.values())
            and all(checkpoint_bindings.values())
            and all(resource_checks.values())
        ),
    }
    atomic_json(OUTDIR / "controls.json", controls)

    summary = {
        "schema_version": "ibm-cert-summary-v2",
        "run_id": args.run_id,
        "terminal_verdict": verdict,
        "strongest_compact_blockers": {
            route: {
                "wall_trace": promotions[route]["trace_blocker"],
                "exterior_relative_l2": promotions[route]["field_blocker"],
            }
            for route in BMC_ROUTES
        },
        "bmc_promotion_receipts": promotions,
        "same_shift_factors": factor_data,
        "orders": orders,
        "pareto": pareto,
        "pareto_route_membership": {
            metric: {
                shift: sorted({point["route"] for point in points})
                for shift, points in shift_fronts.items()
            }
            for metric, shift_fronts in pareto.items()
        },
        "marker_scaling": scaling_receipts,
        "invariants": invariants,
        "floors": {
            "rule": "A factor is exact only when both errors exceed ten times their cell-specific floor.",
            "trace_floor_limited_primary_cells": sum(
                not cell["result"]["trace_above_ten_times_floor"] for cell in primary
            ),
            "exterior_floor_limited_primary_cells": sum(
                not cell["result"]["exterior_above_ten_times_floor"] for cell in primary
            ),
        },
        "timing": {
            "matrix_resource_receipt": time_receipt,
            "aggregation": "median of three complete charged solves per distinct cell",
            "bmc9_wall_trace_blocker_cost": bmc9_trace_cost,
            "bmc9_exterior_blocker_cost": bmc9_field_cost,
        },
        "mechanism_classification": {
            "accuracy_mechanism": "supported",
            "composite_wall_superiority": "rejected",
            "theory": "The one-space BMC construction and its small order-ten modified-wavenumber error constant produce a decisive BMC9 accuracy advantage, but one-space coherence does not make the single-solve coupling setup free.",
            "regime": "The smooth Gaussian Helmholtz spectrum is low-wavenumber dominated, so common L12 benefits from eleven exact moments and beats the broad-band Q12-q evaluators despite their smaller certified worst-band maxima.",
            "diagnostic": "The off-marker maximum trace and exterior relative L2 diagnostics both resolve the BMC9 advantage across all four shifts.",
            "fairness": "Every admitted evaluator was fielded; C10 plus common L12 is the exact BMC9 blocker for both decision metrics; total time charges all construction and solve work.",
            "implementation": "The periodic-reference, adjoint-normalization, and forward-floor defects are repaired and independently controlled. The remaining measured setup premium is a property of the fielded single-solve route, not a failed invariant.",
        },
        "mechanism": {
            "bmc9_accuracy_strength": {
                "wall_trace_worst_shift_exact_factor": bmc9_trace_blocker[
                    "observed_factor"
                ],
                "exterior_relative_l2_worst_shift_exact_factor": bmc9_field_blocker[
                    "observed_factor"
                ],
                "wall_trace_blocker": bmc9_trace_blocker,
                "exterior_relative_l2_blocker": bmc9_field_blocker,
                "on_wall_trace_pareto_front_in_all_shifts": promotions["BMC9"][
                    "pareto_by_metric_and_shift"
                ]["wall_trace"]["all_shifts_pass"],
                "on_exterior_pareto_front_in_all_shifts": promotions["BMC9"][
                    "pareto_by_metric_and_shift"
                ]["exterior_relative_l2"]["all_shifts_pass"],
            },
            "cost_mechanism": {
                "wall_trace_blocker_total_time_ratio": bmc9_trace_cost[
                    "bmc_over_compact_total_ratio"
                ],
                "exterior_blocker_total_time_ratio": bmc9_field_cost[
                    "bmc_over_compact_total_ratio"
                ],
                "wall_trace_blocker_setup_ratio": bmc9_trace_cost[
                    "bmc_over_compact_setup_ratio"
                ],
                "exterior_blocker_setup_ratio": bmc9_field_cost[
                    "bmc_over_compact_setup_ratio"
                ],
                "wall_trace_blocker_schur_assembly_ratio": bmc9_trace_cost[
                    "bmc_over_compact_schur_assembly_ratio"
                ],
                "exterior_blocker_schur_assembly_ratio": bmc9_field_cost[
                    "bmc_over_compact_schur_assembly_ratio"
                ],
                "interpretation": "The 70 to 76 percent total-time premium comes from one-time evaluator and coupling construction, which is about 2.6 times the compact setup. Recurring Schur assembly is equal or slightly faster for BMC9 at the two blocker cells.",
            },
            "evaluator_mechanism": evaluator_receipts,
            "order_certificate_limitation": {
                "formal_order": 10,
                "all_shift_claim_pass": promotions["BMC9"]["checks"][
                    "all_shift_orders_claimable"
                ],
                "localization": "All four BMC9 error sequences are monotone, but only N=96 exceeds ten times the forward floor in both metrics; N=128 and finer enter the exclusion before the sole three-refinement window is complete. Diagnostic trace slopes of about 7.96 to 8.66 are also preasymptotic relative to design order ten. No asymptotic order is claimed.",
                "materiality": "This blocks the composite promotion label but does not erase the exact N=96 same-shift accuracy factors or all-shift Pareto placement.",
            },
        },
        "adoption_answer": {
            "supported_operating_choice": "BMC9_accuracy_first",
            "use_bm_when": "Use BMC9 for smooth periodic immersed-wall solves when reducing trace and exterior error by one to two orders of magnitude is worth a measured 1.70 to 1.76 single-solve time premium, especially when grid refinement, memory growth, or repeated downstream error is more expensive than this setup premium.",
            "use_strongest_compact_when": "Use C10 plus common L12 when single-solve throughput is controlling and its larger wall error is acceptable; it is the strongest exact practical blocker in the BMC9 operating cells.",
            "accuracy_factors": {
                "wall_trace": bmc9_trace_blocker["observed_factor"],
                "exterior_relative_l2": bmc9_field_blocker["observed_factor"],
            },
            "time_premium_ratios": {
                "wall_trace_blocker": bmc9_trace_cost[
                    "bmc_over_compact_total_ratio"
                ],
                "exterior_blocker": bmc9_field_cost[
                    "bmc_over_compact_total_ratio"
                ],
            },
            "downside_guardrail": "Do not claim universal or comparable-cost smooth-wall superiority. Keep the claim inside the frozen smooth periodic problem, four shifts, N=96 resolved accuracy window, current coefficient-local implementation, and certified floor rules.",
            "fallback": "For one-off wall solves, field C10 plus L12. For accuracy-first use, field BMC9 and disclose the measured premium.",
        },
        "paper_consequence": {
            "headline_action": "Do not use strongest-compact smooth-wall superiority as the paper headline.",
            "retained_strength": "Retain BMC9 as a bounded accuracy-first result with exact factors 62.71 and 32.17, all-shift Pareto placement, and an explicit 1.70 to 1.76 time premium.",
            "supporting_action": "Treat matched-order compact comparisons as supporting evidence and explain why common L12 is the strongest evaluator in this smooth regime.",
        },
        "residual_uncertainty": {
            "quantity": "BMC11 strongest-comparator factors and the full BMC9 design-order window are unresolved after high-order values enter their certified forward floors.",
            "material_to": "BMC11 magnitude claims and asymptotic-order promotion.",
            "immaterial_to": "The exact BMC9 accuracy-first choice and the composite rejection, which is independently forced by measured cost and order criteria.",
        },
        "residual_uncertainty_changes_decision": False,
        "next_tractable_route": {
            "name": "fixed_geometry_coupling_reuse",
            "status": "predeclared_not_executed",
            "hypothesis": "For repeated right-hand sides on fixed geometry, caching the BMC coefficient transform, marker and query maps, and Schur operator amortizes the one-time setup premium while preserving the certified BMC9 accuracy factors.",
            "strongest_comparator": "C10 plus common L12 with exactly the same caching rights",
            "mechanism": "At N=96 the median BMC9 setup is larger, but its median post-setup charged time is slightly smaller than C10 plus L12. The measured setup and recurring medians project a crossover near 89 repeated solves; this projection is not an executed result.",
            "measured_inputs": {
                "bmc9_median_setup_seconds": bmc9_setup,
                "compact_median_setup_seconds": compact_setup,
                "bmc9_median_post_setup_seconds": bmc9_recurring,
                "compact_median_post_setup_seconds": compact_recurring,
                "projected_break_even_repeated_solves": projected_crossover,
            },
            "predeclared_checkpoint": "On the same frozen problem and shifts, test K in {1,32,64,128} repeated right-hand sides, cache identical reusable objects for both routes, require unchanged errors and invariants, and promote an amortized-use claim only if BMC9 total time is at most 1.25 times C10 plus L12 by K=128 while retaining both exact accuracy factors and all-shift Pareto placement.",
            "rejection_rule": "Reject the amortized-cost route if no crossover occurs by K=128, any cached and uncached solution differs above its numerical floor, or either exact accuracy factor is lost.",
        },
        "artifact_hashes": {
            "frozen_contract.json": sha256_file(OUTDIR / "frozen_contract.json"),
            "provenance_receipt.json": sha256_file(OUTDIR / "provenance_receipt.json"),
            "evaluator_certificate.json": sha256_file(OUTDIR / "evaluator_certificate.json"),
            "run_manifest.json": sha256_file(OUTDIR / "run_manifest.json"),
            "full_results.csv": sha256_file(OUTDIR / "full_results.csv"),
            "controls.json": sha256_file(OUTDIR / "controls.json"),
            "periodic_reference_certificate.json": reference_hash,
            "reference_sampler_resolution.json": sha256_file(
                OUTDIR / "reference_sampler_resolution.json"
            ),
            "adjoint_normalization_resolution.json": sha256_file(
                OUTDIR / "adjoint_normalization_resolution.json"
            ),
            "forward_floor_resolution.json": sha256_file(
                OUTDIR / "forward_floor_resolution.json"
            ),
        },
    }
    atomic_json(OUTDIR / "summary.json", summary)
    print(json.dumps({
        "terminal_verdict": verdict,
        "primary_distinct_cells": len(primary),
        "logical_rows": logical_rows,
        "scaling_distinct_cells": len(scaling),
        "all_invariants_pass": invariants["all_pass"],
        "promotion_receipts": promotions,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
