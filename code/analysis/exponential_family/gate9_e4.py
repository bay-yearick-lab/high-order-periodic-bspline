"""Gate 9 section E4: coupled off-grid reads for the two fielded members.

The producer of the measurement is gate5a_offgrid.py, imported unchanged. It
already carries the five original routes, the amendment routes at the Lagrange
widths 8, 12 and 16, the anchor packet on N in {64, 128, 256} with 4,096 stored
query positions, and the resolution-scaled band-edge packet on
N in {48, 64, 96, 128, 192, 256}. This module adds two routes built on the
general evaluator of the Gate 8 members,

  pert5   X8_n10_C_SLC1_pert5, X_4(i 0.3628 pi; quad) x P_3, C^2,
  core6   X9_m6_C_SLC1_SLC1992_6IIq2, X_6(i 0.5870 pi, i 0.7757 pi) x P_2, C^4,

and runs the same three measurements on them beside the stored routes:

  * the symbol-level e_c(theta) and the coherence residuals C_1 and C_2,
  * the realized anchor-packet channel errors at N = 64, 128 and 256 with the
    fitted orders,
  * the resolution-scaled band-edge packet against champion, BMC9, BMC11 and
    the SLC augmentation at the widths 8, 12 and 16.

A member route is one field: the coefficients come from one circulant solve
with the member's own sample symbol and the value, the first derivative and the
literal second derivative are read from that one coefficient vector, so the
coherence residuals vanish to rounding by construction. The route class
therefore inherits every method of gate5a_offgrid.SplineRoute and replaces only
the pair (evaluator, generator) with the general evaluator and the Gate 8
polynomial generator of the member record.

Stages. `--stage predict` writes e4_prediction.json, which carries the symbol
level of every route, the contract's stated symbol numbers for the two members,
and the modal prediction of every realized cell. `--stage realize` refuses to
run unless that file exists, and writes e4_offgrid.json, e4_offgrid.md and
e4_runs.csv. The stored champion rows of results/exponential_family/gate5/5a/
runs.csv are recomputed through the imported code and compared bit for bit as
the regression.

Decision, from the Gate 9 contract section E4: PASS if the realized errors lie
within a factor of two of the symbol prediction at every N and below SLC plus
L12 on every channel at every N, PARTIAL if the factor lies between two and
four, REJECT otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate5a_offgrid as g5a  # noqa: E402
import gate8_family as g8  # noqa: E402
import metrics as mt  # noqa: E402
from evaluator import as_evaluator  # noqa: E402

GATE8_MEMBERS = ROOT / "results" / "exponential_family" / "gate8" / "members"
GATE9 = ROOT / "results" / "exponential_family" / "gate9"
GATE5A = ROOT / "results" / "exponential_family" / "gate5" / "5a"

MEMBERS = {
    "pert5": GATE8_MEMBERS / "X8_n10_C_SLC1_pert5.json",
    "core6": GATE8_MEMBERS / "X9_m6_C_SLC1_SLC1992_6IIq2.json",
}
MEMBER_NAMES = ("pert5", "core6")

# The contract's stated symbol numbers, from the verified metrics of the Gate 8
# members. Section E4 for pert5 and Amendment 1 for the core-6 member.
CONTRACT_SYMBOL = {
    "pert5": {"ec_max_B0": 5.6e-4, "reach_ec_1e-2_over_pi": 0.856,
              "reach_ec_1e-3_over_pi": None},
    "core6": {"ec_max_B0": 7.98e-4, "reach_ec_1e-2_over_pi": 0.858,
              "reach_ec_1e-3_over_pi": 0.824},
}
CONTRACT_SLC_L12 = {"ec_max_B0": 6.9e-2, "reach_ec_1e-2_over_pi": 0.540}

BASELINE = "slc_operator_L12"
CHANNELS = g5a.CHANNEL_NAMES
PASS_FACTOR = 2.0
PARTIAL_FACTOR = 4.0
REACH_TOLS = (1e-2, 1e-3, 1e-4)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# the member route
# ----------------------------------------------------------------------

class MemberRoute(g5a.SplineRoute):
    """One Gate 8 member read as a single field through the general evaluator."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = Path(path)
        self.record = json.loads(self.path.read_text())
        self.generator = g8.from_record(self.record)
        self.evaluator = as_evaluator(self.generator)
        self.taps_per_query = int(self.evaluator.n)
        self.solves_per_field = 1
        self.fields = 1
        self.kind = "one field"
        self._samples = {}

    def description(self) -> str:
        ev = self.evaluator
        return (f"{self.record['name']}, {ev.label()}, degree {ev.p}, support "
                f"[-{ev.n / 2:g}, {ev.n / 2:g}], class C^{ev.smoothness}, "
                f"cond {float(self.record['cond']):.3f}")

    def member_block(self) -> dict:
        ev = self.evaluator
        return {"route": self.name, "member": self.record["name"],
                "file": str(self.path.relative_to(ROOT)),
                "m": int(self.record["m"]), "omegas": list(self.record["omegas"]),
                "real_pairs": list(self.record["real_pairs"]),
                "quads": [list(q) for q in self.record["quads"]],
                "polynomial": list(self.record["polynomial"]),
                "n": int(ev.n), "p": int(ev.p),
                "smoothness": int(ev.smoothness),
                "cond": float(self.record["cond"]),
                "taps_per_query": int(self.taps_per_query)}


def build_member_routes():
    return [MemberRoute(name, MEMBERS[name]) for name in MEMBER_NAMES]


# ----------------------------------------------------------------------
# symbol level
# ----------------------------------------------------------------------

def symbol_rows(routes):
    """Return the symbol row and the extra reaches of every route."""
    rows = []
    curves = {}
    for route in routes:
        row, theta, ec = g5a.symbol_level(route)
        for tol in REACH_TOLS:
            got = mt.reach(theta, ec, tol)
            tag = f"{tol:.0e}".replace("e-0", "e-")
            row[f"reach_ec_{tag}_over_pi"] = got["linear"] / np.pi
            row[f"reach_ec_{tag}_saturated"] = bool(got["saturated"])
        rows.append(row)
        curves[route.name] = (theta, ec)
    return rows, curves


# ----------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------

def prediction_payload(all_routes, member_routes, x, scaled_modes) -> dict:
    rows, _ = symbol_rows(all_routes)
    index = {row["route"]: row for row in rows}
    stated = {}
    for name in MEMBER_NAMES:
        want = CONTRACT_SYMBOL[name]
        got = index[name]
        stated[name] = {
            "contract_ec_max_B0": want["ec_max_B0"],
            "computed_ec_max_B0": got["ec_max_B0"],
            "ratio_computed_over_contract": got["ec_max_B0"] / want["ec_max_B0"],
            "contract_reach_ec_1e-2_over_pi": want["reach_ec_1e-2_over_pi"],
            "computed_reach_ec_1e-2_over_pi": got["reach_ec_1e-2_over_pi"],
            "contract_reach_ec_1e-3_over_pi": want["reach_ec_1e-3_over_pi"],
            "computed_reach_ec_1e-3_over_pi": got["reach_ec_1e-3_over_pi"],
        }
    return {
        "producer": "code/analysis/exponential_family/gate9_e4.py",
        "written_utc": now(),
        "gate": "9.E4",
        "imports": "code/analysis/exponential_family/gate5a_offgrid.py, unchanged",
        "members": [route.member_block() for route in member_routes],
        "contract_symbol_prediction": stated,
        "contract_baseline": {"route": BASELINE, **CONTRACT_SLC_L12},
        "decision_rule": (
            "PASS if the realized errors lie within a factor of two of the "
            "symbol prediction at every N and below SLC plus L12 on every "
            "channel at every N; PARTIAL if that factor lies between two and "
            "four; REJECT otherwise."),
        "pass_factor": PASS_FACTOR,
        "partial_factor": PARTIAL_FACTOR,
        "anchor_packet": {"grids": list(g5a.GRIDS), "queries": g5a.QUERIES,
                          "seed": g5a.SEED,
                          "top_mode": [g5a.anchor_top_mode(n) for n in g5a.GRIDS]},
        "resolution_scaled_packet": {
            "grids": list(g5a.SCALED_GRIDS), "queries": g5a.QUERIES,
            "seed": g5a.SEED, "phase_seed": g5a.SCALED_PHASE_SEED,
            "band_divisor": g5a.SCALED_BAND_DIVISOR,
            "top_mode": [g5a.scaled_top_mode(n) for n in g5a.SCALED_GRIDS]},
        "symbol_level": rows,
        "predicted_anchor": {
            route.name: {str(n): g5a.predicted_errors(route, n, x)
                         for n in g5a.GRIDS}
            for route in all_routes},
        "predicted_resolution_scaled": {
            route.name: {str(n): g5a.predicted_errors(route, n, x, scaled_modes[n])
                         for n in g5a.SCALED_GRIDS}
            for route in all_routes},
    }


# ----------------------------------------------------------------------
# realized
# ----------------------------------------------------------------------

def realized_block(routes, grids, x, modes_by_n=None) -> dict:
    out = {}
    for route in routes:
        for npts in grids:
            modes = None if modes_by_n is None else modes_by_n[npts]
            out[(route.name, npts)] = g5a.realized_errors(route, npts, x, modes)
    return out


def rows_for(routes, realized, predicted, grids, tag) -> list:
    rows = []
    for route in routes:
        for npts in grids:
            got = realized[(route.name, npts)]
            pred = predicted[route.name][str(npts)]
            for q in range(3):
                expected = pred["modal_exact"][f"channel_{q}"]
                value = got[f"channel_{q}_rel_sup"]
                ratio = value / expected if expected > 0.0 else float("nan")
                rows.append({
                    "route": route.name, "regime": tag, "N": npts,
                    "h": 2.0 * np.pi / npts, "channel": CHANNELS[q],
                    "max_abs_error": got[f"channel_{q}_max_abs"],
                    "rel_sup_error": value,
                    "rel_l2_error": got[f"channel_{q}_rel_l2"],
                    "predicted_rel_sup_modal_exact": expected,
                    "predicted_rel_sup_modal_bound": pred["modal_bound"][f"channel_{q}"],
                    "realized_over_predicted": ratio,
                    "taps_per_query": route.taps_per_query,
                    "solves_per_field": route.solves_per_field,
                })
    return rows


def ratio_to(routes, realized, grids, reference: str) -> dict:
    out = {}
    for route in routes:
        entry = {}
        for q, channel in enumerate(CHANNELS):
            cells = {}
            for npts in grids:
                mine = realized[(route.name, npts)][f"channel_{q}_rel_sup"]
                theirs = realized[(reference, npts)][f"channel_{q}_rel_sup"]
                cells[str(npts)] = (float(mine / theirs) if theirs > 0.0
                                    else float("nan"))
            entry[channel] = cells
        out[route.name] = entry
    return out


def orders_for(routes, realized, grids) -> dict:
    out = {}
    for route in routes:
        entry = {}
        for q, channel in enumerate(CHANNELS):
            errors = [realized[(route.name, n)][f"channel_{q}_rel_sup"] for n in grids]
            entry[channel] = {
                "all_grids": g5a.fitted_order(grids, errors),
                "pairwise": g5a.pairwise_orders(list(grids), errors),
                "pairs": [f"{grids[i]}->{grids[i + 1]}" for i in range(len(grids) - 1)],
            }
        out[route.name] = entry
    return out


def regime_verdict(name: str, rows, realized, grids, regime: str) -> dict:
    """Apply the contract rule to one route in one realized regime."""
    worst_factor = 0.0
    worst_cell = None
    for row in rows:
        if row["route"] != name or row["regime"] != regime:
            continue
        expected = row["predicted_rel_sup_modal_exact"]
        if expected < g5a.PREDICTION_FLOOR:
            continue
        ratio = row["realized_over_predicted"]
        if not np.isfinite(ratio) or ratio <= 0.0:
            continue
        deviation = max(ratio, 1.0 / ratio)
        if deviation > worst_factor:
            worst_factor = deviation
            worst_cell = {k: row[k] for k in ("route", "regime", "N", "channel",
                                              "rel_sup_error",
                                              "predicted_rel_sup_modal_exact",
                                              "realized_over_predicted")}
    above = []
    worst_ratio = 0.0
    worst_ratio_cell = None
    for npts in grids:
        for q, channel in enumerate(CHANNELS):
            mine = realized[(name, npts)][f"channel_{q}_rel_sup"]
            theirs = realized[(BASELINE, npts)][f"channel_{q}_rel_sup"]
            cell = {"regime": regime, "N": npts, "channel": channel,
                    "route": mine, "baseline": theirs,
                    "route_over_baseline": mine / theirs}
            if mine / theirs > worst_ratio:
                worst_ratio = mine / theirs
                worst_ratio_cell = cell
            if mine >= theirs:
                above.append(cell)
    below = not above
    if worst_factor <= PASS_FACTOR and below:
        verdict = "PASS"
    elif worst_factor <= PARTIAL_FACTOR and below:
        verdict = "PARTIAL"
    else:
        verdict = "REJECT"
    return {
        "regime": regime,
        "verdict": verdict,
        "worst_realized_over_predicted_deviation": worst_factor,
        "worst_cell": worst_cell,
        "prediction_floor": g5a.PREDICTION_FLOOR,
        "baseline_route": BASELINE,
        "below_baseline_on_every_channel_and_grid": bool(below),
        "cells_not_below_the_baseline": above,
        "worst_route_over_baseline_ratio": worst_ratio,
        "worst_route_over_baseline_cell": worst_ratio_cell,
    }


ORDER = ("REJECT", "PARTIAL", "PASS")


def worst_verdict(values) -> str:
    return min(values, key=ORDER.index)


def route_verdict(name: str, rows, regimes) -> dict:
    """Return the rule applied per regime and the literal verdict over both."""
    per_regime = {regime: regime_verdict(name, rows, realized, grids, regime)
                  for regime, (realized, grids) in regimes.items()}
    literal = worst_verdict([block["verdict"] for block in per_regime.values()])
    return {
        "route": name,
        "verdict": literal,
        "verdict_literal_both_regimes": literal,
        "verdict_per_regime": {k: v["verdict"] for k, v in per_regime.items()},
        "worst_realized_over_predicted_deviation": max(
            block["worst_realized_over_predicted_deviation"]
            for block in per_regime.values()),
        "per_regime": per_regime,
    }


# ----------------------------------------------------------------------
# the Gate 5a regression
# ----------------------------------------------------------------------

def regression(x, scaled_modes) -> dict:
    """Recompute the stored champion rows of gate5/5a/runs.csv, bit for bit.

    The stored rows are written by csv.DictWriter from the row dictionaries of
    gate5a_offgrid.main, so the recomputation writes the same dictionaries
    through the same writer and compares the resulting lines as text.
    """
    stored_path = GATE5A / "runs.csv"
    stored = stored_path.read_text().splitlines()
    header = stored[0]
    fields = header.split(",")

    original = [r for r in g5a.build_routes() if r.name == "champion"]
    amended = [r for r in g5a.build_amendment_routes() if r.name == "champion"]

    anchor = {}
    for route in original:
        for npts in g5a.GRIDS:
            anchor[(route.name, npts)] = g5a.realized_errors(route, npts, x)
    scaled = {}
    for route in amended:
        for npts in g5a.SCALED_GRIDS:
            scaled[(route.name, npts)] = g5a.realized_errors(route, npts, x,
                                                             scaled_modes[npts])

    rebuilt = []
    for route in original:
        for npts in g5a.GRIDS:
            got = anchor[(route.name, npts)]
            pred = g5a.predicted_errors(route, npts, x)
            for q in range(3):
                expected = pred["modal_exact"][f"channel_{q}"]
                value = got[f"channel_{q}_rel_sup"]
                rebuilt.append({
                    "route": route.name, "N": npts, "h": 2.0 * np.pi / npts,
                    "channel": ("value", "d1", "d2")[q],
                    "max_abs_error": got[f"channel_{q}_max_abs"],
                    "rel_sup_error": value,
                    "rel_l2_error": got[f"channel_{q}_rel_l2"],
                    "predicted_rel_sup_modal_exact": expected,
                    "predicted_rel_sup_modal_bound":
                        pred["modal_bound"][f"channel_{q}"],
                    "realized_over_predicted": (value / expected if expected > 0.0
                                                else float("nan")),
                    "taps_per_query": route.taps_per_query,
                    "solves_per_field": route.solves_per_field,
                })
    amendment_prediction = {
        "predicted_resolution_scaled": {
            route.name: {str(n): g5a.predicted_errors(route, n, x, scaled_modes[n])
                         for n in g5a.SCALED_GRIDS}
            for route in amended}}
    rebuilt += g5a.amendment_rows(amended, scaled, amendment_prediction,
                                  g5a.SCALED_GRIDS, "resolution_scaled")

    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writerows(rebuilt)
    produced = buffer.getvalue().splitlines()

    wanted = [line for line in stored[1:]
              if line.split(",")[0] in ("champion", "champion|resolution_scaled")]
    differences = [{"stored": a, "recomputed": b}
                   for a, b in zip(wanted, produced) if a != b]
    return {
        "file": str(stored_path.relative_to(ROOT)),
        "rows_compared": len(wanted),
        "rows_recomputed": len(produced),
        "identical": bool(len(wanted) == len(produced) and not differences),
        "differences": differences[:8],
        "routes": ["champion", "champion|resolution_scaled"],
        "statement": ("The stored champion rows of part 5a are reproduced "
                      "character for character by the imported producer."),
    }


# ----------------------------------------------------------------------
# the markdown receipt
# ----------------------------------------------------------------------

def fmt(value, spec="{:.4e}") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return spec.format(value)


def write_markdown(payload: dict, path: Path) -> None:
    lines = ["# Gate 9 section E4: coupled off-grid reads", "",
             f"Produced {payload['written_utc']} by "
             "code/analysis/exponential_family/gate9_e4.py, which imports "
             "gate5a_offgrid.py unchanged.", "",
             "## Members", ""]
    for block in payload["members"]:
        lines.append(f"* {block['route']}: {block['member']}, m = {block['m']}, "
                     f"n = {block['n']}, degree {block['p']}, class "
                     f"C^{block['smoothness']}, cond {block['cond']:.3f}, "
                     f"{block['taps_per_query']} taps per query")
    lines += ["", "## Symbol level", "",
              "| route | e_c max B0 | C_1 max | C_2 max | reach 1e-2 / pi | "
              "reach 1e-3 / pi |", "| --- | --- | --- | --- | --- | --- |"]
    for row in payload["symbol_level"]:
        lines.append(f"| {row['route']} | {fmt(row['ec_max_B0'])} | "
                     f"{fmt(row['c1_max_theta_above_floor'])} | "
                     f"{fmt(row['c2_max_theta_above_floor'])} | "
                     f"{row['reach_ec_1e-2_over_pi']:.4f} | "
                     f"{row['reach_ec_1e-3_over_pi']:.4f} |")
    lines += ["", "The symbol prediction of the contract and the computed value:", "",
              "| member | contract e_c max B0 | computed | ratio | contract reach "
              "1e-2 | computed |", "| --- | --- | --- | --- | --- | --- |"]
    for name, block in payload["contract_symbol_prediction"].items():
        lines.append(f"| {name} | {fmt(block['contract_ec_max_B0'])} | "
                     f"{fmt(block['computed_ec_max_B0'])} | "
                     f"{block['ratio_computed_over_contract']:.3f} | "
                     f"{block['contract_reach_ec_1e-2_over_pi']:.3f} | "
                     f"{block['computed_reach_ec_1e-2_over_pi']:.4f} |")

    for tag, title, grids in (("anchor", "Anchor packet", g5a.GRIDS),
                              ("resolution_scaled", "Resolution-scaled band-edge packet",
                               g5a.SCALED_GRIDS)):
        lines += ["", f"## {title}", "",
                  "Relative sup error, and the ratio to " + BASELINE + ".", "",
                  "| route | channel | " + " | ".join(f"N = {n}" for n in grids) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in grids) + " |"]
        realized = payload[f"realized_{tag}"]
        for route in payload["route_order"]:
            for channel in CHANNELS:
                cells = [fmt(realized[f"{route}_N{n}"][
                    f"channel_{CHANNELS.index(channel)}_rel_sup"]) for n in grids]
                lines.append(f"| {route} | {channel} | " + " | ".join(cells) + " |")
        lines += ["", f"Ratio to {BASELINE}:", "",
                  "| route | channel | " + " | ".join(f"N = {n}" for n in grids) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in grids) + " |"]
        ratios = payload[f"ratio_to_baseline_{tag}"]
        for route in payload["route_order"]:
            for channel in CHANNELS:
                cells = [fmt(ratios[route][channel][str(n)], "{:.3e}") for n in grids]
                lines.append(f"| {route} | {channel} | " + " | ".join(cells) + " |")
        lines += ["", "Fitted orders over all grids:", "",
                  "| route | " + " | ".join(CHANNELS) + " |",
                  "| --- | " + " | ".join("---" for _ in CHANNELS) + " |"]
        orders = payload[f"fitted_orders_{tag}"]
        for route in payload["route_order"]:
            cells = [f"{orders[route][c]['all_grids']:.2f}" for c in CHANNELS]
            lines.append(f"| {route} | " + " | ".join(cells) + " |")

    lines += ["", "## Decision", ""]
    for block in payload["decision"]["members"] + payload["decision"][
            "calibration_stored_routes"]:
        per = block["verdict_per_regime"]
        worst = {k: v["worst_route_over_baseline_ratio"]
                 for k, v in block["per_regime"].items()}
        lines.append(
            f"* {block['route']}: literal {block['verdict']}; anchor "
            f"{per['anchor']}, resolution scaled {per['resolution_scaled']}. "
            f"Worst realized over predicted deviation "
            f"{block['worst_realized_over_predicted_deviation']:.4f}; worst "
            f"ratio to {BASELINE} {worst['anchor']:.3e} on the anchor packet "
            f"and {worst['resolution_scaled']:.3e} on the band-edge packet.")
    lines += ["", payload["decision"]["regime_note"], "",
              f"Overall literal: {payload['decision']['verdict']}. "
              f"Per regime: " + ", ".join(
                  f"{k} {v}" for k, v in
                  payload["decision"]["verdict_per_regime"].items()) + ".", "",
              "## Regression", "",
              f"{payload['regression']['rows_compared']} stored champion rows of "
              f"{payload['regression']['file']} recomputed through the imported "
              f"producer: identical = {payload['regression']['identical']}.", ""]
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("all", "predict", "realize"),
                        default="all")
    args = parser.parse_args()

    GATE9.mkdir(parents=True, exist_ok=True)
    member_routes = build_member_routes()
    stored_routes = g5a.build_amendment_routes()
    all_routes = member_routes + stored_routes
    x = g5a.query_positions(2.0 * np.pi / g5a.GRIDS[0], g5a.GRIDS[0])
    scaled_modes = {n: g5a.scaled_packet_modes(n) for n in g5a.SCALED_GRIDS}

    prediction_file = GATE9 / "e4_prediction.json"
    if args.stage in ("all", "predict"):
        prediction = prediction_payload(all_routes, member_routes, x, scaled_modes)
        prediction_file.write_text(json.dumps(prediction, indent=2) + "\n")
        print(f"prediction written {prediction_file.relative_to(ROOT)}", flush=True)
        if args.stage == "predict":
            return 0
    else:
        if not prediction_file.exists():
            raise SystemExit("the prediction must be written before the realized stage")
        prediction = json.loads(prediction_file.read_text())

    started = now()
    anchor = realized_block(all_routes, g5a.GRIDS, x)
    scaled = realized_block(all_routes, g5a.SCALED_GRIDS, x, scaled_modes)

    rows = (rows_for(all_routes, anchor, prediction["predicted_anchor"],
                     g5a.GRIDS, "anchor")
            + rows_for(all_routes, scaled, prediction["predicted_resolution_scaled"],
                       g5a.SCALED_GRIDS, "resolution_scaled"))
    with (GATE9 / "e4_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    regimes = {"anchor": (anchor, g5a.GRIDS),
               "resolution_scaled": (scaled, g5a.SCALED_GRIDS)}
    verdicts = [route_verdict(name, rows, regimes) for name in MEMBER_NAMES]
    calibration = [route_verdict(name, rows, regimes)
                   for name in ("champion", "BMC9", "BMC11")]
    overall = worst_verdict([v["verdict"] for v in verdicts])
    by_regime = {regime: worst_verdict([v["verdict_per_regime"][regime]
                                        for v in verdicts])
                 for regime in regimes}

    payload = {
        "producer": "code/analysis/exponential_family/gate9_e4.py",
        "written_utc": now(),
        "realized_started_utc": started,
        "gate": "9.E4",
        "prediction": str(prediction_file.relative_to(ROOT)),
        "prediction_written_utc": prediction["written_utc"],
        "prediction_before_runs": True,
        "members": prediction["members"],
        "contract_symbol_prediction": prediction["contract_symbol_prediction"],
        "decision_rule": prediction["decision_rule"],
        "route_order": [route.name for route in all_routes],
        "symbol_level": prediction["symbol_level"],
        "anchor_packet": prediction["anchor_packet"],
        "resolution_scaled_packet": prediction["resolution_scaled_packet"],
        "realized_anchor": {f"{name}_N{n}": value
                            for (name, n), value in anchor.items()},
        "realized_resolution_scaled": {f"{name}_N{n}": value
                                       for (name, n), value in scaled.items()},
        "fitted_orders_anchor": orders_for(all_routes, anchor, g5a.GRIDS),
        "fitted_orders_resolution_scaled": orders_for(all_routes, scaled,
                                                      g5a.SCALED_GRIDS),
        "ratio_to_baseline_anchor": ratio_to(all_routes, anchor, g5a.GRIDS, BASELINE),
        "ratio_to_baseline_resolution_scaled": ratio_to(all_routes, scaled,
                                                        g5a.SCALED_GRIDS, BASELINE),
        "ratio_to_champion_anchor": ratio_to(all_routes, anchor, g5a.GRIDS, "champion"),
        "ratio_to_champion_resolution_scaled": ratio_to(all_routes, scaled,
                                                        g5a.SCALED_GRIDS, "champion"),
        "decision": {
            "rule": prediction["decision_rule"],
            "verdict": overall,
            "verdict_literal_both_regimes": overall,
            "verdict_per_regime": by_regime,
            "members": verdicts,
            "calibration_stored_routes": calibration,
            "regime_note": (
                "The rule is applied twice, once per realized regime. The "
                "anchor packet of the original part 5a has its top mode at "
                "0.469 pi on N = 64 and 0.234 pi on N = 128, so refining it "
                "walks the packet into the range where formal order alone "
                "decides, and the twelve-node Lagrange layer of the "
                "augmentation is of formal order 12 against 4 and 6 for the "
                "two members and 6 for the champion. The stored champion is "
                "carried through the same rule beside the members so that the "
                "anchor-regime outcome is read against a route whose Gate 5 "
                "receipt is already in hand. The resolution-scaled packet "
                "holds the top mode at the dealiased band edge on every grid, "
                "which is the regime the symbol prediction of the contract "
                "describes."),
        },
        "regression": regression(x, scaled_modes),
        "runs": "results/exponential_family/gate9/e4_runs.csv",
    }
    (GATE9 / "e4_offgrid.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(payload, GATE9 / "e4_offgrid.md")

    for verdict in verdicts + calibration:
        per = verdict["verdict_per_regime"]
        print(f"{verdict['route']:10s} literal {verdict['verdict']:8s} "
              f"anchor {per['anchor']:8s} band-edge "
              f"{per['resolution_scaled']:8s} worst prediction deviation "
              f"{verdict['worst_realized_over_predicted_deviation']:.4f}",
              flush=True)
    print(f"overall literal {overall}; per regime {by_regime}")
    print(f"regression identical {payload['regression']['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
