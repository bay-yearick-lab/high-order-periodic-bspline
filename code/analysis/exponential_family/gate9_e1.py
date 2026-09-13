"""Gate 9, E1: the static battery on the fielded members and its receipt check.

Two modes, both thin drivers over frozen code.

--battery assembles the six generator specifications of the Gate 9 contract
and hands them to static_battery.py unchanged:

    pert5        X8_n10_C_SLC1_pert5, the everything member (m = 4, one
                 imaginary slot, one quad, P of degree 3), read from its
                 Gate 8 member file;
    pert12       X8_n10_C_BMC11_SLC1992_pert12, the same shape at the BMC11
                 budget, read from its Gate 8 member file;
    champion     the Gate 1 band champion E_{6,2}(1.7701927977475826,
                 2.0606855355641316) at budget C_SLC1 on band B0;
    SLC_realized  the exact realization of SLC's 1992 pentadiagonal D1 as a
                 one-field member, the n = 8, m = 4, two imaginary slots,
                 q = 2 cell of gate8/realization.json whose sample residual is
                 9.331e-9, taken with its stored slots and polynomial at full
                 precision;
    BMC9, BMC11  the pure polynomial cores m = 10 and m = 12, which are the
                 B-spline collocation references of degree 9 and 11.

--check reads the receipt the battery wrote and compares it with the frozen
records:

    pert5 and pert12 against every shared yardstick of the verified block of
    results/exponential_family/gate8/dominance.json at 1e-6 relative;
    the champion against results/exponential_family/gate1/champions.json at
    1e-9 relative;
    the realized SLC field against SLC's printed D1 target vector, the
    sample ratios psi(l)/psi(0) and psi'(s)/psi(0) of gate8_realization.py,
    at 1e-8 absolute, with the realized literal-D2 band maximum E2max(B0)
    divided by the band maximum of SLC's own tuned fourth-order D2.

A member fielded by --extra-member is checked against any Gate 8 dominance
receipt with

    --check-member <member file> --against <dominance json> --key <locator>

where the locator is either the summary key n|budget|target of a per-cell
receipt of the E0 search, whose verified block is keyed that way, or the
member name of dominance.json, whose verified block is keyed by name. Both
shapes are read by the same routine and compared at the same 1e-6. With
--tag <name> the receipt of the check is written to e1_check_<name>.json and
e1_check_<name>.md; without it, to the frozen e1_check.json and e1_check.md.

The comparison of float fields is static_battery._compare_floats, the same
routine the frozen reproduction mode uses, so a deviation reported here is
measured exactly as Gate 1 and Gate 7 measure theirs.

Slot order of an extra member. A member file written by the E0 search names
the receipt and summary key it came from in its source field, so an
--extra-member is fielded in the slot order of that verified record, exactly
as pert5 and pert12 are, and the battery reproduces the verified row. The
file order is measured beside it in the slot order block of the check.

Slot order. A member file records the slot multiset as separate omegas and
quads lists and loses the order in which the slot factors multiplied when
Gate 8 verified the member; the verified record of dominance.json keeps that
order. The product of the slot factors is commutative in exact arithmetic and
not in floating point, so the member files of pert5 and pert12 are fielded
with the slot order of their verified records, which makes the battery
reproduce the Gate 8 row bit for bit. The check measures the alternative
order, omegas before quads, and reports the keys it moves.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
for _path in (str(CODE), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate8_realization as g8r  # noqa: E402
import static_battery as sb  # noqa: E402

EXP = ROOT / "results" / "exponential_family"
GATE9 = EXP / "gate9"
GATE8 = EXP / "gate8"
MEMBERS = GATE8 / "members"
REALIZATION = GATE8 / "realization.json"
DOMINANCE = GATE8 / "dominance.json"
CHAMPIONS = EXP / "gate1" / "champions.json"

NAME_PERT5 = "X8_n10_C_SLC1_pert5"
NAME_PERT12 = "X8_n10_C_BMC11_SLC1992_pert12"
PERT5 = MEMBERS / (NAME_PERT5 + ".json")
PERT12 = MEMBERS / (NAME_PERT12 + ".json")

CHAMPION_BAND = "B0"
CHAMPION_BUDGET = "C_SLC1"
CHAMPION_OMEGAS = (1.7701927977475826, 2.0606855355641316)

LABEL_PERT5 = "pert5"
LABEL_PERT12 = "pert12"
LABEL_CHAMPION = "champion"
LABEL_SLC = "SLC_realized"
LABEL_BMC9 = "BMC9_m10"
LABEL_BMC11 = "BMC11_m12"

GATE8_RTOL = 1.0e-6
GATE1_RTOL = 1.0e-9
TARGET_ATOL = 1.0e-8
D2_RATIO_RANGE = (6.0, 8.0)

# Stored keys of the Gate 8 verified metric row that the battery does not
# carry, with the reason. They are listed in the receipt so that the set of
# matched yardsticks is auditable.
GATE8_UNMATCHED = {
    "fine_e1_max_B0": "Gate 8 confirmation grid of 200,001 angles",
    "fine_e2_max_B0": "Gate 8 confirmation grid of 200,001 angles",
    "fine_reach_d1_0.01": "Gate 8 confirmation grid of 200,001 angles",
    "fine_reach_d1_0.001": "Gate 8 confirmation grid of 200,001 angles",
    "fine_reach_d1_0.0001": "Gate 8 confirmation grid of 200,001 angles",
    "fine_gv_reach_1e-2": "Gate 8 confirmation grid of 200,001 angles",
    "fine_cond": "Gate 8 confirmation grid of 200,001 angles",
    "fine_max_b": "Gate 8 confirmation grid of 200,001 angles",
    "fine_min_b": "Gate 8 confirmation grid of 200,001 angles",
    "kappa2_min_over_grid": ("minimum of kappa_2 on the 200,001 point grid; "
                             "the battery reports kappa2_min on the 20,001 "
                             "point metrics grid, a different quantity"),
    "e1_zeros_over_pi": "a list, not a scalar yardstick",
    "order_d1_window": "a list, not a scalar yardstick",
    "order_d2_window": "a list, not a scalar yardstick",
    "sign_N64": "a boolean, carried by the battery under the same name",
    "sign_N127": "a boolean, carried by the battery under the same name",
    "sign_N128": "a boolean, carried by the battery under the same name",
    "name": "a label",
    "coupled": "a label",
}

NAME_MAP = {
    "metrics": ("every Gate 0 metric name of metrics.py, identical in both "
                "receipts: e1_*, e2_*, reach_d1_*, reach_d2_*, reach_ec_*, "
                "ec_max_*, gv_reach_1e-2, gv_error, rho1, rho2, rk4_*, "
                "order_d1_*, order_d2_*, kappa2_min_nonzero_N*, theta_points"),
    "conditioning": ("stored metrics.cond, cond_N64, cond_N127, cond_N128, "
                     "min_left, max_left map to the battery conditioning "
                     "block under the same names"),
    "stencil": ("stored metrics.bandwidth_left, bandwidth_d1, bandwidth_d2, "
                "solves map to the battery stencil block under the same "
                "names"),
    "positivity": ("the verified positivity certificate of the record maps "
                   "field for field to the battery positivity block"),
}


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

def member_file_spec(path: Path, label: str) -> dict:
    """Return the specification of a member file in its own field order.

    The slot factors multiply in the order the battery builds them from the
    kind lists, that is every imaginary pair, then every real pair, then every
    quad.
    """
    record = json.loads(Path(path).read_text())
    return {"label": label, "m": int(record["m"]),
            "omegas": [float(w) for w in record.get("omegas", ())],
            "real_pairs": [float(a) for a in record.get("real_pairs", ())],
            "quads": [[float(a), float(w)] for a, w in record.get("quads", ())],
            "polynomial": [float(p) for p in record.get("polynomial", ())],
            "backend": "gate8"}


def member_spec(path: Path, label: str, name: str) -> dict:
    """Return the specification of a member file in its frozen slot order.

    The slot multiset of the member file is checked against the slots of the
    Gate 8 verified record value for value, and the verified order is the one
    fielded, so that the battery reproduces the frozen row bit for bit.
    """
    spec = member_file_spec(path, label)
    record = verified_record(name)
    frozen = [[float(a), float(w)] for a, w in record["slots"]]
    from_file = [[0.0, w] for w in spec["omegas"]]
    from_file += [[a, 0.0] for a in spec["real_pairs"]]
    from_file += [[float(a), float(w)] for a, w in spec["quads"]]
    if sorted(frozen) != sorted(from_file):
        raise SystemExit(f"{name}: the member file slots and the verified "
                         "record slots differ")
    if [float(p) for p in record["polynomial"]] != spec["polynomial"]:
        raise SystemExit(f"{name}: the member file polynomial and the verified "
                         "record polynomial differ")
    spec["slots"] = frozen
    return spec


SOURCE_PATTERN = re.compile(r"^(?P<path>\S+)\s+summary\[(?P<key>.+)\]$")


def source_locator(record: dict):
    """Return the receipt path and summary key a member file names, or None.

    A member file written by gate9_e0_members.py carries the line
    "<receipt> summary[<key>]" in its source field.
    """
    match = SOURCE_PATTERN.match(str(record.get("source", "")))
    if match is None:
        return None
    return match.group("path"), match.group("key")


def stored_slots(path, key):
    """Return the slot order of one verified record of a dominance receipt."""
    data = json.loads(locate(path).read_text())
    summary = data.get("summary", {})
    if key in summary and "slots" in summary[key]:
        return [[float(a), float(w)] for a, w in summary[key]["slots"]]
    for entries in data.get("verified", {}).values():
        for entry in entries:
            if entry.get("name") == key and "slots" in entry:
                return [[float(a), float(w)] for a, w in entry["slots"]]
    return None


def extra_member_spec(path) -> dict:
    """Return the specification of an extra member in its verified slot order.

    The order is the one of the verified record the member file names in its
    source field, checked against the slot multiset of the file itself. A
    member file that names no record is fielded in its own field order.
    """
    path = locate(path)
    record = json.loads(path.read_text())
    label = str(record.get("name", path.stem))
    spec = member_file_spec(path, label)
    locator = source_locator(record)
    if locator is None:
        return spec
    slots = stored_slots(locator[0], locator[1])
    if slots is None:
        return spec
    from_file = [[0.0, w] for w in spec["omegas"]]
    from_file += [[a, 0.0] for a in spec["real_pairs"]]
    from_file += [[float(a), float(w)] for a, w in spec["quads"]]
    if sorted(slots) != sorted(from_file):
        raise SystemExit(f"{label}: the member file slots and the slots of "
                         f"{locator[0]} [{locator[1]}] differ")
    spec["slots"] = slots
    spec["slot_order"] = f"{under_root(locator[0])} [{locator[1]}]"
    return spec


def champion_entry() -> dict:
    """Return the stored Gate 1 champion entry of band B0 at budget C_SLC1."""
    data = json.loads(CHAMPIONS.read_text())
    return data["champions"][CHAMPION_BAND][CHAMPION_BUDGET]["champion"]


def champion_spec() -> dict:
    """Return the battery specification of the band champion."""
    champion = champion_entry()
    omegas = [float(w) for w in champion["omegas"]]
    if len(omegas) != len(CHAMPION_OMEGAS) or any(
            abs(a - b) > 0.0 for a, b in zip(omegas, CHAMPION_OMEGAS)):
        raise SystemExit("the stored champion frequencies are not the "
                         "contract's band champion")
    return {"label": LABEL_CHAMPION, "m": int(champion["m"]), "omegas": omegas,
            "real_pairs": [], "quads": [], "polynomial": []}


def slc_cell() -> dict:
    """Return the exact realization cell of SLC1992 of gate8/realization.json.

    The cell is the n = 8, m = 4, two imaginary slots, q = 2 one, the only
    SLC1992 cell flagged exact_realization, whose sample residual is 9.3e-9.
    """
    data = json.loads(REALIZATION.read_text())
    picked = [cell for cell in data["cells"]
              if cell["target"] == "SLC1992" and cell.get("exact_realization")]
    if len(picked) != 1:
        raise SystemExit("exactly one exact SLC1992 realization cell is expected")
    cell = picked[0]
    if not (cell["m"] == 4 and cell["q"] == 2 and cell["slots_count"] == 2
            and cell["generator"]["slot_kinds"] == ["imaginary", "imaginary"]):
        raise SystemExit("the exact SLC1992 cell is not the m = 4, q = 2 cell")
    return cell


def slc_spec() -> dict:
    """Return the battery specification of SLC's realized one-field member."""
    gen = slc_cell()["generator"]
    return {"label": LABEL_SLC, "m": int(gen["m"]),
            "slots": [[float(a), float(w)] for a, w in gen["slots"]],
            "polynomial": [float(p) for p in gen["polynomial"]],
            "backend": "gate8"}


def build_specs(extra=()) -> list:
    """Return the specifications of the Gate 9 E1 battery, in order.

    Any path in extra is a further member file, fielded in its own field
    order under the name it carries. That is the hook the contract leaves
    open for a member section E0 reports later.
    """
    specs = [member_spec(PERT5, LABEL_PERT5, NAME_PERT5),
             member_spec(PERT12, LABEL_PERT12, NAME_PERT12),
             champion_spec(),
             slc_spec(),
             {"label": LABEL_BMC9, "m": 10, "omegas": [], "real_pairs": [],
              "quads": [], "polynomial": []},
             {"label": LABEL_BMC11, "m": 12, "omegas": [], "real_pairs": [],
              "quads": [], "polynomial": []}]
    for path in extra:
        specs.append(extra_member_spec(path))
    return specs


# ---------------------------------------------------------------------------
# The battery run
# ---------------------------------------------------------------------------

def run_battery(processes: int, extra=()) -> int:
    """Write the specifications and call static_battery.py on them."""
    GATE9.mkdir(parents=True, exist_ok=True)
    specs_path = GATE9 / "battery_specs.json"
    specs_path.write_text(json.dumps(build_specs(extra), indent=1) + "\n")
    argv = ["--specs", str(specs_path), "--decimal",
            "--processes", str(int(processes)),
            "--out", str(GATE9 / "battery.json"),
            "--md-out", str(GATE9 / "battery.md"), "--quiet"]
    print("static_battery.py " + " ".join(argv))
    return sb.main(argv)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def battery_rows() -> dict:
    """Return the battery results keyed by label."""
    payload = json.loads((GATE9 / "battery.json").read_text())
    return {row["label"]: row for row in payload["results"]}


def flat_row(result: dict) -> dict:
    """Return the battery metrics, conditioning and stencil as one flat row."""
    row = dict(result["metrics"])
    row.update(result["conditioning"])
    row.update(result["stencil"])
    return row


def verified_record(name: str) -> dict:
    """Return the Gate 8 verified record of one member name."""
    data = json.loads(DOMINANCE.read_text())
    for entries in data["verified"].values():
        for entry in entries:
            if entry["name"] == name:
                return entry
    raise SystemExit(f"{name} is not in the verified block of dominance.json")


def locate(path) -> Path:
    """Return an existing path, resolved against the repository root if need be."""
    candidate = Path(path)
    if not candidate.exists() and not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def under_root(path) -> str:
    """Return a path written relative to the repository root when it lies there."""
    resolved = locate(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def dominance_record(path, key: str) -> dict:
    """Return metrics, positivity and cond of one verified Gate 8 record.

    Two receipt shapes are read. dominance.json carries a verified block of
    lists keyed by budget, every entry named by its member name, with cond
    beside the metrics. A per-cell receipt of the E0 search carries a
    verification block keyed by the summary key n|budget|target, with cond
    inside the member record it stores.
    """
    path = locate(path)
    data = json.loads(path.read_text())
    entry = None
    block = data.get("verification")
    if isinstance(block, dict) and key in block:
        entry = block[key]
    else:
        for entries in data.get("verified", {}).values():
            for candidate in entries:
                if candidate.get("name") == key:
                    entry = candidate
    if entry is None:
        raise SystemExit(f"{key} is not in the verified block of {path}")
    cond = entry.get("cond")
    if cond is None:
        cond = entry.get("record", {}).get("cond")
    if cond is None:
        raise SystemExit(f"{key} of {path} carries no conditioning")
    return {"metrics": entry["metrics"], "positivity": entry["positivity"],
            "cond": float(cond)}


def record_case(label: str, record: dict, source: str, rows: dict) -> dict:
    """Return the comparison of one battery row with one verified record."""
    result = rows[label]
    row = flat_row(result)
    stored = record["metrics"]
    matched = sorted(key for key, value in stored.items()
                     if isinstance(value, (int, float))
                     and not isinstance(value, bool)
                     and isinstance(row.get(key), (int, float))
                     and not isinstance(row.get(key), bool))
    metrics = sb._compare_floats(stored, row, GATE8_RTOL)
    positivity = sb._compare_floats(record["positivity"],
                                    result.get("positivity", {}), GATE8_RTOL)
    cond = sb._compare_floats({"cond": record["cond"]},
                              {"cond": result["conditioning"]["cond"]},
                              GATE8_RTOL)
    unmatched = sorted(key for key in stored
                       if key not in matched and key in GATE8_UNMATCHED)
    leftover = sorted(key for key in stored
                      if key not in matched and key not in GATE8_UNMATCHED)
    return {"label": label, "record": source, "rtol": GATE8_RTOL,
            "matched_yardsticks": matched,
            "matched_count": len(matched),
            "unmatched_stored_keys": {key: GATE8_UNMATCHED[key]
                                      for key in unmatched},
            "stored_keys_without_a_reason": leftover,
            "metrics": metrics, "positivity": positivity,
            "record_cond": cond,
            "worst": max(metrics["worst"], positivity["worst"], cond["worst"]),
            "ok": bool(metrics["ok"] and positivity["ok"] and cond["ok"])}


def gate8_case(label: str, name: str, rows: dict) -> dict:
    """Return the comparison of one fielded member with its Gate 8 record."""
    return record_case(label, verified_record(name), name, rows)


def extra_member_case(path, against, key, rows: dict) -> tuple:
    """Return the label and the comparison of one further fielded member.

    The label is the one build_specs gives the member file, its stored name,
    so the battery row is found without a further option.
    """
    path = locate(path)
    label = str(json.loads(path.read_text()).get("name", path.stem))
    if label not in rows:
        raise SystemExit(f"{label} is not a label of the battery receipt; "
                         "field the member with --extra-member first")
    locator = key if key else label
    record = dominance_record(against, locator)
    source = f"{under_root(against)} [{locator}]"
    return label, record_case(label, record, source, rows)


def champion_case(rows: dict) -> dict:
    """Return the comparison of the band champion with its Gate 1 record."""
    stored = champion_entry()["metrics"]
    result = rows[LABEL_CHAMPION]
    row = flat_row(result)
    matched = sorted(key for key, value in stored.items()
                     if isinstance(value, (int, float))
                     and not isinstance(value, bool)
                     and isinstance(row.get(key), (int, float))
                     and not isinstance(row.get(key), bool))
    metrics = sb._compare_floats(stored, row, GATE1_RTOL)
    return {"label": LABEL_CHAMPION,
            "record": f"gate1/champions.json champions.{CHAMPION_BAND}."
                      f"{CHAMPION_BUDGET}.champion.metrics",
            "rtol": GATE1_RTOL, "matched_yardsticks": matched,
            "matched_count": len(matched), "metrics": metrics,
            "worst": metrics["worst"], "ok": bool(metrics["ok"])}


def slc_case(rows: dict) -> dict:
    """Return the D1 target reproduction and the literal-D2 ratio of SLC.

    The sample ratio vector is the one of gate8_realization.sample_ratios,
    psi(l)/psi(0) for l = 1, 2, 3 followed by psi'(s)/psi(0) for s = 1, 2, 3,
    and the target vector is gate8_realization.target_vector of the printed
    SLC1992 coefficients, alpha_l then -a_s.
    """
    cell = slc_cell()
    spec = sb.normalise_spec(slc_spec())
    gen, backend = sb.make_generator(spec)
    if backend != "gate8":
        raise SystemExit("the realized SLC field needs the gate8 back end")
    ratios = g8r.sample_ratios(gen)
    target = g8r.target_vector(g8r.TARGETS["SLC1992"])
    residual = np.asarray(ratios) - np.asarray(target)
    names = ["psi(1)/psi(0)", "psi(2)/psi(0)", "psi(3)/psi(0)",
             "psi'(1)/psi(0)", "psi'(2)/psi(0)", "psi'(3)/psi(0)"]

    targets = sb.comparator_targets()
    slc_d2 = float(targets["SLC1992"]["e2_max_B0"])
    realized_d2 = float(rows[LABEL_SLC]["metrics"]["e2_max_B0"])
    ratio = realized_d2 / slc_d2
    return {
        "label": LABEL_SLC,
        "cell": {"target": cell["target"], "n": cell["n"], "m": cell["m"],
                 "q": cell["q"], "slots": cell["generator"]["slots"],
                 "polynomial": cell["generator"]["polynomial"],
                 "stored_residual_max": cell["residual_max"],
                 "stored_exact_realization": cell["exact_realization"]},
        "d1_target": {
            "atol": TARGET_ATOL,
            "components": names,
            "target": [float(v) for v in target],
            "realized": [float(v) for v in ratios],
            "residual": [float(v) for v in residual],
            "residual_max": float(np.max(np.abs(residual))),
            "stored_residual_max": float(cell["residual_max"]),
            "deviation_from_stored_residual": float(
                abs(np.max(np.abs(residual)) - cell["residual_max"])),
            "ok": bool(np.max(np.abs(residual)) <= TARGET_ATOL),
        },
        "d2_ratio": {
            "realized_literal_d2_e2_max_B0": realized_d2,
            "slc_tuned_d2_e2_max_B0": slc_d2,
            "ratio": ratio,
            "expected_range": list(D2_RATIO_RANGE),
            "ok": bool(D2_RATIO_RANGE[0] <= ratio <= D2_RATIO_RANGE[1]),
        },
        "ok": bool(np.max(np.abs(residual)) <= TARGET_ATOL
                   and D2_RATIO_RANGE[0] <= ratio <= D2_RATIO_RANGE[1]),
    }


def slot_order_sensitivity(processes: int = 2, extra=()) -> dict:
    """Return the Gate 8 deviation of the alternative slot order.

    The alternative order is the one a member file alone dictates, imaginary
    pairs before quads. The generator is the same in exact arithmetic, so the
    deviation this block reports is the floating point sensitivity of each
    stored key to the order of one product of two slot factors.
    """
    specs = [member_file_spec(PERT5, LABEL_PERT5 + "_file_order"),
             member_file_spec(PERT12, LABEL_PERT12 + "_file_order")]
    cases = [(LABEL_PERT5, NAME_PERT5, verified_record(NAME_PERT5)["metrics"])]
    cases.append((LABEL_PERT12, NAME_PERT12,
                  verified_record(NAME_PERT12)["metrics"]))
    for path, against, key in extra:
        member = locate(path)
        label = str(json.loads(member.read_text()).get("name", member.stem))
        locator = key if key else label
        specs.append(member_file_spec(member, label + "_file_order"))
        cases.append((label, f"{under_root(against)} [{locator}]",
                      dominance_record(against, locator)["metrics"]))
    options = sb.default_options(decimal=False, comparison=False)
    results = sb.evaluate_specs(specs, options, processes)
    out = {"definition": ("the slot factors multiplied as the member file "
                          "lists them, imaginary pairs before quads, against "
                          "the frozen order of the verified record"),
           "rtol": GATE8_RTOL, "members": {}}
    for (label, name, stored), result in zip(cases, results):
        comparison = sb._compare_floats(stored, flat_row(result), GATE8_RTOL)
        out["members"][label] = {
            "record": name, "worst": comparison["worst"],
            "worst_key": comparison["worst_key"],
            "keys_above_rtol": comparison["failures"],
            "ok": comparison["ok"]}
    return out


def run_check(processes: int = 2, extra=()) -> dict:
    """Return the full E1 check payload.

    Every entry of extra is a triple of a member file, the dominance receipt
    holding its verified record and the locator of that record inside it.
    """
    rows = battery_rows()
    cases = {
        LABEL_PERT5: gate8_case(LABEL_PERT5, NAME_PERT5, rows),
        LABEL_PERT12: gate8_case(LABEL_PERT12, NAME_PERT12, rows),
        LABEL_CHAMPION: champion_case(rows),
        LABEL_SLC: slc_case(rows),
    }
    extra_labels = []
    for path, against, key in extra:
        label, case = extra_member_case(path, against, key, rows)
        cases[label] = case
        extra_labels.append(label)
    failures = sorted(key for key, case in cases.items() if not case["ok"])
    errors = sorted(label for label, row in rows.items() if row.get("error"))
    summary = {
        "producer": "code/analysis/exponential_family/gate9_e1.py --check",
        "contract": "results/exponential_family/gate9/contract.md, section E1",
        "battery": "results/exponential_family/gate9/battery.json",
        "name_mapping": NAME_MAP,
        "labels_in_the_battery": sorted(rows),
        "battery_errors": errors,
        "slot_order_sensitivity": slot_order_sensitivity(processes, extra),
        "cases": cases,
        "verdict": "PASS" if not failures and not errors else "FAIL",
        "failed_cases": failures,
    }
    if extra_labels:
        summary["extra_members"] = extra_labels
    return summary


def check_markdown(report: dict) -> str:
    """Return a short table of the E1 check."""
    lines = ["# Gate 9 E1 check", "",
             f"Verdict: {report['verdict']}", "",
             "| case | record | tolerance | matched | worst relative | ok |",
             "| --- | --- | --- | --- | --- | --- |"]
    for key in ((LABEL_PERT5, LABEL_PERT12, LABEL_CHAMPION)
                + tuple(report.get("extra_members", ()))):
        case = report["cases"][key]
        lines.append(f"| {key} | {case['record']} | {case['rtol']:g} | "
                     f"{case['matched_count']} | {case['worst']:.3e} | "
                     f"{'yes' if case['ok'] else 'no'} |")
    slc = report["cases"][LABEL_SLC]
    lines += ["", "SLC's realized field:", "",
              f"* D1 target residual max {slc['d1_target']['residual_max']:.4e} "
              f"against the tolerance {slc['d1_target']['atol']:g}, "
              f"Gate 8 stored {slc['d1_target']['stored_residual_max']:.4e};",
              f"* literal D2 E2max(B0) "
              f"{slc['d2_ratio']['realized_literal_d2_e2_max_B0']:.6e} over "
              f"SLC's tuned D2 "
              f"{slc['d2_ratio']['slc_tuned_d2_e2_max_B0']:.6e} is "
              f"{slc['d2_ratio']['ratio']:.4f}, expected in "
              f"[{D2_RATIO_RANGE[0]:g}, {D2_RATIO_RANGE[1]:g}]."]
    sens = report["slot_order_sensitivity"]
    lines += ["", "Slot order sensitivity, the member file order against the",
              "frozen order of dominance.json:", ""]
    for label, entry in sens["members"].items():
        moved = ", ".join(f["key"] for f in entry["keys_above_rtol"]) or "none"
        lines.append(f"* {label}: worst {entry['worst']:.3e} on "
                     f"{entry['worst_key']}, above {sens['rtol']:g}: {moved};")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--battery", action="store_true",
                        help="run static_battery.py on the six specifications")
    parser.add_argument("--check", action="store_true",
                        help="compare the battery with the frozen receipts")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--extra-member", dest="extra_member", action="append",
                        default=[],
                        help="path of a further member file to field, "
                             "repeatable; the hook for a section E0 member")
    parser.add_argument("--check-member", dest="check_member", action="append",
                        default=[],
                        help="path of a further member file to check against "
                             "a verified Gate 8 record, repeatable")
    parser.add_argument("--against", default=str(DOMINANCE),
                        help="dominance receipt holding the verified records "
                             "of the --check-member files")
    parser.add_argument("--key", dest="keys", action="append", default=[],
                        help="locator of the verified record inside --against, "
                             "the summary key n|budget|target of a per-cell "
                             "receipt or a member name; repeatable, taken in "
                             "the order of the --check-member options, and "
                             "defaulting to the member name")
    parser.add_argument("--tag", default=None,
                        help="suffix of the check receipt names, "
                             "e1_check_<tag>.json and e1_check_<tag>.md; "
                             "default none, the frozen names")
    args = parser.parse_args(argv)
    if not (args.battery or args.check):
        parser.error("give --battery or --check")
    if len(args.keys) > len(args.check_member):
        parser.error("more --key options than --check-member options")
    status = 0
    if args.battery:
        status = run_battery(args.processes, args.extra_member)
        print(f"wrote {(GATE9 / 'battery.json').relative_to(ROOT)} and "
              f"{(GATE9 / 'battery.md').relative_to(ROOT)}")
    if args.check:
        keys = list(args.keys) + [None] * (len(args.check_member)
                                           - len(args.keys))
        extra = [(member, args.against, key)
                 for member, key in zip(args.check_member, keys)]
        report = run_check(min(2, max(1, args.processes)), extra)
        suffix = f"_{args.tag}" if args.tag else ""
        path = GATE9 / f"e1_check{suffix}.json"
        path.write_text(json.dumps(report, indent=1, default=float) + "\n")
        text = check_markdown(report)
        (GATE9 / f"e1_check{suffix}.md").write_text(text)
        print(text)
        print(f"wrote {path.relative_to(ROOT)}")
        status = status or (0 if report["verdict"] == "PASS" else 1)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
