"""Gate 9, E0: collect the core-6 cell receipts of the Gate 8 producer and field the best member.

Reads every results/exponential_family/gate8/dominance_m6_<cell>.json written by
gate8_dominance.py (one invocation per cell, contract E0 of gate9/contract.md),
ranks the verified members against SLC1992 and TunedQ8 per budget by the
smallest margin, applies the Gate 8 dominance predicate (smallest margin >= 0,
coupled ratio <= 1, b and kappa_2 b positive by the grid-plus-Lipschitz bound,
conditioning within budget) and writes the dominating members in the Gate 8
member schema (gate8_dominance.write_members format) to
results/exponential_family/gate8/members/X9_<key>.json, plus
results/exponential_family/gate9/e0_summary.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gate8_dominance as gdom  # noqa: E402

GATE8 = ROOT / "results" / "exponential_family" / "gate8"
GATE9 = ROOT / "results" / "exponential_family" / "gate9"
MEMBERS = GATE8 / "members"


def main() -> int:
    files = sorted(GATE8.glob("dominance_m6_*.json"))
    rows = []
    for path in files:
        cell = path.stem.replace("dominance_m6_", "")
        data = json.loads(path.read_text())
        for key, rec in data["summary"].items():
            if rec["yardstick"] is not None or key.endswith("|C4"):
                continue  # the C4 restriction repeats the q = 2 cells
            ver = data["verification"][key]
            vm = ver["metrics"]
            budget = data["budgets"][rec["budget"]]
            cond = vm["cond"] if "cond" in vm else ver["record"]["cond"]
            dom = gdom.dominates(rec, ver) and cond <= budget * (1.0 + gdom.BUDGET_SLACK)
            rows.append({"cell": cell, "key": key, "budget": rec["budget"], "C": budget,
                         "target": rec["target_name"], "smoothness": rec["smoothness"],
                         "label": ver["label"], "cond": cond, "value": rec["value"],
                         "margins": rec["margins"], "binding": min(rec["margins"], key=rec["margins"].get),
                         "ec_ratio": vm["ec_max_B0"] / (gdom.COUPLED_FACTOR * max(vm["e1_max_B0"], vm["e2_max_B0"])),
                         "b_certified": ver["positivity"]["b_certified"],
                         "kappa2b_certified": ver["positivity"]["kappa2b_certified"],
                         "decimal_deviation": ver["routes"]["decimal_deviation"],
                         "residue_deviation": ver["routes"]["residue_deviation"],
                         "dominates": bool(dom), "seconds": rec.get("seconds"),
                         "R1_1e-2": vm["reach_d1_0.01"], "R1_1e-3": vm["reach_d1_0.001"],
                         "R1_1e-4": vm["reach_d1_0.0001"], "G": vm["gv_reach_1e-2"],
                         "E1max": vm["e1_max_B0"], "E2max": vm["e2_max_B0"],
                         "order_d1": vm["order_d1_fitted"], "order_d2": vm["order_d2_fitted"],
                         "_rec": rec, "_ver": ver})
    rows.sort(key=lambda r: (r["budget"], r["target"], -r["value"]))
    written = []
    # Every run rewrites the current best per (budget, target). A member file that
    # already carries the additive evaluator block (eval_*, gate9_evaluator.py) keeps
    # it when its lattice samples are unchanged, so a rerun never strips it.
    preserved = {}
    for stale in MEMBERS.glob("X9_m6_*.json"):
        old = json.loads(stale.read_text())
        preserved[stale.name] = {"beta": old.get("beta"),
                                 "extra": {k: v for k, v in old.items() if k.startswith("eval_")}}
        stale.unlink()
    best = {}
    for r in rows:
        k = (r["budget"], r["target"])
        if k not in best:
            best[k] = r
    for (budget, target), r in best.items():
        if not r["dominates"]:
            continue
        name = f"X9_m6_{budget}_{target}_{r['cell']}"
        record = dict(r["_ver"]["record"])
        record["name"] = name
        record["dominates"] = target
        record["budget"] = budget
        record["margins"] = r["_rec"].get("margins")
        record["verification"] = {"routes": {k: v for k, v in r["_ver"]["routes"].items() if k != "beta_digits"},
                                  "positivity": r["_ver"]["positivity"],
                                  "sufficient_conditions": r["_ver"]["sufficient_conditions"],
                                  "metrics": r["_ver"]["metrics"]}
        record["source"] = f"results/exponential_family/gate8/dominance_m6_{r['cell']}.json summary[{r['key']}]"
        kept = preserved.get(f"{name}.json")
        if kept and kept["beta"] == record["beta"]:
            record.update(kept["extra"])
        MEMBERS.mkdir(parents=True, exist_ok=True)
        (MEMBERS / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        written.append(name)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    GATE9.mkdir(parents=True, exist_ok=True)
    (GATE9 / "e0_summary.json").write_text(json.dumps({"cells": [f.stem for f in files], "rows": public,
                                                        "written_members": written}, indent=1, default=float) + "\n")
    lines = ["# Gate 9 E0: core-6 cells at bandwidth 9 (receipts dominance_m6_<cell>.json)", "",
             f"Cells finished: {', '.join(f.stem.replace('dominance_m6_', '') for f in files)}.", "",
             "| cell | budget | target | member | C^k | cond | smallest margin | binding | R1(1e-2) | R1(1e-3) | R1(1e-4) | G | E1max | E2max | ec/2max | orders | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in public:
        lines.append(f"| {r['cell']} | {r['budget']} | {r['target']} | {r['label']} | {r['smoothness']} | {r['cond']:.3f} | "
                     f"{r['value']:+.4f} | {r['binding']} | {r['R1_1e-2']:.4f} | {r['R1_1e-3']:.4f} | {r['R1_1e-4']:.4f} | "
                     f"{r['G']:.4f} | {r['E1max']:.3e} | {r['E2max']:.3e} | {r['ec_ratio']:.3f} | "
                     f"{r['order_d1']:.2f}/{r['order_d2']:.2f} | {'DOMINATES' if r['dominates'] else 'no'} |")
    lines += ["", f"Members written: {', '.join(written) if written else 'none'}."]
    (GATE9 / "e0_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
