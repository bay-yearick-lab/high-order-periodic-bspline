"""Wall time per step of the C++ Taylor-Green solver at 128^3, per derivative route.

Prints the table from the replicated timing receipt
results/exponential_family/gate6/timing_N128.json (three sequential replicates per
route, 917 SSP-RK3 steps to t* = 9.0026, Re = 1600, 4 OpenMP threads). To re-measure
on your machine:

    make -C code/solver tgv_re1600
    OMP_NUM_THREADS=4 code/solver/tgv_re1600 128 exp 1600 9.00262644794325 9 0.2 \\
        9.00262644794325 none - --exact-time \\
        --exp-member results/exponential_family/members/E62_champion.json
    OMP_NUM_THREADS=4 code/solver/tgv_re1600 128 sl  1600 9.00262644794325 5 0.2 \\
        9.00262644794325 none - --exact-time

The solver prints the wall time per step on its VERDICT line.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "exp": "band member E(6,2), nine-diagonal",
    "sl": "spectral-like compact, pentadiagonal",
    "opt": "eighth-order tridiagonal pair",
    "bmc_p9": "BMC9, nine-diagonal",
    "bmc_p11": "BMC11, eleven-diagonal",
}


def main() -> None:
    d = json.loads((ROOT / "results/exponential_family/gate6/timing_N128.json").read_text())
    base = d["routes"]["sl"]["median_wall_per_step_s"]
    print("| route | median s / step | min | max | relative to spectral-like compact |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for key, label in LABELS.items():
        r = d["routes"][key]
        print(f"| {label} | {r['median_wall_per_step_s']:.3f} | {r['min_wall_per_step_s']:.3f} | "
              f"{r['max_wall_per_step_s']:.3f} | {r['median_wall_per_step_s'] / base:.2f} |")


if __name__ == "__main__":
    main()
