"""Write one exponential-family member to the JSON file the DNS solver reads.

The member E_{m,r}(omega_1, ..., omega_r) has degree p = m + 2 r - 1 and its
generator is supported on [-(p+1)/2, (p+1)/2], so the lattice samples

    beta(n), beta'(n), beta''(n),      |n| <= R = floor((p+1)/2),

determine the whole periodic route: the left side is the symmetric cyclic
banded matrix with entries beta(n), the first-derivative right side is the
antisymmetric stencil (1/h) beta'(n), and the second-derivative right side is
the symmetric stencil (1/h^2) beta''(n).

Because beta is even, beta' odd and beta'' even, only the one-sided samples
n = 0, 1, ..., R are stored. The symbols follow from them as

    b(theta)   = beta(0)   + 2 sum_{n=1}^{R} beta(n)   cos(n theta),
    a(theta)   =             2 sum_{n=1}^{R} beta'(n)  sin(n theta),
    c(theta)   = beta''(0) + 2 sum_{n=1}^{R} beta''(n) cos(n theta),

so that on the grid mode exp(i n theta) the applied first derivative has
eigenvalue -(i/h) a(theta) / b(theta) and the applied second derivative has
eigenvalue (1/h^2) c(theta) / b(theta). The recorded "cond" is the symbol
level left side condition number max b / min b over theta in [0, pi].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .family import ExponentialGenerator
except ImportError:  # direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from family import ExponentialGenerator


def member_record(m: int, omegas, name: str | None = None) -> dict:
    """Return the JSON record of E_{m,r}(omegas) with one-sided lattice samples."""
    generator = ExponentialGenerator(m, omegas)
    samples = generator.integer_samples()
    radius = generator.nmax
    center = int(np.searchsorted(samples.n, 0))
    if samples.n[center] != 0:
        raise ValueError("The sample lattice does not contain n = 0.")
    one_sided = slice(center, center + radius + 1)
    beta = np.asarray(samples.beta[one_sided], dtype=float)
    dbeta = np.asarray(samples.dbeta[one_sided], dtype=float)
    ddbeta = np.asarray(samples.ddbeta[one_sided], dtype=float)
    if beta.size != radius + 1:
        raise ValueError("The one-sided sample block has the wrong length.")
    return {
        "name": name if name is not None else generator.label(),
        "m": generator.m,
        "r": generator.r,
        "omegas": [float(w) for w in generator.omegas],
        "p": generator.p,
        "radius": int(radius),
        "cond": float(generator.cond()["cond"]),
        "beta": [float(v) for v in beta],
        "dbeta": [float(v) for v in dbeta],
        "ddbeta": [float(v) for v in ddbeta],
    }


def write_member(m: int, omegas, path: Path, name: str | None = None) -> dict:
    """Write the member record to path and return it."""
    record = member_record(m, omegas, name=name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m", type=int, required=True,
                        help="polynomial core multiplicity m")
    parser.add_argument("--omega", type=float, action="append", default=[],
                        help="one exponential frequency; repeat for r > 1")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--name", type=str, default=None,
                        help="member name; defaults to the generator label")
    args = parser.parse_args(argv)
    record = write_member(args.m, args.omega, args.out, name=args.name)
    print(f"{record['name']}: p={record['p']} radius={record['radius']} "
          f"cond={record['cond']:.6f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
