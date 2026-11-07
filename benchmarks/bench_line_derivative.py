"""Matrix-free banded line derivative on a periodic N^3 grid: accuracy versus cost.

Each route applies its first derivative along x of a smooth 3D periodic field
as a compact relation  M d = A u / h:  the right side A u is a radius-R
stencil evaluated with periodic shifts, and the left side is one cyclic
banded Cholesky solve (banded factor plus a rank-2R wrap-around update) of M, applied to all N^2 grid lines at
once. No N x N matrix is formed. The Fourier route uses a real FFT along x.

Writes benchmarks/results/line_derivative.csv and line_derivative.md.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_line_derivative.py --reps 9
    uv run python benchmarks/bench_line_derivative.py --quick    # N = 32, 64
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import spline_analysis as sa  # noqa: E402

OUT = ROOT / "benchmarks" / "results"


def routes():
    """Return (label, formal order, mass entries, advection entries) per banded route."""
    out = []
    mass, adv = sa.compact_scheme_entries(6)
    out.append(("C6 tridiagonal compact", 6, np.asarray(mass, float), np.asarray(adv, float)))
    for p in (5, 7, 9, 11):
        mass, adv = sa.collocation_mass_and_advection_entries(p)
        out.append((f"BMC{p}", 2 * ((p + 1) // 2), mass, adv))
    return out


def field(n: int):
    """Smooth periodic test field and its exact x-derivative on [0, 2 pi)^3."""
    h = 2.0 * np.pi / n
    x = np.arange(n) * h
    X = x[:, None, None]
    Y = x[None, :, None]
    Z = x[None, None, :]
    g = np.exp(np.sin(X))
    u = g * np.cos(Y) * np.sin(2 * Z)
    du = np.cos(X) * g * np.cos(Y) * np.sin(2 * Z)
    return h, u, du


class CyclicBandedSolver:
    """Solve the symmetric positive definite cyclic banded system M x = r for many lines.

    M = T + C, with T the non-cyclic band (banded Cholesky, LAPACK pbtrf/pbtrs) and C
    the 2R wrap-around corner columns, handled by a low-rank update. Storage is
    O(nR); each solve is one banded triangular pair plus a rank-2R update.
    """

    def __init__(self, mass, n):
        r = len(mass) - 1
        self.r = r
        ab = np.zeros((r + 1, n))
        for k in range(r + 1):
            ab[r - k, k:] = mass[k]
        self.cb = cholesky_banded(ab)
        idx = np.r_[0:r, n - r:n]
        corner = np.zeros((n, 2 * r))
        for c, j in enumerate(idx):
            for i in range(n):
                d = (i - j) % n
                d = min(d, n - d)
                if d <= r and abs(i - j) > r:
                    corner[i, c] = mass[d]
        self.idx = idx
        self.z = cho_solve_banded((self.cb, False), corner)
        self.small = np.linalg.inv(np.eye(2 * r) + self.z[idx, :])
        self.nbytes = self.cb.nbytes + self.z.nbytes + self.small.nbytes

    def solve(self, rhs):
        y = cho_solve_banded((self.cb, False), rhs)
        return y - self.z @ (self.small @ y[self.idx, :])


def banded_apply(u, h, adv, lu):
    """Matrix-free compact derivative along axis 0: stencil right side plus one cyclic banded solve."""
    n, r = u.shape[0], len(adv) - 1
    up = np.concatenate([u[n - r:], u, u[:r]], axis=0)  # periodic halo of width R
    rhs = np.zeros_like(u)
    for k in range(1, r + 1):
        if adv[k] != 0.0:
            rhs += adv[k] * (up[r + k:r + k + n] - up[r - k:r - k + n])
    return lu.solve(rhs.reshape(n, -1)).reshape(u.shape) / h


def fourier_apply(u, n):
    k = np.fft.rfftfreq(n, d=1.0 / n)
    k[-1] = 0.0 if n % 2 == 0 else k[-1]
    return np.fft.irfft(1j * k[:, None, None] * np.fft.rfft(u, axis=0), n=n, axis=0)


def timed(fn, reps):
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
    return out, float(np.min(samples))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    grids = (32, 64) if args.quick else (64, 128, 192)
    rows = []
    for n in grids:
        h, u, du = field(n)
        scale = float(np.max(np.abs(du)))
        for label, order, mass, adv in routes():
            lu = CyclicBandedSolver(mass, n)
            d, t = timed(lambda: banded_apply(u, h, adv, lu), args.reps)
            stored = (len(mass) + len(adv)) * 8 + lu.nbytes
            rows.append(dict(route=label, order=order, N=n, radius=len(mass) - 1,
                             rel_max_error=float(np.max(np.abs(d - du))) / scale,
                             seconds=t, ns_per_point=1e9 * t / n**3,
                             operator_bytes=stored, dense_bytes=8 * n * n))
        d, t = timed(lambda: fourier_apply(u, n), args.reps)
        rows.append(dict(route="Fourier (rFFT)", order=0, N=n, radius=n // 2,
                         rel_max_error=float(np.max(np.abs(d - du))) / scale,
                         seconds=t, ns_per_point=1e9 * t / n**3,
                         operator_bytes=0, dense_bytes=8 * n * n))
        print(f"N = {n} done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else ""
    with open(OUT / f"line_derivative{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        f"Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, "
        f"numpy {np.__version__}, threads {os.environ.get('OMP_NUM_THREADS', 'default')}. "
        f"Best of {args.reps} repetitions.",
        "",
        "| N | route | order | radius R | rel. max error | time per 3D derivative (ms) | ns / point | operator storage (B) | dense N x N (B) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        order = "spectral" if r["order"] == 0 else str(r["order"])
        store = "-" if r["operator_bytes"] == 0 else f"{r['operator_bytes']:,}"
        lines.append(f"| {r['N']} | {r['route']} | {order} | {r['radius']} | {r['rel_max_error']:.2e} | "
                     f"{1e3 * r['seconds']:.1f} | {r['ns_per_point']:.1f} | {store} | {r['dense_bytes']:,} |")
    (OUT / f"line_derivative{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
