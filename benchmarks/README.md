# Benchmarks

## Matrix-free banded line derivative

`bench_line_derivative.py` applies each route's first derivative along one axis of a smooth $N^3$ periodic field, $u = e^{\sin x}\cos y \sin 2z$. The right side is a radius-$R$ stencil on a periodic halo; the left side is one cyclic banded Cholesky solve (banded factor plus a rank-$2R$ low-rank update for the wrap-around corners) applied to all $N^2$ lines at once. No $N \times N$ matrix is formed, and the stored operator is $O(NR)$.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_line_derivative.py --reps 9
```

Machine: x86_64, 20 logical cores, numpy 2.5.4, threads 1. Best of 9 repetitions.

| N | route | order | radius R | rel. max error | time per 3D derivative (ms) | ns / point | operator storage (B) | dense N x N (B) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | C6 tridiagonal compact | 6 | 1 | 7.53e-08 | 5.8 | 22.3 | 2,120 | 32,768 |
| 64 | BMC5 | 6 | 2 | 3.19e-08 | 5.9 | 22.5 | 3,760 | 32,768 |
| 64 | BMC7 | 8 | 3 | 1.57e-10 | 6.2 | 23.7 | 5,472 | 32,768 |
| 64 | BMC9 | 10 | 4 | 1.05e-12 | 7.4 | 28.4 | 7,248 | 32,768 |
| 64 | BMC11 | 12 | 5 | 1.69e-14 | 6.9 | 26.3 | 9,088 | 32,768 |
| 64 | Fourier (rFFT) | spectral | 32 | 1.04e-14 | 1.3 | 5.1 | - | 32,768 |
| 128 | C6 tridiagonal compact | 6 | 1 | 1.16e-09 | 65.4 | 31.2 | 4,168 | 131,072 |
| 128 | BMC5 | 6 | 2 | 4.86e-10 | 66.0 | 31.5 | 7,344 | 131,072 |
| 128 | BMC7 | 8 | 3 | 5.87e-13 | 70.9 | 33.8 | 10,592 | 131,072 |
| 128 | BMC9 | 10 | 4 | 1.99e-14 | 84.4 | 40.2 | 13,904 | 131,072 |
| 128 | BMC11 | 12 | 5 | 2.32e-14 | 79.3 | 37.8 | 17,280 | 131,072 |
| 128 | Fourier (rFFT) | spectral | 64 | 2.80e-14 | 19.0 | 9.1 | - | 131,072 |
| 192 | C6 tridiagonal compact | 6 | 1 | 1.02e-10 | 264.0 | 37.3 | 6,216 | 294,912 |
| 192 | BMC5 | 6 | 2 | 4.25e-11 | 263.2 | 37.2 | 10,928 | 294,912 |
| 192 | BMC7 | 8 | 3 | 4.51e-14 | 299.6 | 42.3 | 15,712 | 294,912 |
| 192 | BMC9 | 10 | 4 | 3.38e-14 | 348.6 | 49.3 | 20,560 | 294,912 |
| 192 | BMC11 | 12 | 5 | 4.08e-14 | 348.2 | 49.2 | 25,472 | 294,912 |
| 192 | Fourier (rFFT) | spectral | 96 | 4.74e-14 | 81.8 | 11.6 | - | 294,912 |

The cost per point grows slowly with the radius $R$ while the error falls by orders of magnitude: at $N = 64$, BMC9 is 7.2e4 times more accurate than the sixth-order tridiagonal compact scheme for 1.27 times the time, and at $N = 128$ BMC9 and BMC11 reach round-off.

## C++ Taylor-Green solver, wall time per step

`tgv_step_timing.py` prints the replicated timing receipt `results/exponential_family/gate6/timing_N128.json`: $128^3$, Re = 1600, CFL 0.2, 917 SSP-RK3 steps to $t^* = 9.0026$, three sequential replicates per route, 4 OpenMP threads on a 20-core x86_64 workstation. Every route shares the same solver and differs only in its banded derivative pair.

```bash
uv run python benchmarks/tgv_step_timing.py
```

| route | median s / step | min | max | relative to spectral-like compact |
| --- | ---: | ---: | ---: | ---: |
| band member E(6,2), nine-diagonal | 1.050 | 1.006 | 1.075 | 0.87 |
| spectral-like compact, pentadiagonal | 1.210 | 1.199 | 1.283 | 1.00 |
| eighth-order tridiagonal pair | 0.896 | 0.891 | 0.952 | 0.74 |
| BMC9, nine-diagonal | 1.011 | 1.008 | 1.073 | 0.84 |
| BMC11, eleven-diagonal | 1.196 | 1.167 | 1.239 | 0.99 |

The script docstring gives the solver command lines that regenerate the receipt.
