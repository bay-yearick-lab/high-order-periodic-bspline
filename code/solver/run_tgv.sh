#!/usr/bin/env bash
# Taylor-Green Re=1600 sweep at a fixed 64^3 grid: the BMC degree ladder
# (p=5,7,9 -> formal order 6,8,10) against the sixth-order compact C6, the DRP
# stencil and explicit central differences. Same solver, route/degree
# swapped each run; each run also dumps the velocity field near the peak (t=9) for
# the energy spectrum and vorticity slices.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
T=20
echo "### BMC degree ladder at fixed 64^3 (p = 5,7,9)"
for p in 5 7 9; do ./tgv_re1600 64 bmc 1600 $T $p 0.20 9.0; done
echo "### fixed-order baselines (C6, DRP, explicit E6/E4/E2)"
for r in c6 drp fd6 fd4 fd2; do ./tgv_re1600 64 $r 1600 $T 5 0.25 9.0; done
echo "### TGV sweep complete"
