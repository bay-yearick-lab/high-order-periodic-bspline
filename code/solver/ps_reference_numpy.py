"""Fourier pseudospectral incompressible Navier-Stokes reference on numpy/scipy.

A line for line port of code/solver/ps_reference.py from torch to numpy with
scipy.fft, so the same dealiased pseudospectral solution can be produced on a
machine without torch. Every numerical convention of the torch code is kept:

  * binary64 throughout, real to complex transform over the last axis (x) and
    full complex transforms over axes 0 and 1 (z, y);
  * integer wavenumbers on the box [0, 2 pi)^3, assigned as
    KZ, KY, KX = meshgrid(fftfreq(N, 1/N), fftfreq(N, 1/N), rfftfreq(N, 1/N));
  * two thirds rule dealiasing as the cube |k_a| <= (2/3) (N/2) in each of the
    three components separately, applied to the velocity before the product and
    to the product afterwards;
  * rotational (Lamb) convection N = u x omega, whose gradient part |u|^2/2 is
    annihilated by the projection;
  * exact spectral Leray projection P = I - k k^T / |k|^2;
  * viscous term -nu |k|^2 u formed from the undealiased velocity spectrum;
  * low storage Shu-Osher SSP-RK3 with a projection after each stage;
  * dt = cfl h / umax0 with umax0 = 1 and h = 2 pi / N, the rule of
    code/solver/tgv_re1600.cpp.

Two additions carry the C++ conventions that the torch driver did not need:

  * --exact-time T lands the final state on T by the rule of tgv_re1600.cpp,
    nst = floor(T/dt (1 - 1e-12)) whole steps of length dt and, when the
    remainder T - nst dt exceeds 1e-10, one final partial step of that length;
  * --dump writes the physical velocity at the landed state in the C++ dump
    layout, u then v then w, each N^3 binary64 with x fastest.

Usage:
  ps_reference_numpy.py tgv N [Tend] [--cfl C] [--exact-time T] [--dump F]
                              [--json F] [--threads W] [--load-ic F]
                              [--dump-field t1,t2,...]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import scipy.fft as sfft

PI = math.pi


class Spectral:
    """Fourier wavenumber operators on the (z, y, x) grid of size N^3, box 2 pi."""

    def __init__(self, N, workers=1):
        self.N = N
        self.workers = workers
        # Real to complex over the last axis (x); full transform over axes 0, 1
        # (z, y). Integer wavenumbers because the box has length 2 pi.
        k1 = np.fft.fftfreq(N, d=1.0 / N).astype(np.float64)   # z, y
        kx = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float64)  # x (last)
        KZ, KY, KX = np.meshgrid(k1, k1, kx, indexing="ij")
        self.KX, self.KY, self.KZ = KX, KY, KZ
        self.K2 = KX * KX + KY * KY + KZ * KZ
        self.K2_inv = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0),
                               np.zeros_like(self.K2))
        # Two thirds rule on the half spectrum grid, as a cube in the three
        # wavenumber components separately.
        kmax = N // 2
        cut = (2.0 / 3.0) * kmax
        self.dealias = ((np.abs(KX) <= cut) & (np.abs(KY) <= cut)
                        & (np.abs(KZ) <= cut)).astype(np.float64)

    def fft(self, f):
        return sfft.rfftn(f, axes=(0, 1, 2), workers=self.workers)

    def ifft(self, F):
        return sfft.irfftn(F, s=(self.N, self.N, self.N), axes=(0, 1, 2),
                           workers=self.workers)

    def project(self, Fx, Fy, Fz):
        """Exact Leray projection: remove the k parallel component."""
        div = self.KX * Fx + self.KY * Fy + self.KZ * Fz
        p = div * self.K2_inv
        return Fx - self.KX * p, Fy - self.KY * p, Fz - self.KZ * p

    def curl(self, Fx, Fy, Fz):
        """omega = curl u in spectral space; returns spectral components."""
        i = 1j
        Wx = i * (self.KY * Fz - self.KZ * Fy)
        Wy = i * (self.KZ * Fx - self.KX * Fz)
        Wz = i * (self.KX * Fy - self.KY * Fx)
        return Wx, Wy, Wz


def rhs(sp, Ux, Uy, Uz, nu):
    """d/dt of the spectral velocity: P[ u x omega ] - nu |k|^2 u.

    Rotational (Lamb) form: the convective term u . grad u equals
    grad(|u|^2/2) - u x omega, and the gradient is annihilated by the Leray
    projection, so only the projected Lamb vector survives. Products are formed
    in physical space on dealiased fields and the product is dealiased again.
    """
    Fx, Fy, Fz = Ux * sp.dealias, Uy * sp.dealias, Uz * sp.dealias
    Wx, Wy, Wz = sp.curl(Fx, Fy, Fz)
    ux, uy, uz = sp.ifft(Fx), sp.ifft(Fy), sp.ifft(Fz)
    del Fx, Fy, Fz
    wx, wy, wz = sp.ifft(Wx), sp.ifft(Wy), sp.ifft(Wz)
    del Wx, Wy, Wz
    lx = uy * wz - uz * wy
    ly = uz * wx - ux * wz
    lz = ux * wy - uy * wx
    del ux, uy, uz, wx, wy, wz
    Lx, Ly, Lz = sp.fft(lx), sp.fft(ly), sp.fft(lz)
    del lx, ly, lz
    Lx, Ly, Lz = Lx * sp.dealias, Ly * sp.dealias, Lz * sp.dealias
    Lx, Ly, Lz = sp.project(Lx, Ly, Lz)
    Rx = Lx - nu * sp.K2 * Ux
    Ry = Ly - nu * sp.K2 * Uy
    Rz = Lz - nu * sp.K2 * Uz
    return Rx, Ry, Rz


def ssp_rk3_step(sp, Ux, Uy, Uz, nu, dt):
    """Shu-Osher SSP-RK3, matching the C++ solver's time integrator."""
    Ux0, Uy0, Uz0 = Ux, Uy, Uz
    Rx, Ry, Rz = rhs(sp, Ux0, Uy0, Uz0, nu)
    U1x = Ux0 + dt * Rx
    U1y = Uy0 + dt * Ry
    U1z = Uz0 + dt * Rz
    U1x, U1y, U1z = sp.project(U1x, U1y, U1z)
    Rx, Ry, Rz = rhs(sp, U1x, U1y, U1z, nu)
    U2x = 0.75 * Ux0 + 0.25 * (U1x + dt * Rx)
    U2y = 0.75 * Uy0 + 0.25 * (U1y + dt * Ry)
    U2z = 0.75 * Uz0 + 0.25 * (U1z + dt * Rz)
    U2x, U2y, U2z = sp.project(U2x, U2y, U2z)
    Rx, Ry, Rz = rhs(sp, U2x, U2y, U2z, nu)
    Ux = (1.0 / 3.0) * Ux0 + (2.0 / 3.0) * (U2x + dt * Rx)
    Uy = (1.0 / 3.0) * Uy0 + (2.0 / 3.0) * (U2y + dt * Ry)
    Uz = (1.0 / 3.0) * Uz0 + (2.0 / 3.0) * (U2z + dt * Rz)
    Ux, Uy, Uz = sp.project(Ux, Uy, Uz)
    return Ux, Uy, Uz


# ----------------------------------------------------------------------------
# Diagnostics (means over the periodic box, matching the C++ conventions).
# ----------------------------------------------------------------------------
def kinetic_energy(sp, Ux, Uy, Uz):
    ux, uy, uz = sp.ifft(Ux), sp.ifft(Uy), sp.ifft(Uz)
    return 0.5 * float((ux * ux + uy * uy + uz * uz).mean())


def mean_enstrophy(sp, Ux, Uy, Uz):
    """<1/2 |omega|^2> over the box (matches tgv_re1600.cpp enstrophy())."""
    Wx, Wy, Wz = sp.curl(Ux, Uy, Uz)
    wx, wy, wz = sp.ifft(Wx), sp.ifft(Wy), sp.ifft(Wz)
    return 0.5 * float((wx * wx + wy * wy + wz * wz).mean())


def max_divergence(sp, Ux, Uy, Uz):
    div = 1j * (sp.KX * Ux + sp.KY * Uy + sp.KZ * Uz)
    d = sp.ifft(div)
    return float(np.abs(d).max())


# ----------------------------------------------------------------------------
# Initial conditions.
# ----------------------------------------------------------------------------
def tgv_ic(sp):
    """Taylor-Green: u=sin x cos y cos z, v=-cos x sin y cos z, w=0."""
    N = sp.N
    h = 2 * PI / N
    idx = np.arange(N, dtype=np.float64)
    Z, Y, X = np.meshgrid(idx * h, idx * h, idx * h, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    Ux, Uy, Uz = sp.fft(u), sp.fft(v), sp.fft(w)
    return sp.project(Ux, Uy, Uz)


def load_ic(sp, path):
    """Load a t=0 velocity dump in the C++ layout and Leray project it.

    The dump is N^3 binary64 for u, then v, then w, with x fastest, so the
    C order reshape to (N, N, N) is indexed [z, y, x].
    """
    N = sp.N
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size != 3 * N**3:
        raise ValueError(f"IC file {path} has {raw.size} doubles, expected {3 * N**3}")
    block = N**3
    u = np.ascontiguousarray(raw[0:block].reshape(N, N, N))
    v = np.ascontiguousarray(raw[block:2 * block].reshape(N, N, N))
    w = np.ascontiguousarray(raw[2 * block:3 * block].reshape(N, N, N))
    Ux, Uy, Uz = sp.fft(u), sp.fft(v), sp.fft(w)
    return sp.project(Ux, Uy, Uz)


# ----------------------------------------------------------------------------
# Drivers.
# ----------------------------------------------------------------------------
def dump_physical_field(sp, Ux, Uy, Uz, path):
    """Write the physical field in the C++ dump layout (u, v, w; x fastest)."""
    with open(path, "wb") as f:
        for F in (Ux, Uy, Uz):
            sp.ifft(F).astype(np.float64).tofile(f)


def step_plan(Tend, dt, exact_time):
    """Return (nst, dt_last, ntot) for the requested landing rule.

    Base rule: a whole number of steps, nst = round(Tend/dt). Exact time rule of
    tgv_re1600.cpp: nst = floor(Tend/dt (1 - 1e-12)) whole steps of length dt
    and one final step of length Tend - nst dt when that remainder is at least
    1e-10, so the state lands on Tend to round off.
    """
    if exact_time:
        nst = int(math.floor(Tend / dt * (1.0 - 1e-12)))
        dt_last = Tend - nst * dt
        if dt_last < 1e-10:
            dt_last = 0.0
    else:
        nst = int(round(Tend / dt))
        dt_last = 0.0
    ntot = nst + (1 if dt_last > 0.0 else 0)
    return nst, dt_last, ntot


def run_tgv(N, Tend, cfl, out_json, dump_times=None, exact_time=False,
            dump_path=None, workers=1, ic_path=None, dump_dir="psref"):
    Re = 1600.0
    nu = 1.0 / Re
    sp = Spectral(N, workers=workers)
    if ic_path:
        Ux, Uy, Uz = load_ic(sp, ic_path)
    else:
        Ux, Uy, Uz = tgv_ic(sp)
    h = 2 * PI / N
    dt = cfl * h / 1.0  # umax0 = 1, matches tgv_re1600.cpp
    nst, dt_last, ntot = step_plan(Tend, dt, exact_time)
    rec = max(1, ntot // 200)
    pending_dumps = sorted(dump_times or [])
    dumped = []
    print(f"[TGV] N={N} Re={Re:.0f} nu={nu:.3e} dt={dt:.10f} steps={ntot} "
          f"dt_last={dt_last:.6e} -> t={Tend!r}", flush=True)
    E0 = kinetic_energy(sp, Ux, Uy, Uz)
    ts, Es, ens_l, eps_ens_l, eps_dEdt_l, div_l = [], [], [], [], [], []
    Eprev, tprev, have_prev = E0, 0.0, False
    t0 = time.time()
    for n in range(ntot + 1):
        t = n * dt if n <= nst else Tend
        if exact_time and n == ntot:
            t = Tend
        if n % rec == 0 or n == ntot:
            E = kinetic_energy(sp, Ux, Uy, Uz)
            ens = mean_enstrophy(sp, Ux, Uy, Uz)
            eps_ens = 2.0 * nu * ens  # nu <|omega|^2>
            eps_dEdt = -(E - Eprev) / (t - tprev) if (have_prev and t > tprev) else 0.0
            dv = max_divergence(sp, Ux, Uy, Uz)
            ts.append(t); Es.append(E); ens_l.append(ens)
            eps_ens_l.append(eps_ens); eps_dEdt_l.append(eps_dEdt); div_l.append(dv)
            if n % (rec * 10) == 0 or n == ntot:
                print(f"  t={t:5.2f} E/E0={E / E0:.5f} eps_ens={eps_ens:.4e} "
                      f"eps_-dEdt={eps_dEdt:.4e} div={dv:.1e}", flush=True)
            Eprev, tprev, have_prev = E, t, True
        while pending_dumps and t >= pending_dumps[0] - 1e-12:
            treq = pending_dumps.pop(0)
            path = f"{dump_dir}/tgvfield_ps_N{N}_t{treq:g}.bin"
            dump_physical_field(sp, Ux, Uy, Uz, path)
            dumped.append({"t_requested": treq, "t_actual": t, "file": path,
                           "step": n})
            print(f"  [dump] t={t:.10f} step={n} -> {path}", flush=True)
        if n == ntot and dump_path:
            dump_physical_field(sp, Ux, Uy, Uz, dump_path)
            dumped.append({"t_requested": Tend, "t_actual": t, "file": dump_path,
                           "step": n})
            print(f"  [dump] landed t={t:.14f} step={n} -> {dump_path}", flush=True)
        if n < ntot:
            Ux, Uy, Uz = ssp_rk3_step(sp, Ux, Uy, Uz, nu,
                                      dt if n < nst else dt_last)
    wall = time.time() - t0
    peak_eps = float(np.max(eps_ens_l))
    t_peak = float(ts[int(np.argmax(eps_ens_l))])
    print(f"[TGV] done ({wall:.1f}s). peak eps_ens={peak_eps:.6e} at t={t_peak:.2f}",
          flush=True)
    result = {
        "case": "tgv", "N": N, "Re": Re, "dt": dt, "steps": ntot, "Tend": Tend,
        "t": ts, "E": Es, "enstrophy": ens_l, "eps_ens": eps_ens_l,
        "eps_dEdt": eps_dEdt_l, "divmax": div_l,
        "peak_eps_ens": peak_eps, "t_peak": t_peak, "wall_s": wall,
        "device": "cpu (numpy/scipy.fft)", "field_dumps": dumped,
        "cfl": cfl, "exact_time": bool(exact_time), "whole_steps": nst,
        "dt_last": dt_last, "threads": workers,
        "ic": str(ic_path) if ic_path else "analytic",
    }
    if out_json:
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[TGV] wrote {out_json}", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="numpy Fourier pseudospectral NS reference")
    sub = ap.add_subparsers(dest="case", required=True)
    pt = sub.add_parser("tgv")
    pt.add_argument("N", type=int)
    pt.add_argument("Tend", type=float, nargs="?", default=20.0)
    pt.add_argument("--cfl", type=float, default=0.30)
    pt.add_argument("--exact-time", type=float, default=None,
                    help="land the final state on this time by the partial "
                         "final step rule of tgv_re1600.cpp")
    pt.add_argument("--dump", type=str, default=None,
                    help="write the landed velocity field to this path")
    pt.add_argument("--json", default=None)
    pt.add_argument("--threads", type=int, default=1,
                    help="scipy.fft worker threads")
    pt.add_argument("--load-ic", type=str, default=None,
                    help="read the t=0 velocity from a C++ layout dump")
    pt.add_argument("--dump-field", type=str, default=None,
                    help="comma separated times; write u,v,w at the first step "
                         "reaching each time, in the torch driver's naming")
    pt.add_argument("--dump-dir", type=str, default="psref",
                    help="directory for the --dump-field files; the default "
                         "reproduces the torch driver's location")
    args = ap.parse_args(argv)
    dumps = ([float(s) for s in args.dump_field.split(",")]
             if args.dump_field else None)
    exact = args.exact_time is not None
    Tend = args.exact_time if exact else args.Tend
    run_tgv(args.N, Tend, args.cfl, args.json, dump_times=dumps,
            exact_time=exact, dump_path=args.dump, workers=args.threads,
            ic_path=args.load_ic, dump_dir=args.dump_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
