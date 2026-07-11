# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Independent Fourier pseudospectral incompressible Navier-Stokes reference.

This is a second, method-independent solver that supplies the reference fields
for the B-spline Taylor-Green and vortex-reconnection runs. It shares no code with the C++ collocated B-spline
solver: derivatives are exact in Fourier space, the pressure projection is the
exact Leray projector, and time advance is the same low-storage SSP-RK3 used by
the C++ code so the two differ only in the SPATIAL operator.

Standard classical pseudospectral method on the periodic torus [0, 2pi)^3:
  * velocity form, rotational (Lamb) convection  N = u x omega  (energy conserving),
  * 2/3-rule dealiasing of the nonlinear product,
  * exact spectral Leray projection  P = I - k k^T / |k|^2,
  * viscous term  -nu |k|^2 u  treated with the same explicit RK3 as convection
    (advective CFL is the binding limit at these Re; matches the C++ SSP-RK3).

Array backend: PyTorch on the GPU (torch.fft, which dispatches to cuFFT), in
float64 with TF32 disabled. Falls back to CPU torch if CUDA is absent.

Grid convention matches the C++ dump layout (bm3d.h Field3D): the flat buffer is
x fastest, then y, then z, i.e. index (i,j,k) -> (k*ny + j)*nx + i. Reshaped to a
C-order (N, N, N) array this is [z, y, x], so FFT axis 0 <-> z, axis 1 <-> y,
axis 2 <-> x. Wavenumbers are assigned accordingly.

Usage:
  uv run ps_reference.py tgv    N [Tend] [--json out.json]     # validation gate
  uv run ps_reference.py recon  N icfile [Tend] [--icN Nsrc] [--json out.json]
      (--icN: IC dump resolution; if < N the field is refined to the run grid by
       exact spectral zero-padding)
"""

import argparse
import json
import math
import sys
import time

import numpy as np
import torch

PI = math.pi


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    sys.stderr.write("[ps_reference] WARNING: CUDA unavailable, running on CPU\n")
    return torch.device("cpu")


class Spectral:
    """Fourier wavenumber operators on the (z, y, x) grid of size N^3, box 2pi."""

    def __init__(self, N, device, dtype=torch.float64):
        self.N = N
        self.device = device
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        # rfft over the last axis (x); full fft over axes 0,1 (z,y).
        # integer wavenumbers on a 2pi box.
        k1 = torch.fft.fftfreq(N, d=1.0 / N).to(device=device, dtype=dtype)   # z, y
        kx = torch.fft.rfftfreq(N, d=1.0 / N).to(device=device, dtype=dtype)  # x (last)
        KZ, KY, KX = torch.meshgrid(k1, k1, kx, indexing="ij")
        self.KX, self.KY, self.KZ = KX, KY, KZ
        self.K2 = KX * KX + KY * KY + KZ * KZ
        self.K2_inv = torch.where(self.K2 > 0, 1.0 / self.K2, torch.zeros_like(self.K2))
        # 2/3-rule dealiasing mask on the half-spectrum grid.
        kmax = N // 2
        cut = (2.0 / 3.0) * kmax
        self.dealias = ((KX.abs() <= cut) & (KY.abs() <= cut) & (KZ.abs() <= cut)).to(dtype)

    def fft(self, f):
        return torch.fft.rfftn(f, dim=(0, 1, 2))

    def ifft(self, F):
        return torch.fft.irfftn(F, s=(self.N, self.N, self.N), dim=(0, 1, 2))

    def project(self, Fx, Fy, Fz):
        """Exact Leray projection: remove the k-parallel component (spectral div-free)."""
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

    Rotational (Lamb) form: convection = omega x u absorbed into a gradient plus
    u x omega; the gradient is annihilated by the projection, so only the
    projected u x omega survives. Products formed in physical space on dealiased
    fields (2/3 rule), then transformed back.
    """
    # dealias the velocity spectrum before forming products
    Fx, Fy, Fz = Ux * sp.dealias, Uy * sp.dealias, Uz * sp.dealias
    Wx, Wy, Wz = sp.curl(Fx, Fy, Fz)
    # to physical space (free the spectral copies as we go to cap peak memory)
    ux, uy, uz = sp.ifft(Fx), sp.ifft(Fy), sp.ifft(Fz)
    del Fx, Fy, Fz
    wx, wy, wz = sp.ifft(Wx), sp.ifft(Wy), sp.ifft(Wz)
    del Wx, Wy, Wz
    # Lamb vector  u x omega
    lx = uy * wz - uz * wy
    ly = uz * wx - ux * wz
    lz = ux * wy - uy * wx
    del ux, uy, uz, wx, wy, wz
    Lx, Ly, Lz = sp.fft(lx), sp.fft(ly), sp.fft(lz)
    del lx, ly, lz
    # dealias the product too, then project (annihilates the |u|^2/2 gradient)
    Lx, Ly, Lz = Lx * sp.dealias, Ly * sp.dealias, Lz * sp.dealias
    Lx, Ly, Lz = sp.project(Lx, Ly, Lz)
    # viscous term (exact spectral Laplacian)
    Rx = Lx - nu * sp.K2 * Ux
    Ry = Ly - nu * sp.K2 * Uy
    Rz = Lz - nu * sp.K2 * Uz
    return Rx, Ry, Rz


def ssp_rk3_step(sp, Ux, Uy, Uz, nu, dt):
    """Shu-Osher SSP-RK3, matching the C++ solver's time integrator."""
    Ux0, Uy0, Uz0 = Ux, Uy, Uz
    # stage 1
    Rx, Ry, Rz = rhs(sp, Ux0, Uy0, Uz0, nu)
    U1x = Ux0 + dt * Rx
    U1y = Uy0 + dt * Ry
    U1z = Uz0 + dt * Rz
    U1x, U1y, U1z = sp.project(U1x, U1y, U1z)
    # stage 2
    Rx, Ry, Rz = rhs(sp, U1x, U1y, U1z, nu)
    U2x = 0.75 * Ux0 + 0.25 * (U1x + dt * Rx)
    U2y = 0.75 * Uy0 + 0.25 * (U1y + dt * Ry)
    U2z = 0.75 * Uz0 + 0.25 * (U1z + dt * Rz)
    U2x, U2y, U2z = sp.project(U2x, U2y, U2z)
    # stage 3
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
    return 0.5 * float((ux * ux + uy * uy + uz * uz).mean().cpu())


def mean_enstrophy(sp, Ux, Uy, Uz):
    """<1/2 |omega|^2> over the box (matches tgv_re1600.cpp enstrophy())."""
    Wx, Wy, Wz = sp.curl(Ux, Uy, Uz)
    wx, wy, wz = sp.ifft(Wx), sp.ifft(Wy), sp.ifft(Wz)
    return 0.5 * float((wx * wx + wy * wy + wz * wz).mean().cpu())


def peak_vorticity(sp, Ux, Uy, Uz):
    Wx, Wy, Wz = sp.curl(Ux, Uy, Uz)
    wx, wy, wz = sp.ifft(Wx), sp.ifft(Wy), sp.ifft(Wz)
    mag2 = wx * wx + wy * wy + wz * wz
    return float(torch.sqrt(mag2.max()).cpu())


def max_divergence(sp, Ux, Uy, Uz):
    div = 1j * (sp.KX * Ux + sp.KY * Uy + sp.KZ * Uz)
    d = sp.ifft(div)
    return float(d.abs().max().cpu())


# ----------------------------------------------------------------------------
# Initial conditions.
# ----------------------------------------------------------------------------
def tgv_ic(sp, device):
    """Taylor-Green: u=sin x cos y cos z, v=-cos x sin y cos z, w=0, on [0,2pi)^3."""
    N = sp.N
    h = 2 * PI / N
    idx = torch.arange(N, device=device, dtype=sp.dtype)
    # grid arrays indexed [z, y, x] to match the reshaped C++ layout
    Z, Y, X = torch.meshgrid(idx * h, idx * h, idx * h, indexing="ij")
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    Ux, Uy, Uz = sp.fft(u), sp.fft(v), sp.fft(w)
    return sp.project(Ux, Uy, Uz)


def spectral_upsample_field(f, Nd):
    """Exact spectral zero-padding of a periodic field from Ns^3 to Nd^3 (Nd > Ns).

    Copies every Fourier mode strictly below the source Nyquist into the finer
    spectrum and zeroes the source Nyquist planes (round-off content for the
    smooth reconnection tubes). Rescales for the FFT normalization so the result
    is the trigonometric interpolant of the source field on the fine grid: the
    identical flow, refined exactly.
    """
    Ns = f.shape[0]
    if Nd == Ns:
        return f
    if Nd < Ns:
        raise ValueError(f"upsample only: Nd={Nd} < Ns={Ns}")
    hs = Ns // 2
    F = np.fft.rfftn(f)
    Fd = np.zeros((Nd, Nd, Nd // 2 + 1), dtype=complex)
    pos = slice(0, hs)                                            # modes 0 .. hs-1
    negs, negd = slice(Ns - hs + 1, Ns), slice(Nd - hs + 1, Nd)   # modes -hs+1 .. -1
    for s0, d0 in ((pos, pos), (negs, negd)):
        for s1, d1 in ((pos, pos), (negs, negd)):
            Fd[d0, d1, 0:hs] = F[s0, s1, 0:hs]
    Fd *= (Nd / Ns) ** 3
    return np.fft.irfftn(Fd, s=(Nd, Nd, Nd), axes=(0, 1, 2))


def load_recon_ic(sp, device, icfile, icN=None):
    """Load the bit-identical t=0 velocity dump from the C++ reconnection solver.

    The dump is N^3 float64 for u, then v, then w, in the x-fastest layout. Read
    as C-order (N, N, N) = [z, y, x]. Spectrally project (Leray) before use: the
    C++ field is divergence-free in the B-spline sense (max|div| ~ 1e-15), so the
    projection is a round-off-level correction that only enforces the constraint
    in the pseudospectral norm.

    If ``icN`` is given and smaller than the run resolution, the dump is read at
    icN^3 and refined to the run grid by exact spectral zero-padding
    (spectral_upsample_field): the identical flow on a finer grid.
    """
    N = sp.N
    Ns = icN if icN else N
    raw = np.fromfile(icfile, dtype=np.float64)
    if raw.size != 3 * Ns * Ns * Ns:
        raise ValueError(
            f"IC file {icfile} has {raw.size} doubles, expected {3 * Ns**3} for N={Ns}"
        )
    u = raw[0 : Ns**3].reshape(Ns, Ns, Ns)
    v = raw[Ns**3 : 2 * Ns**3].reshape(Ns, Ns, Ns)
    w = raw[2 * Ns**3 : 3 * Ns**3].reshape(Ns, Ns, Ns)
    if Ns != N:
        u = spectral_upsample_field(u, N)
        v = spectral_upsample_field(v, N)
        w = spectral_upsample_field(w, N)
        print(f"[RECON] IC spectrally upsampled {Ns}^3 -> {N}^3 (exact zero-padding)",
              flush=True)
    tu = torch.from_numpy(np.ascontiguousarray(u)).to(device=device, dtype=sp.dtype)
    tv = torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=sp.dtype)
    tw = torch.from_numpy(np.ascontiguousarray(w)).to(device=device, dtype=sp.dtype)
    Ux, Uy, Uz = sp.fft(tu), sp.fft(tv), sp.fft(tw)
    return sp.project(Ux, Uy, Uz)


# ----------------------------------------------------------------------------
# Drivers.
# ----------------------------------------------------------------------------
def dump_physical_field(sp, Ux, Uy, Uz, path):
    """Write the physical-space field in the C++ dump layout (u,v,w; x-fastest)."""
    with open(path, "wb") as f:
        for F in (Ux, Uy, Uz):
            sp.ifft(F).cpu().numpy().astype(np.float64).tofile(f)


def run_tgv(N, Tend, cfl, out_json, dump_times=None):
    device = get_device()
    Re = 1600.0
    nu = 1.0 / Re
    sp = Spectral(N, device)
    Ux, Uy, Uz = tgv_ic(sp, device)
    h = 2 * PI / N
    dt = cfl * h / 1.0  # umax0 = 1, matches tgv_re1600.cpp
    nst = int(round(Tend / dt))
    rec = max(1, nst // 200)
    pending_dumps = sorted(dump_times or [])
    dumped = []
    print(f"[TGV] N={N} Re={Re:.0f} nu={nu:.3e} dt={dt:.5f} steps={nst} -> t={Tend}", flush=True)
    E0 = kinetic_energy(sp, Ux, Uy, Uz)
    ts, Es, ens_l, eps_ens_l, eps_dEdt_l, div_l = [], [], [], [], [], []
    Eprev, tprev, have_prev = E0, 0.0, False
    t0 = time.time()
    for n in range(nst + 1):
        if n % rec == 0 or n == nst:
            t = n * dt
            E = kinetic_energy(sp, Ux, Uy, Uz)
            ens = mean_enstrophy(sp, Ux, Uy, Uz)
            eps_ens = 2.0 * nu * ens  # nu <|omega|^2>
            eps_dEdt = -(E - Eprev) / (t - tprev) if have_prev else 0.0
            dv = max_divergence(sp, Ux, Uy, Uz)
            ts.append(t); Es.append(E); ens_l.append(ens)
            eps_ens_l.append(eps_ens); eps_dEdt_l.append(eps_dEdt); div_l.append(dv)
            if n % (rec * 10) == 0 or n == nst:
                print(f"  t={t:5.2f} E/E0={E/E0:.5f} eps_ens={eps_ens:.4e} "
                      f"eps_-dEdt={eps_dEdt:.4e} div={dv:.1e}", flush=True)
            Eprev, tprev, have_prev = E, t, True
        while pending_dumps and n * dt >= pending_dumps[0] - 1e-12:
            treq = pending_dumps.pop(0)
            tact = n * dt
            path = f"psref/tgvfield_ps_N{N}_t{treq:g}.bin"
            dump_physical_field(sp, Ux, Uy, Uz, path)
            dumped.append({"t_requested": treq, "t_actual": tact, "file": path})
            print(f"  [dump] t={tact:.5f} -> {path}", flush=True)
        if n < nst:
            Ux, Uy, Uz = ssp_rk3_step(sp, Ux, Uy, Uz, nu, dt)
    wall = time.time() - t0
    peak_eps = float(np.max(eps_ens_l))
    t_peak = float(ts[int(np.argmax(eps_ens_l))])
    print(f"[TGV] done ({wall:.1f}s). peak eps_ens={peak_eps:.6e} at t={t_peak:.2f}", flush=True)
    result = {
        "case": "tgv", "N": N, "Re": Re, "dt": dt, "steps": nst, "Tend": Tend,
        "t": ts, "E": Es, "enstrophy": ens_l, "eps_ens": eps_ens_l,
        "eps_dEdt": eps_dEdt_l, "divmax": div_l,
        "peak_eps_ens": peak_eps, "t_peak": t_peak, "wall_s": wall,
        "device": str(device), "field_dumps": dumped,
    }
    if out_json:
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[TGV] wrote {out_json}", flush=True)
    return result


def run_recon(N, icfile, Tend, out_json, icN=None, dump_times=None):
    device = get_device()
    Re = 8000.0
    Gam = 1.0
    nu = Gam / Re
    acore = 0.45
    sp = Spectral(N, device)
    Ux, Uy, Uz = load_recon_ic(sp, device, icfile, icN=icN)
    peak0 = peak_vorticity(sp, Ux, Uy, Uz)
    ens0 = 2.0 * mean_enstrophy(sp, Ux, Uy, Uz) * (2 * PI) ** 3  # sum|omega|^2 * h^3 = <|omega|^2>*V
    E0 = kinetic_energy(sp, Ux, Uy, Uz)
    div0 = max_divergence(sp, Ux, Uy, Uz)
    print(f"[RECON] N={N} Re={Re:.0f} from {icfile}", flush=True)
    print(f"  init: peak|omega|={peak0:.4f} enstrophy={ens0:.4f} KE={E0:.4f} div={div0:.2e}",
          flush=True)
    h = 2 * PI / N
    umax = peak0 * acore
    dt = 0.4 * h / max(umax, 0.2)  # matches ns_reconnection.cpp
    nst = int(round(Tend / dt))
    rec = max(1, nst // 50)
    print(f"  dt={dt:.4f} steps={nst} -> t={Tend}", flush=True)
    ts, pk_l, pkrel_l, ens_l, ke_l, div_l = [], [], [], [], [], []
    pending_dumps = sorted(dump_times or [])
    dumped = []
    t0 = time.time()
    for n in range(nst + 1):
        if n % rec == 0 or n == nst:
            t = n * dt
            pk = peak_vorticity(sp, Ux, Uy, Uz)
            en = 2.0 * mean_enstrophy(sp, Ux, Uy, Uz) * (2 * PI) ** 3
            ke = kinetic_energy(sp, Ux, Uy, Uz) / E0
            dv = max_divergence(sp, Ux, Uy, Uz)
            ts.append(t); pk_l.append(pk); pkrel_l.append(pk / peak0)
            ens_l.append(en); ke_l.append(ke); div_l.append(dv)
            print(f"  t={t:5.1f} peak|omega|={pk:.5f} rel={pk/peak0:.5f} "
                  f"enstrophy={en:.5f} KE/KE0={ke:.5f} div={dv:.1e}", flush=True)
        while pending_dumps and n * dt >= pending_dumps[0] - 1e-12:
            treq = pending_dumps.pop(0)
            path = f"psref/reconfield_ps_N{N}_t{treq:g}.bin"
            dump_physical_field(sp, Ux, Uy, Uz, path)
            dumped.append({"t_requested": treq, "t_actual": n * dt, "file": path})
            print(f"  [dump] t={n * dt:.5f} -> {path}", flush=True)
        if n < nst:
            Ux, Uy, Uz = ssp_rk3_step(sp, Ux, Uy, Uz, nu, dt)
    wall = time.time() - t0
    peak_amp = float(np.max(pkrel_l))
    t_at_peak = float(ts[int(np.argmax(pkrel_l))])
    print(f"[RECON] done ({wall:.1f}s). peak amplification={peak_amp:.5f} at t={t_at_peak:.1f}",
          flush=True)
    result = {
        "case": "recon", "N": N, "Re": Re, "dt": dt, "steps": nst, "Tend": Tend,
        "icfile": icfile, "peak0": peak0, "E0": E0,
        "t": ts, "peak": pk_l, "peak_rel": pkrel_l, "enstrophy": ens_l,
        "KErel": ke_l, "divmax": div_l,
        "peak_amplification": peak_amp, "t_at_peak": t_at_peak, "wall_s": wall,
        "device": str(device), "field_dumps": dumped,
    }
    if out_json:
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[RECON] wrote {out_json}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Fourier pseudospectral NS reference")
    sub = ap.add_subparsers(dest="case", required=True)

    pt = sub.add_parser("tgv")
    pt.add_argument("N", type=int)
    pt.add_argument("Tend", type=float, nargs="?", default=20.0)
    pt.add_argument("--cfl", type=float, default=0.30)
    pt.add_argument("--json", default=None)
    pt.add_argument("--dump-field", type=str, default=None,
                    help="comma-separated times; write u,v,w (x-fastest float64, "
                         "C++ dump layout) at the first step reaching each time")

    pr = sub.add_parser("recon")
    pr.add_argument("N", type=int)
    pr.add_argument("icfile", type=str)
    pr.add_argument("Tend", type=float, nargs="?", default=45.0)
    pr.add_argument("--icN", type=int, default=None,
                    help="resolution of the IC dump; if < N, spectrally upsample "
                         "(exact zero-padding) to the run grid")
    pr.add_argument("--json", default=None)
    pr.add_argument("--dump-field", type=str, default=None,
                    help="comma-separated times; write u,v,w (x-fastest float64, "
                         "C++ dump layout) at the first step reaching each time")

    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False  # keep float64 exact
    dumps = ([float(s) for s in args.dump_field.split(",")]
             if getattr(args, "dump_field", None) else None)
    if args.case == "tgv":
        run_tgv(args.N, args.Tend, args.cfl, args.json, dump_times=dumps)
    else:
        run_recon(args.N, args.icfile, args.Tend, args.json, icN=args.icN,
                  dump_times=dumps)


if __name__ == "__main__":
    main()
