#!/usr/bin/env python3
from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scikit-image"]
# ///
"""Taylor-Green frontier and off-grid figures for the periodic B-spline paper.

The 3D incompressible Navier-Stokes solver in ``code/solver`` is run with the
derivative operator swapped by a one-word change of route or degree. The headline
result is the spline *degree ladder* at a fixed coarse grid: the same banded
cardinal-spline construction yields BMC5/BMC7/BMC9 (formal orders 6/8/10), and at
64^3 they march a resolution-sensitive diagnostic to its fully resolved value,
overtaking the best fixed-order compact (C6) and the explicit stencils.

Figures:
  fig_tgv_setup.pdf         : 3D vorticity isosurfaces of the Taylor-Green run --
      the structured initial vortices and the turbulent vortex tangle at the peak.
  fig_tgv_dissipation.pdf   : Taylor-Green Re=1600. (a) dissipation rate eps(t)
      for the BMC ladder vs C6/E6 against the published 512^3 spectral peak;
      (b) the peak recovered vs the operator's formal order -- the ladder reaches
      the resolved value where the fixed-order compact is stuck.
  fig_reconnection_setup.pdf: 3D vorticity isosurfaces -- the antiparallel tubes
      and the reconnected vortices near the peak.
  fig_reconnection_peak.pdf : vortex reconnection Re=8000. Peak vorticity
      amplification for the BMC ladder vs C6/E6, with the independent Fourier
      pseudospectral reference (cross-method verified at 128^3; ps_reference.py).
  fig_offgrid_turbulence.pdf: off-grid value/derivative at the differentiation
      order from one banded solve vs a bolt-on interpolant, in the house
      convergence style (slope guides + formal/measured order table).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THISDIR = Path(__file__).resolve().parent
REPO_ROOT = THISDIR.parent if THISDIR.name == "code" else THISDIR
FIGDIR = REPO_ROOT / "figures_paper"
DATADIR = REPO_ROOT / "code" / "solver" / "data"
RESULTSDIR = REPO_ROOT / "code" / "results"
FIGDIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THISDIR))

from plotting_helpers import (  # noqa: E402
    add_formal_order_triangle,
    apply_classy_defaults,
    make_overhead_legend,
    quiet_axis,
)

# Published resolved reference for the Taylor-Green Re=1600 peak kinetic-energy
# dissipation rate, from the 512^3 dealiased pseudospectral DNS of van Rees,
# Leonard, Pullin & Koumoutsakos, J. Comput. Phys. 230 (2011) 2794, as
# distributed by the 1st High-Order CFD Workshop (spectral_Re1600_512.gdiag,
# peak -dE/dt = 1.28575e-2 at t=8.97). The earlier 1.25e-2 anchor was wrong.
TGV_REF_PEAK = 1.28575e-2
TGV_REF_TPEAK = 8.97
PSREFDIR = REPO_ROOT / "code" / "solver" / "psref"

# Blue gradient for the spline degree ladder; warm/neutral for the fixed-order
# baselines. Colours chosen for print legibility.
BMC_LADDER = {5: "#c6dbef", 7: "#6baed6", 9: "#2171b5", 11: "#08306b"}
C6_COLOR, E6_COLOR, DRP_COLOR = "#d6602c", "#6b6b6b", "#9467bd"
# The compact ladder above C6: darker warm tones that pair with C6 while
# staying distinct in print.
C8_COLOR, C10_COLOR, C12_COLOR = "#8c2d04", "#cc4c02", "#993404"
COMPACT_LADDER = {6: C6_COLOR, 8: C8_COLOR, 10: C10_COLOR, 12: C12_COLOR}


def _first_deriv_order(p):
    """Periodic BMC first-derivative order: 2*ceil(p/2) = p+1 for odd p."""
    return 2 * ((p + 1) // 2)


def load_field_any(name, N, *dirs):
    """Load a velocity-field dump searching the given directories in order."""
    for base in dirs:
        p = base / name
        if p.exists():
            raw = np.fromfile(p, dtype=np.float64)
            if raw.size == 3 * N**3:
                return tuple(raw[i * N**3:(i + 1) * N**3].reshape(N, N, N)
                             for i in range(3))
    return None


def _field_error_records():
    """(route, p, N) -> record from the operator-free field-error diagnostic."""
    fp = RESULTSDIR / "s8_field_error.json"
    if not fp.exists():
        return {}
    recs = json.loads(fp.read_text())["records"]
    return {(r["route"], r["p"], r["N"]): r for r in recs}


def _native_peak_pct():
    """(route, p, N) -> native dissipation peak as % of the corrected anchor."""
    fp = RESULTSDIR / "s8_common_diagnostic.json"
    if not fp.exists():
        return {}
    tgv = json.loads(fp.read_text())["tgv"]
    out = {}
    for r in tgv:
        pct = r.get("pct_of_reference_at_t_dump", {}).get("native")
        if pct is not None:
            out[(r["route"], r["p"], r["N"])] = pct
    return out


# ---- CSV + field-dump loading --------------------------------------------
def load_csv(name: str):
    p = DATADIR / name
    if not p.exists():
        return None
    try:
        d = np.genfromtxt(p, delimiter=",", names=True, invalid_raise=False)
    except (IndexError, ValueError):
        return None
    return d if d is not None and getattr(d, "size", 0) > 1 else None


def load_field(name: str, N: int = 64):
    """Load a velocity-field dump (3 x N^3 doubles, u then v then w). Array
    layout is [z, y, x] (x is the fastest index in the solver)."""
    p = DATADIR / name
    if not p.exists():
        return None
    raw = np.fromfile(p, dtype=np.float64)
    if raw.size != 3 * N**3:
        return None
    u, v, w = (raw[i * N**3:(i + 1) * N**3].reshape(N, N, N) for i in range(3))
    return u, v, w


def spectral_upsample(field2d, M, smooth=0.0):
    """Smooth, faithful upsampling of a periodic 2D field from N x N to M x M by
    zero-padding its Fourier transform (interpolation of the underlying field).
    ``smooth`` in (0,1) applies a raised-cosine spectral taper over the top
    ``smooth`` fraction of modes, suppressing the Gibbs ringing that a sharp
    cutoff produces when the field carries energy near the grid Nyquist (e.g. an
    under-resolved turbulent slice). This is a rendering choice, not a result."""
    N = field2d.shape[0]
    if M <= N:
        return field2d
    F = np.fft.fftshift(np.fft.fft2(field2d))
    if smooth > 0.0:
        c = N / 2.0
        ii = np.arange(N) - c
        KR = np.sqrt(ii[:, None] ** 2 + ii[None, :] ** 2) / c     # 0..~1.4
        k0 = 1.0 - smooth
        t = np.clip((KR - k0) / max(1.0 - k0, 1e-9), 0.0, 1.0)
        F = F * (0.5 * (1.0 + np.cos(np.pi * t)))
    pad = (M - N) // 2
    Fp = np.zeros((M, M), dtype=complex)
    Fp[pad:pad + N, pad:pad + N] = F
    return np.fft.ifft2(np.fft.ifftshift(Fp)).real * (M / N) ** 2


def vorticity_components(u, v, w, L=2 * np.pi):
    """Spectrally exact vorticity components (wx, wy, wz). Axes: 0->z, 1->y, 2->x.
    Each component is band-limited, so a slice of it upsamples without ringing."""
    N = u.shape[0]
    k1 = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    kz, ky, kx = np.meshgrid(k1, k1, k1, indexing="ij")
    uh, vh, wh = (np.fft.fftn(q) for q in (u, v, w))
    dy = lambda fh: np.fft.ifftn(1j * ky * fh).real
    dz = lambda fh: np.fft.ifftn(1j * kz * fh).real
    dx = lambda fh: np.fft.ifftn(1j * kx * fh).real
    return dy(wh) - dz(vh), dz(uh) - dx(wh), dx(vh) - dy(uh)


def vorticity_mag(u, v, w, L=2 * np.pi):
    wx, wy, wz = vorticity_components(u, v, w, L)
    return np.sqrt(wx**2 + wy**2 + wz**2)


def vortmag_slice_upsampled(fld, z_index, M, roll_y=False, smooth=0.45):
    """|omega| on a z-slice, rendered at M x M by upsampling each band-limited
    vorticity component and then combining. A spectral taper (``smooth``) keeps
    the render ring-free when the slice carries near-Nyquist content."""
    comps = vorticity_components(*fld)
    out = 0.0
    for c in comps:
        s = c[z_index, :, :]
        if roll_y:
            s = np.roll(s, s.shape[0] // 2, axis=0)
        s = spectral_upsample(s, M, smooth=smooth)
        out = out + s**2
    return np.sqrt(out)


# ---------------------------------------------------------------------------
# Taylor-Green Re=1600: degree ladder + energy spectrum.
# ---------------------------------------------------------------------------
def figure_tgv_dissipation() -> None:
    apply_classy_defaults()
    Re = 1600
    ladder = [("bmc", 5, "BMC5 (order 6)", BMC_LADDER[5]),
              ("bmc", 7, "BMC7 (order 8)", BMC_LADDER[7]),
              ("bmc", 9, "BMC9 (order 10)", BMC_LADDER[9])]
    baselines = [("c6", 5, "C6 (order 6)", C6_COLOR, "--"),
                 ("c8", 5, "C8 (order 8)", C8_COLOR, "-."),
                 ("fd6", 5, "E6 explicit (order 6)", E6_COLOR, ":")]

    del ladder, baselines  # this figure now reads the operator-free diagnostics
    fe = _field_error_records()
    npct = _native_peak_pct()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"wspace": 0.30})

    # Panel (a): the operator-free resolved-band spectrum error against the
    # 384^3 reference, spline degree ladder vs the full compact ladder, versus
    # first-derivative order. The spline falls monotonically; the compact
    # saturates.
    ax = axes[0]
    spline = [(p, _first_deriv_order(p)) for p in (5, 7, 9, 11)]
    sx, sy = [], []
    for p, order in spline:
        r = fe.get(("bmc", p, 64))
        if r is None:
            continue
        sx.append(order); sy.append(r["resolved_band_spectrum_error"])
    if sx:
        ax.plot(sx, sy, color="#08306b", lw=1.9, marker="o", ms=8, mec="white",
                mew=1.1, label="BMC spline ladder", zorder=4)
        for x, y in zip(sx, sy):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5, color="#08306b")
    cx, cy = [], []
    for order in (6, 8, 10, 12):
        r = fe.get((f"c{order}", 5, 64))
        if r is None:
            continue
        cx.append(order); cy.append(r["resolved_band_spectrum_error"])
    if cx:
        ax.plot(cx, cy, color=C6_COLOR, lw=1.7, ls="--", marker="s", ms=7,
                mec="white", mew=1.1, label="compact Padé ladder", zorder=4)
        for x, y in zip(cx, cy):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8.5, color=C6_COLOR)
    ax.set_xlabel("first-derivative order")
    ax.set_ylabel(r"resolved-band spectrum error vs $384^3$")
    ax.set_xticks([6, 8, 10, 12]); ax.set_xlim(5.5, 12.5)
    ax.text(0.035, 0.06, r"(a) common field diagnostic, $64^3$, $t\approx9$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    # Panel (b): the operator-native dissipation peak as % of the corrected
    # anchor, spline and compact interleaved, to show this single number does
    # not separate the families (why the common diagnostic is the verdict).
    ax = axes[1]
    ax.axhline(100.0, color="0.35", ls=(0, (5, 2)), lw=1.0)
    ax.text(5.7, 100.5, "resolved rate", color="0.35", fontsize=8.5,
            va="bottom", ha="left")
    for label, routes, color, mk in (
            ("BMC (own curl)", [("bmc", p) for p in (5, 7, 9, 11)], "#2171b5", "o"),
            ("compact (own curl)", [(f"c{o}", 5) for o in (6, 8, 10, 12)],
             C6_COLOR, "s")):
        xs, ys = [], []
        for route, p in routes:
            key = (route, p, 64)
            if key not in npct:
                continue
            order = _first_deriv_order(p) if route == "bmc" else int(route[1:])
            xs.append(order); ys.append(npct[key])
        if xs:
            ax.plot(xs, ys, color=color, lw=1.5, marker=mk, ms=7, mec="white",
                    mew=1.0, label=label, ls="-" if mk == "o" else "--", zorder=3)
    ax.set_xlabel("first-derivative order")
    ax.set_ylabel(r"native peak $\varepsilon$ ($\%$ of reference)")
    ax.set_xticks([6, 8, 10, 12]); ax.set_xlim(5.5, 12.5); ax.set_ylim(70, 105)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.text(0.035, 0.06, "(b) native peak does not rank the families",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    make_overhead_legend(fig, handles, labels, ncol=2,
                         bbox=(0.5, 1.02), rect=(0, 0, 1, 0.90))
    fig.savefig(FIGDIR / "fig_tgv_dissipation.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  fig_tgv_dissipation: spline band-err {sy}  compact band-err {cy}")


def energy_spectrum(u, v, w):
    """Shell-averaged 3D kinetic-energy spectrum E(k) of a periodic velocity
    field. Coefficients are normalized so that sum_k E(k) = <1/2|u|^2>, making
    E(k) comparable across grid sizes at each integer wavenumber shell."""
    N = u.shape[0]
    uh = np.fft.fftn(u) / N**3
    vh = np.fft.fftn(v) / N**3
    wh = np.fft.fftn(w) / N**3
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2 + np.abs(wh) ** 2)
    kax = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kax, kax, kax, indexing="ij")
    kbin = np.rint(np.sqrt(KX**2 + KY**2 + KZ**2)).astype(int)
    kmax = N // 2
    keep = kbin <= kmax
    Ek = np.zeros(kmax + 1)
    np.add.at(Ek, kbin[keep].ravel(), e[keep].ravel())
    return np.arange(kmax + 1), Ek


# ---------------------------------------------------------------------------
# Taylor-Green Re=1600: the 102% recovered by BMC9 at 64^3 is not aliasing.
# Panel (a): the field's energy spectrum at the peak is comparably populated for
# every 64^3 operator and tracks the resolved 128^3 reference, with only a mild
# high-k bottleneck (no runaway cutoff spike). Panel (b): grid convergence of the
# recovered peak. BMC9 brackets the resolved value from just above at 64^3 (102%)
# and settles onto it at 128^3 (100%, the published peak), so the overshoot is a
# small coarse-grid feature that refinement removes; the fixed-order compact
# approaches from well below and is still short at 128^3.
# ---------------------------------------------------------------------------
def figure_tgv_spectrum_diagnostic() -> None:
    apply_classy_defaults()
    N = 64
    curves = [
        ("tgvfield_bmc_p5_N64.bin", "BMC5 (order 6)", BMC_LADDER[5], "-"),
        ("tgvfield_bmc_p7_N64.bin", "BMC7 (order 8)", BMC_LADDER[7], "-"),
        ("tgvfield_bmc_p9_N64.bin", "BMC9 (order 10)", BMC_LADDER[9], "-"),
        ("tgvfield_c6_p5_N64.bin", "C6 (order 6)", C6_COLOR, "--"),
        ("tgvfield_fd6_p5_N64.bin", "E6 explicit (order 6)", E6_COLOR, ":"),
    ]
    ref = None
    rfld = load_field_any("tgvfield_ps_N384_t9.bin", 384, PSREFDIR)
    ref_label = r"resolved $384^3$"
    if rfld is None:  # fall back to the finest B-spline field if the ref dump is absent
        rfld = load_field("tgvfield_bmc_p9_N128.bin", 128)
        ref_label = r"resolved $128^3$"
    if rfld is not None:
        kEref, Eref = energy_spectrum(*rfld)
        ref = (kEref, Eref)
    data = []
    for nm, label, color, ls in curves:
        fld = load_field(nm, N)
        if fld is None:
            continue
        kE, E = energy_spectrum(*fld)
        data.append((label, color, ls, kE, E))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5),
                             gridspec_kw={"wspace": 0.30})
    kny = N // 2
    xhi = kny + 6

    # Panel (a): energy spectrum E(k), log-log.
    ax = axes[0]
    ymax = 1e-3
    if ref is not None:
        kEref, Eref = ref
        sel = kEref <= xhi
        ax.loglog(kEref[1:][sel[1:]], Eref[1:][sel[1:]], color="0.25", lw=3.4,
                  alpha=0.30, solid_capstyle="round",
                  label=ref_label, zorder=1)
        ymax = max(ymax, float(np.max(Eref[1:])))
    for label, color, ls, kE, E in data:
        ax.loglog(kE[1:], E[1:], color=color, lw=1.7, ls=ls, label=label,
                  zorder=3)
    ax.axvline(kny, color="0.5", ls=(0, (2, 2)), lw=0.9)
    ax.set_ylim(ymax * 8e-5, ymax * 2.2)
    ax.text(kny * 0.95, ymax * 3e-4, r"$64^3$ Nyquist", rotation=90,
            ha="right", va="bottom", fontsize=8, color="0.45")
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"energy spectrum $E(k)$")
    ax.set_xlim(1, xhi)
    ax.text(0.035, 0.06, r"(a) energy spectrum, $64^3$, $t\approx9$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    # Panel (b): grid convergence of the operator-free resolved-band spectrum
    # error. BMC9 is lowest at every grid; the compact rungs stay above.
    ax = axes[1]
    fe = _field_error_records()
    Ns = [64, 128]

    def band_err(route, p):
        out = []
        for n in Ns:
            r = fe.get((route, p, n))
            out.append(r["resolved_band_spectrum_error"] if r else np.nan)
        return out

    series = [("BMC9 (order 10)", band_err("bmc", 9), BMC_LADDER[9], "o", "-"),
              ("C8 (order 8)", band_err("c8", 5), C8_COLOR, "^", "-."),
              ("C6 (order 6)", band_err("c6", 5), C6_COLOR, "s", "--")]
    for label, ys, color, mk, ls in series:
        ax.plot(Ns, ys, color=color, lw=1.7, ls=ls, marker=mk, ms=7,
                mec="white", mew=1.0, label=label, zorder=3)
        # Label endpoints horizontally (left at N=64, right at N=128), vertically
        # centered, so the three close values never collide on the log axis.
        if np.isfinite(ys[0]):
            ax.annotate(f"{ys[0]:.3f}", (Ns[0], ys[0]), textcoords="offset points",
                        xytext=(-8, 0), ha="right", va="center", fontsize=8.5,
                        color=color)
        if np.isfinite(ys[-1]):
            ax.annotate(f"{ys[-1]:.3f}", (Ns[-1], ys[-1]), textcoords="offset points",
                        xytext=(9, 0), ha="left", va="center", fontsize=8.5,
                        color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlim(Ns[0] * 0.72, Ns[-1] * 1.34)
    ax.set_xlabel(r"grid resolution $N$ (per direction)")
    ax.set_ylabel(r"resolved-band spectrum error vs $384^3$")
    ax.text(0.035, 0.06, "(b) the spline leads at every grid",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    make_overhead_legend(fig, handles, labels, ncol=3,
                         bbox=(0.5, 1.04), rect=(0, 0, 1, 0.86))
    fig.savefig(FIGDIR / "tgv_spectrum_diagnostic.pdf", bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"  tgv_spectrum_diagnostic band-err BMC9: {band_err('bmc', 9)}  "
          f"C6: {band_err('c6', 5)}")


# ---------------------------------------------------------------------------
# Vortex reconnection Re=8000: degree ladder + setup snapshots.
# ---------------------------------------------------------------------------
def load_recon_txt(name: str):
    p = DATADIR / name
    if not p.exists():
        return None
    ts, pk, en = [], [], []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            t, peak, ens = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        ts.append(t); pk.append(peak); en.append(ens)
    if not ts:
        return None
    pk = np.array(pk)
    return {"t": np.array(ts), "peak_rel": pk / pk[0], "enstrophy": np.array(en)}


# ---------------------------------------------------------------------------
# Independent pseudospectral reference. ps_reference.py writes one JSON
# per run (recon_ps128/256/384.json); the peak-amplification trace is loaded for
# the reconnection figure.
# ---------------------------------------------------------------------------
PS_REF_DIR = REPO_ROOT / "code" / "solver" / "psref"


def load_ps_json(name: str):
    p = PS_REF_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def figure_reconnection_peak() -> None:
    apply_classy_defaults()
    Re = 8000
    ladder = [("bmc", 5, "BMC5 (order 6)", BMC_LADDER[5]),
              ("bmc", 7, "BMC7 (order 8)", BMC_LADDER[7]),
              ("bmc", 9, "BMC9 (order 10)", BMC_LADDER[9])]
    baselines = [("c6", "C6 (order 6)", C6_COLOR, "--"),
                 ("c8", "C8 (order 8)", C8_COLOR, "-."),
                 ("fd6", "E6 explicit (order 6)", E6_COLOR, ":")]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"wspace": 0.28})
    # Independent Fourier pseudospectral reference, built by a second,
    # method-independent solver from the bit-identical t=0 field. The 384^3 run is
    # the converged reference (256^3 -> 384^3 moves the peak by less than 0.1%);
    # cross-method agreement verifies the value at matched resolution (PS vs BMC:
    # 1.3% at 128^3, 0.1% at 256^3), so the ladder climbs toward a verified value.
    ps_ref = (load_ps_json("recon_ps384.json") or load_ps_json("recon_ps256.json")
              or load_ps_json("recon_ps128.json"))

    # Panel (a): omega_max(t)/omega_0.
    ax = axes[0]
    for route, p, label, color in ladder:
        d = load_csv(f"recon_{route}_p{p}_N64_re{Re}.csv")
        if d is None:
            continue
        ax.plot(d["t"], d["peak_rel"], color=color, lw=1.8, label=label)
    for route, label, color, lsd in baselines:
        d = load_csv(f"recon_{route}_p5_N64_re{Re}.csv")
        if d is None:
            continue
        ax.plot(d["t"], d["peak_rel"], color=color, lw=1.5, ls=lsd, label=label)
    if ps_ref is not None:
        ax.plot(ps_ref["t"], ps_ref["peak_rel"], color="0.15", lw=1.2,
                ls=(0, (5, 2)),
                label=r"pseudospectral $%d^3$ reference" % ps_ref["N"])
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"peak amplification $\omega_{\max}(t)/\omega_0$")
    ax.text(0.035, 0.96, r"(a) $64^3$, Re$=8000$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="#2F4858", fontstyle="italic")
    ax.set_xlim(0, None); ax.set_ylim(1.0, None)
    quiet_axis(ax)

    # Panel (b): COMMON-curl peak amplification vs formal order, spline ladder
    # against the full compact ladder, every scheme measured by one exact
    # Fourier curl of its saved fields (operator-free). The converged reference
    # (4.17) sits far above every 64^3 operator; the percentage annotations
    # keep that honest.
    ax = axes[1]
    fp = RESULTSDIR / "s8_common_diagnostic.json"
    recs = {}
    if fp.exists():
        for r in json.loads(fp.read_text())["reconnection"]:
            if "amplification_common_unfiltered" in r:
                recs[(r["route"], r["p"], r["N"])] = r["amplification_common_unfiltered"]
    REF = 4.166  # converged cross-method reference (ps 384^3; BMC 256^3 within 0.1%)
    sx, sy = [], []
    for p in (5, 7, 9):
        a = recs.get(("bmc", p, 64))
        if a is not None:
            sx.append(_first_deriv_order(p)); sy.append(a)
    if sx:
        ax.plot(sx, sy, color="#08306b", lw=1.9, marker="o", ms=8, mec="white",
                mew=1.1, zorder=3, label="BMC spline ladder")
        for x, y in zip(sx, sy):
            # at order 8 the two ladders nearly touch; push the BMC label down-left
            off = (-14, -11) if x == 8 else (0, 10)
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=off, ha="center", fontsize=8.5, color="#08306b")
        ax.annotate(rf"$= {100*sy[-1]/REF:.0f}\%$ of the converged $4.17$",
                    (sx[-1], sy[-1]), textcoords="offset points", xytext=(-4, 24),
                    ha="right", fontsize=8.5, color="#08306b", fontstyle="italic")
    cx, cy = [], []
    for order in (6, 8, 10, 12):
        a = recs.get((f"c{order}", 5, 64))
        if a is not None:
            cx.append(order); cy.append(a)
    if cx:
        ax.plot(cx, cy, color=C6_COLOR, lw=1.7, ls="--", marker="s", ms=7,
                mec="white", mew=1.1, zorder=3, label="compact Padé ladder")
        for x, y in zip(cx, cy):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8.5, color=C6_COLOR)
    ax.set_xlabel("first-derivative order")
    ax.set_ylabel(r"common-curl peak $\omega_{\max}/\omega_0$ at $64^3$")
    ax.set_xticks([6, 8, 10, 12]); ax.set_xlim(5.5, 12.5)
    ax.set_ylim(2.60, 3.02)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.text(0.97, 0.05, "(b) common postprocessor, both ladders",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#2F4858", fontstyle="italic")
    quiet_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    make_overhead_legend(fig, handles, labels, ncol=3,
                         bbox=(0.5, 1.04), rect=(0, 0, 1, 0.86))
    fig.savefig(FIGDIR / "fig_reconnection_peak.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("  fig_reconnection_peak written")


def _polish_field_axis(ax, im, fig, title, L=2 * np.pi, ycentered=False):
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_xticks([0, np.pi, L]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    if ycentered:
        ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    else:
        ax.set_yticks([0, np.pi, L]); ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_title(title, fontsize=11, color="#2F4858", style="italic", pad=6)
    for s in ax.spines.values():
        s.set_color("#888888"); s.set_linewidth(0.6)
    ax.tick_params(length=3, width=0.5, colors="#444444")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(r"$|\boldsymbol{\omega}|$", fontsize=10)
    cb.ax.tick_params(labelsize=8, length=2.5, width=0.5)
    cb.outline.set_linewidth(0.5)


def _smooth3d(vol, smooth):
    """Radial Fourier low-pass of a periodic 3D field (raised-cosine taper over
    the top ``smooth`` fraction of modes), for a clean isosurface."""
    if smooth <= 0:
        return vol
    N = vol.shape[0]
    k = np.fft.fftfreq(N) * N
    KR = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2) / (N / 2)
    k0 = 1.0 - smooth
    t = np.clip((KR - k0) / max(1.0 - k0, 1e-9), 0.0, 1.0)
    return np.fft.ifftn(np.fft.fftn(vol) * (0.5 * (1 + np.cos(np.pi * t)))).real


def _isosurface(ax, vol, level, color, L=2 * np.pi, light=(0.3, 0.45, 0.82),
                min_frac=0.0):
    """Render a shaded isosurface of a 3D scalar field. Axes of ``vol`` are
    (z, y, x); the plot shows (x, y, z) with Lambertian shading for depth.
    ``min_frac`` drops connected components of the thresholded region smaller
    than that fraction of the largest one (speckle cleanup for a clean render)."""
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgb
    if min_frac > 0.0:
        lbl = measure.label(vol > level)
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        if sizes.size > 1 and sizes.max() > 0:
            keep = sizes >= min_frac * sizes.max()
            vol = np.where(keep[lbl], vol, 0.0)
    verts, faces, _, _ = measure.marching_cubes(vol, level)
    N = vol.shape[0]
    P = verts / N * L
    tri = np.stack([P[faces][:, :, 2], P[faces][:, :, 1], P[faces][:, :, 0]], axis=2)  # (x,y,z)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    ld = np.array(light, float); ld /= np.linalg.norm(ld)
    inten = 0.4 + 0.6 * np.clip(np.abs(n @ ld), 0, 1)
    base = np.array(to_rgb(color))
    fc = np.clip(inten[:, None] * base[None, :], 0, 1)
    coll = Poly3DCollection(tri, facecolors=fc, edgecolor="none", linewidths=0)
    coll.set_rasterized(True)              # keep the vector PDF small for dense surfaces
    ax.add_collection3d(coll)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)


def figure_reconnection_setup() -> None:
    """3D vorticity-magnitude isosurfaces: the antiparallel tubes at t=0 and the
    reconnected state near the peak. The isosurface threshold isolates the
    coherent high-vorticity cores, so it is robust to the grid-scale noise of the
    deliberately under-resolved collocated 64^3 run."""
    apply_classy_defaults()
    N = 64
    f0 = load_field("reconfield_bmc_p9_N64_t0.bin", N)
    fp = load_field("reconfield_bmc_p9_N64_peak.bin", N)
    if f0 is None or fp is None:
        print("  fig_reconnection_setup: no field dumps, skipped")
        return
    w0 = vorticity_mag(*f0)
    wp = _smooth3d(vorticity_mag(*fp), 0.55)
    vcol = "#c0392b"

    fig = plt.figure(figsize=(11.6, 5.2))
    for i, (vol, frac, title) in enumerate(
            ((w0, 0.45, r"(a) initial antiparallel tubes, $t=0$"),
             (wp, 0.40, r"(b) reconnected vortices, $t\approx 38$"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _isosurface(ax, vol, frac * float(vol.max()), vcol)
        ax.set_title(title, fontsize=11, color="#2F4858", style="italic", y=0.98)
        ax.set_xlabel(r"$x$", labelpad=-8); ax.set_ylabel(r"$y$", labelpad=-8)
        ax.set_zlabel(r"$z$", labelpad=-8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1, 1, 1))
        try:
            ax.set_proj_type("persp")
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_facecolor((1, 1, 1, 0)); pane.pane.set_edgecolor("0.85")
        except Exception:
            pass
    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.0, wspace=0.02)
    fig.savefig(FIGDIR / "fig_reconnection_setup.pdf", bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("  fig_reconnection_setup written (3D isosurface)")


# ---------------------------------------------------------------------------
# Off-grid evaluation in a turbulence-like field.
# ---------------------------------------------------------------------------
def figure_offgrid_turbulence() -> None:
    import convergence_style as cs
    from matplotlib.lines import Line2D
    from plotting_helpers import make_overhead_legend, method_style
    apply_classy_defaults()
    d = load_csv("offgrid_turbulence.csv")
    if d is None:
        print("  fig_offgrid_turbulence: no data, skipped")
        return
    degs = [3, 5, 7]
    dcol = {3: cs.DEG_COLOR[3], 5: cs.DEG_COLOR[5], 7: cs.DEG_COLOR[7]}
    dmark = {p: cs.DEG_MARKER[p] for p in degs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 5.2))

    measured = {"value": {}, "derivative": {}}

    def draw_metric(ax, degree, error_key, expected_order, metric_style):
        selected = d["p"] == degree
        h = d["h"][selected]
        error = d[error_key][selected]
        n = 2.0 * np.pi / h
        fit = cs.fit_indices(h, error, expected_order=expected_order)
        last = int(fit[-1])
        line_style, marker, width = metric_style
        ax.loglog(n[:last + 1], error[:last + 1], line_style, marker=marker,
                  color=dcol[degree], ms=5.7, lw=width, mec="white", mew=0.7)
        if last + 1 < len(error):
            ax.loglog(n[last:], error[last:], linestyle=":", marker=marker,
                      color=dcol[degree], ms=5.2, lw=1.0, mfc="white",
                      mec=dcol[degree], mew=0.9, alpha=0.48)
        measured_order = cs.measured_order(h, error, expected_order=expected_order)
        return measured_order, (n[:last + 1], error[:last + 1])

    value_guides = {}
    for p in degs:
        l2_order, l2_curve = draw_metric(
            ax1, p, "val_err", p + 1, ("-", dmark[p], 2.1)
        )
        max_order, _ = draw_metric(
            ax1, p, "val_max_q", p + 1, ("--", "^", 1.5)
        )
        measured["value"][p] = (l2_order, max_order)
        value_guides[p + 1] = l2_curve
    cs.style_axis(ax1, r"grid resolution $N$", "relative query error",
                  r"(a)", headroom=1.8)
    cs.resolution_axis(ax1, [32, 48, 64, 96, 128, 192, 256])
    value_guide_positions = {4: (0.42, 0.40), 6: (0.55, 0.38), 8: (0.62, 0.42)}
    for order, curve in sorted(value_guides.items()):
        anchor_frac, vertical_offset = value_guide_positions[order]
        add_formal_order_triangle(
            ax1, order, anchor_curve=curve, anchor_frac=anchor_frac,
            size_frac=0.050, offset_decades=(0.025, vertical_offset), alpha=0.30,
        )

    derivative_guides = {}
    for p in degs:
        l2_order, l2_curve = draw_metric(
            ax2, p, "der_err", p, ("-", dmark[p], 2.1)
        )
        max_order, _ = draw_metric(
            ax2, p, "der_max_q", p, ("--", "^", 1.5)
        )
        measured["derivative"][p] = (l2_order, max_order)
        derivative_guides[p] = l2_curve
    baseline_handles = []
    selected = d["p"] == 3
    for key, label, style_key in (
        ("bolton4_der_err", "E6 + Lagrange-4", "C4"),
        ("bolton6_der_err", "E6 + Lagrange-6", "C6"),
    ):
        h = d["h"][selected]
        error = d[key][selected]
        n = 2.0 * np.pi / h
        style = method_style(style_key)
        ax2.loglog(n, error, style["ls"], marker=style["marker"],
                   color=style["color"], ms=4.8, lw=1.25, alpha=0.72)
        derivative_guides[4 if key == "bolton4_der_err" else 6] = (n, error)
        baseline_handles.append(Line2D([], [], color=style["color"],
                                       marker=style["marker"], lw=1.25,
                                       ls=style["ls"], label=label))
    cs.style_axis(ax2, r"grid resolution $N$", "relative query error",
                  r"(b)", headroom=1.8)
    cs.resolution_axis(ax2, [32, 48, 64, 96, 128, 192, 256])
    derivative_guide_positions = {
        3: (0.38, 0.45),
        4: (0.75, -0.35),
        5: (0.60, -0.50),
        6: (0.45, -0.45),
        7: (0.55, 0.55),
    }
    for order, curve in sorted(derivative_guides.items()):
        anchor_frac, vertical_offset = derivative_guide_positions[order]
        add_formal_order_triangle(
            ax2, order, anchor_curve=curve, anchor_frac=anchor_frac,
            size_frac=0.044, offset_decades=(0.020, vertical_offset), alpha=0.30,
        )

    handles = [Line2D([], [], color=dcol[p], marker=dmark[p], lw=2.0,
                      label=f"BMC{p}") for p in degs]
    handles.extend([
        Line2D([], [], color="0.25", marker="s", lw=2.1,
               label=r"relative $\ell^2(Q)$"),
        Line2D([], [], color="0.25", marker="^", lw=1.5, ls="--",
               label=r"relative $\ell^\infty(Q)$"),
        Line2D([], [], color="0.45", marker="o", mfc="white", lw=1.0, ls=":",
               label="dotted/open: excluded from fit"),
    ])
    handles.extend(baseline_handles)
    labels = [handle.get_label() for handle in handles]
    make_overhead_legend(fig, handles, labels, ncol=3,
                         bbox=(0.5, 0.995), rect=(0, 0, 1, 0.82),
                         fontsize=15.2, handslcngth=2.0,
                         columnspacing=1.0, handletextpad=0.4)
    fig.savefig(FIGDIR / "fig_offgrid_turbulence.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  fig_offgrid_turbulence written (reference style)")


def figure_tgv_setup() -> None:
    """The Taylor-Green configuration as 3D vorticity isosurfaces: the structured
    initial vortices (analytic IC on 64^3) and the turbulent vortex tangle near
    the dissipation peak from BMC9 on 64^3."""
    apply_classy_defaults()
    plt.rcParams.update({"font.size": 16.0, "axes.labelsize": 16.0})
    N = 64
    fp = load_field("tgvfield_bmc_p9_N64.bin", N)
    if fp is None:
        print("  fig_tgv_setup: no field dump, skipped")
        return
    L = 2 * np.pi
    # analytic Taylor-Green initial velocity on the grid, then its vorticity.
    x = np.arange(N) * (L / N)
    Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
    u0 = np.sin(X) * np.cos(Y) * np.cos(Z)
    v0 = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w0vel = np.zeros_like(u0)
    w0 = vorticity_mag(u0, v0, w0vel)
    wp = _smooth3d(vorticity_mag(*fp), 0.45)
    vcol = "#1f6f6f"

    fig = plt.figure(figsize=(11.6, 5.2))
    for i, (vol, frac, title) in enumerate(
            ((w0, 0.55, r"(a)"),
             (wp, 0.42, r"(b)"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _isosurface(ax, vol, frac * float(vol.max()), vcol, light=(0.35, 0.5, 0.78))
        ax.set_title(title, fontsize=17, color="#2F4858", style="italic", y=0.98)
        ax.set_xlabel(r"$x$", labelpad=-8); ax.set_ylabel(r"$y$", labelpad=-8)
        ax.set_zlabel(r"$z$", labelpad=-8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1, 1, 1))
        try:
            ax.set_proj_type("persp")
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_facecolor((1, 1, 1, 0)); pane.pane.set_edgecolor("0.85")
        except Exception:
            pass
    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.0, wspace=0.02)
    fig.savefig(FIGDIR / "fig_tgv_setup.pdf", bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("  fig_tgv_setup written (3D isosurface)")


def main() -> None:
    figure_tgv_setup()
    figure_tgv_dissipation()
    figure_reconnection_setup()
    figure_reconnection_peak()
    figure_offgrid_turbulence()
    print("Wrote Taylor-Green and off-grid figures to", FIGDIR)


if __name__ == "__main__":
    main()
