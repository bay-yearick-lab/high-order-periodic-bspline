# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Long-horizon transport of a narrow-band packet at four points per wavelength.

Periodic linear advection with c = 1 on N points sends the mode at theta
through exp(-i H N theta e_1(theta)) after H box crossings relative to the
exact solution, so the relative L2 error of a packet with spectrum A is

    e_H = sqrt( sum |A|^2 |exp(-i H N theta e_1) - 1|^2 / sum |A|^2 ),

a closed form in the symbol alone. The packet is Gaussian with 99 percent of
its energy inside theta_c +- 0.1 pi, and the crossing of a route is the first
horizon at which e_H reaches one percent.

Panel (a) carries e_H against H at the carrier theta_c = 0.5 pi on N = 128,
that is four points per wavelength, with the one percent level drawn and each
route's crossing marked on it. Panel (b) carries those crossings against the
three carriers 0.4 pi, 0.5 pi and 0.6 pi, so the whole band picture reads at
once: a route buys horizon by holding e_1 small across the packet, not by
holding it small at the carrier alone.

Every curve is recomputed here through the closed-form routines of
gate4b_long_horizon.py, with the six routes assembled by gate9_e3.routes_by_key
so that the joint member enters exactly as it does in the receipt. The
crossings and the two quoted horizons of every route and carrier are then
checked against results/exponential_family/gate9/e3_transport_core6.json, whose
stored routes in turn re-reduce the Gate 4b tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate4b_long_horizon as g4b  # noqa: E402
import gate9_e3 as g9e3  # noqa: E402
import plotting_helpers as ph  # noqa: E402

EF = ROOT / "results" / "exponential_family"
TRANSPORT = EF / "gate9" / "e3_transport_core6.json"
JOINT_MEMBER = EF / "gate8" / "members" / "X9_m6_C_SLC1_SLC1992_6IIq2.json"
MEMBER_KEY = "core6"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"

ROUTES = (MEMBER_KEY, "champion", "bmc11", "cw12b", "slc_q4", "tuned_q8")
ROUTE_LABEL = {
    MEMBER_KEY: JOINT_LABEL,
    "champion": r"band member $E_{6,2}$",
    "bmc11": "BMC11",
    "cw12b": "CW12B",
    "slc_q4": "SLC q4",
    "tuned_q8": "tuned q8",
}
ROUTE_COLOR = {
    MEMBER_KEY: "#7B3294",
    "champion": "#0072B2",
    "bmc11": "#000000",
    "cw12b": "#8C510A",
    "slc_q4": "#D55E00",
    "tuned_q8": "#CC79A7",
}
ROUTE_MARKER = {
    MEMBER_KEY: "o",
    "champion": "s",
    "bmc11": "P",
    "cw12b": "D",
    "slc_q4": "v",
    "tuned_q8": "^",
}

N_1D = 128
CARRIERS = (0.4, 0.5, 0.6)
MAIN_CARRIER = 0.5
LEVEL = 0.01
HORIZONS = np.logspace(-3.0, 2.0, 1001)
QUOTED = (10.0, 100.0)
RELATIVE_TOLERANCE = 1.0e-9


def _cell(route: dict, theta_c: float):
    """Return the packet band and the symbol error of one route and carrier."""
    packet = g4b.packet(N_1D, theta_c)
    live = g4b.active_mask(packet["amp"])
    theta = packet["theta"][live]
    amp = packet["amp"][live]
    return amp, theta, g4b.relative_symbol_error(route["kappa1"], theta)


def _stored() -> dict:
    """Return the stored one-dimensional cells keyed by carrier and route."""
    cells = json.loads(TRANSPORT.read_text())["cells_1d"]
    out = {}
    for key, block in cells.items():
        out[round(float(key) / np.pi, 3)] = block
    return out


def main() -> int:
    g9e3.configure(member=JOINT_MEMBER, tag=MEMBER_KEY)
    table = g9e3.routes_by_key()
    stored = _stored()

    curves = {}
    crossings = {name: {} for name in ROUTES}
    receipt = []
    for carrier in CARRIERS:
        theta_c = carrier * np.pi
        block = stored[round(carrier, 3)]
        for name in ROUTES:
            amp, theta, e1 = _cell(table[name], theta_c)
            horizon = g4b.crossing_horizon(amp, theta, e1, N_1D, LEVEL)
            reference = float(block[name]["horizon_at_1_percent"])
            if abs(horizon - reference) > RELATIVE_TOLERANCE * reference:
                raise RuntimeError(
                    f"{name} at {carrier} pi crossing drift: "
                    f"{horizon} against {reference}")
            crossings[name][carrier] = horizon
            receipt.append((carrier, name, horizon, reference))
            for quoted in QUOTED:
                fresh = g4b.predicted_error(amp, theta, e1, N_1D, quoted)
                held = float(block[name][f"error_at_H{int(quoted)}"])
                if abs(fresh - held) > RELATIVE_TOLERANCE * held:
                    raise RuntimeError(
                        f"{name} at {carrier} pi error at H = {int(quoted)} "
                        f"drift: {fresh} against {held}")
            if carrier == MAIN_CARRIER:
                curves[name] = np.array(
                    [g4b.predicted_error(amp, theta, e1, N_1D, float(H))
                     for H in HORIZONS])

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 12.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
    })
    fig, (ax_run, ax_band) = plt.subplots(
        1, 2, figsize=(7.9, 3.7),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.26})

    for name in ROUTES:
        ax_run.loglog(HORIZONS, np.maximum(curves[name], 1.0e-9),
                      color=ROUTE_COLOR[name], lw=1.4,
                      label=ROUTE_LABEL[name])
        ax_run.plot([crossings[name][MAIN_CARRIER]], [LEVEL], marker="o",
                    ms=4.6, mfc="white", mec=ROUTE_COLOR[name], mew=1.2,
                    ls="none", zorder=6)
    ax_run.axhline(LEVEL, color="0.55", lw=0.7, ls=(0, (4, 2.5)), zorder=2)
    ax_run.text(1.6e-3, LEVEL * 1.35, "one percent", fontsize=12.5,
                color="0.45", ha="left", va="bottom")
    ax_run.set_xlabel("box crossings $H$")
    ax_run.set_ylabel("relative packet error")
    ax_run.set_xlim(1.0e-3, 1.0e2)
    ax_run.set_ylim(1.0e-5, 3.0)
    ax_run.grid(axis="y", color="0.93", lw=0.5)
    ph.quiet_axis(ax_run)
    ax_run.set_title("(a) carrier $\\theta_c = 0.5\\,\\pi$, four points per\n"
                     "wavelength, $N = 128$", loc="left", fontsize=12.5,
                     color=ph.CLASSY_ACCENT, fontstyle="italic", pad=4.0)

    for name in ROUTES:
        values = [crossings[name][carrier] for carrier in CARRIERS]
        ax_band.semilogy(CARRIERS, values, color=ROUTE_COLOR[name], lw=1.4,
                         marker=ROUTE_MARKER[name], ms=4.6)
    ax_band.set_xlabel(r"carrier $\theta_c/\pi$")
    ax_band.set_ylabel("crossings within one percent")
    ax_band.set_xlim(0.37, 0.63)
    ax_band.set_xticks(list(CARRIERS))
    ax_band.set_xticklabels([f"{c:.1f}" for c in CARRIERS])
    ax_band.set_ylim(3.0e-3, 1.0e2)
    ax_band.grid(axis="y", color="0.93", lw=0.5)
    ph.quiet_axis(ax_band)
    ax_band.set_title("(b) crossings against the carrier", loc="left",
                      fontsize=12.5, color=ph.CLASSY_ACCENT, fontstyle="italic",
                      pad=4.0)

    handles, labels = ax_run.get_legend_handles_labels()
    legend = ph.make_overhead_legend(fig, handles, labels, ncol=6,
                                     bbox=(0.5, 1.005), rect=(0, 0, 1, 0.86),
                                     fontsize=12.5, handslcngth=1.7,
                                     columnspacing=1.1, handletextpad=0.4)
    for handle in legend.legend_handles:
        handle.set_linewidth(2.2)
    for name, marker in ROUTE_MARKER.items():
        index = ROUTES.index(name)
        legend.legend_handles[index].set_marker(marker)
        legend.legend_handles[index].set_markersize(4.6)
    fig.subplots_adjust(top=0.795, bottom=0.135, left=0.085, right=0.99,
                        wspace=0.26)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_transport.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("crossings within one percent, recomputed against "
          "gate9/e3_transport_core6.json:")
    for carrier, name, fresh, reference in receipt:
        print(f"  theta_c = {carrier:.1f} pi  {name:10s} "
              f"{fresh:.9f} against {reference:.9f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
