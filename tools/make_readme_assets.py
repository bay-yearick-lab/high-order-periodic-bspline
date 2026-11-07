"""Regenerate the README figures from the receipts in results/.

    uv run python tools/make_readme_assets.py

Writes assets/hero.png, assets/card_tgv_accuracy.png, assets/card_transport.png
and assets/card_wall_cost.png. The hero uses the three Taylor-Green vorticity
frames in assets/source/.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
EF = ROOT / "results" / "exponential_family"
STYLE = ROOT / "tools" / "lab.mplstyle"

INK, TEAL, AMBER, SLATE, CORAL, GRAY = "#1B2A41", "#0F9D8A", "#F2A541", "#3D5A80", "#E07A5F", "#8D99AE"
DEEP_TEAL = "#0A6B5E"

if STYLE.exists():
    plt.style.use(str(STYLE))


def hero() -> None:
    fig = plt.figure(figsize=(16, 5.6), dpi=100)
    fig.patch.set_facecolor(INK)
    frames = [("8.25", "tgv_vorticity_t8.25.png"), ("9.66", "tgv_vorticity_t9.66.png")]
    hgt = 0.80
    w, gap = hgt * 5.6 / 16, 0.008
    x0 = 0.985 - 2 * w - gap
    for i, (t, name) in enumerate(frames):
        ax = fig.add_axes([x0 + i * (w + gap), (1 - hgt) / 2, w, hgt])
        ax.imshow(mpimg.imread(ASSETS / "source" / name))
        ax.set_axis_off()
        ax.text(0.035, 0.965, f"t = {t}", transform=ax.transAxes, color="white", fontsize=12,
                va="top", ha="left", fontweight="bold")
    fig.text(0.035, 0.74, "high-order-periodic-bspline", color="white", fontsize=26, fontweight="bold")
    fig.text(0.035, 0.625, "Exponential B-spline operators for Navier-Stokes", color="#D7DEE8",
             fontsize=16)
    lines = ["One banded solve: compatible first and second derivatives,",
             "summation-by-parts conservation, spectral-like resolution,",
             "and off-grid reads of one field.",
             "",
             "Taylor-Green vortex DNS, vorticity magnitude",
             "(maximum-intensity projection), degree-4 collocation."]
    for k, line in enumerate(lines):
        fig.text(0.035, 0.49 - k * 0.066, line, color="#AEB9C9" if k < 3 else GRAY, fontsize=13 if k < 3 else 11.5)
    fig.add_artist(plt.Line2D([0.035, 0.36], [0.585, 0.585], color=TEAL, lw=3, transform=fig.transFigure))
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(ASSETS / "hero.png", dpi=100, facecolor=INK)
    plt.close(fig)


def card_tgv() -> None:
    d = json.loads((EF / "gate9" / "e2_decision_core6.json").read_text())["e_full"]
    routes = [("sl", "spectral-like compact\n(pentadiagonal, cond 93.4)", AMBER),
              ("core6", "enriched member\nX6 x P2 (cond 93.4)", TEAL),
              ("bmc_p11", "polynomial rung\nBMC11", SLATE),
              ("exp", "band member\nE(6,2) (cond 88.7)", DEEP_TEAL)]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    width = 0.19
    for j, (key, label, color) in enumerate(routes):
        vals = [d["N64"][key], d["N128"][key]]
        xs = np.arange(2) + (j - 1.5) * width
        ax.bar(xs, vals, width * 0.92, color=color, label=label, zorder=3)
        if key != "sl":
            for x, n, v in zip(xs, ("N64", "N128"), vals):
                ax.text(x, v * 1.12, f"{d[n]['sl'] / v:.1f}x", ha="center", va="bottom", fontsize=9.5,
                        color=INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 2.2e-2)
    ax.set_xticks([0, 1], ["$64^3$", "$128^3$"])
    ax.set_ylabel("relative RMS velocity error at $t^*$")
    ax.set_title("Dealiased Taylor-Green: 4.4x and 38-43x lower error", loc="left")
    ax.legend(loc="upper right", fontsize=8.8, ncol=2, handslcngth=1.2, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "card_tgv_accuracy.png", dpi=200)
    plt.close(fig)


def card_transport() -> None:
    d = json.loads((EF / "gate9" / "e3_transport_core6.json").read_text())["cells_1d"]
    cells = [("1.256637", "0.4$\\pi$"), ("1.570796", "0.5$\\pi$"), ("1.884956", "0.6$\\pi$")]
    routes = [("slc_q4", "spectral-like compact", AMBER), ("core6", "enriched member X6 x P2", TEAL),
              ("champion", "band member E(6,2)", DEEP_TEAL)]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    width = 0.25
    for j, (key, label, color) in enumerate(routes):
        vals = [d[c][key]["horizon_at_1_percent"] for c, _ in cells]
        xs = np.arange(3) + (j - 1) * width
        ax.bar(xs, vals, width * 0.92, color=color, label=label, zorder=3)
        if key != "slc_q4":
            for x, (c, _), v in zip(xs, cells, vals):
                ax.text(x, v * 1.12, f"{d[c][key]['crossing_ratio_over_slc_q4']:.1f}x", ha="center",
                        va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 20)
    ax.set_xticks(range(3), [f"carrier $\\theta_c$ = {lab}" for _, lab in cells])
    ax.set_ylabel("box crossings before 1% packet error")
    ax.set_title("Dispersive transport: 37x longer horizon at 0.5$\\pi$", loc="left")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "card_transport.png", dpi=200)
    plt.close(fig)


def card_wall() -> None:
    rows = list(csv.DictReader(open(ROOT / "results" / "ibm_cert" / "full_results.csv")))
    worst = defaultdict(lambda: [0.0, 0.0])  # (route, N) -> [worst trace error, worst total time]
    kind = {}
    for r in rows:
        key = (r["logical_route_label"], int(r["N"]))
        worst[key][0] = max(worst[key][0], float(r["trace_error"]))
        worst[key][1] = max(worst[key][1], float(r["total_time_seconds"]))
        kind[r["logical_route_label"]] = r["route_kind"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    labels = sorted(kind)
    strongest = "C10|common_L12"
    for lab in labels:
        pts = sorted((n, *worst[(lab, n)]) for (l2, n) in worst if l2 == lab)
        t = [p[2] for p in pts]
        e = [p[1] for p in pts]
        if lab == "BMC9":
            ax.plot(t, e, "-o", color=TEAL, lw=2.6, ms=7, zorder=5, label="BMC9")
        elif kind[lab] == "bmc":
            ax.plot(t, e, "-o", color=TEAL, lw=1.0, ms=4, alpha=0.45, zorder=4)
        elif lab == strongest:
            ax.plot(t, e, "-s", color=AMBER, lw=2.6, ms=7, zorder=5,
                    label="strongest compact: C10 + 12-point layer")
        elif kind[lab] == "fourier":
            continue
        else:
            ax.plot(t, e, "-s", color=AMBER, lw=0.9, ms=3.5, alpha=0.35, zorder=3)
    ax.plot([], [], "-o", color=TEAL, lw=1.0, ms=4, alpha=0.45, label="BMC5, BMC7, BMC11")
    ax.plot([], [], "-s", color=AMBER, lw=0.9, ms=3.5, alpha=0.35, label="other compact routes (C6-CW12)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("complete one-solve time (s), worst marker shift")
    ax.set_ylabel("wall-trace error, worst marker shift")
    ax.set_title("Immersed wall: BMC9 leads the error-time front", loc="left")
    ax.set_xticks([0.03, 0.05, 0.1, 0.2], ["0.03", "0.05", "0.1", "0.2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="lower left", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(ASSETS / "card_wall_cost.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    hero()
    card_tgv()
    card_transport()
    card_wall()
    print("wrote", ", ".join(p.name for p in sorted(ASSETS.glob("*.png"))))


if __name__ == "__main__":
    main()
