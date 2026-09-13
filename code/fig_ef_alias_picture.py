# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""The alias picture: transform zeros, shifted copies, and the alias average.

Panel (a) draws the generator transform on xi/pi in [-3, 3] for two fields of
width ten: the polynomial rung E_{10,0} (BMC9), whose transform sinc^{10}(xi/2)
vanishes only at the lattice points 2 pi k, and the band member
E_{6,2}(0.5635 pi, 0.6559 pi) of results/exponential_family/members/
E62_champion.json, whose two conjugate pairs split each lattice zero into the
pair 2 pi k +- omega_j. The band (0, pi] carried by the grid is shaded.

Panel (b) draws the three shifted copies phi_hat(theta + 2 pi k) for
k = -1, 0, +1 of the band member over theta in (0, pi]. The copies with k != 0
vanish at omega_1 and omega_2, which is the statement that each conjugate pair
annihilates its own aliases. At theta = 0.6 pi the three stems carry the alias
weights w_k = phi_hat(theta + 2 pi k) / b(theta), which sum to one, and the
modified wavenumber is their average kappa_1 = sum_k (theta + 2 pi k) w_k.

Panel (c) repeats the construction for the dominance member
X9_m6_C_SLC1_SLC1992_6IIq2 of results/exponential_family/gate8/members/,
whose transform carries the further factor P(xi^2). The real root
xi_0 = 1.1919 pi of P is reached by the k = -1 copy at
theta_0 = 2 pi - xi_0 = 0.8081 pi, where that copy vanishes: the soft contact.

Every curve comes from family.ExponentialGenerator and
gate8_family.PolynomialGenerator applied to the tracked member records, and the
integer samples each generator returns are checked against the samples stored
in those records before the figure is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import family as fam  # noqa: E402
import gate8_family as g8  # noqa: E402
import plotting_helpers as ph  # noqa: E402

BAND_MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
DOMINANCE = (ROOT / "results" / "exponential_family" / "gate8" / "members"
             / "X9_m6_C_SLC1_SLC1992_6IIq2.json")
OUTDIR = ROOT / "figures_paper"

C_RUNG = "#009E73"
C_BAND = "#0072B2"
C_DOM = "#7B3294"
C_GUIDE = "0.45"
C_SHADE = "#E8EEF4"

BASE_FONT = 11.1
SAMPLE_TOLERANCE = 1.0e-12
FLOOR_A = 1.0e-8
FLOOR_BC = 1.0e-12
TOP_BC = 3.0e3
LABEL_STEM = 12.0
LABEL_TEXT = 16.0
PROBE = 0.6 * np.pi


def _refined(lo: float, hi: float, zeros, points: int = 4001) -> np.ndarray:
    """Return a uniform grid on [lo, hi] refined geometrically around zeros."""
    grid = [np.linspace(lo, hi, points)]
    offsets = np.concatenate([-np.geomspace(1.0e-9, 0.04, 70)[::-1], [0.0],
                              np.geomspace(1.0e-9, 0.04, 70)])
    for zero in zeros:
        grid.append(zero + offsets)
    values = np.unique(np.concatenate(grid))
    return values[(values >= lo) & (values <= hi)]


def _check_samples(generator, stored) -> float:
    """Return the drift of the generator samples against the stored samples."""
    samples = generator.integer_samples()
    half = samples.n >= 0
    drift = float(np.max(np.abs(samples.beta[half] - np.asarray(stored, dtype=float))))
    if drift > SAMPLE_TOLERANCE:
        raise RuntimeError(f"integer sample drift {drift:.3e}")
    return drift


def _copies(generator, theta: np.ndarray) -> dict:
    """Return the three shifted transform copies and the alias weights."""
    b = generator.symbols(theta)[0]
    out = {"b": b}
    for k in (-1, 0, 1):
        out[k] = generator.fourier(theta + 2.0 * np.pi * k)
    probe = np.array([PROBE])
    b_probe = generator.symbols(probe)[0][0]
    out["weights"] = {k: float(generator.fourier(probe + 2.0 * np.pi * k)[0] / b_probe)
                      for k in (-1, 0, 1)}
    out["kappa1"] = float(generator.symbols(probe)[1][0])
    return out


def _copy_panel(ax, theta, data, color, contacts, soft, tag, title) -> None:
    """Draw the three shifted copies, the probe line and the three alias stems."""
    styles = {0: ("-", 1.8), -1: ((0, (5, 2)), 1.5), 1: ((0, (1, 1.6)), 1.5)}
    for k in (0, -1, 1):
        style, width = styles[k]
        ax.semilogy(theta / np.pi, np.maximum(np.abs(data[k]), FLOOR_BC),
                    color=color, lw=width, ls=style)
    ax.axvline(PROBE / np.pi, color=C_GUIDE, lw=0.8)
    for offset, k, name in ((-0.135, -1, r"$w_{-1}$"), (0.0, 0, r"$w_{0}$"),
                            (0.135, 1, r"$w_{+1}$")):
        value = max(abs(data["weights"][k]), FLOOR_BC)
        x = PROBE / np.pi + offset
        ax.plot([x, x], [FLOOR_BC, value], color="#444444", lw=1.0, zorder=5)
        ax.plot([x, x], [value, LABEL_STEM], color="#999999", lw=0.6,
                ls=(0, (1, 2)), zorder=5)
        ax.plot([x], [value], marker="o", ms=4.2, mfc="white", mec="#444444",
                mew=1.1, ls="none", zorder=6)
        ax.text(x, LABEL_TEXT, name, ha="center", va="bottom",
                fontsize=BASE_FONT, color="#444444", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.9))
    ax.plot([value / np.pi for value in contacts], [FLOOR_BC] * len(contacts),
            ls="none", marker="^", ms=5.0, color=color, clip_on=False, zorder=7)
    ax.plot([value / np.pi for value in soft], [FLOOR_BC] * len(soft),
            ls="none", marker="^", ms=5.0, mfc="white", mec=color, mew=1.1,
            clip_on=False, zorder=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(FLOOR_BC, TOP_BC)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0])
    ax.set_xlabel(r"$\theta/\pi$")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.set_title(f"{tag} {title}", fontsize=BASE_FONT, color=color, pad=3.0)
    ph.quiet_axis(ax)


def main() -> int:
    band_record = json.loads(BAND_MEMBER.read_text())
    dom_record = json.loads(DOMINANCE.read_text())

    band = fam.ExponentialGenerator(int(band_record["m"]), band_record["omegas"])
    rung = fam.ExponentialGenerator(int(band_record["p"]) + 1, ())
    dominance = g8.from_record(dom_record)
    drift_band = _check_samples(band, band_record["beta"])
    drift_dom = _check_samples(dominance, dom_record["beta"])

    band_omegas = [float(w) for w in band_record["omegas"]]
    dom_omegas = [float(w) for w in dom_record["omegas"]]
    soft_theta = [float(v) for v in dom_record["polynomial_roots"]["soft_contacts_theta"]]
    xi_0 = [float(v) for v in dom_record["polynomial_roots"]["real_positive_xi"]]

    rung_zeros = [-2.0 * np.pi, 2.0 * np.pi]
    band_zeros = sorted(sign * (2.0 * np.pi + pick * w)
                        for w in band_omegas for pick in (-1.0, 1.0)
                        for sign in (-1.0, 1.0))
    xi = _refined(-3.0 * np.pi, 3.0 * np.pi, rung_zeros + band_zeros)
    theta = _refined(1.0e-4, np.pi, band_omegas + dom_omegas + soft_theta)
    theta = theta[theta > 0.0]

    band_copies = _copies(band, theta)
    dom_copies = _copies(dominance, theta)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    })
    fig = plt.figure(figsize=(7.6, 5.70))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], hspace=0.52,
                            wspace=0.30)
    ax_hat = fig.add_subplot(grid[0, :])
    ax_band = fig.add_subplot(grid[1, 0])
    ax_dom = fig.add_subplot(grid[1, 1])

    ax_hat.axvspan(0.0, 1.0, color=C_SHADE, lw=0.0, zorder=0)
    ax_hat.semilogy(xi / np.pi, np.maximum(np.abs(rung.fourier(xi)), FLOOR_A),
                    color=C_RUNG, lw=1.5, ls=(0, (5, 2)),
                    label=r"polynomial rung $E_{10,0}$ (BMC9)")
    ax_hat.semilogy(xi / np.pi, np.maximum(np.abs(band.fourier(xi)), FLOOR_A),
                    color=C_BAND, lw=1.8, label=r"band member $E_{6,2}$")
    ax_hat.plot([value / np.pi for value in rung_zeros], [FLOOR_A] * 2,
                ls="none", marker="^", ms=5.4, color=C_RUNG, clip_on=False,
                zorder=6)
    ax_hat.plot([value / np.pi for value in band_zeros],
                [FLOOR_A] * len(band_zeros), ls="none", marker="^", ms=5.4,
                color=C_BAND, clip_on=False, zorder=6)
    ax_hat.set_xlim(-3.0, 3.0)
    ax_hat.set_ylim(FLOOR_A, 3.0)
    ax_hat.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax_hat.set_yticks([1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    ax_hat.set_xlabel(r"$\xi/\pi$")
    ax_hat.set_ylabel(r"$|\hat{\varphi}(\xi)|$")
    ax_hat.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax_hat)
    ax_hat.text(0.5, 0.055, "band", transform=ax_hat.transAxes, ha="center",
                va="bottom", fontsize=BASE_FONT, color="#3A5A75")
    ax_hat.set_title("(a) generator transforms and their zeros",
                     fontsize=BASE_FONT, color=ph.CLASSY_ACCENT, pad=3.0)
    for x_key, y_key, color, text in ((-2.90, 0.55, C_RUNG, r"$2\pi k$"),
                                      (-2.90, 0.045, C_BAND, r"$2\pi\pm\omega_j$")):
        ax_hat.plot([x_key], [y_key], ls="none", marker="^", ms=5.4,
                    color=color, zorder=6)
        ax_hat.text(x_key + 0.13, y_key, text, ha="left", va="center",
                    fontsize=BASE_FONT, color=color)

    _copy_panel(ax_band, theta, band_copies, C_BAND, band_omegas, [], "(b)",
                "band member")
    _copy_panel(ax_dom, theta, dom_copies, C_DOM, dom_omegas, soft_theta, "(c)",
                "dominance member")
    ax_band.set_ylabel(r"$|\hat{\varphi}(\theta+2\pi k)|$")
    ax_dom.set_ylabel(r"$|\hat{\psi}(\theta+2\pi k)|$")
    ax_dom.annotate("soft contact", xy=(soft_theta[0] / np.pi, 2.0e-12),
                    xytext=(0.035, 4.0e-10), fontsize=BASE_FONT, color=C_DOM,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_DOM, lw=0.7,
                                    shrinkA=1.0, shrinkB=2.0))

    neutral = "0.35"
    handles = [
        Line2D([], [], color=C_RUNG, lw=1.5, ls=(0, (5, 2))),
        Line2D([], [], color=C_BAND, lw=1.8),
        Line2D([], [], color=neutral, lw=1.8),
        Line2D([], [], color=neutral, lw=1.5, ls=(0, (5, 2))),
        Line2D([], [], color=neutral, lw=1.5, ls=(0, (1, 1.6))),
        Line2D([], [], color="#444444", lw=1.0, marker="o", ms=4.2,
               mfc="white", mec="#444444", mew=1.1),
        Line2D([], [], color=neutral, ls="none", marker="^", ms=5.0),
        Line2D([], [], color=neutral, ls="none", marker="^", ms=5.0,
               mfc="white", mew=1.1),
    ]
    labels = [r"$E_{10,0}$ (BMC9)", r"$E_{6,2}$ band member", r"$k=0$",
              r"$k=-1$", r"$k=+1$", r"alias weight $w_k$", "transform zero",
              "soft contact"]
    ph.make_overhead_legend(fig, handles, labels, ncol=4, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.87), fontsize=BASE_FONT,
                            handslcngth=2.0, columnspacing=1.3,
                            handletextpad=0.45)
    fig.text(0.5, -0.035,
             r"$\kappa_1(\theta)=\sum_k(\theta+2\pi k)\,w_k(\theta)$",
             ha="center", va="bottom", fontsize=BASE_FONT,
             color=ph.CLASSY_ACCENT)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_alias_picture.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("alias picture summary:")
    print(f"  band member omegas   {band_omegas[0] / np.pi:.4f} pi, "
          f"{band_omegas[1] / np.pi:.4f} pi; sample drift {drift_band:.2e}")
    print(f"  dominance member omegas {dom_omegas[0] / np.pi:.4f} pi, "
          f"{dom_omegas[1] / np.pi:.4f} pi; sample drift {drift_dom:.2e}")
    print(f"  polynomial root xi_0 = {xi_0[0] / np.pi:.4f} pi reaches "
          f"theta_0 = {soft_theta[0] / np.pi:.4f} pi")
    for name, data in (("band member", band_copies),
                       ("dominance member", dom_copies)):
        weights = data["weights"]
        total = sum(weights.values())
        print(f"  {name} at 0.6 pi: w_-1 = {weights[-1]:.4e}, "
              f"w_0 = {weights[0]:.10f}, w_+1 = {weights[1]:.4e}, "
              f"three-term sum {total:.10f}, kappa_1 = {data['kappa1']:.10f}, "
              f"theta = {PROBE:.10f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
