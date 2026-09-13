# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""The dials: conjugate pairs split the transform zeros, a polynomial adds one.

Panel (a) draws the generator transform on [pi, 3 pi] for three fields with the
same polynomial core m = 6: the bare core E_{6,0}, the single-pair field
E_{6,1}(omega_1) and the two-pair field E_{6,2}(omega_1, omega_2), at the
frequencies 0.5635 pi and 0.6559 pi of the band member
results/exponential_family/members/E62_champion.json. The bare core vanishes
only at 2 pi. Each conjugate pair moves one zero off the lattice: the first
pair opens the pair 2 pi +- omega_1, the second opens 2 pi +- omega_2, and the
order of contact at 2 pi drops accordingly.

Panel (b) draws the first-derivative error e_1 on (0, pi) for those three, for
the confluent field E_{6,2}(omega_c, omega_c) at the mean
omega_c = 0.6097 pi of the band member's two frequencies, and for the
dominance member X9_m6_C_SLC1_SLC1992_6IIq2. Solid curves are the full alias
sum of the alias-average theorem; dashed curves are the one-alias model, which
keeps only k = -1 and carries the polynomial weight P((2 pi - theta)^2). Two
separated frequencies give two sign changes, the confluent pair gives one
tangential touch, and the polynomial root of the dominance member gives a
further sign change at the soft contact 0.8081 pi.

Every curve is evaluated by gate8_family.PolynomialGenerator, whose alias route
carries an explicit tail; the one-alias model uses the same slot factors
through gate7_family.slot_rational, and it is checked against gate2_rule's own
one-alias formula for the fields with no polynomial factor.
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

import gate2_rule as g2  # noqa: E402
import gate7_family as g7  # noqa: E402
import gate8_family as g8  # noqa: E402
import plotting_helpers as ph  # noqa: E402

BAND_MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
DOMINANCE = (ROOT / "results" / "exponential_family" / "gate8" / "members"
             / "X9_m6_C_SLC1_SLC1992_6IIq2.json")
OUTDIR = ROOT / "figures_paper"

C_CORE = "#6B6B6B"
C_ONE = "#009E73"
C_TWO = "#0072B2"
C_CONF = "#D55E00"
C_DOM = "#7B3294"
C_GUIDE = "0.45"

BASE_FONT = 11.1
FLOOR_HAT = 1.0e-12
LINTHRESH = 1.0e-8
MODEL_TOLERANCE = 1.0e-10
SAMPLE_TOLERANCE = 1.0e-12


def _refined(lo: float, hi: float, zeros, points: int = 4001) -> np.ndarray:
    """Return a uniform grid on [lo, hi] refined geometrically around zeros."""
    grid = [np.linspace(lo, hi, points)]
    offsets = np.concatenate([-np.geomspace(1.0e-9, 0.05, 80)[::-1], [0.0],
                              np.geomspace(1.0e-9, 0.05, 80)])
    for zero in zeros:
        grid.append(zero + offsets)
    values = np.unique(np.concatenate(grid))
    return values[(values >= lo) & (values <= hi)]


def _one_alias(generator, theta: np.ndarray) -> np.ndarray:
    """Return the one-alias model of e_1, with the polynomial weight of k = -1."""
    theta = np.asarray(theta, dtype=float)

    def rational(x: np.ndarray) -> np.ndarray:
        out = x ** (-float(generator.m))
        for slot in generator.slots:
            out = out * g7.slot_rational(slot, x)
        return out

    shifted = theta - 2.0 * np.pi
    numerator = generator.polynomial(shifted * shifted) * rational(shifted)
    denominator = generator.polynomial(theta * theta) * rational(theta)
    return -(2.0 * np.pi / theta) * numerator / denominator


def _generator(m: int, omegas, poly=()) -> g8.PolynomialGenerator:
    """Return the polynomial-augmented generator of one imaginary-pair field."""
    return g8.PolynomialGenerator(m, tuple((0.0, float(w)) for w in omegas), poly)


def main() -> int:
    band_record = json.loads(BAND_MEMBER.read_text())
    dom_record = json.loads(DOMINANCE.read_text())
    omega_1, omega_2 = (float(w) for w in band_record["omegas"])
    omega_c = 0.5 * (omega_1 + omega_2)
    soft = [float(v) for v in dom_record["polynomial_roots"]["soft_contacts_theta"]]

    core = _generator(6, ())
    one_pair = _generator(6, (omega_1,))
    two_pair = _generator(6, (omega_1, omega_2))
    confluent = _generator(6, (omega_c, omega_c))
    dominance = g8.from_record(dom_record)

    samples = dominance.integer_samples()
    drift = float(np.max(np.abs(samples.beta[samples.n >= 0]
                                - np.asarray(dom_record["beta"], dtype=float))))
    if drift > SAMPLE_TOLERANCE:
        raise RuntimeError(f"dominance member sample drift {drift:.3e}")

    hat_fields = [
        (r"$E_{6,0}$", C_CORE, (0, (1, 1.6)), 1.5, core, [2.0 * np.pi]),
        (r"$E_{6,1}(\omega_1)$", C_ONE, (0, (5, 2)), 1.5, one_pair,
         [2.0 * np.pi, 2.0 * np.pi - omega_1, 2.0 * np.pi + omega_1]),
        (r"$E_{6,2}(\omega_1,\omega_2)$", C_TWO, "-", 1.8, two_pair,
         [2.0 * np.pi, 2.0 * np.pi - omega_1, 2.0 * np.pi + omega_1,
          2.0 * np.pi - omega_2, 2.0 * np.pi + omega_2]),
    ]
    zeros_all = sorted({value for _, _, _, _, _, zeros in hat_fields for value in zeros})
    xi = _refined(np.pi, 3.0 * np.pi, zeros_all)

    error_fields = [
        (r"$E_{6,0}$", C_CORE, core),
        (r"$E_{6,1}(\omega_1)$", C_ONE, one_pair),
        (r"$E_{6,2}(\omega_1,\omega_2)$", C_TWO, two_pair),
        (r"confluent $E_{6,2}(\omega_c,\omega_c)$", C_CONF, confluent),
        ("dominance member", C_DOM, dominance),
    ]
    theta = _refined(1.0e-4, np.pi - 1.0e-4,
                     [omega_1, omega_2, omega_c] + soft + list(dominance.omegas))

    curves = {}
    for label, _, generator in error_fields:
        full = generator.symbols(theta)[1] / theta - 1.0
        model = _one_alias(generator, theta)
        curves[label] = (full, model)

    check = float(np.max(np.abs(
        _one_alias(two_pair, theta)
        - g2.one_alias_e1(theta, 6, (omega_1, omega_2)))))
    if check > MODEL_TOLERANCE:
        raise RuntimeError(f"one-alias model deviation {check:.3e}")

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    })
    fig, (ax_hat, ax_err) = plt.subplots(1, 2, figsize=(7.6, 3.65))

    for row, (label, color, style, width, generator, zeros) in enumerate(hat_fields):
        ax_hat.semilogy(xi / np.pi, np.maximum(np.abs(generator.fourier(xi)), FLOOR_HAT),
                        color=color, lw=width, ls=style, label=label)
        height = 10.0 ** (-11.4 + 0.85 * row)
        ax_hat.plot([value / np.pi for value in zeros], [height] * len(zeros),
                    ls="none", marker="v", ms=5.0, color=color, zorder=6)
    ax_hat.axvline(2.0, color=C_GUIDE, lw=0.6, ls=(0, (1, 2.2)))
    ax_hat.set_xlim(1.0, 3.0)
    ax_hat.set_ylim(FLOOR_HAT, 1.0)
    ax_hat.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax_hat.set_yticks([1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0])
    ax_hat.set_xlabel(r"$\xi/\pi$")
    ax_hat.set_ylabel(r"$|\hat{\varphi}(\xi)|$")
    ax_hat.grid(axis="y", color="0.92", lw=0.5)
    ax_hat.set_title("(a) the zeros at $2\\pi$ split", fontsize=BASE_FONT,
                     color=ph.CLASSY_ACCENT, pad=3.0)
    ph.quiet_axis(ax_hat)
    ax_hat.annotate(r"$2\pi-\omega_2$", xy=((2.0 * np.pi - omega_2) / np.pi, 3.0e-7),
                    xytext=(1.02, 0.30), fontsize=BASE_FONT, color=C_TWO,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_TWO, lw=0.7,
                                    shrinkA=2.0, shrinkB=2.0))
    ax_hat.annotate(r"$2\pi+\omega_2$", xy=((2.0 * np.pi + omega_2) / np.pi, 3.0e-7),
                    xytext=(2.98, 0.30), fontsize=BASE_FONT, color=C_TWO,
                    ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_TWO, lw=0.7,
                                    shrinkA=2.0, shrinkB=2.0))

    for label, color, _ in error_fields:
        full, model = curves[label]
        ax_err.plot(theta / np.pi, full, color=color, lw=1.6)
        ax_err.plot(theta / np.pi, model, color=color, lw=1.1, ls=(0, (4, 2)))
    ax_err.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.55)
    ax_err.axhline(0.0, color="0.75", lw=0.5)
    for value, color in ((omega_1, C_TWO), (omega_2, C_TWO), (omega_c, C_CONF),
                         (soft[0], C_DOM)):
        ax_err.plot([value / np.pi], [0.0], ls="none", marker="o", ms=4.4,
                    mfc="white", mec=color, mew=1.2, zorder=7)
    ax_err.set_xlim(0.0, 1.0)
    ax_err.set_ylim(-2.0, 2.0)
    ax_err.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_err.set_yticks([-1.0, -1.0e-4, 0.0, 1.0e-4])
    ax_err.set_xlabel(r"$\theta/\pi$")
    ax_err.set_ylabel(r"$e_1(\theta)$")
    ax_err.grid(axis="y", color="0.92", lw=0.5)
    ax_err.set_title("(b) where the error crosses zero", fontsize=BASE_FONT,
                     color=ph.CLASSY_ACCENT, pad=3.0)
    ph.quiet_axis(ax_err)

    neutral = "0.35"
    handles = [Line2D([], [], color=color, lw=1.6) for _, color, _ in error_fields]
    handles += [Line2D([], [], color=neutral, lw=1.6),
                Line2D([], [], color=neutral, lw=1.1, ls=(0, (4, 2)))]
    labels = [label for label, _, _ in error_fields] + ["full alias sum",
                                                        "one-alias model"]
    ph.make_overhead_legend(fig, handles, labels, ncol=4, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.80), fontsize=BASE_FONT,
                            handslcngth=2.0, columnspacing=1.2,
                            handletextpad=0.45)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_dials.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    band = theta <= 2.0 * np.pi / 3.0
    print("dial summary:")
    print(f"  frequencies {omega_1 / np.pi:.4f} pi, {omega_2 / np.pi:.4f} pi, "
          f"confluent {omega_c / np.pi:.4f} pi, soft contact {soft[0] / np.pi:.4f} pi")
    print(f"  one-alias model against gate2_rule on the two-pair field {check:.2e}")
    print(f"  dominance member sample drift {drift:.2e}")
    for label, _, _ in error_fields:
        full, model = curves[label]
        print(f"  {label}: max |e_1| on the band {np.max(np.abs(full[band])):.3e}, "
              f"largest gap to the one-alias model on the band "
              f"{np.max(np.abs(full[band] - model[band])):.3e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
