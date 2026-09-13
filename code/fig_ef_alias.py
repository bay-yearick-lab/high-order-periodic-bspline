# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Alias anatomy: cancellation against small alias mass.

Both fields drawn here carry the same conditioning (93.37). The first is the
one-field realization of the published spectral-like first derivative,

    psi_hat = P(xi^2) phi_hat,   phi = E_{4,2}(0.6366 pi, 0.7321 pi),
    P(t) = 1 - 0.020209 t - 0.0030557 t^2,

read from results/exponential_family/gate8/realization.json. The second is the
member X9_m6_C_SLC1_SLC1992_6IIq2 of
results/exponential_family/gate8/members/, the everything member of the
manuscript: a core-6 generator with two imaginary pairs at 0.5870 pi and
0.7757 pi and a quadratic polynomial factor with a soft contact at 0.8081 pi,
of class C^4.

Panels (a) and (b) draw the alias weights

    w_k(theta) = psi_hat(theta + 2 pi k) / b(theta),   sum_k w_k = 1,

for k = 0, -1, +1 on (0, pi]. Panel (c) draws the errors they produce,

    e_1 = sum_{k != 0} (2 pi k / theta) w_k,
    e_2 = sum_{k != 0} ((theta + 2 pi k)^2 / theta^2 - 1) w_k.

The coefficients of e_1 have opposite signs for k = -1 and k = +1, so a large
alias weight can still leave a small first-derivative error; the coefficients
of e_2 are all positive, so the same weights are summed without cancellation.
That is why the realized field of the published pair, whose k = -1 weight runs
above the member's through the interior of the band (1.26e-4 against 2.61e-5 at
0.4 pi), is only about 1.5 times worse on the band maximum of e_1 and about 7.6
times worse on the band maximum of e_2. The producer prints the band maxima of
all four quantities.

Every curve is produced by gate8_family.PolynomialGenerator from the tracked
receipts; the integer samples the generator returns are checked against the
samples stored in those receipts before the figure is written.
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

import gate8_family as g8  # noqa: E402
import plotting_helpers as ph  # noqa: E402

REALIZATION = ROOT / "results" / "exponential_family" / "gate8" / "realization.json"
MEMBER = (ROOT / "results" / "exponential_family" / "gate8" / "members"
          / "X9_m6_C_SLC1_SLC1992_6IIq2.json")
OUTDIR = ROOT / "figures_paper"

C_SLC = "#D55E00"
C_MEMBER = "#7B3294"
C_ZERO = "0.55"
C_GUIDE = "0.45"

BAND_EDGE = 2.0 / 3.0
SAMPLE_TOLERANCE = 1.0e-12
FLOOR = 1.0e-8


def _records() -> tuple[dict, dict]:
    """Return the realized-field record and the member record."""
    realization = json.loads(REALIZATION.read_text())
    slc = realization["realizations"]["SLC1992"]["analysis"]["record"]
    member = json.loads(MEMBER.read_text())
    return slc, member


def _check_samples(generator, record: dict) -> None:
    """Verify the generator samples against the samples stored in the receipt."""
    samples = generator.integer_samples()
    half = samples.n >= 0
    drift = float(np.max(np.abs(samples.beta[half]
                                - np.asarray(record["beta"], dtype=float))))
    if drift > SAMPLE_TOLERANCE:
        raise RuntimeError(f"{record['name']}: sample drift {drift:.3e}")


def _refined_grid(zeros) -> np.ndarray:
    """Return a grid on (0, pi] refined geometrically around the listed zeros."""
    grid = [np.linspace(1.0e-4, np.pi, 3001)]
    offsets = np.concatenate([-np.geomspace(1.0e-7, 0.03, 90)[::-1], [0.0],
                              np.geomspace(1.0e-7, 0.03, 90)])
    for zero in zeros:
        grid.append(zero + offsets)
    values = np.unique(np.concatenate(grid))
    return values[(values > 0.0) & (values <= np.pi)]


def _weights(generator, theta: np.ndarray) -> dict:
    """Return the alias weights for k = 0, -1, +1 and the two nodal errors."""
    b, kappa1, kappa2 = generator.symbols(theta)
    out = {"b": b, "e1": kappa1 / theta - 1.0, "e2": kappa2 / theta**2 - 1.0}
    for k in (0, -1, 1):
        out[k] = generator.fourier(theta + 2.0 * np.pi * k) / b
    total = np.zeros(theta.shape)
    for k in range(-64, 65):
        total = total + generator.fourier(theta + 2.0 * np.pi * k) / b
    out["closure"] = float(np.max(np.abs(total - 1.0)))
    return out


def _panel(ax, theta: np.ndarray, data: dict, color: str, tag: str,
           title: str, exact, soft) -> None:
    """Draw the three alias weights of one field on a logarithmic axis."""
    ax.semilogy(theta / np.pi, np.maximum(np.abs(data[0]), FLOOR),
                color=C_ZERO, lw=1.0, ls=(0, (1, 1.6)))
    ax.semilogy(theta / np.pi, np.maximum(np.abs(data[-1]), FLOOR),
                color=color, lw=1.7)
    ax.semilogy(theta / np.pi, np.maximum(np.abs(data[1]), FLOOR),
                color=color, lw=1.2, ls=(0, (5, 2)))
    ax.axvline(BAND_EDGE, color=C_GUIDE, lw=0.6, ls=(0, (1, 2.2)))
    ax.plot([value / np.pi for value in exact], [FLOOR] * len(exact),
            ls="none", marker="^", ms=4.0, color=color, clip_on=False, zorder=6)
    ax.plot([value / np.pi for value in soft], [FLOOR] * len(soft),
            ls="none", marker="^", ms=4.0, mfc="none", mec=color, mew=0.9,
            clip_on=False, zorder=6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(FLOOR, 4.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"$\theta/\pi$")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.set_title(title, fontsize=10.9, color=color, pad=3.0)
    ph.quiet_axis(ax)
    ax.text(0.985, 0.03, tag, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.9, color=ph.CLASSY_ACCENT)


def main() -> int:
    slc_record, member_record = _records()
    slc = g8.from_record(slc_record)
    member = g8.from_record(member_record)
    _check_samples(slc, slc_record)
    _check_samples(member, member_record)

    contacts = [float(value) for generator in (slc, member)
                for value in generator.omegas]
    contacts += [float(value) for record in (slc_record, member_record)
                 for value in record["polynomial_roots"]["soft_contacts_theta"]]
    theta = _refined_grid(sorted(set(contacts)))

    slc_data = _weights(slc, theta)
    member_data = _weights(member, theta)

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 10.9,
        "axes.labelsize": 10.9,
        "xtick.labelsize": 10.9,
        "ytick.labelsize": 10.9,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.05))
    ax_slc, ax_member, ax_err = axes

    _panel(ax_slc, theta, slc_data, C_SLC, "(a)",
           "realized spectral-like field", slc.omegas,
           slc_record["polynomial_roots"]["soft_contacts_theta"])
    _panel(ax_member, theta, member_data, C_MEMBER, "(b)",
           "dominance member", member.omegas,
           member_record["polynomial_roots"]["soft_contacts_theta"])
    ax_slc.set_ylabel(r"$|w_k(\theta)|$")
    ax_member.set_yticklabels([])

    ax_err.semilogy(theta / np.pi, np.maximum(np.abs(slc_data["e1"]), 1e-9),
                    color=C_SLC, lw=1.7)
    ax_err.semilogy(theta / np.pi, np.maximum(np.abs(slc_data["e2"]), 1e-9),
                    color=C_SLC, lw=1.2, ls=(0, (5, 2)))
    ax_err.semilogy(theta / np.pi, np.maximum(np.abs(member_data["e1"]), 1e-9),
                    color=C_MEMBER, lw=1.7)
    ax_err.semilogy(theta / np.pi, np.maximum(np.abs(member_data["e2"]), 1e-9),
                    color=C_MEMBER, lw=1.2, ls=(0, (5, 2)))
    ax_err.axvline(BAND_EDGE, color=C_GUIDE, lw=0.6, ls=(0, (1, 2.2)))
    ax_err.set_xlim(0.0, 1.0)
    ax_err.set_ylim(1.0e-7, 4.0)
    ax_err.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_err.set_xlabel(r"$\theta/\pi$")
    ax_err.set_ylabel(r"$|e_1|$, $|e_2|$")
    ax_err.grid(axis="y", color="0.92", lw=0.5)
    ax_err.set_title("nodal errors", fontsize=10.9, color=ph.CLASSY_ACCENT,
                     pad=3.0)
    ax_err.text(0.035, 0.945, "realized field", transform=ax_err.transAxes,
                ha="left", va="top", fontsize=10.9, color=C_SLC)
    ax_err.text(0.035, 0.845, "member", transform=ax_err.transAxes,
                ha="left", va="top", fontsize=10.9, color=C_MEMBER)
    ph.quiet_axis(ax_err)
    ax_err.text(0.985, 0.03, "(c)", transform=ax_err.transAxes, ha="right",
                va="bottom", fontsize=10.9, color=ph.CLASSY_ACCENT)

    neutral = "0.35"
    handles = [
        Line2D([], [], color=C_ZERO, lw=1.0, ls=(0, (1, 1.6))),
        Line2D([], [], color=neutral, lw=1.7),
        Line2D([], [], color=neutral, lw=1.2, ls=(0, (5, 2))),
        Line2D([], [], color=neutral, ls="none", marker="^", ms=4.0),
        Line2D([], [], color=neutral, ls="none", marker="^", ms=4.0,
               mfc="none", mew=0.9),
    ]
    labels = [r"$k=0$", r"$k=-1$ and $|e_1|$", r"$k=+1$ and $|e_2|$",
              "exact contact", "soft contact"]
    ph.make_overhead_legend(fig, handles, labels, ncol=5, bbox=(0.5, 1.005),
                            rect=(0, 0, 1, 0.88), fontsize=10.9,
                            handslcngth=2.0, columnspacing=1.5,
                            handletextpad=0.5)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_alias.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    band = theta <= 2.0 * np.pi / 3.0
    print("alias anatomy receipt:")
    for name, data in (("realized spectral-like field", slc_data),
                       ("dominance member", member_data)):
        print(f"  {name}: max |w_-1| on the band "
              f"{np.max(np.abs(data[-1][band])):.3e}, "
              f"max |e_1| {np.max(np.abs(data['e1'][band])):.3e}, "
              f"max |e_2| {np.max(np.abs(data['e2'][band])):.3e}, "
              f"head closure over |k| <= 64 {data['closure']:.2e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
