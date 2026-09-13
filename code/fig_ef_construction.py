# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""The three dials of the exponential cardinal B-spline family.

Panel (a) draws the compactly supported generator of four fields of the same
width: the polynomial rung E_{10,0} of degree nine, the enriched band champion
E_{6,2}(omega_1, omega_2) of the same degree, the everything member
X9_m6_C_SLC1_SLC1992_6IIq2 (core six, two imaginary pairs at 0.5870 pi and
0.7757 pi, quadratic polynomial factor, class C^4), and the one-field
realization of the published spectral-like first derivative,
E_{4,2}(0.6366 pi, 0.7321 pi) with a quadratic polynomial factor, whose support
is [-4, 4].

Panel (b) draws the generator transform on [0, 3 pi]. Without a polynomial
factor,

    phi_hat(xi) = sinc(xi/2)^m prod_j sinc((xi - omega_j)/2) sinc((xi + omega_j)/2)
                  / sinc(omega_j/2)^2,

and a conjugate pair at omega_j contributes the factors sinc((xi -+ omega_j)/2),
whose zero sets are xi = 2 pi n +- omega_j. Those are exactly the alias
abscissae reached from theta = +- omega_j, so every alias of the two contact
frequencies is annihilated and kappa_1(omega_j) = omega_j,
kappa_2(omega_j) = omega_j^2 hold exactly (Theorem A3). A polynomial factor
psi_hat = P(xi^2) phi_hat with a real root xi_0 in (pi, 2 pi) adds one further
transform zero there, which the k = -1 alias reaches from theta = 2 pi - xi_0:
a soft contact. The markers under the top of panel (b) are the exact contacts
(filled) and the soft contacts (open) of each field, one row per field.

The polynomial rung and the champion are evaluated in closed form by the
exponential evaluator; the two polynomial-augmented fields are recovered from
their transforms by the Poisson identity

    psi(x) = (1 / 2M) sum_l psi_hat(pi l / M) exp(i pi l x / M),   |x| <= M,

with M = 16 well beyond the support and the sum carried to |l| < 2^19 by one
inverse transform. Every frequency, polynomial coefficient and integer sample
is read from the tracked receipts, and the samples each route returns are
checked against the receipt before the figure is written.
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

import evaluator as ev  # noqa: E402
import family as fam  # noqa: E402
import gate8_family as g8  # noqa: E402
import plotting_helpers as ph  # noqa: E402

MEMBER = ROOT / "results" / "exponential_family" / "members" / "E62_champion.json"
GATE8 = ROOT / "results" / "exponential_family" / "gate8"
REALIZATION = GATE8 / "realization.json"
EVERYTHING = GATE8 / "members" / "X9_m6_C_SLC1_SLC1992_6IIq2.json"
OUTDIR = ROOT / "figures_paper"

C_CHAMPION = "#0072B2"
C_POLY = "#009E73"
C_MEMBER = "#7B3294"
C_SLC = "#D55E00"
C_GUIDE = "0.45"

SAMPLE_TOLERANCE = 1.0e-12
PERIOD_HALF = 16.0
TRANSFORM_POINTS = 1 << 20


def _refined_grid(lo: float, hi: float, zeros) -> np.ndarray:
    """Return a uniform grid refined geometrically around the listed zeros."""
    grid = [np.linspace(lo, hi, 4001)]
    offsets = np.concatenate([-np.geomspace(1.0e-9, 0.05, 60)[::-1], [0.0],
                              np.geomspace(1.0e-9, 0.05, 60)])
    for zero in zeros:
        grid.append(zero + offsets)
    values = np.unique(np.concatenate(grid))
    return values[(values >= lo) & (values <= hi)]


def _receipt() -> dict:
    """Return the tracked champion member record."""
    return json.loads(MEMBER.read_text())


def _check_samples(evaluator, beta_half: list[float]) -> np.ndarray:
    """Return the integer samples and verify them against the receipt."""
    index = np.arange(-(len(beta_half) - 1), len(beta_half), dtype=float)
    samples = evaluator.value(index)
    stored = np.asarray(beta_half[::-1] + beta_half[1:], dtype=float)
    drift = float(np.max(np.abs(samples - stored)))
    if drift > SAMPLE_TOLERANCE:
        raise RuntimeError(f"integer sample drift {drift:.3e} against the receipt")
    return samples


def _augmented_generator(record: dict):
    """Return the augmented generator, its abscissae and its values on a period."""
    generator = g8.from_record(record)
    index = np.fft.fftfreq(TRANSFORM_POINTS, d=1.0 / TRANSFORM_POINTS)
    xi = np.pi * index / PERIOD_HALF
    coefficients = generator.fourier(xi)
    values = np.fft.ifft(coefficients) * TRANSFORM_POINTS / (2.0 * PERIOD_HALF)
    x = 2.0 * PERIOD_HALF * np.arange(TRANSFORM_POINTS) / TRANSFORM_POINTS
    x = np.where(x > PERIOD_HALF, x - 2.0 * PERIOD_HALF, x)
    order = np.argsort(x)
    return generator, x[order], values[order].real


def _check_augmented(record: dict, x: np.ndarray, values: np.ndarray,
                     radius: int) -> float:
    """Verify the recovered field against the stored integer samples."""
    stored = np.asarray(record["beta"], dtype=float)
    step = TRANSFORM_POINTS // int(2 * PERIOD_HALF)
    origin = int(np.argmin(np.abs(x)))
    recovered = np.array([values[origin + k * step] for k in range(stored.size)])
    drift = float(np.max(np.abs(recovered - stored)))
    outside = float(np.max(np.abs(values[np.abs(x) > radius + 1.0e-9])))
    if max(drift, outside) > SAMPLE_TOLERANCE:
        raise RuntimeError(f"{record['name']}: sample drift {drift:.3e}, "
                           f"support leakage {outside:.3e}")
    return drift


def _contacts(record: dict) -> tuple[list[float], list[float]]:
    """Return the exact and soft transform zeros of one field inside (0, 3 pi]."""
    exact = sorted(value for omega in record.get("omegas", ())
                   for value in (2.0 * np.pi - omega, 2.0 * np.pi + omega)
                   if 0.0 < value <= 3.0 * np.pi)
    roots = record.get("polynomial_roots", {}).get("real_positive_xi", [])
    soft = sorted(value for value in roots if 0.0 < value <= 3.0 * np.pi)
    return exact, soft


def main() -> int:
    record = _receipt()
    m_core = int(record["m"])
    omegas = tuple(float(w) for w in record["omegas"])
    degree = int(record["p"])
    half = (degree + 1) // 2

    champion = ev.evaluator_for(m_core, omegas)
    polynomial = ev.evaluator_for(degree + 1, ())
    _check_samples(champion, list(record["beta"]))

    realized_record = (json.loads(REALIZATION.read_text())["realizations"]
                       ["SLC1992"]["analysis"]["record"])
    everything_record = json.loads(EVERYTHING.read_text())
    realized, x_realized, phi_realized = _augmented_generator(realized_record)
    everything, x_everything, phi_everything = _augmented_generator(everything_record)
    drift_realized = _check_augmented(realized_record, x_realized, phi_realized,
                                      realized.n // 2)
    drift_everything = _check_augmented(everything_record, x_everything,
                                        phi_everything, everything.n // 2)

    index = np.arange(-half, half + 1, dtype=float)
    x = np.linspace(-half, half, 4001)
    phi_champion = champion.value(x)
    phi_polynomial = polynomial.value(x)

    fields = [
        ("polynomial rung $E_{10,0}$ (BMC9)", C_POLY, (0, (5, 2)), 1.3,
         {"omegas": (), "polynomial_roots": {"real_positive_xi": []}}),
        ("band member $E_{6,2}$", C_CHAMPION, "-", 1.7, record),
        ("dominance member", C_MEMBER, (0, (1, 1.4)), 1.7, everything_record),
        ("realized spectral-like field", C_SLC, (0, (4, 1.4, 1, 1.4)), 1.3,
         realized_record),
    ]
    zeros = {2.0 * np.pi}
    for _, _, _, _, source in fields:
        exact, soft = _contacts(source)
        zeros |= set(exact) | set(soft)
    xi = _refined_grid(0.0, 3.0 * np.pi, sorted(zeros))
    hat = {
        "polynomial rung $E_{10,0}$ (BMC9)":
            np.abs(fam.ExponentialGenerator(degree + 1, ()).fourier(xi)),
        "band member $E_{6,2}$":
            np.abs(fam.ExponentialGenerator(m_core, omegas).fourier(xi)),
        "dominance member": np.abs(everything.fourier(xi)),
        "realized spectral-like field": np.abs(realized.fourier(xi)),
    }
    floor = 1.0e-12

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 10.9,
        "axes.labelsize": 10.9,
        "xtick.labelsize": 10.9,
        "ytick.labelsize": 10.9,
    })
    fig, (ax_phi, ax_hat) = plt.subplots(1, 2, figsize=(7.6, 3.5))

    inside = np.abs(x_realized) <= half
    inside_member = np.abs(x_everything) <= half
    ax_phi.plot(x, phi_polynomial, color=C_POLY, lw=1.3, ls=(0, (5, 2)),
                label=fields[0][0])
    ax_phi.plot(x, phi_champion, color=C_CHAMPION, lw=1.7, label=fields[1][0])
    ax_phi.plot(x_everything[inside_member], phi_everything[inside_member],
                color=C_MEMBER, lw=1.7, ls=(0, (1, 1.4)), label=fields[2][0])
    ax_phi.plot(x_realized[inside], phi_realized[inside], color=C_SLC, lw=1.3,
                ls=(0, (4, 1.4, 1, 1.4)), label=fields[3][0])
    ax_phi.plot(index, polynomial.value(index), ls="none", marker="s", ms=3.2,
                mfc="none", mec=C_POLY, mew=0.8)
    ax_phi.plot(index, champion.value(index), ls="none", marker="o", ms=3.6,
                mfc="none", mec=C_CHAMPION, mew=0.8)
    ax_phi.axhline(0.0, color="0.75", lw=0.5)
    ax_phi.set_xlabel(r"$x$")
    ax_phi.set_ylabel(r"$\varphi(x)$")
    ax_phi.set_xlim(-half - 0.15, half + 0.15)
    ax_phi.set_xticks(np.arange(-half, half + 1, 1))
    ax_phi.set_ylim(-0.055, 0.52)
    ax_phi.grid(axis="y", color="0.90", lw=0.5)
    ph.quiet_axis(ax_phi)
    ax_phi.text(0.985, 0.985, "(a)", transform=ax_phi.transAxes, ha="right",
                va="top", fontsize=10.9, color=ph.CLASSY_ACCENT)

    for label, color, style, width, _ in fields:
        ax_hat.semilogy(xi / np.pi, np.maximum(hat[label], floor), color=color,
                        lw=width, ls=style)
    for row, (label, color, _, _, source) in enumerate(fields):
        exact, soft = _contacts(source)
        height = 10.0 ** (-0.15 - 0.60 * row)
        ax_hat.plot([value / np.pi for value in exact],
                    [height] * len(exact), ls="none", marker="v", ms=4.0,
                    color=color, zorder=6)
        ax_hat.plot([value / np.pi for value in soft], [height] * len(soft),
                    ls="none", marker="v", ms=4.0, mfc="none", mec=color,
                    mew=0.9, zorder=6)
    ax_hat.text(0.03, 0.055, "filled: exact contact\nopen: soft contact",
                transform=ax_hat.transAxes, ha="left", va="bottom",
                fontsize=9.4, color=C_GUIDE, linespacing=1.25, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.2))
    ax_hat.set_xlabel(r"$\xi/\pi$")
    ax_hat.set_ylabel(r"$|\hat{\varphi}(\xi)|$")
    ax_hat.set_xlim(0.0, 3.0)
    ax_hat.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax_hat.set_ylim(floor, 3.0)
    ax_hat.grid(axis="y", color="0.92", lw=0.5)
    ph.quiet_axis(ax_hat)
    ax_hat.text(0.985, 0.985, "(b)", transform=ax_hat.transAxes, ha="right",
                va="top", fontsize=10.9, color=ph.CLASSY_ACCENT)

    handles, labels = ax_phi.get_legend_handles_labels()
    ph.make_overhead_legend(fig, handles, labels, ncol=2, bbox=(0.5, 1.01),
                            rect=(0, 0, 1, 0.855), fontsize=10.9,
                            handslcngth=2.2, columnspacing=1.6,
                            handletextpad=0.5)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_construction.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("construction receipt:")
    print(f"  champion omegas = {omegas[0]:.12f}, {omegas[1]:.12f}")
    for label, _, _, _, source in fields[1:]:
        exact, soft = _contacts(source)
        print(f"  {label}: exact contacts "
              + ", ".join(f"{value / np.pi:.4f} pi" for value in exact)
              + "; soft contacts "
              + (", ".join(f"{value / np.pi:.4f} pi" for value in soft) or "none"))
    print(f"  sample drift, realized field {drift_realized:.2e}, "
          f"everything member {drift_everything:.2e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
