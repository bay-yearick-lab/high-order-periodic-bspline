# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","matplotlib"]
# ///
"""fig_resolving_power_band: first-derivative modified-wavenumber error
|k'(w)-w| over [0,pi] for the spline degree ladder, the SLC spectral-like
scheme, and the tridiagonal Pade compacts, with the resolved band [0, 2pi/3]
shaded. Visualizes the finding-10 resolving-power result: on the resolved band
the twelfth-order BMC11 carries a lower band error than the fourth-order
spectral-like scheme (which has a higher CUT but is less accurate where the
band lives), and orders of magnitude lower deeper in. Numbers are gated by
code/analysis/resolving_power_band.py (G15)."""
import sys
from math import comb, factorial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plotting_helpers import apply_classy_defaults  # noqa
apply_classy_defaults()
OUT = HERE.parent / "figures_paper/fig_resolving_power_band.pdf"
w = np.linspace(1e-6, np.pi, 6000)


def bspline_samples(p):
    K = int(np.ceil((p + 1) / 2)); ks = np.arange(-K, K + 1)
    def tp(x, n): return np.where(x > 0, np.power(np.maximum(x, 0.0), n), 0.0)
    b = np.zeros(len(ks)); db = np.zeros(len(ks))
    for i, x in enumerate(ks):
        for k in range(p + 2):
            a = x + (p + 1) / 2.0 - k
            b[i] += (-1) ** k * comb(p + 1, k) * tp(a, p)
            db[i] += (-1) ** k * comb(p + 1, k) * tp(a, p - 1)
        b[i] /= factorial(p); db[i] /= factorial(p - 1)
    return ks, b, db


def bmc(p):
    ks, b, db = bspline_samples(p)
    num = np.array([np.sum(db * np.sin(ks * ww)) for ww in w])
    den = np.array([np.sum(b * np.cos(ks * ww)) for ww in w])
    return -num / den


def compact(alpha, a_s, beta2=0.0):
    num = sum(a * np.sin(s * w) for s, a in a_s.items())
    return num / (1 + 2 * alpha * np.cos(w) + 2 * beta2 * np.cos(2 * w))


CURVES = [
    ("BMC5", bmc(5), "#9ecae1", "-"), ("BMC7", bmc(7), "#6baed6", "-"),
    ("BMC9", bmc(9), "#3182bd", "-"), ("BMC11", bmc(11), "#08519c", "-"),
    ("SLC SL", compact(0.5771439, {1: 1.3025166, 2: 0.9935500/2, 3: 0.03750245/3}, 0.0896406),
     "#c0392b", "--"),
    ("C6", compact(1/3, {1: 14/9, 2: (1/9)/2}), "#969696", ":"),
    ("C8", compact(3/8, {1: 25/16, 2: (1/5)/2, 3: (-1/80)/3}), "#636363", ":"),
]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.axvspan(0, 2/3, color="#f0f0f0", zorder=0)
ax.text(2/3/2, 3e-8, "resolved band", ha="center", fontsize=8, color="0.45")
for name, kp, c, ls in CURVES:
    ax.semilogy(w/np.pi, np.abs(kp - w) + 1e-16, ls, color=c, lw=1.6, label=name)
ax.set_xlim(0, 1); ax.set_ylim(1e-8, 2)
ax.axvline(2/3, color="0.6", lw=0.8, ls="-", zorder=1)
ax.set_xlabel(r"reduced wavenumber $\omega/\pi$")
ax.set_ylabel(r"modified-wavenumber error $|k'(\omega)-\omega|$")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
