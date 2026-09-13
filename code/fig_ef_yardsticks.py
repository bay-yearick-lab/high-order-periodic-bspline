# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Every Gate 7 yardstick of every scheme, normalized to the SLC 1992 pair.

The Gate 7 contract compares two schemes on seven numbers. Six are nodal: the
three reaches of the first derivative

    R_1(tol) = sup { theta : |e_1| <= tol on [0, theta] } / pi,

the group-velocity reach G(10^{-2}) built the same way from d kappa_1 / d theta,
and the two band maxima E_1^max(B_0), E_2^max(B_0) of |e_1| and |e_2| on the
dealiased band B_0 = 2 pi / 3. The seventh is the coupled off-grid reach
R_c(10^{-2}) of e_c(theta) = max_xi max_q |T_q(theta, xi) - 1|, which only a
scheme carrying a field can answer.

Each bar carries the value of one scheme on one yardstick divided so that the
SLC 1992 pair sits at one and larger is better: a reach is divided by SLC's
reach, a band maximum is SLC's maximum divided by the scheme's. The bars grow
from the unit line, so a bar above the line is a scheme ahead of SLC on that
yardstick and a bar below is a scheme behind. The coupled group is normalized
by the realized SLC field of the battery, since the compact pairs carry no
field of their own and therefore no bar in that group.

The nodal yardsticks of the SLC 1992 and tuned eighth-order pairs are the
stored targets of results/exponential_family/gate8/dominance.json; the member
yardsticks are the rows of results/exponential_family/gate9/battery.json; CW12B
is evaluated from comparators.py. Every member row is independently recomputed
here from its generator, the champion through family.ExponentialGenerator and
the polynomial members through gate8_family.from_record, and both comparator
pairs are recomputed from comparators.py, before the figure is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(HERE), str(HERE / "analysis" / "exponential_family")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import comparators as comp  # noqa: E402
import family as fam  # noqa: E402
import gate8_family as g8  # noqa: E402
import metrics as met  # noqa: E402
import plotting_helpers as ph  # noqa: E402
import static_battery as sb  # noqa: E402

EF = ROOT / "results" / "exponential_family"
BATTERY = EF / "gate9" / "battery.json"
DOMINANCE = EF / "gate8" / "dominance.json"
JOINT_MEMBER = EF / "gate8" / "members" / "X9_m6_C_SLC1_SLC1992_6IIq2.json"
RECORD_MEMBER = EF / "gate8" / "members" / "X8_n10_C_SLC1_pert5.json"
CHAMPION_MEMBER = EF / "members" / "E62_champion.json"
OUTDIR = ROOT / "figures_paper"

# One place to rename the joint member if the manuscript renames it.
JOINT_LABEL = "dominance member"
RECORD_LABEL = r"$C^2$ record"

C_SLC = "#D55E00"
C_Q8 = "#CC79A7"
C_CW12B = "#8C510A"
C_BMC9 = "#3A3A3A"
C_BMC11 = "#000000"
C_CHAMPION = "#0072B2"
C_RECORD = "#009E73"
C_JOINT = "#7B3294"

XI_POINTS = 129
RELATIVE_TOLERANCE = 1.0e-9

# (key, tick label, larger is better in the raw value)
YARDSTICKS = (
    ("reach_d1_0.01_refined", r"$R_1(10^{-2})$", True),
    ("reach_d1_0.001_refined", r"$R_1(10^{-3})$", True),
    ("reach_d1_0.0001_refined", r"$R_1(10^{-4})$", True),
    ("gv_reach_1e-2", r"$G(10^{-2})$", True),
    ("e1_max_B0", r"$E_1^{\max}(I_0)$", False),
    ("e2_max_B0", r"$E_2^{\max}(I_0)$", False),
    ("reach_ec_0.01_refined", r"$R_c(10^{-2})$", True),
)
NODAL = tuple(key for key, _, _ in YARDSTICKS[:6])
COUPLED = YARDSTICKS[6][0]

# The stored dominance targets use the unrefined key names.
TARGET_KEY = {"reach_d1_0.01_refined": "reach_d1_0.01",
              "reach_d1_0.001_refined": "reach_d1_0.001",
              "reach_d1_0.0001_refined": "reach_d1_0.0001",
              "gv_reach_1e-2": "gv_reach_1e-2",
              "e1_max_B0": "e1_max_B0",
              "e2_max_B0": "e2_max_B0"}


def _compact_row(name: str) -> dict:
    """Return the full metric row of one compact comparator pair."""
    scheme = next(s for s in comp.comparator_schemes() if s.name == name)
    return met.evaluate(name, scheme.kappa1, scheme.kappa2, scheme.left,
                        analytic_gv=scheme.analytic_gv,
                        bandwidth_left=scheme.bandwidth_left,
                        bandwidth_d1=scheme.bandwidth_d1,
                        bandwidth_d2=scheme.bandwidth_d2,
                        solves=scheme.solves)


def _member_row(name: str, generator) -> dict:
    """Return the full metric row of one member, coupled channel included."""
    xi = np.linspace(-0.5, 0.5, XI_POINTS)
    kappa1, kappa2, left, gv, coupled = sb.symbol_callables(generator, xi)
    return met.evaluate(name, kappa1, kappa2, left, analytic_gv=gv,
                        coupled=coupled)


def _check(tag: str, fresh: dict, stored: dict, keys) -> list:
    """Compare recomputed yardsticks against a stored row and return the pairs."""
    out = []
    for key in keys:
        a = float(fresh[key])
        b = float(stored[key])
        if abs(a - b) > RELATIVE_TOLERANCE * abs(b):
            raise RuntimeError(f"{tag} {key} drift: {a} against {b}")
        out.append((key, a, b))
    return out


def _rows() -> tuple[dict, dict, list]:
    """Return the plotted rows, the conditioning and the recomputation log."""
    battery = {row["label"]: row for row in json.loads(BATTERY.read_text())["results"]}
    targets = json.loads(DOMINANCE.read_text())["targets"]
    log = []

    slc = _compact_row("SLC1992")
    q8 = _compact_row("TunedQ8")
    cw12b = _compact_row("CW12B")
    for tag, fresh, stored in (("SLC1992", slc, targets["SLC1992"]),
                               ("TunedQ8", q8, targets["TunedQ8"])):
        for key in NODAL:
            a = float(fresh[key])
            b = float(stored[TARGET_KEY[key]])
            if abs(a - b) > RELATIVE_TOLERANCE * abs(b):
                raise RuntimeError(f"{tag} {key} drift: {a} against {b}")
            log.append((tag, key, a, b))
        if abs(fresh["cond"] - float(stored["cond"])) > RELATIVE_TOLERANCE:
            raise RuntimeError(f"{tag} cond drift")

    champion_record = json.loads(CHAMPION_MEMBER.read_text())
    generators = {
        "champion": fam.ExponentialGenerator(int(champion_record["m"]),
                                             tuple(champion_record["omegas"])),
        "BMC9_m10": fam.ExponentialGenerator(10, ()),
        "BMC11_m12": fam.ExponentialGenerator(12, ()),
        "pert5": g8.from_record(json.loads(RECORD_MEMBER.read_text())),
        "X9_m6_C_SLC1_SLC1992_6IIq2":
            g8.from_record(json.loads(JOINT_MEMBER.read_text())),
        "SLC_realized": sb.make_generator(battery["SLC_realized"]["spec"])[0],
    }
    keys = NODAL + (COUPLED, "cond")
    for label, generator in generators.items():
        fresh = _member_row(label, generator)
        stored = battery[label]["metrics"]
        for key, a, b in _check(label, fresh, stored, keys):
            log.append((label, key, a, b))

    reference = {key: float(targets["SLC1992"][TARGET_KEY[key]]) for key in NODAL}
    reference[COUPLED] = float(battery["SLC_realized"]["metrics"][COUPLED])

    rows = {
        "tuned q8": battery_free(q8),
        "CW12B": battery_free(cw12b),
        "BMC9": battery["BMC9_m10"]["metrics"],
        "BMC11": battery["BMC11_m12"]["metrics"],
        "band member": battery["champion"]["metrics"],
        RECORD_LABEL: battery["pert5"]["metrics"],
        JOINT_LABEL: battery["X9_m6_C_SLC1_SLC1992_6IIq2"]["metrics"],
    }
    rows["tuned q8"] = dict(rows["tuned q8"])
    for key in NODAL:
        rows["tuned q8"][key] = float(targets["TunedQ8"][TARGET_KEY[key]])
    conditioning = {name: float(rows[name]["cond"]) for name in rows}
    conditioning["SLC q4 pair"] = float(targets["SLC1992"]["cond"])
    return reference, rows, (conditioning, log)


def battery_free(row: dict) -> dict:
    """Return a comparator row with no coupled entry, as the pairs carry none."""
    out = dict(row)
    out[COUPLED] = None
    return out


def _normalized(reference: dict, row: dict, key: str, higher: bool):
    """Return the scheme value divided so that SLC reads one and more is better."""
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    base = reference[key]
    return value / base if higher else base / value


def main() -> int:
    reference, rows, (conditioning, log) = _rows()

    order = ["tuned q8", "CW12B", "BMC9", "BMC11", "band member",
             RECORD_LABEL, JOINT_LABEL]
    color = {"tuned q8": C_Q8, "CW12B": C_CW12B, "BMC9": C_BMC9,
             "BMC11": C_BMC11, "band member": C_CHAMPION,
             RECORD_LABEL: C_RECORD, JOINT_LABEL: C_JOINT}

    ph.apply_classy_defaults()
    plt.rcParams.update({
        "font.size": 11.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
    })
    fig, (ax_all, ax_zoom) = plt.subplots(
        1, 2, figsize=(7.9, 4.5), sharey=True,
        gridspec_kw={"width_ratios": [1.72, 1.0], "wspace": 0.06})

    rank = {name: slot for slot, name in enumerate(order)}
    height = 0.113
    offsets = (np.arange(len(order)) - 0.5 * (len(order) - 1)) * height
    receipt = []
    for row_index, (key, _, higher) in enumerate(YARDSTICKS):
        for name in order:
            value = _normalized(reference, rows[name], key, higher)
            if value is None:
                continue
            ax_all.barh(row_index - offsets[rank[name]], value - 1.0, left=1.0,
                        height=height, color=color[name], linewidth=0.0,
                        zorder=3)
            receipt.append((name, key, value))
        value = _normalized(reference, rows[JOINT_LABEL], key, higher)
        ax_zoom.barh(row_index, value - 1.0, left=1.0, height=0.44,
                     color=C_JOINT, linewidth=0.0, zorder=3)
        ax_zoom.text(value * 1.012, row_index, f"{value:.2f}", fontsize=11.5,
                     color=ph.CLASSY_ACCENT, ha="left", va="center", zorder=5)

    for ax, lo, hi in ((ax_all, 4.0e-2, 3.0e1), (ax_zoom, 0.998, 1.78)):
        ax.axvline(1.0, color=C_SLC, lw=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(len(YARDSTICKS) - 0.55, -0.55)
        ax.grid(axis="x", color="0.93", lw=0.5, zorder=0)
        ph.quiet_axis(ax)
    ax_all.set_yticks(range(len(YARDSTICKS)))
    ax_all.set_yticklabels([label for _, label, _ in YARDSTICKS])
    ax_all.set_xlabel("value relative to SLC q4, larger is better")
    ax_zoom.set_xlabel("magnified, same normalization")
    ax_zoom.set_xticks([1.0, 1.2, 1.4, 1.6])
    ax_zoom.set_xticklabels(["1.0", "1.2", "1.4", "1.6"])
    ax_zoom.xaxis.set_minor_locator(NullLocator())
    ax_zoom.xaxis.set_minor_formatter(NullFormatter())
    ax_zoom.tick_params(axis="y", length=0.0)
    ax_all.set_title("(a) every yardstick of every scheme", loc="left",
                     fontsize=11.5, color=ph.CLASSY_ACCENT, fontstyle="italic",
                     pad=4.0)
    ax_zoom.set_title(f"(b) the {JOINT_LABEL} alone", loc="left", fontsize=11.5,
                      color=ph.CLASSY_ACCENT, fontstyle="italic", pad=4.0)
    ax_all.text(0.5, -0.195,
                "the coupled reach is normalized by the realized SLC field",
                transform=ax_all.transAxes, ha="center", va="top",
                fontsize=11.5, color="0.45")

    handles = [Line2D([], [], color=C_SLC, lw=2.2,
                      label=f"SLC q4 pair ({conditioning['SLC q4 pair']:.1f}), "
                            "the unit line")]
    handles += [Line2D([], [], color=color[name], lw=2.2,
                       label=f"{name} ({conditioning[name]:.1f})")
                for name in order]
    legend = ph.make_overhead_legend(fig, handles, [h.get_label() for h in handles],
                                     ncol=4, bbox=(0.5, 1.005),
                                     rect=(0, 0, 1, 0.80), fontsize=10.0,
                                     handslcngth=1.8, columnspacing=1.4,
                                     handletextpad=0.45)
    legend.set_title("scheme, with the symbol conditioning in parentheses")
    legend.get_title().set_fontsize(10.0)
    legend.get_title().set_color(ph.CLASSY_ACCENT)
    fig.subplots_adjust(top=0.755, bottom=0.155, left=0.105, right=0.985,
                        wspace=0.06)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig_ef_yardsticks.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("recomputed yardsticks against the stored rows "
          "(gate9/battery.json, gate8/dominance.json):")
    for tag, key, fresh, stored in log:
        print(f"  {tag:30s} {key:24s} {fresh:.12e} against {stored:.12e}")
    print("normalized yardsticks, SLC q4 at one:")
    for name, key, value in receipt:
        print(f"  {name:16s} {key:24s} {value:.6f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
