"""Section 7 sub-capability (5a): parity-driven nodal superconvergence.

Measures the nodal convergence slope of the BMC first-derivative
operators on a smooth periodic test
  ``u(x) = sin(2 pi x) + 0.5 sin(4 pi x + 0.7)``
on grids of size ``N = 16, 24, 32, 48, 64, 96, 128``. Seven curves are
plotted (``BMC_2`` through ``BMC_8``), with the measured order printed
beside the final resolved point of each curve. The reader sees the
parity-driven ladder as a visual fact: each odd-degree curve sits on the
slope of the next even degree, not its own degree.

Output: ``figures_paper/fig_superconvergence.pdf``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import plotting_helpers as ph
import project_paths as pp
import spline_analysis as sa
import convergence_style as cs


GRID_SIZES = [16, 24, 32, 48, 64, 96, 128]


@dataclass(frozen=True)
class SuperConvRecord:
    label: str
    family: str
    degree: int
    fitted_slope: float
    formal_order: int
    h_values: list[float]
    errors: list[float]
    grid_sizes: list[int]
    fit_indices: list[int]
    excluded_coarse_indices: list[int]
    excluded_tail_indices: list[int]


def _build(label: str, npts: int, h: float) -> np.ndarray:
    if label.startswith("BMC"):
        return sa.collocation_derivative_matrix(int(label[3:]), npts, h=h)
    if label.startswith("BMG"):
        return sa.iga_derivative_matrix(int(label[3:]), npts, h=h)
    raise ValueError(label)


def _formal_order(label: str) -> int:
    if label.startswith("BMC"):
        return sa.formal_order_collocation(int(label[3:]))
    if label.startswith("BMG"):
        return sa.formal_order_iga(int(label[3:]))
    raise ValueError(label)


def _nodal_error(label: str, npts: int) -> tuple[float, float]:
    h = 1.0 / npts
    D = _build(label, npts, h)
    x = np.arange(npts) * h
    u = np.sin(2.0 * np.pi * x) + 0.5 * np.sin(4.0 * np.pi * x + 0.7)
    du_exact = (
        2.0 * np.pi * np.cos(2.0 * np.pi * x)
        + 0.5 * 4.0 * np.pi * np.cos(4.0 * np.pi * x + 0.7)
    )
    err_inf = float(np.max(np.abs(D @ u - du_exact)))
    return h, err_inf


def _nodal_error_d2(degree: int, npts: int) -> tuple[float, float]:
    h = 1.0 / npts
    D2 = sa.collocation_second_derivative_matrix(degree, npts, h=h)
    x = np.arange(npts) * h
    u = np.sin(2.0 * np.pi * x) + 0.5 * np.sin(4.0 * np.pi * x + 0.7)
    d2u_exact = (
        -((2.0 * np.pi) ** 2) * np.sin(2.0 * np.pi * x)
        - 0.5 * ((4.0 * np.pi) ** 2) * np.sin(4.0 * np.pi * x + 0.7)
    )
    return h, float(np.max(np.abs(D2 @ u - d2u_exact)))


def _audit(label: str) -> SuperConvRecord:
    h_vals = []
    errs = []
    for n in GRID_SIZES:
        h, err = _nodal_error(label, n)
        h_vals.append(h)
        errs.append(max(err, 1e-300))
    log_h = np.log(np.array(h_vals))
    log_e = np.log(np.array(errs))
    keep = log_e > np.log(1e-13)
    if np.sum(keep) >= 3:
        slope = float(np.polyfit(log_h[keep], log_e[keep], 1)[0])
    else:
        slope = float("nan")
    family = "BMC" if label.startswith("BMC") else "BMG"
    degree = int(label[3:])
    fit = cs.fit_indices(np.asarray(h_vals), np.asarray(errs),
                         expected_order=_formal_order(label))
    fit_indices = [int(i) for i in fit]
    excluded_coarse = list(range(fit_indices[0]))
    excluded_tail = list(range(fit_indices[-1] + 1, len(GRID_SIZES)))
    return SuperConvRecord(
        label=label, family=family, degree=degree,
        fitted_slope=slope, formal_order=_formal_order(label),
        h_values=h_vals, errors=errs,
        grid_sizes=list(GRID_SIZES), fit_indices=fit_indices,
        excluded_coarse_indices=excluded_coarse,
        excluded_tail_indices=excluded_tail,
    )


def _second_derivative_records() -> list[dict]:
    records = []
    for degree in range(2, 9):
        h_values, errors = [], []
        for npts in GRID_SIZES:
            h, error = _nodal_error_d2(degree, npts)
            h_values.append(h)
            errors.append(max(error, 1.0e-300))
        formal_order = 2 * (degree // 2)
        h_array = np.asarray(h_values)
        error_array = np.asarray(errors)
        fit = cs.fit_indices(h_array, error_array,
                             expected_order=formal_order)
        fit_indices = [int(i) for i in fit]
        excluded_coarse = list(range(fit_indices[0]))
        excluded_tail = list(range(fit_indices[-1] + 1, len(GRID_SIZES)))
        records.append({
            "label": f"BMC{degree}",
            "family": "BMC",
            "degree": degree,
            "formal_order": formal_order,
            "fitted_slope": cs.measured_order(
                h_array, error_array, expected_order=formal_order
            ),
            "grid_sizes": list(GRID_SIZES),
            "h_values": h_values,
            "errors": errors,
            "fit_indices": fit_indices,
            "excluded_coarse_indices": excluded_coarse,
            "excluded_tail_indices": excluded_tail,
        })
    return records


def _annotate_endpoint_orders(ax, endpoints: list[tuple[int, float, float, float, str]],
                              min_sep_points: float = 13.5) -> None:
    """Place measured orders beside resolved curve endpoints without overlap."""
    ax.figure.canvas.draw()
    pixels_per_point = ax.figure.dpi / 72.0
    min_sep = min_sep_points * pixels_per_point
    half_height = 6.0 * pixels_per_point

    placed = []
    for degree, x, y, order, color in endpoints:
        _, y_display = ax.transData.transform((x, y))
        placed.append({
            "degree": degree,
            "x": x,
            "y": y,
            "order": order,
            "color": color,
            "original_y": y_display,
            "label_y": y_display,
        })
    placed.sort(key=lambda item: item["original_y"])

    lower = ax.bbox.y0 + half_height
    upper = ax.bbox.y1 - half_height
    for i, item in enumerate(placed):
        if i == 0:
            item["label_y"] = max(item["label_y"], lower)
        else:
            item["label_y"] = max(
                item["label_y"], placed[i - 1]["label_y"] + min_sep
            )
    if placed and placed[-1]["label_y"] > upper:
        placed[-1]["label_y"] = upper
        for i in range(len(placed) - 2, -1, -1):
            placed[i]["label_y"] = min(
                placed[i]["label_y"], placed[i + 1]["label_y"] - min_sep
            )

    for item in placed:
        offset_points = (
            (item["label_y"] - item["original_y"]) / pixels_per_point
        )
        connector = None
        if abs(offset_points) > 1.0:
            connector = {
                "arrowstyle": "-",
                "color": item["color"],
                "lw": 0.6,
                "alpha": 0.65,
                "shrinkA": 1.5,
                "shrinkB": 2.0,
            }
        ax.annotate(
            f"{item['order']:.2f}",
            (item["x"], item["y"]),
            xytext=(7.0, offset_points),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=11.5,
            color=item["color"],
            annotation_clip=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                  "pad": 0.12},
            arrowprops=connector,
        )


def _figure(records: list[SuperConvRecord], d2_records: list[dict]) -> None:
    """Parity-driven nodal superconvergence of the BMC operators, degrees 2-8,
    for both the first and the direct second derivative."""
    ph.apply_classy_defaults()

    degrees = [2, 3, 4, 5, 6, 7, 8]
    par_d1 = {p: 2 * ((p + 1) // 2) for p in degrees}
    par_d2 = {p: 2 * (p // 2) for p in degrees}

    # gather D1 (from records) and recompute D2 nodal errors
    d1 = {r.degree: (np.asarray(r.h_values), np.asarray(r.errors))
          for r in records if r.family == "BMC"}
    d2 = {record["degree"]: (np.asarray(record["h_values"]),
                              np.asarray(record["errors"]))
          for record in d2_records}

    fig = plt.figure(figsize=(10.2, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.34), hspace=0.42,
                            wspace=0.28)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    axt = fig.add_subplot(grid[1, :])

    mo1, mo2 = {}, {}
    for panel, ax, store, parity, mo in (
            ("(a)", ax1, d1, par_d1, mo1),
            ("(b)", ax2, d2, par_d2, mo2)):
        endpoints = []
        for p in degrees:
            h, e = store[p]
            col = cs.DEG_COLOR[p]
            n = 1.0 / h
            fit = cs.fit_indices(h, e, expected_order=parity[p])
            last = int(fit[-1])
            ax.loglog(n[:last + 1], e[:last + 1], color=col,
                      marker=cs.DEG_MARKER[p], ms=6.0, lw=2.0,
                      mec="white", mew=0.7, alpha=0.98,
                      label=f"BMC{p}")
            if last + 1 < len(e):
                ax.loglog(n[last:], e[last:], color=col, marker=cs.DEG_MARKER[p],
                          ms=5.2, lw=1.0, ls=":", mfc="white", mec=col,
                          alpha=0.45, zorder=1)
            mo[p] = cs.measured_order(h, e, expected_order=parity[p])
            if mo[p] is not None:
                endpoints.append((p, n[last], e[last], mo[p], col))
        cs.style_axis(ax, r"grid resolution $N$", "maximum-norm error", panel,
                      headroom=1.9)
        cs.resolution_axis(ax, GRID_SIZES)
        _annotate_endpoint_orders(ax, endpoints)

    axt.axis("off")
    axt.set_title(
        r"formal/measured order: $q_1=2\lceil p/2\rceil$, "
        r"$q_2=2\lfloor p/2\rfloor$",
        fontsize=16.0,
        pad=3.0,
    )
    columns = ["channel"] + [f"p={p}" for p in degrees]
    rows = [
        [r"$D$"] + [f"{par_d1[p]}/{mo1[p]:.2f}" for p in degrees],
        [r"$D^{(2)}$"] + [f"{par_d2[p]}/{mo2[p]:.2f}" for p in degrees],
    ]
    table = axt.table(cellText=rows, colLabels=columns, cellLoc="center",
                      colLoc="center", bbox=(0.01, 0.20, 0.98, 0.58))
    table.auto_set_font_size(False)
    table.set_fontsize(15.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("0.78")
        if row == 0 or col == 0:
            cell.set_facecolor("#F6F1E6")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False, fontsize=15.2, handslcngth=2.0,
               columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.09, right=0.98)
    fig.savefig(pp.figure_path("fig_superconvergence.pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    labels = [f"BMC{p}" for p in range(2, 9)]
    records = [_audit(label) for label in labels]
    d2_records = _second_derivative_records()
    print("BMC fitted nodal slopes vs formal order:")
    for r in records:
        print(f"  BMC{r.degree}: fitted={r.fitted_slope:.3f}, formal={r.formal_order}, "
              f"bonus={r.fitted_slope - r.formal_order:+.2f}")
    _figure(records, d2_records)
    out = {
        "records": [{**r.__dict__} for r in records],
        "second_derivative_records": d2_records,
    }
    with open(pp.results_path("s7_superconvergence.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
