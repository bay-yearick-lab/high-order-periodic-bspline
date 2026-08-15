"""Section 7 sub-capability (4): local finite-volume balance in the spline metric.

The eliminated first derivative D = B^{-1}Q (BMC) or M^{-1}K (BMG) is a DENSE
circulant: its entries decay exponentially but never vanish, so no fixed-radius
point-value flux exists for D itself. The corrected local conservation statement
lives one level up, at the banded compact relation, where bandedness is real:

    B (du/dt) = -Q f        (BMC; M dc/dt = -K g for BMG; compact relations too)

with B symmetric positive circulant of radius R_B and Q antisymmetric circulant
of radius R_Q. The flux lemma applied to the BANDED Q (where its finite-band
hypothesis actually holds) gives face fluxes F_{j+1/2} of fixed radius R_Q, and
the evolved cell content is the metric-weighted local average

    m_j = (B u)_j = sum_s b_s u_{j+s},   b_s >= 0,  sum_s b_s = 1,

a positive-weight average supported on radius R_B whose total tiles the nodal
mass exactly. In native BMG coefficient coordinates, m_j = (M c)_j/h is the
continuum test-function moment (1/h) int B_j(x) u_h(x) dx. The two coordinate
readings are kept distinct even though their eliminated arrays coincide.

This audit verifies, for BMC2..BMC11, BMG1..BMG5, and the compact Pade C4/C6/C8
relations (the mass-metric balance is generic to every centered compact relation
with positive symmetric left-hand side; the paper scores it for the baselines
too):

  1. relation exactness:  || B (hD) - Q ||_inf  (machine zero);
  2. metric positivity and normalization of the b_s weights;
  3. flux identity: (Q f)_j = F_{j+1/2} - F_{j-1/2} entrywise, radius R_Q;
  4. per-cell balance residual  max_j | h d/dt (Bu)_j + F_{j+1/2} - F_{j-1/2} |;
  5. mass tiling  sum_j m_j = sum_j u_j  and its drift under an RK4 march;
  6. density of the eliminated D: smallest-magnitude entry, entry at the
     antipode j = N/2, and the fitted exponential decay rate per node
     (the honesty numbers behind the paper's banded-relation/dense-operator
     taxonomy);
  7. the matched pair BMC_{2p+1} = BMG_p shares one dense D but carries two
     different banded conservative factorizations (B,Q) vs (M,K); the audit
     reports both relation radii and the entrywise D agreement.

Outputs: code/results/s7_local_fv.json (no figure: every check lands at machine
zero, and machine-zero confirmations are prose by house style).
"""

from __future__ import annotations

import json

import numpy as np

import project_paths as pp
import spline_analysis as sa

RNG = np.random.default_rng(20260710)


def relation_entries(label: str) -> tuple[np.ndarray, np.ndarray]:
    """One-sided (mass, advection) entries of the compact relation B(hD) = Q."""
    if label.startswith("BMC"):
        return sa.collocation_mass_and_advection_entries(int(label[3:]))
    if label.startswith("BMG"):
        return sa.iga_mass_and_advection_entries(int(label[3:]))
    if label.startswith("C"):
        return sa.compact_scheme_entries(int(label[1:]))
    raise ValueError(label)


def eliminated_matrix(label: str, npts: int) -> np.ndarray:
    if label.startswith("BMC"):
        return sa.collocation_derivative_matrix(int(label[3:]), npts, h=1.0)
    if label.startswith("BMG"):
        return sa.iga_derivative_matrix(int(label[3:]), npts, h=1.0)
    mass, adv = relation_entries(label)
    return sa.compact_derivative_matrix(mass, adv, npts, h=1.0)


def flux_coefficients(adv: np.ndarray) -> np.ndarray:
    """phi_r = 2 sum_{t=r+1}^{R} alpha_t for the one-sided antisymmetric entries."""
    radius = len(adv) - 1
    return np.array([2.0 * adv[r + 1 : radius + 1].sum() for r in range(radius)])


def face_flux(f: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """F_{j+1/2} = sum_r phi_r (f_{j+r+1} + f_{j-r}) / 2 on the periodic line."""
    n = f.size
    j = np.arange(n)
    F = np.zeros(n)
    for r, coeff in enumerate(phi):
        F += 0.5 * coeff * (f[(j + r + 1) % n] + f[(j - r) % n])
    return F


def dense_decay(D: np.ndarray, band_radius: int) -> dict:
    """Honesty numbers for the eliminated operator's dense circulant profile.

    The antipode entry (offset N/2) is a STRUCTURAL zero for any antisymmetric
    circulant with even N, so it is not a density witness. The witnesses are
    the entry one node past the relation band and the entry a quarter period
    away, plus the fitted exponential decay rate per node.
    """
    row = D[0].copy()
    n = row.size
    s = np.arange(1, n // 2)
    mags = np.abs(row[1 : n // 2])
    window = (s >= 2) & (s <= min(20, n // 2 - 1)) & (mags > 1e-300)
    if window.sum() >= 2:
        slope = np.polyfit(s[window], np.log(mags[window]), 1)[0]
        decay = float(np.exp(slope))
    else:
        decay = 0.0
    return {
        "max_entry": float(np.abs(row).max()),
        "band_plus_one_entry": float(abs(row[band_radius + 1])),
        "quarter_period_entry": float(abs(row[n // 4])),
        "decay_rate_per_node": decay,
    }


def audit_family(label: str, npts: int = 64) -> dict:
    mass, adv = relation_entries(label)
    B = sa.cyclic_compact_mass_matrix(mass, npts)
    Q = sa.cyclic_compact_advection_matrix(adv, npts)
    D = eliminated_matrix(label, npts)

    relation_residual = float(np.abs(B @ D - Q).max())

    row_sum = float(mass[0] + 2.0 * mass[1:].sum())
    weights = mass / row_sum
    metric_min_weight = float(weights.min())

    phi = flux_coefficients(adv)
    f = RNG.standard_normal(npts)
    F = face_flux(f, phi)
    flux_residual = float(np.abs(Q @ f - (F - np.roll(F, 1))).max())

    # per-cell balance under u_t + D f = 0 (h = 1): h d/dt(Bu) + dF = B(-Df) + Qf
    balance_residual = float(np.abs(B @ (-(D @ f)) + Q @ f).max())

    # mass tiling and its drift under an RK4 advection march
    u = RNG.standard_normal(npts)
    tiling_residual = float(abs((B @ u).sum() / row_sum - u.sum()))
    m0 = (B @ u).sum()
    dt = 0.2
    for _ in range(50):
        k1 = -(D @ u)
        k2 = -(D @ (u + 0.5 * dt * k1))
        k3 = -(D @ (u + 0.5 * dt * k2))
        k4 = -(D @ (u + dt * k3))
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    march_mass_drift = float(abs((B @ u).sum() - m0) / abs(m0))

    return {
        "family": label,
        "N": npts,
        "relation_radius_mass": int(len(mass) - 1),
        "relation_radius_flux": int(len(adv) - 1),
        "relation_residual": relation_residual,
        "metric_min_weight": metric_min_weight,
        "metric_row_sum": row_sum,
        "flux_entrywise_residual": flux_residual,
        "per_cell_balance_residual": balance_residual,
        "mass_tiling_residual": tiling_residual,
        "rk4_march_mass_drift": march_mass_drift,
        "eliminated_operator": dense_decay(D, int(len(adv) - 1)),
    }


def audit_matched_pairs(npts: int = 64) -> list[dict]:
    pairs = []
    for p in range(1, 6):
        Dc = sa.collocation_derivative_matrix(2 * p + 1, npts, h=1.0)
        Dg = sa.iga_derivative_matrix(p, npts, h=1.0)
        bm, _ = sa.collocation_mass_and_advection_entries(2 * p + 1)
        gm, _ = sa.iga_mass_and_advection_entries(p)
        pairs.append({
            "pair": f"BMC{2 * p + 1} = BMG{p}",
            "eliminated_agreement": float(np.abs(Dc - Dg).max()),
            "bmc_relation_radius": int(len(bm) - 1),
            "bmg_relation_radius": int(len(gm) - 1),
        })
    return pairs


def main() -> int:
    families = ([f"BMC{p}" for p in range(2, 12)]
                + [f"BMG{p}" for p in range(1, 6)]
                + ["C4", "C6", "C8"])
    records = [audit_family(label) for label in families]
    worst = {
        key: max(r[key] for r in records)
        for key in ("relation_residual", "flux_entrywise_residual",
                    "per_cell_balance_residual", "mass_tiling_residual",
                    "rk4_march_mass_drift")
    }
    payload = {
        "description": __doc__.strip().split("\n")[0],
        "statement": "Local FV balance holds at the banded compact relation: "
                     "h d/dt (Bu)_j = -(F_{j+1/2} - F_{j-1/2}) with fixed-radius "
                     "flux from the banded antisymmetric Q; the eliminated D is "
                     "dense (exponentially decaying circulant) and admits no "
                     "fixed-radius point-value flux.",
        "records": records,
        "matched_pairs": audit_matched_pairs(),
        "worst_residuals": worst,
        "min_metric_weight_across_catalog": min(r["metric_min_weight"] for r in records),
    }
    out = pp.results_path("s7_local_fv.json")
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    for r in records:
        d = r["eliminated_operator"]
        print(f"  {r['family']:>5}: relation {r['relation_residual']:.2e}  "
              f"flux {r['flux_entrywise_residual']:.2e}  "
              f"balance {r['per_cell_balance_residual']:.2e}  "
              f"minw {r['metric_min_weight']:+.3f}  "
              f"D decay/node {d['decay_rate_per_node']:.3f}  "
              f"band+1 {d['band_plus_one_entry']:.1e}  "
              f"N/4 {d['quarter_period_entry']:.1e}")
    print("worst:", {k: f"{v:.2e}" for k, v in worst.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
