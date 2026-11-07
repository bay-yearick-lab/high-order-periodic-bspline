"""Numerical validation of the periodic B-spline operators (fast, CPU-only).

Asserts the discrete structure the conservation argument rests on (SBP,
degree-shift identity, self-adjoint nonpositive second derivative,
energy-neutral skew-symmetric convection), the formal convergence orders of
the collocation ladder, and the headline receipts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import spline_analysis as sa  # noqa: E402

N = 48
H = 1.0 / N


@pytest.mark.parametrize("p", range(1, 6))
def test_degree_shift_identity_is_exact(p):
    """BMC_{2p+1} and BMG_p coincide entrywise, first and direct second derivative."""
    assert np.abs(sa.collocation_derivative_matrix(2 * p + 1, N) - sa.iga_derivative_matrix(p, N)).max() < 1e-12
    assert np.abs(sa.collocation_second_derivative_matrix(2 * p + 1, N)
                  - sa.iga_second_derivative_matrix(p, N)).max() < 1e-12


@pytest.mark.parametrize("p", range(2, 12))
def test_periodic_sbp_to_round_off(p):
    D = sa.collocation_derivative_matrix(p, N, H)
    Hn = sa.periodic_sbp_norm(N, H)
    assert np.abs(Hn @ D + D.T @ Hn).max() < 1e-13 * np.abs(D).max()


@pytest.mark.parametrize("p", range(2, 12))
def test_second_derivative_self_adjoint_nonpositive(p):
    D2 = sa.collocation_second_derivative_matrix(p, N, H)
    assert np.abs(D2 - D2.T).max() < 1e-12 * np.abs(D2).max()
    ev = np.linalg.eigvalsh(0.5 * (D2 + D2.T))
    assert ev.max() < 1e-9 * np.abs(ev).max()
    assert np.sum(np.abs(ev) < 1e-9 * np.abs(ev).max()) == 1  # kernel = constants only


@pytest.mark.parametrize("p", (3, 5, 9))
def test_skew_convection_injects_no_energy(p):
    """Split (skew-symmetric) convection of a periodic Burgers field conserves the discrete energy."""
    n = 64
    D = sa.collocation_derivative_matrix(p, n, 2 * np.pi / n)
    u = np.random.default_rng(p).standard_normal(n)
    conv = (D @ (u * u) + u * (D @ u)) / 3.0
    assert abs(u @ conv) < 1e-12 * np.linalg.norm(u) ** 3


@pytest.mark.parametrize("p", range(2, 10))
def test_first_derivative_reaches_formal_order(p):
    errs = []
    for n in (48, 64):
        h = 2 * np.pi / n
        x = np.arange(n) * h
        u = np.sin(2 * x) + 0.5 * np.cos(3 * x)
        du = 2 * np.cos(2 * x) - 1.5 * np.sin(3 * x)
        errs.append(np.abs(sa.collocation_derivative_matrix(p, n, h) @ u - du).max())
    q = 2 * ((p + 1) // 2)
    assert abs(np.log(errs[0] / errs[1]) / np.log(64 / 48) - q) < 0.25


def test_taylor_green_receipts():
    d = json.loads((ROOT / "results/exponential_family/gate9/e2_decision_core6.json").read_text())["e_full"]
    for n in ("N64", "N128"):
        assert 4.3 < d[n]["sl"] / d[n]["core6"] < 4.6
        assert 37.0 < d[n]["sl"] / d[n]["exp"] < 43.0


def test_wall_receipts():
    a = json.loads((ROOT / "results/ibm_cert/summary.json").read_text())["adoption_answer"]
    assert a["accuracy_factors"]["wall_trace"] == pytest.approx(62.7075, rel=1e-5)
    assert a["accuracy_factors"]["exterior_relative_l2"] == pytest.approx(32.1659, rel=1e-5)
    rows = list(csv.DictReader(open(ROOT / "results/ibm_cert/full_results.csv")))
    assert all(r["invariants_pass"] == "True" for r in rows)
