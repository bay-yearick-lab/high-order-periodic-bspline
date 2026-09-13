"""Checks for the generic exponential-family route.

The route is defined by the lattice samples of one generator E_{m,r}(omega),
stored in a JSON member file. The operators assembled from that file must agree
with family.py's own alias-sum symbols, and must carry the parity the samples
have: beta' is odd, so the first-derivative matrix is antisymmetric; beta'' is
even, so the second-derivative matrix is symmetric.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
ANALYSIS_DIR = CODE_DIR / "analysis" / "exponential_family"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import spline_analysis as sa
import experiment_d1 as ed
from family import ExponentialGenerator

MEMBER_PATH = (
    CODE_DIR.parent / "results" / "exponential_family" / "members" / "E62_provisional.json"
)
MEMBER_LABEL = f"{sa.EXP_LABEL_PREFIX}{MEMBER_PATH}"
TOL = 1e-12


class ExponentialMemberRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MEMBER_PATH.exists():
            raise unittest.SkipTest(
                "Regenerate the member with code/analysis/exponential_family/write_member.py."
            )
        cls.generator = ExponentialGenerator(6, [1.769951, 2.060426])

    def test_member_file_reproduces_the_generator_samples(self) -> None:
        record = sa.exp_member_record(MEMBER_LABEL)
        samples = self.generator.integer_samples()
        center = record["radius"]
        self.assertEqual(record["p"], self.generator.p)
        self.assertEqual(record["radius"], self.generator.nmax)
        np.testing.assert_allclose(record["beta"], samples.beta[center:], atol=1e-15)
        np.testing.assert_allclose(record["dbeta"], samples.dbeta[center:], atol=1e-15)
        np.testing.assert_allclose(record["ddbeta"], samples.ddbeta[center:], atol=1e-15)

    def test_first_derivative_symbol_matches_the_alias_sum_symbol(self) -> None:
        # family.py reports kappa_1: the nodal first derivative has symbol
        # (i/h) kappa_1, which is the modified wavenumber this module returns.
        theta = np.linspace(0.0, np.pi, 401)
        _, kappa1, _ = self.generator.symbols(theta)
        measured = sa.first_derivative_symbol_by_label(MEMBER_LABEL, theta)
        self.assertLess(float(np.max(np.abs(measured - kappa1))), TOL)

    def test_second_derivative_symbol_matches_the_alias_sum_symbol(self) -> None:
        # family.py reports kappa_2: the nodal second derivative has symbol
        # -(1/h^2) kappa_2, and ksecond_* returns kappa_2 with the module's
        # convention (the positive modified wavenumber squared).
        theta = np.linspace(0.0, np.pi, 401)
        _, _, kappa2 = self.generator.symbols(theta)
        measured = sa.ksecond_exp_member(theta, MEMBER_LABEL)
        self.assertLess(float(np.max(np.abs(measured - kappa2))), TOL)

    def test_matrix_eigenvalues_match_the_symbol_on_every_grid_mode(self) -> None:
        npts = 32
        theta = 2.0 * np.pi * np.arange(npts) / npts
        d1 = sa.first_derivative_matrix_by_label(MEMBER_LABEL, npts)
        d2 = sa.exp_member_second_derivative_matrix(MEMBER_LABEL, npts)
        j = np.arange(npts)
        for mode in range(npts):
            vector = np.exp(1j * theta[mode] * j)
            _, kappa1, kappa2 = self.generator.symbols(np.array([theta[mode]]))
            np.testing.assert_allclose(
                d1 @ vector, 1j * kappa1[0] * vector, atol=TOL
            )
            np.testing.assert_allclose(
                d2 @ vector, -kappa2[0] * vector, atol=TOL
            )

    def test_first_derivative_matrix_is_antisymmetric_and_second_is_symmetric(self) -> None:
        npts = 24
        d1 = sa.first_derivative_matrix_by_label(MEMBER_LABEL, npts)
        d2 = sa.exp_member_second_derivative_matrix(MEMBER_LABEL, npts)
        self.assertLess(float(np.max(np.abs(d1 + d1.T))), TOL)
        self.assertLess(float(np.max(np.abs(d2 - d2.T))), TOL)

    def test_registered_label_and_path_label_agree(self) -> None:
        label = sa.register_exp_member("E62_provisional", MEMBER_PATH)
        np.testing.assert_allclose(
            sa.first_derivative_matrix_by_label(label, 16),
            sa.first_derivative_matrix_by_label(MEMBER_LABEL, 16),
        )
        for entries_registered, entries_path in zip(
            ed.unique_scheme_entries(label), ed.unique_scheme_entries(MEMBER_LABEL)
        ):
            np.testing.assert_allclose(entries_registered, entries_path)
        for entries_registered, entries_path in zip(
            ed.unique_second_derivative_scheme_entries(label),
            ed.unique_second_derivative_scheme_entries(MEMBER_LABEL),
        ):
            np.testing.assert_allclose(entries_registered, entries_path)

    def test_existing_labels_are_unchanged_by_the_exp_prefix(self) -> None:
        # "E6" is the explicit centered sixth-order baseline, not a member.
        self.assertFalse(sa.is_exp_member_label("E6"))
        np.testing.assert_allclose(
            sa.first_derivative_matrix_by_label("E6", 16),
            sa.compact_derivative_matrix(*sa.explicit_centered_fd_entries(6), 16),
        )
        np.testing.assert_allclose(
            ed.unique_scheme_entries("E6")[1], sa.explicit_centered_fd_entries(6)[1]
        )


if __name__ == "__main__":
    unittest.main()
