import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import spline_analysis as sa


class PeriodicSBPStructureTests(unittest.TestCase):
    def _assert_h_self_adjoint(
        self, operator: np.ndarray, h: float, tol: float = 1e-12
    ) -> None:
        norm = sa.periodic_sbp_norm(operator.shape[0], h=h)
        residual = norm @ operator - operator.T @ norm
        self.assertLess(np.linalg.norm(residual, ord=np.inf), tol)

    def _assert_h_nonpositive_semidefinite(
        self, operator: np.ndarray, h: float, tol: float = 1e-12
    ) -> None:
        normed = sa.periodic_sbp_norm(operator.shape[0], h=h) @ operator
        symmetric_part = 0.5 * (normed + normed.T)
        eigenvalues = np.linalg.eigvalsh(symmetric_part)
        self.assertLessEqual(eigenvalues[-1], tol)

    def test_periodic_sbp_norm_is_scaled_identity(self) -> None:
        np.testing.assert_allclose(
            sa.periodic_sbp_norm(6, h=0.125),
            0.125 * np.eye(6),
        )

    def test_periodic_sbp_residual_matches_hid_plus_dt_h(self) -> None:
        derivative = np.array(
            [
                [1.0, 2.0, -1.0],
                [0.0, -3.0, 4.0],
                [2.0, 1.0, 0.5],
            ]
        )
        h = 0.25
        expected = h * (derivative + derivative.T)
        self.assertGreater(np.linalg.norm(expected, ord=np.inf), 0.0)
        np.testing.assert_allclose(
            sa.periodic_sbp_residual(derivative, h=h),
            expected,
        )

    def test_collocation_first_derivative_is_periodic_sbp(self) -> None:
        for degree, npts in ((2, 16), (6, 24), (7, 32)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                derivative = sa.collocation_derivative_matrix(degree, npts, h=h)
                residual = sa.periodic_sbp_residual(derivative, h=h)
                self.assertLess(np.linalg.norm(residual, ord=np.inf), 1e-12)

    def test_iga_first_derivative_is_periodic_sbp(self) -> None:
        for degree, npts in ((1, 16), (2, 24), (4, 32)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                derivative = sa.iga_derivative_matrix(degree, npts, h=h)
                residual = sa.periodic_sbp_residual(derivative, h=h)
                self.assertLess(np.linalg.norm(residual, ord=np.inf), 1e-12)

    def test_collocation_second_derivative_is_h_self_adjoint(self) -> None:
        for degree, npts in ((2, 16), (7, 32)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                operator = sa.collocation_second_derivative_matrix(degree, npts, h=h)
                self._assert_h_self_adjoint(operator, h=h)

    def test_collocation_second_derivative_is_h_nonpositive_semidefinite(self) -> None:
        for degree, npts in ((2, 16), (7, 32)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                operator = sa.collocation_second_derivative_matrix(degree, npts, h=h)
                self._assert_h_nonpositive_semidefinite(operator, h=h)

    def test_iga_second_derivative_is_h_self_adjoint(self) -> None:
        for degree, npts in ((1, 16), (3, 24)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                operator = sa.iga_second_derivative_matrix(degree, npts, h=h)
                self._assert_h_self_adjoint(operator, h=h)

    def test_iga_second_derivative_is_h_nonpositive_semidefinite(self) -> None:
        for degree, npts in ((1, 16), (3, 24)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                operator = sa.iga_second_derivative_matrix(degree, npts, h=h)
                self._assert_h_nonpositive_semidefinite(operator, h=h)

    def test_periodic_direct_second_derivative_identity_matches_collocation(self) -> None:
        for degree, npts in ((1, 16), (2, 24), (4, 32)):
            with self.subTest(degree=degree, npts=npts):
                h = 2.0 * np.pi / npts
                np.testing.assert_allclose(
                    sa.iga_second_derivative_matrix(degree, npts, h=h),
                    sa.collocation_second_derivative_matrix(2 * degree + 1, npts, h=h),
                    atol=1e-12,
                    rtol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
