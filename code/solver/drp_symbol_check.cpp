// drp_symbol_check.cpp  --  Gate: verify the DRP (Tam & Webb, JCP 107, 1993)
// first-derivative operator used by the DNS engine against the PUBLISHED
// coefficients and the closed-form modified-wavenumber symbol. Two checks:
//
//   (1) COEFFICIENT DRIFT: extract the production stencil weights from the
//       impulse response of FastDeriv::drp_d1 (bm_apply.h) and require exact
//       equality (1e-14 h-scaled) with the published JCP 107 values, which are
//       also the values in the Python symbol library (spline_analysis.py).
//       This gate exists because until 2026-07-13 the production block carried
//       Tam's later REVISED coefficients (0.770882380518, ...) mislabeled as
//       the 1993 operator, while every symbol-level paper artifact used the
//       true 1993 set: two different operators under one label and citation,
//       invisible to order-of-accuracy tests (both sets are 4th order).
//   (2) SYMBOL: apply the production stencil to discrete Fourier modes and
//       compare to  k_eff h = 2 [ a1 sin w + a2 sin 2w + a3 sin 3w ].
//
// PASS if both hold; returns nonzero otherwise.

#include <cstdio>
#include <cmath>
#include <vector>
#include "bm_apply.h"

int main() {
    const int    N = 48;
    const int    p = 5;                  // spline degree irrelevant for FD routes
    const double L = 2.0 * M_PI;
    const double h = L / N;
    FastDeriv f = FastDeriv::build(/*periodic=*/true, N, p, L);

    // Published Tam & Webb JCP 107 (1993) coefficients (= spline_analysis.py).
    const double a1 = 0.79926643, a2 = -0.18941314, a3 = 0.02651995;

    // (1) stencil extraction from the production impulse response
    std::vector<double> imp(N, 0.0), resp(N);
    imp[0] = 1.0;
    f.drp_d1(imp, resp);
    // D phi |_j for phi = e_0 gives the REVERSED stencil column: resp[j] = w_{-j}
    const double got_a1 = -resp[1] * h, got_a2 = -resp[2] * h, got_a3 = -resp[3] * h;
    const double ga1 = resp[N - 1] * h, ga2 = resp[N - 2] * h, ga3 = resp[N - 3] * h;
    double drift = 0.0;
    drift = std::fmax(drift, std::fabs(got_a1 - a1));
    drift = std::fmax(drift, std::fabs(got_a2 - a2));
    drift = std::fmax(drift, std::fabs(got_a3 - a3));
    drift = std::fmax(drift, std::fabs(ga1 - a1));   // antisymmetry
    drift = std::fmax(drift, std::fabs(ga2 - a2));
    drift = std::fmax(drift, std::fabs(ga3 - a3));
    std::printf(" DRP stencil from production impulse response:\n");
    std::printf("   a1 = %.12f (published %.8f)\n", got_a1, a1);
    std::printf("   a2 = %.12f (published %.8f)\n", got_a2, a2);
    std::printf("   a3 = %.12f (published %.8f)\n", got_a3, a3);
    std::printf("   max coefficient drift = %.3e\n", drift);

    // (2) modified-wavenumber symbol on every resolvable mode
    double max_err = 0.0;
    std::vector<double> re(N), im(N), dre(N), dim(N);
    for (int m = 1; m <= N / 2; ++m) {
        const double w = 2.0 * M_PI * m / N;
        for (int j = 0; j < N; ++j) { re[j] = std::cos(j * w); im[j] = std::sin(j * w); }
        f.drp_d1(re, dre); f.drp_d1(im, dim);
        // D e^{i j w} = i keff e^{i j w}: at j=0, dre = -keff*sin(0)=... use both parts
        const double keff = dim[0];                        // Im part at j=0: keff*cos(0)
        const double sym = (2.0 / h) * (a1 * std::sin(w) + a2 * std::sin(2 * w) + a3 * std::sin(3 * w));
        max_err = std::fmax(max_err, std::fabs(keff - sym));
        max_err = std::fmax(max_err, std::fabs(dre[0]));   // real part must vanish
    }
    std::printf(" DRP symbol sweep: max|applied - closed form| = %.3e\n", max_err);

    const bool pass = drift < 1e-14 && max_err < 1e-10;
    std::printf(" %s\n", pass ? "DRP GATE: PASS" : "** DRP GATE: FAIL **");
    return pass ? 0 : 1;
}
