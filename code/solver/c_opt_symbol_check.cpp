// c_opt_symbol_check.cpp -- verify the paper's THRESHOLD-OPTIMIZED TRIDIAGONAL
// compact first- and second-derivative operators (route "opt") against their
// closed-form modified-wavenumber symbols. Applies the EXACT production stencils
// (FastDeriv::compactOPT_d1 / compactOPT_d2 from bm_apply.h) to a discrete
// Fourier mode and compares the recovered eigenvalue to the analytic tridiagonal
// symbol. D1 is frontier cell L1r5q8 (verified 1% cut 0.8810*pi); D2 is its
// matched threshold-maximal partner (1% cut 0.9582*pi).
//
//   D1:  k'h   = (sum_{s=1..5} 2 a_s sin(s w)) / (1 + 2 alpha cos w)
//   D2:  k''h^2 = (a_0 + sum_{s=1..5} 2 a_s cos(s w)) / (1 + 2 alpha2 cos w)  (~ -w^2)
//
// PASS if max|applied - closed form| < 1e-11 over all resolvable modes, the
// low-mode error slopes match formal order 8 (D1 ~ w^9 => slope 9, D2 error
// against -w^2 ~ w^10 => slope 10), and the D2 symbol is strictly negative on
// (0, pi] (required by the SAN audit).

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>
#include "bm_apply.h"

// OPT coefficients (must match the production build block in bm_apply.h).
static const double AL1 = 0.4742169929451113;
static const double A1[5] = {0.7598825426879071, 0.13489507072322338,
                             -0.023260830361993556, 0.004069991868971751,
                             -0.00039063351582985587};
static const double AL2 = 0.4320698009820234;
static const double A0  = -1.555224078876143;
static const double A2[5] = {0.35890831739343104, 0.45913001629633177,
                             -0.04560561305199756, 0.00559124812826596,
                             -0.0004119293279594219};

static double d1_symbol(double w) {
    double num = 0.0;
    for (int s = 1; s <= 5; ++s) num += 2.0 * A1[s - 1] * std::sin(s * w);
    return num / (1.0 + 2.0 * AL1 * std::cos(w));
}
static double d2_symbol(double w) {
    double num = A0;
    for (int s = 1; s <= 5; ++s) num += 2.0 * A2[s - 1] * std::cos(s * w);
    return num / (1.0 + 2.0 * AL2 * std::cos(w));
}

int main() {
    const int N = 64, p = 5;
    const double L = 2.0 * M_PI, h = L / N;
    FastDeriv f = FastDeriv::build(true, N, p, L);
    const std::complex<double> I(0.0, 1.0);

    // 1. Applied production kernel vs closed-form symbol over all resolvable modes.
    double e1max = 0.0, e2max = 0.0;
    for (int m = 1; m <= N / 2; ++m) {
        const double w = 2.0 * M_PI * m / N;
        std::vector<double> re(N), im(N), d1re, d1im, d2re, d2im;
        for (int j = 0; j < N; ++j) { re[j] = std::cos(j * w); im[j] = std::sin(j * w); }
        f.compactOPT_d1(re, d1re); f.compactOPT_d1(im, d1im);
        f.compactOPT_d2(re, d2re); f.compactOPT_d2(im, d2im);
        std::complex<double> lam1(d1re[0], d1im[0]), lam2(d2re[0], d2im[0]);
        std::complex<double> lam1_cf = I * d1_symbol(w) / h;
        std::complex<double> lam2_cf = d2_symbol(w) / (h * h);
        e1max = std::max(e1max, std::abs(lam1 - lam1_cf));
        e2max = std::max(e2max, std::abs(lam2 - lam2_cf));
    }

    // 2. Formal-order slopes against the true derivative at low modes.
    // Formal order 8 => D1 symbol error ~ w^9 (slope 9), D2 error ~ w^10 (slope 10).
    const double w1 = 0.4, w2 = 0.2;
    const double q1 = std::log(std::fabs(d1_symbol(w1) - w1) /
                               std::fabs(d1_symbol(w2) - w2)) / std::log(2.0);
    const double q2 = std::log(std::fabs(d2_symbol(w1) + w1 * w1) /
                               std::fabs(d2_symbol(w2) + w2 * w2)) / std::log(2.0);

    // 3. D2 symbol strict negativity on (0, pi] (SAN audit): scan finely, bounds.
    double d2min = 1e300, d2max = -1e300;
    const int M = 20000;
    for (int i = 1; i <= M; ++i) {
        const double w = M_PI * i / M;
        const double v = d2_symbol(w);
        d2min = std::min(d2min, v);
        d2max = std::max(d2max, v);
    }

    const bool sym_ok  = e1max < 1e-11 && e2max < 1e-11;
    const bool ord_ok  = std::fabs(q1 - 9.0) < 0.2 && std::fabs(q2 - 10.0) < 0.2;
    const bool sgn_ok  = d2max < 0.0;
    const bool ok = sym_ok && ord_ok && sgn_ok;

    std::printf("OPT: max|applied-symbol| D1 %.2e  D2 %.2e ; observed orders D1 %.3f  D2 %.3f\n",
                e1max, e2max, q1, q2);
    std::printf("OPT: D2 symbol on (0,pi]  min %.6f  max %.3e  -> %s\n",
                d2min, d2max, sgn_ok ? "strictly negative" : "SIGN VIOLATION");
    std::printf("OPT: %s (symbol %s, order %s, sign %s)\n",
                ok ? "PASS" : "FAIL",
                sym_ok ? "ok" : "FAIL", ord_ok ? "ok" : "FAIL", sgn_ok ? "ok" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
