// c_sl_symbol_check.cpp -- verify the SLC spectral-like PENTADIAGONAL compact
// first- and second-derivative operators (route "sl") against their closed-form
// modified-wavenumber symbols. Applies the EXACT production stencils
// (FastDeriv::compactSL_d1 / compactSL_d2 from bm_apply.h) to a discrete Fourier
// mode and compares the recovered eigenvalue to the analytic pentadiagonal symbol
// (SLC 1992 JCP 103:16-42, eqs. 3.1.6-3.1.7 and 3.1.12).
//
//   D1:  k'h = (a sin w + (b/2) sin 2w + (c/3) sin 3w)
//                   / (1 + 2 alpha cos w + 2 beta cos 2w)
//   D2: -k''h^2 = 2[a2(cos w - 1) + (b2/4)(cos 2w - 1) + (c2/9)(cos 3w - 1)]
//                   / (1 + 2 alpha2 cos w + 2 beta2 cos 2w)
//
// PASS if max|applied - closed form| < 1e-11 over all resolvable modes, the
// low-mode error slopes match the formal order 4 (D1 ~ w^5, D2 ~ w^6), and the
// D2 symbol is nonpositive on (0, pi] (required by the SAN audit).

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>
#include "bm_apply.h"

// SL coefficients (must match the production build block in bm_apply.h).
static const double AL1 = 0.5771439,   BE1 = 0.0896406;
static const double A1  = 1.3025166,   B1  = 0.9935500,  C1 = 0.03750245;
static const double AL2 = 0.50209266,  BE2 = 0.05569169;
static const double A2  = 0.21564935,  B2  = 1.7233220,  C2 = 0.17659730;

static double d1_symbol(double w) {
    const double num = A1 * std::sin(w) + (B1 / 2) * std::sin(2 * w) + (C1 / 3) * std::sin(3 * w);
    const double den = 1.0 + 2.0 * AL1 * std::cos(w) + 2.0 * BE1 * std::cos(2 * w);
    return num / den;
}
static double d2_symbol(double w) {
    const double num = 2.0 * (A2 * (std::cos(w) - 1.0) + (B2 / 4) * (std::cos(2 * w) - 1.0)
                              + (C2 / 9) * (std::cos(3 * w) - 1.0));
    const double den = 1.0 + 2.0 * AL2 * std::cos(w) + 2.0 * BE2 * std::cos(2 * w);
    return num / den;
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
        f.compactSL_d1(re, d1re); f.compactSL_d1(im, d1im);
        f.compactSL_d2(re, d2re); f.compactSL_d2(im, d2im);
        std::complex<double> lam1(d1re[0], d1im[0]), lam2(d2re[0], d2im[0]);
        std::complex<double> lam1_cf = I * d1_symbol(w) / h;
        std::complex<double> lam2_cf = d2_symbol(w) / (h * h);
        e1max = std::max(e1max, std::abs(lam1 - lam1_cf));
        e2max = std::max(e2max, std::abs(lam2 - lam2_cf));
    }

    // 2. Formal-order slopes against the true derivative at low modes.
    // Formal order 4 => D1 symbol error ~ w^5 (slope 5), D2 error ~ w^6 (slope 6).
    const double w1 = 0.4, w2 = 0.2;
    const double q1 = std::log(std::fabs(d1_symbol(w1) - w1) /
                               std::fabs(d1_symbol(w2) - w2)) / std::log(2.0);
    const double q2 = std::log(std::fabs(d2_symbol(w1) + w1 * w1) /
                               std::fabs(d2_symbol(w2) + w2 * w2)) / std::log(2.0);

    // 3. D2 symbol nonpositivity on (0, pi] (SAN audit): scan finely, report bounds.
    double d2min = 1e300, d2max = -1e300;
    const int M = 20000;
    for (int i = 1; i <= M; ++i) {
        const double w = M_PI * i / M;
        const double v = d2_symbol(w);
        d2min = std::min(d2min, v);
        d2max = std::max(d2max, v);
    }

    const bool sym_ok  = e1max < 1e-11 && e2max < 1e-11;
    const bool ord_ok  = std::fabs(q1 - 5.0) < 0.2 && std::fabs(q2 - 6.0) < 0.2;
    const bool sgn_ok  = d2max <= 0.0;
    const bool ok = sym_ok && ord_ok && sgn_ok;

    std::printf("SL: max|applied-symbol| D1 %.2e  D2 %.2e ; observed orders D1 %.3f  D2 %.3f\n",
                e1max, e2max, q1, q2);
    std::printf("SL: D2 symbol on (0,pi]  min %.6f  max %.3e  -> %s\n",
                d2min, d2max, sgn_ok ? "nonpositive" : "SIGN VIOLATION");
    std::printf("SL: %s (symbol %s, order %s, sign %s)\n",
                ok ? "PASS" : "FAIL",
                sym_ok ? "ok" : "FAIL", ord_ok ? "ok" : "FAIL", sgn_ok ? "ok" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
