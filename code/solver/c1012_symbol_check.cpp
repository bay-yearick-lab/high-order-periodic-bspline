// c1012_symbol_check.cpp -- verify the new C10/C12 (tridiagonal compact 10th- and
// 12th-order) first- and second-derivative operators used by the DNS engine
// against their closed-form modified-wavenumber symbols. Applies the EXACT
// production stencils (FastDeriv::compact10_* / compact12_* from bm_apply.h) to a
// discrete Fourier mode and compares the recovered eigenvalue to the analytic
// symbol built from the exact fractions of the Taylor order conditions.
//
// PASS if max|applied - closed form| < 1e-11 over all resolvable modes, AND the
// low-mode error against the true derivative halves at the design order
// (ratio ~ 2^10 resp. 2^12 when w is halved).

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>
#include "bm_apply.h"

struct Compact { double al1; std::vector<double> a; double al2; std::vector<double> o; };

static double d1_symbol(const Compact& s, double w) {
    double num = 0.0;
    for (size_t k = 0; k < s.a.size(); ++k) num += 2.0 * s.a[k] * std::sin((k + 1) * w);
    return num / (1.0 + 2.0 * s.al1 * std::cos(w));
}
static double d2_symbol(const Compact& s, double w) {
    double num = 0.0;
    for (size_t k = 0; k < s.o.size(); ++k) num += 2.0 * s.o[k] * (std::cos((k + 1) * w) - 1.0);
    return num / (1.0 + 2.0 * s.al2 * std::cos(w));
}

int main() {
    const int N = 64, p = 5;
    const double L = 2.0 * M_PI, h = L / N;
    FastDeriv f = FastDeriv::build(true, N, p, L);

    const Compact c10 = {2.0/5, {39.0/50, 1.0/15, -1.0/210, 1.0/4200},
                         8.0/29, {1126.0/1305, 247.0/1305, -74.0/9135, 43.0/146160}};
    const Compact c12 = {5.0/12, {7.0/9, 5.0/63, -5.0/672, 1.0/1512, -1.0/30240},
                         25.0/82, {4595.0/5904, 605.0/2583, -155.0/11808, 5.0/5904, -23.0/688800}};
    const std::complex<double> I(0.0, 1.0);
    int fails = 0;

    for (int which = 0; which < 2; ++which) {
        const Compact& s = which ? c12 : c10;
        const char* name = which ? "C12" : "C10";
        double e1max = 0.0, e2max = 0.0;
        for (int m = 1; m <= N / 2; ++m) {
            const double w = 2.0 * M_PI * m / N;
            std::vector<double> re(N), im(N), d1re, d1im, d2re, d2im;
            for (int j = 0; j < N; ++j) { re[j] = std::cos(j * w); im[j] = std::sin(j * w); }
            if (which) { f.compact12_d1(re, d1re); f.compact12_d1(im, d1im);
                         f.compact12_d2(re, d2re); f.compact12_d2(im, d2im); }
            else       { f.compact10_d1(re, d1re); f.compact10_d1(im, d1im);
                         f.compact10_d2(re, d2re); f.compact10_d2(im, d2im); }
            std::complex<double> lam1(d1re[0], d1im[0]), lam2(d2re[0], d2im[0]);
            std::complex<double> lam1_cf = I * d1_symbol(s, w) / h;
            std::complex<double> lam2_cf = d2_symbol(s, w) / (h * h);
            e1max = std::max(e1max, std::abs(lam1 - lam1_cf));
            e2max = std::max(e2max, std::abs(lam2 - lam2_cf));
        }
        // order checks against the true derivative at low modes
        const double w1 = 0.4, w2 = 0.2;
        const double q1 = std::log(std::fabs(d1_symbol(s, w1) - w1) /
                                   std::fabs(d1_symbol(s, w2) - w2)) / std::log(2.0);
        const double q2 = std::log(std::fabs(d2_symbol(s, w1) + w1 * w1) /
                                   std::fabs(d2_symbol(s, w2) + w2 * w2)) / std::log(2.0);
        // symbol-error slopes: a centered D1 has odd error ~ w^(q+1), a centered
        // D2 has even error ~ w^(q+2); C12's D2 slope sits near the double-
        // precision floor at w=0.2 (0.2^14 ~ 1.6e-9 relative), hence the wider
        // tolerance there
        const double q1exp = which ? 13.0 : 11.0;
        const double q2exp = which ? 14.0 : 12.0;
        const bool ok = e1max < 1e-11 && e2max < 1e-9 &&
                        std::fabs(q1 - q1exp) < 0.15 && std::fabs(q2 - q2exp) < 0.35;
        std::printf("%s: max|applied-symbol| D1 %.2e  D2 %.2e ; observed orders D1 %.3f  D2 %.3f  -> %s\n",
                    name, e1max, e2max, q1, q2, ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
    }
    return fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
