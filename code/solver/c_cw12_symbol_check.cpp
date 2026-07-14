// c_cw12_symbol_check.cpp -- verify the two CERTIFIED WIDE PENTADIAGONAL
// compact routes ("cw12a", "cw12b") against their closed-form
// modified-wavenumber symbols. Applies the EXACT production stencils
// (FastDeriv::compactCWA_d1/d2, compactCWB_d1/d2 from bm_apply.h) to discrete
// Fourier modes and compares the recovered eigenvalue to the analytic
// pentadiagonal symbol. D1 coefficients are the certified frontier cells
// (cw12a = L2r7q12 floor 0.05,
// cw12b = L2r5q12 floor 0.025); D2 partners are the matched threshold-maximal
// pentadiagonal members.
//
//   D1:  k'h    = (sum 2 a_s sin(s w)) / (1 + 2 al1 cos w + 2 al2 cos 2w)
//   D2:  k''h^2 = (b_0 + sum 2 b_s cos(s w)) / (1 + 2 bl1 cos w + 2 bl2 cos 2w)
//
// PASS requires, per route:
//   1. max|applied - closed form| < 1e-11 over all resolvable modes (N=64);
//   2. the certified 1% dispersion cut is REPRODUCED on a dense grid: the D1
//      error stays <= 1% on all of (0, certified_cut], and the dense-grid cut
//      itself is >= certified_cut - 1e-3. (The certified value is an interval-
//      arithmetic LOWER bound; the dense-grid cut may exceed it by up to the
//      cell's recorded certification gap -- cw12a: cut 0.8616*pi certified,
//      0.9002*pi on the grid, gap 0.039 in the sweep record; cw12b is tight,
//      0.8225 certified vs 0.8228 grid.)
//   3. low-mode error slopes match the exact-series predictions for formal
//      order 12 at anchors w=0.8/0.4 (pre-verified in Python from the exact
//      Taylor coefficients; the bare exponents are 13 (D1) and 14 (D2), the
//      series predictions fold in the honest higher-order contamination):
//      D1 12.554 / 13.004, D2 13.211 / 14.024 (cw12a / cw12b), tol 0.3;
//   4. the D2 symbol is strictly negative on (0, pi] (SAN audit), scanned at
//      20000 points, and its dense-grid 1% cut matches the partner JSON
//      (0.9701*pi / 0.9367*pi) within 1e-3.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>
#include "bm_apply.h"

struct CW {
    const char* name;
    int R;                                   // RHS radius (D1 and D2)
    double al1, al2;                         // D1 pentadiagonal LHS
    const double* a;                         // D1 RHS a_1..a_R
    double bl1, bl2, b0;                     // D2 LHS + center
    const double* b;                         // D2 RHS b_1..b_R
    double cert_cut_pi;                      // certified D1 1% cut / pi
    double d2_cut_pi;                        // partner JSON D2 1% cut / pi
    double slope1, slope2;                   // series-predicted order slopes
};

// cw12a (must match the production build block in bm_apply.h)
static const double A1A[7] = {0.8359351614953894, 0.06465601447530303,
                              -0.05254555071859491, 0.011976865033900547,
                              -0.0025753384161538, 0.00038591694604337513,
                              -2.8343087156344804e-05};
static const double B2A[7] = {0.9043854478128212, 0.5970618140260426,
                              -0.3252810784893023, 0.040779787773292164,
                              -0.006239267401751708, 0.0007337128953475037,
                              -4.4574635453931686e-05};
// cw12b
static const double A1B[5] = {0.6474244112956682, 0.25095267402010113,
                              0.00827175994799237, -0.0003731748662601286,
                              1.3116508343528986e-05};
static const double B2B[5] = {0.15392188750639246, 0.43804625186979695,
                              0.03355551188077371, -0.001018485995992867,
                              2.8059003771746482e-05};

static const CW CWS[2] = {
    {"cw12a", 7, 0.40937878082640167, -0.06462037441519262, A1A,
     0.1601749806496769, -0.21726126868097, -2.4227916839619907, B2A,
     0.8615559156697674, 0.9700671205929359, 12.554, 13.211},
    {"cw12b", 5, 0.5796083747693036, 0.09310954748722111, A1B,
     0.5273008328764947, 0.06895526765898032, -1.2490664485294838, B2B,
     0.8225422332152466, 0.9366842935036545, 13.004, 14.024},
};

static double d1_symbol(const CW& c, double w) {
    double num = 0.0;
    for (int s = 1; s <= c.R; ++s) num += 2.0 * c.a[s - 1] * std::sin(s * w);
    return num / (1.0 + 2.0 * c.al1 * std::cos(w) + 2.0 * c.al2 * std::cos(2.0 * w));
}
static double d2_symbol(const CW& c, double w) {
    double num = c.b0;
    for (int s = 1; s <= c.R; ++s) num += 2.0 * c.b[s - 1] * std::cos(s * w);
    return num / (1.0 + 2.0 * c.bl1 * std::cos(w) + 2.0 * c.bl2 * std::cos(2.0 * w));
}

int main() {
    const int N = 64, p = 5;
    const double L = 2.0 * M_PI, h = L / N;
    FastDeriv f = FastDeriv::build(true, N, p, L);
    const std::complex<double> I(0.0, 1.0);
    bool all_ok = true;

    for (const CW& c : CWS) {
        const bool isA = (c.R == 7);

        // 1. Applied production kernel vs closed-form symbol, all resolvable modes.
        double e1max = 0.0, e2max = 0.0;
        for (int m = 1; m <= N / 2; ++m) {
            const double w = 2.0 * M_PI * m / N;
            std::vector<double> re(N), im(N), d1re, d1im, d2re, d2im;
            for (int j = 0; j < N; ++j) { re[j] = std::cos(j * w); im[j] = std::sin(j * w); }
            if (isA) { f.compactCWA_d1(re, d1re); f.compactCWA_d1(im, d1im);
                       f.compactCWA_d2(re, d2re); f.compactCWA_d2(im, d2im); }
            else     { f.compactCWB_d1(re, d1re); f.compactCWB_d1(im, d1im);
                       f.compactCWB_d2(re, d2re); f.compactCWB_d2(im, d2im); }
            std::complex<double> lam1(d1re[0], d1im[0]), lam2(d2re[0], d2im[0]);
            std::complex<double> lam1_cf = I * d1_symbol(c, w) / h;
            std::complex<double> lam2_cf = d2_symbol(c, w) / (h * h);
            e1max = std::max(e1max, std::abs(lam1 - lam1_cf));
            e2max = std::max(e2max, std::abs(lam2 - lam2_cf));
        }

        // 2. Certified D1 1% cut on a dense grid (60000 points, as the sweep).
        const int M = 60000;
        double cut1 = M_PI, err_below_cert = 0.0;
        bool past = false;
        for (int i = 1; i <= M; ++i) {
            const double w = 1e-4 + (M_PI - 1e-4) * (i - 1) / (M - 1);
            const double err = std::fabs(d1_symbol(c, w) / w - 1.0);
            if (w <= c.cert_cut_pi * M_PI) err_below_cert = std::max(err_below_cert, err);
            if (!past && err > 0.01) {
                cut1 = 1e-4 + (M_PI - 1e-4) * (i - 2) / (M - 1);
                past = true;
            }
        }
        // ... and the D2 partner's own 1% cut against the partner JSON value.
        double cut2 = M_PI;
        past = false;
        for (int i = 1; i <= M && !past; ++i) {
            const double w = 1e-4 + (M_PI - 1e-4) * (i - 1) / (M - 1);
            const double err = std::fabs(d2_symbol(c, w) / (w * w) + 1.0);
            if (err > 0.01) {
                cut2 = 1e-4 + (M_PI - 1e-4) * (i - 2) / (M - 1);
                past = true;
            }
        }

        // 3. Order slopes at w=0.8 / 0.4 vs the exact-series predictions.
        const double w1 = 0.8, w2 = 0.4;
        const double q1 = std::log(std::fabs(d1_symbol(c, w1) - w1) /
                                   std::fabs(d1_symbol(c, w2) - w2)) / std::log(2.0);
        const double q2 = std::log(std::fabs(d2_symbol(c, w1) + w1 * w1) /
                                   std::fabs(d2_symbol(c, w2) + w2 * w2)) / std::log(2.0);

        // 4. D2 symbol strict negativity on (0, pi].
        double d2min = 1e300, d2max = -1e300;
        const int MS = 20000;
        for (int i = 1; i <= MS; ++i) {
            const double w = M_PI * i / MS;
            const double v = d2_symbol(c, w);
            d2min = std::min(d2min, v);
            d2max = std::max(d2max, v);
        }

        const bool sym_ok  = e1max < 1e-11 && e2max < 1e-11;
        const bool cut1_ok = err_below_cert <= 0.01 + 1e-9
                          && cut1 / M_PI >= c.cert_cut_pi - 1e-3;
        const bool cut2_ok = std::fabs(cut2 / M_PI - c.d2_cut_pi) < 1e-3;
        const bool ord_ok  = std::fabs(q1 - c.slope1) < 0.3 && std::fabs(q2 - c.slope2) < 0.3;
        const bool sgn_ok  = d2max < 0.0;
        const bool ok = sym_ok && cut1_ok && cut2_ok && ord_ok && sgn_ok;
        all_ok = all_ok && ok;

        std::printf("%s: max|applied-symbol| D1 %.2e  D2 %.2e ; observed orders D1 %.3f (pred %.3f)  D2 %.3f (pred %.3f)\n",
                    c.name, e1max, e2max, q1, c.slope1, q2, c.slope2);
        std::printf("%s: D1 1%% cut %.6f*pi (certified lower bound %.6f, max err below it %.6f)\n",
                    c.name, cut1 / M_PI, c.cert_cut_pi, err_below_cert);
        std::printf("%s: D2 1%% cut %.6f*pi (partner json %.6f); D2 symbol on (0,pi] min %.6f max %.3e -> %s\n",
                    c.name, cut2 / M_PI, c.d2_cut_pi, d2min, d2max,
                    sgn_ok ? "strictly negative" : "SIGN VIOLATION");
        std::printf("%s: %s (symbol %s, d1cut %s, d2cut %s, order %s, sign %s)\n\n",
                    c.name, ok ? "PASS" : "FAIL",
                    sym_ok ? "ok" : "FAIL", cut1_ok ? "ok" : "FAIL",
                    cut2_ok ? "ok" : "FAIL", ord_ok ? "ok" : "FAIL",
                    sgn_ok ? "ok" : "FAIL");
    }
    std::printf("CW12 symbol check: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
