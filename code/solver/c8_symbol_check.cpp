// c8_symbol_check.cpp  --  Gate 1: verify the C8 (compact Pade 8th-order) first-
// and second-derivative operators used by the DNS engine against their closed-form
// modified-wavenumber symbols. Applies the EXACT production stencils (FastDeriv::
// compact8_d1 / compact8_d2 from bm_apply.h) to a discrete Fourier mode and
// compares the recovered eigenvalue to the analytic Pade symbol.
//
//   D1 symbol (paper Eq. 42): i k_eff,  k_eff*h =
//        ( a sin w + (b/2) sin 2w + (c/3) sin 3w ) / (1 + 2 alpha cos w)
//   with alpha=3/8, a=25/16, b=1/5, c=-1/80.
//   D2 symbol: kk_eff*h^2 = ( 2 o1 (1-cos w) + 2 o2 (1-cos 2w) + 2 o3 (1-cos 3w) )
//                            / (1 + 2 alpha2 cos w),  the eigenvalue of -D2.
//
// PASS if max|applied - closed form| < 1e-12 over a sweep of w in (0, pi).

#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>
#include "bm_apply.h"

int main() {
    const int    N = 48;                 // periodic grid, plenty of modes
    const int    p = 5;                  // spline degree is irrelevant for compact routes
    const double L = 2.0 * M_PI;
    const double h = L / N;
    FastDeriv f = FastDeriv::build(/*periodic=*/true, N, p, L);

    // C8 coefficients (must match bm_apply.h).
    const double alpha1 = 3.0/8, a = 25.0/16, b = 1.0/5, c = -1.0/80;
    const double alpha2 = 9.0/38, o1 = 147.0/152, o2 = 51.0/380, o3 = -23.0/6840;

    double max_err_d1 = 0.0, max_err_d2 = 0.0;
    const std::complex<double> I(0.0, 1.0);

    // Sweep the resolvable wavenumbers m = 1 .. N/2 (w = 2 pi m / N).
    for (int m = 1; m <= N/2; ++m) {
        const double w = 2.0 * M_PI * m / N;   // phase per point
        // Build real/imag parts of the mode phi_j = exp(i * m * x_j) = exp(i * j * w).
        std::vector<double> re(N), im(N);
        for (int j = 0; j < N; ++j) { re[j] = std::cos(j*w); im[j] = std::sin(j*w); }

        // Apply D1 to real and imaginary parts (operator is real-linear).
        std::vector<double> d1re, d1im, d2re, d2im;
        f.compact8_d1(re, d1re); f.compact8_d1(im, d1im);
        f.compact8_d2(re, d2re); f.compact8_d2(im, d2im);

        // Recover the complex eigenvalue at node 0: (D phi)_0 / phi_0, phi_0 = 1.
        std::complex<double> lam1(d1re[0], d1im[0]);   // should be i * k_eff
        std::complex<double> lam2(d2re[0], d2im[0]);   // should be -k_eff^2 (approx)

        // Closed-form D1 symbol.
        const double num1 = a*std::sin(w) + (b/2)*std::sin(2*w) + (c/3)*std::sin(3*w);
        const double den1 = 1.0 + 2*alpha1*std::cos(w);
        std::complex<double> lam1_cf = I * (num1/den1) / h;

        // Closed-form D2 symbol (eigenvalue of the applied second derivative).
        const double num2 = 2*o1*(std::cos(w)-1) + 2*o2*(std::cos(2*w)-1) + 2*o3*(std::cos(3*w)-1);
        const double den2 = 1.0 + 2*alpha2*std::cos(w);
        std::complex<double> lam2_cf = (num2/den2) / (h*h);

        double e1 = std::abs(lam1 - lam1_cf);
        double e2 = std::abs(lam2 - lam2_cf);
        if (e1 > max_err_d1) max_err_d1 = e1;
        if (e2 > max_err_d2) max_err_d2 = e2;
    }

    // Order check: the Pade symbols must match the TRUE derivative symbols
    //   D1: k_eff*h -> w      (exact = w)
    //   D2: -kk*h^2 -> -w^2   (exact = -w^2)
    // to 8th order. Halve w twice and confirm the error ratio ~ 2^8 = 256.
    auto d1_symbol_err = [&](double w){
        double num = a*std::sin(w)+(b/2)*std::sin(2*w)+(c/3)*std::sin(3*w);
        return std::fabs(num/(1+2*alpha1*std::cos(w)) - w);        // (k_eff h) - w
    };
    auto d2_symbol_err = [&](double w){
        double num = 2*o1*(std::cos(w)-1)+2*o2*(std::cos(2*w)-1)+2*o3*(std::cos(3*w)-1);
        return std::fabs(num/(1+2*alpha2*std::cos(w)) - (-w*w));   // (-kk h^2) - (-w^2)
    };
    // Use w in the truncation-dominated band (well above the ~1e-15 roundoff floor
    // that a second-derivative symbol hits by w~0.1). w0=0.8, w1=0.4 => ratio ~2^8.
    const double w0 = 0.8, w1 = 0.4;
    const double r1 = d1_symbol_err(w0)/d1_symbol_err(w1);
    const double r2 = d2_symbol_err(w0)/d2_symbol_err(w1);

    std::printf("C8 symbol check (N=%d):\n", N);
    std::printf("  max|D1_applied - D1_symbol| = %.3e\n", max_err_d1);
    std::printf("  max|D2_applied - D2_symbol| = %.3e\n", max_err_d2);
    // D1 symbol error ~ w^9/17640 (>=8th order), D2 ~ w^10 (8th order): the two-point
    // ratio is >= 2^8 = 256 (higher here because the leading term is w^9 / w^10).
    std::printf("  D1 error ratio w=0.8->0.4   = %.1f (>=8th order => >=256)\n", r1);
    std::printf("  D2 error ratio w=0.8->0.4   = %.1f (>=8th order => >=256)\n", r2);

    // Symbol-match gate: the applied stencil must reproduce its closed form to
    // near machine precision (cyclic-solve roundoff sets the floor ~1e-12).
    const bool pass = (max_err_d1 < 1e-12) && (max_err_d2 < 5e-12)
                    && (r1 > 256*0.95) && (r2 > 256*0.95);
    std::printf("  GATE 1: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
