// c_exp_symbol_check.cpp -- verify the GENERIC exponential-family route ("exp")
// against the trigonometric-polynomial symbols of its own member file. Applies
// the EXACT production kernels (FastDeriv::exp_d1 / exp_d2 from bm_apply.h) to
// every discrete Fourier mode exp(i theta_l x) of the grid and compares the
// recovered eigenvalue to the closed form built from the JSON samples.
//
// With one-sided lattice samples beta(n), beta'(n), beta''(n), n = 0..R, and
// theta_l = 2 pi l / N,
//
//   b(theta)  = beta(0)   + 2 sum_{n=1}^{R} beta(n)   cos(n theta),
//   a(theta)  =             2 sum_{n=1}^{R} beta'(n)  sin(n theta),
//   c(theta)  = beta''(0) + 2 sum_{n=1}^{R} beta''(n) cos(n theta),
//
// the left side has eigenvalue b(theta) on that mode and the right-side
// stencils, whose coefficient at offset o is beta^{(q)}(-o), have eigenvalues
// -i a(theta) / h and c(theta) / h^2. Hence
//
//   D1 eigenvalue = -(i/h) a(theta) / b(theta),
//   D2 eigenvalue =  (1/h^2) c(theta) / b(theta),
//
// the same sign and normalization convention the BMC branch of poisson.h uses:
// the eigenvalue is the APPLIED operator's, so D2 is nonpositive (the identity
// sum_n beta''(n) = 0 rewrites c as -4 sum_{n>0} beta''(n) sin^2(n theta / 2)).
//
// The deviation is measured on the DIMENSIONLESS symbols h * lambda_1 and
// h^2 * lambda_2, the modified wavenumbers themselves. Comparing the dimensional
// eigenvalues instead would multiply the same floating point noise by 1/h^2 and
// report a grid-dependent number that says nothing about the operator.
//
// The grid is swept at N = 128 (even, so theta = pi is a grid mode) and N = 127
// (odd, so no mode lands on pi and the cyclic wrap correction is exercised on an
// n that is not a power of two). PASS if the deviation is at most 1e-12.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
#include "bm_apply.h"

static const double TOL = 1e-12;

struct SymbolResult { double e1 = 0.0, e2 = 0.0, d2max = -1e300, d1skew = 0.0; };

static SymbolResult sweep(int N, const ExpMember& em) {
    const double L = 2.0 * M_PI, h = L / N;
    const int R = em.radius;
    FastDeriv f = FastDeriv::build(true, N, /*p=*/em.p, L);
    const std::complex<double> I(0.0, 1.0);
    SymbolResult out;

    std::vector<double> re(N), im(N), d1re(N), d1im(N), d2re(N), d2im(N), sc(N);
    for (int l = 0; l < N; ++l) {
        const double th = 2.0 * M_PI * l / N;
        double b = em.beta[0], a = 0.0, c = em.ddbeta[0];
        for (int nn = 1; nn <= R; ++nn) {
            b += 2.0 * em.beta[nn] * std::cos(nn * th);
            a += 2.0 * em.dbeta[nn] * std::sin(nn * th);
            c += 2.0 * em.ddbeta[nn] * std::cos(nn * th);
        }
        for (int j = 0; j < N; ++j) { re[j] = std::cos(j * th); im[j] = std::sin(j * th); }
        f.exp_d1(re, d1re, sc); f.exp_d1(im, d1im, sc);
        f.exp_d2(re, d2re, sc); f.exp_d2(im, d2im, sc);
        // Dimensionless measured symbols: h * lambda_1 and h^2 * lambda_2.
        const std::complex<double> s1(d1re[0] * h, d1im[0] * h);
        const std::complex<double> s2(d2re[0] * h * h, d2im[0] * h * h);
        const std::complex<double> s1_cf = -I * (a / b);
        const std::complex<double> s2_cf = (c / b);
        out.e1 = std::max(out.e1, std::abs(s1 - s1_cf));
        out.e2 = std::max(out.e2, std::abs(s2 - s2_cf));
        out.d2max = std::max(out.d2max, c / b);
        // D1 is antisymmetric, so its eigenvalue is purely imaginary.
        out.d1skew = std::max(out.d1skew, std::fabs(s1.real()));
    }
    return out;
}

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--exp-member") exp_member_set_path(argv[i + 1]);

    const char* why = "";
    if (!exp_member_ready(why)) {
        std::printf("EXP: cannot load member (%s)\n", why);
        return EXIT_FAILURE;
    }
    const ExpMember& em = exp_member_store();
    std::printf("EXP: member %s  m=%d r=%d p=%d radius=%d cond=%.6f\n",
                em.name.c_str(), em.m, em.r, em.p, em.radius, em.cond);

    const int grids[2] = {128, 127};
    bool ok = true;
    for (int g = 0; g < 2; ++g) {
        const SymbolResult s = sweep(grids[g], em);
        const bool pass = s.e1 <= TOL && s.e2 <= TOL;
        // The constant mode has D2 symbol exactly zero (sum_n beta''(n) = 0), so
        // nonpositivity is asserted up to roundoff, not as a strict inequality.
        ok = ok && pass && s.d2max <= TOL;
        std::printf("EXP: N=%3d  max|h^q applied - symbol| D1 %.3e  D2 %.3e ; "
                    "max Re(h D1 symbol) %.3e ; max h^2 D2 symbol %.3e  -> %s\n",
                    grids[g], s.e1, s.e2, s.d1skew, s.d2max, pass ? "ok" : "FAIL");
    }
    std::printf("EXP: %s (tolerance %.0e)\n", ok ? "PASS" : "FAIL", TOL);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
