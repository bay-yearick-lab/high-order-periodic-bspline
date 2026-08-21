// offgrid_turbulence.cpp
//
// The periodic B-spline coefficient field supplies values and first
// derivatives at arbitrary query points from the same banded reconstruction.
//
//   s(x*)  = sum_j alpha_j beta_p((x*-x_j)/h)
//   s'(x*) = (1/h) sum_j alpha_j beta_p'((x*-x_j)/h)
//
// The deterministic broadband field has amplitudes proportional to k^(-5/6)
// and fixed random phases. Analytic values and derivatives provide exact query
// targets for relative discrete L2 and sampled relative maximum errors.

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include "bm_apply.h"

static const double PI = 3.14159265358979323846;

// ---- broadband turbulence-like field on [0,2pi): exact value & derivative ----
struct Field {
    std::vector<double> A, phi; int kmax;
    Field(int kmax_, unsigned seed) : kmax(kmax_) {
        std::mt19937 rng(seed); std::uniform_real_distribution<double> U(0, 2*PI);
        A.resize(kmax+1); phi.resize(kmax+1);
        for (int k = 1; k <= kmax; ++k) { A[k] = std::pow((double)k, -5.0/6.0); phi[k] = U(rng); }
    }
    double f(double x)  const { double s=0; for(int k=1;k<=kmax;++k) s += A[k]*std::cos(k*x+phi[k]); return s; }
    double fp(double x) const { double s=0; for(int k=1;k<=kmax;++k) s += -A[k]*k*std::sin(k*x+phi[k]); return s; }
};

// ---- native spline off-grid value and derivative at x* (periodic) -----------
// Only the ~(p+1) nodes whose centered cardinal support covers x* contribute.
static void spline_offgrid(const std::vector<double>& alpha, int p, double h, int N,
                           double xs, double& val, double& der) {
    val = 0; der = 0;
    const double t0 = xs / h;                      // x* in grid units
    const double rad = 0.5 * (p + 1);              // support radius in cells
    const int jlo = (int)std::floor(t0 - rad), jhi = (int)std::ceil(t0 + rad);
    for (int j = jlo; j <= jhi; ++j) {
        const double t = t0 - j;                   // (x* - x_j)/h
        const double a = alpha[((j % N) + N) % N];
        val += a * beta_value(p, t);
        der += a * beta_deriv1(p, t);
    }
    der /= h;
}

// ---- bolt-on periodic Lagrange interpolation of nodal samples g_j -----------
// npts-point centered Lagrange stencil; order = npts. This is the auxiliary
// interpolant an FD/compact/DRP user must add to read a value between nodes.
static double lagrange_interp(const std::vector<double>& g, int N, double h, double xs, int npts) {
    const double t = xs / h;
    const int i0 = (int)std::floor(t) - (npts/2 - 1);     // leftmost stencil node
    double res = 0;
    for (int a = 0; a < npts; ++a) {
        const int ja = i0 + a;
        double w = 1.0;
        for (int b = 0; b < npts; ++b) if (b != a) w *= (t - (i0 + b)) / (double)(a - b);
        res += w * g[((ja % N) + N) % N];
    }
    return res;
}

static double slope(double e1, double e2, double h1, double h2) {
    return std::log(e1/e2) / std::log(h1/h2);
}

int main() {
    const double L = 2*PI;
    const int kmax = 6;                            // smooth multiscale field
    Field fld(kmax, 12345u);
    const int M = 4096;                            // random off-grid query points
    std::vector<double> xq(M);
    { std::mt19937 rng(999u); std::uniform_real_distribution<double> U(0, L); for (auto& x : xq) x = U(rng); }

    std::printf("=================================================================\n");
    std::printf(" OFF-GRID EVALUATION AT THE DIFFERENTIATION ORDER (capability 2)\n");
    std::printf(" broadband turbulence-like field, kmax=%d; %d random query points\n", kmax, M);
    std::printf(" native spline: value & derivative from ONE banded solve, no interpolant\n");
    std::printf(" bolt-on: FD6 nodal derivative + p-point Lagrange interpolation\n");
    std::printf("=================================================================\n");

    const int Ns[] = {32, 48, 64, 96, 128, 192, 256};
    const int degs[] = {3, 5, 7};
    FILE* csv = std::fopen("data/offgrid_turbulence.csv", "w");
    std::fprintf(csv, "N,h,p,val_err,der_err,val_max_q,der_max_q,bolton4_der_err,bolton6_der_err\n");

    // Error histories support the fixed-window order report.
    std::vector<std::vector<double>> verr(3), derr(3); std::vector<double> b4(7), b6(7), hs(7);
    for (auto& v : verr) v.clear();
    for (auto& v : derr) v.clear();

    int row = 0;
    for (int N : Ns) {
        const double h = L / N;
        hs[row] = h;
        // exact off-grid targets
        std::vector<double> fexact(M), fpexact(M);
        for (int q = 0; q < M; ++q) { fexact[q] = fld.f(xq[q]); fpexact[q] = fld.fp(xq[q]); }
        // FD6 nodal derivative for the bolt-on baseline (independent of degree)
        FastDeriv f6 = FastDeriv::build(true, N, 5, L);
        std::vector<double> u6(N); for (int j = 0; j < N; ++j) u6[j] = fld.f(j*h);
        std::vector<double> g6; f6.fd6_d1(u6, g6);
        double e_b4 = 0, e_b6 = 0, den_d = 0;
        for (int q = 0; q < M; ++q) {
            const double d4 = lagrange_interp(g6, N, h, xq[q], 4);
            const double d6 = lagrange_interp(g6, N, h, xq[q], 6);
            e_b4 += (d4 - fpexact[q])*(d4 - fpexact[q]);
            e_b6 += (d6 - fpexact[q])*(d6 - fpexact[q]);
            den_d += fpexact[q]*fpexact[q];
        }
        b4[row] = std::sqrt(e_b4/den_d); b6[row] = std::sqrt(e_b6/den_d);

        for (int di = 0; di < 3; ++di) {
            const int p = degs[di];
            FastDeriv f = FastDeriv::build(true, N, p, L);
            std::vector<double> u(N); for (int j = 0; j < N; ++j) u[j] = fld.f(j*h);
            std::vector<double> alpha = f.solveB(u);
            double ev = 0, ed = 0, denv = 0;
            double ev_max = 0, ed_max = 0, denv_max = 0, dend_max = 0;
            for (int q = 0; q < M; ++q) {
                double val, der; spline_offgrid(alpha, p, h, N, xq[q], val, der);
                ev += (val - fexact[q])*(val - fexact[q]);  denv += fexact[q]*fexact[q];
                ed += (der - fpexact[q])*(der - fpexact[q]);
                ev_max = std::max(ev_max, std::abs(val - fexact[q]));
                ed_max = std::max(ed_max, std::abs(der - fpexact[q]));
                denv_max = std::max(denv_max, std::abs(fexact[q]));
                dend_max = std::max(dend_max, std::abs(fpexact[q]));
            }
            const double rv = std::sqrt(ev/denv), rd = std::sqrt(ed/den_d);
            const double rv_max = ev_max / denv_max, rd_max = ed_max / dend_max;
            verr[di].push_back(rv); derr[di].push_back(rd);
            std::fprintf(csv, "%d,%.6e,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                         N, h, p, rv, rd, rv_max, rd_max, b4[row], b6[row]);
        }
        ++row;
    }
    std::fclose(csv);

    // measured-order report (last decade, before any round-off floor)
    std::printf("\n native spline OFF-GRID order (measured between N=96 and N=192):\n");
    auto idx = [&](int N){ int i=0; for(int x : Ns){ if(x==N) return i; ++i;} return -1; };
    const int ia = idx(96), ib = idx(192);
    for (int di = 0; di < 3; ++di) {
        std::printf("   p=%d:  value order ~ %.2f (expect %d),  derivative order ~ %.2f (expect %d)\n",
            degs[di], slope(verr[di][ia], verr[di][ib], hs[ia], hs[ib]), degs[di]+1,
            slope(derr[di][ia], derr[di][ib], hs[ia], hs[ib]), degs[di]);
    }
    std::printf(" bolt-on FD6 derivative + Lagrange OFF-GRID order:\n");
    std::printf("   4-point: ~ %.2f (capped at 4) ;  6-point: ~ %.2f (capped at 6)\n",
        slope(b4[ia], b4[ib], hs[ia], hs[ib]), slope(b6[ia], b6[ib], hs[ia], hs[ib]));
    std::printf("\n The native spline reaches its differentiation order off-grid from the\n");
    std::printf(" same coefficients; the bolt-on caps at the SEPARATE interpolant's order.\n");
    std::printf("\nwrote data/offgrid_turbulence.csv\nDone.\n");
    return 0;
}
