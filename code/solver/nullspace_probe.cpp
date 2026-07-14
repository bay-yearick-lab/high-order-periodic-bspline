// nullspace_probe.cpp
//
// Task-5 adversarial probe of the collocated pressure operator D_iD_i on the
// periodic B-spline (BMC) grid, and its competitor C6 (compact Pade, SLC 1992).
//
// The composed pressure operator is a Kronecker SUM of the per-axis composed
// symbol (D1 D1); the route's centered first derivative D1 annihilates BOTH the
// constant (omega=0) and the grid Nyquist (omega=pi) modes, so at even N the
// kernel of D_iD_i in 3D is the 2^3 = 8-dimensional span of all per-direction
// products of {constant, Nyquist}: the 7 nontrivial checkerboard modes plus the
// constant. This driver:
//
//   (A) prints the direct-D2 Nyquist eigenvalue lambda_route(pi) used by the
//       viscous term (the predicted damping symbol), for BMC p5 and C6, and
//       scans the composed-operator per-axis spectrum to confirm EXACTLY two
//       machine-zero eigenvalues per axis (k=0 and k=N/2);
//   (B) statically demonstrates the operator's blindness: the pressure solve
//       returns machine zero on a checkerboard right-hand side, grad of a
//       checkerboard pressure is machine zero, and the divergence of a
//       checkerboard velocity is machine zero (the divergence diagnostic that
//       the solver reports cannot see these modes);
//   (C) injects a controlled-amplitude (1e-3) velocity checkerboard into u and
//       marches the FULL nonlinear semidiscrete Navier-Stokes (skew-symmetric
//       convection + projection + viscous D2 + SSP-RK3), recording the Fourier
//       amplitude of all 7 nontrivial kernel modes in every velocity component
//       at each output step, together with the kinetic energy and the maximum
//       divergence. Two base states are used: a quiescent base (base flow = 0,
//       the clean invariant-subspace test) and the live TGV Re=1600 turbulent
//       base (the adversarial nonlinear-coupling test), for routes BMC and C6.
//
// Nothing in the solver headers is modified; this is a standalone driver added
// to the Makefile PROGRAMS list.

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include "ns.h"

static const double PI = 3.14159265358979323846;

// The 7 nontrivial per-direction Nyquist (checkerboard) patterns, plus labels.
static const std::array<std::array<int,3>,7> MODES = {{
    {{1,0,0}}, {{0,1,0}}, {{0,0,1}},
    {{1,1,0}}, {{1,0,1}}, {{0,1,1}},
    {{1,1,1}}
}};
static const char* MODE_ID[7] = {"x","y","z","xy","xz","yz","xyz"};

static inline double sgn(int p, int i) { return (p && (i & 1)) ? -1.0 : 1.0; }

// SIGNED modal coefficient c = (1/Ntot) sum f * pattern, so that f's projection
// onto the checkerboard pattern is c*pattern. This sum is LINEAR in f, so the
// perturbation-only coefficient is (injected run) - (baseline run) mode by mode;
// the mode amplitude reported in the paper is |c|.
static double mode_amp(const Field3D& f, int N, int px, int py, int pz) {
    double s = 0.0;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
        s += f.a[f.idx(i,j,k)] * sgn(px,i) * sgn(py,j) * sgn(pz,k);
    return s / ((double)N*N*N);
}

// The route's direct-D2 Nyquist eigenvalue: (D2 applied to (-1)^j)[0].
static double lambda_d2_nyquist(int N, double L, int p, Route route) {
    FastDeriv f = FastDeriv::build(true, N, p, L);
    std::vector<double> nyq(N), out(N), a(N);
    for (int j = 0; j < N; ++j) nyq[j] = (j & 1) ? -1.0 : 1.0;
    if      (route == Route::C6) f.compact_d2(nyq, out);
    else                         f.bmc_d2(nyq, out, a);
    return out[0];
}

// Composed-operator per-axis eigenvalue (D1 D1 cos_k)[0] = -mu_k^2.
static std::vector<double> divgrad_spectrum(int N, double L, int p, Route route) {
    FastDeriv f = FastDeriv::build(true, N, p, L);
    std::vector<double> eig(N), cosk(N), t1(N), out(N), a(N);
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) cosk[j] = std::cos(2.0*PI*k*j/N);
        if (route == Route::C6) { f.compact_d1(cosk, t1); f.compact_d1(t1, out); }
        else                    { f.bmc_d1(cosk, t1, a);  f.bmc_d1(t1, out, a); }
        eig[k] = out[0];
    }
    return eig;
}

// Initialise the TGV field (projected divergence-free) or a quiescent base.
static void set_base(IncompressibleNS& ns, int N, double h, bool tgv) {
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i*h, y = j*h, z = k*h; const size_t id = ns.u.idx(i,j,k);
        if (tgv) {
            ns.u.a[id] =  std::sin(x)*std::cos(y)*std::cos(z);
            ns.v.a[id] = -std::cos(x)*std::sin(y)*std::cos(z);
            ns.w.a[id] =  0.0;
        } else {
            ns.u.a[id] = ns.v.a[id] = ns.w.a[id] = 0.0;
        }
    }
    ns.project();
}

int main(int argc, char** argv) {
    const int    N    = (argc > 1) ? std::atoi(argv[1]) : 64;
    const int    pdeg = (argc > 2) ? std::atoi(argv[2]) : 5;
    const double Re   = (argc > 3) ? std::atof(argv[3]) : 1600.0;
    const double Tend = (argc > 4) ? std::atof(argv[4]) : 5.0;
    const double cfl  = (argc > 5) ? std::atof(argv[5]) : 0.30;
    const double eps  = (argc > 6) ? std::atof(argv[6]) : 1.0e-3;
    const char*  outp = (argc > 7) ? argv[7] : "checkerboard_injection.csv";
    const double L = 2*PI, nu = 1.0/Re, h = L/N;

    std::printf("=================================================================\n");
    std::printf(" Nullspace probe  (N=%d^3, p=%d, Re=%.0f, Tend=%.1f, eps=%.1e)\n",
                N, pdeg, Re, Tend, eps);
    std::printf("=================================================================\n\n");

    // --- (A) direct-D2 Nyquist eigenvalues and composed-operator zero scan ----
    std::printf("[A] Direct second-derivative Nyquist eigenvalue lambda_route(pi)\n");
    std::printf("    (viscous damping symbol; 1/h^2 = %.6f, spectral -k_N^2 = %d)\n",
                1.0/(h*h), -(N/2)*(N/2));
    for (Route r : {Route::BMC, Route::C6}) {
        const double lam = lambda_d2_nyquist(N, L, pdeg, r);
        const double gamma1 = nu*std::fabs(lam);
        std::printf("    %-4s : lambda(pi) = %.8e   nu|lambda(pi)| = %.8e  (gamma_1)\n",
                    route_name(r), lam, gamma1);
    }
    std::printf("\n[A] Composed operator D1D1 per-axis spectrum: count of machine-zero modes\n");
    std::printf("    (RELATIVE cut 1e-10*max|eig|, matching SeparablePoisson::build; the\n");
    std::printf("     old ABSOLUTE 1e-12 census was blind above N~128 / p~7 where the\n");
    std::printf("     numerically-assembled kernel round-off outgrows a fixed threshold)\n");
    for (Route r : {Route::BMC, Route::C6}) {
        std::vector<double> eig = divgrad_spectrum(N, L, pdeg, r);
        double mx = 0.0; for (double v : eig) mx = std::fmax(mx, std::fabs(v));
        int nz = 0; int zk[8]; int nzk = 0;
        for (int k = 0; k < N; ++k) if (std::fabs(eig[k]) <= 1e-10 * mx) { ++nz; if (nzk<8) zk[nzk++]=k; }
        std::printf("    %-4s : %d kernel eigenvalues (rel cut %.1e)  at k = ", route_name(r), nz, 1e-10 * mx);
        for (int q = 0; q < nzk; ++q) std::printf("%d ", zk[q]);
        std::printf(" | eig[0]=%.3e eig[N/2]=%.3e eig[1]=%.3e\n",
                    eig[0], eig[N/2], eig[1]);
        if (nz != 2 || zk[0] != 0 || zk[1] != N/2) {
            std::printf("    ** KERNEL CENSUS FAIL: expected exactly {0, N/2} **\n");
            return 1;
        }
    }

    // --- (B) static blindness demonstration -----------------------------------
    std::printf("\n[B] Static blindness of the pressure operator (BMC p%d, N=%d)\n", pdeg, N);
    {
        SimConfig c; c.p = pdeg; c.nu = nu; c.route = Route::BMC;
        for (int a = 0; a < 3; ++a) c.ax[a] = {N, L, BC::Periodic, true};
        const char* why = nullptr;
        SeparablePoisson pois; pois.build(c, why, /*divgrad=*/true);
        Deriv3D D(c);
        Field3D cb(N,N,N), out(N,N,N), tmp(N,N,N), zero(N,N,N);
        // checkerboard scalar (x-Nyquist)
        for (int k=0;k<N;++k) for (int j=0;j<N;++j) for (int i=0;i<N;++i)
            cb.a[cb.idx(i,j,k)] = sgn(1,i);
        // (i) pressure solve on a checkerboard RHS -> zero pressure correction
        pois.solve(cb, out, 0.0, 1.0);
        double mx_solve = 0.0; for (double v : out.a) mx_solve = std::fmax(mx_solve, std::fabs(v));
        // (ii) grad of a checkerboard pressure -> machine zero
        D.deriv(cb, out, 0, 1);
        double mx_gx = 0.0; for (double v : out.a) mx_gx = std::fmax(mx_gx, std::fabs(v));
        // (iii) divergence of a checkerboard velocity (u=cb, v=w=0) -> machine zero
        D.divergence(cb, zero, zero, out, tmp);
        double mx_div = 0.0; for (double v : out.a) mx_div = std::fmax(mx_div, std::fabs(v));
        // sanity: a resolved cosine RHS is inverted (non-kernel) -> nonzero
        Field3D cs(N,N,N);
        for (int k=0;k<N;++k) for (int j=0;j<N;++j) for (int i=0;i<N;++i)
            cs.a[cs.idx(i,j,k)] = std::cos(2.0*PI*i/N);
        pois.solve(cs, out, 0.0, 1.0);
        double mx_cos = 0.0; for (double v : out.a) mx_cos = std::fmax(mx_cos, std::fabs(v));
        std::printf("    max| solve(D1D1)^{-1} [checkerboard] | = %.3e   (kernel -> zero)\n", mx_solve);
        std::printf("    max| grad(checkerboard pressure)     | = %.3e   (grad annihilates)\n", mx_gx);
        std::printf("    max| div(checkerboard velocity)      | = %.3e   (divergence blind)\n", mx_div);
        std::printf("    max| solve(D1D1)^{-1} [cos x]        | = %.3e   (non-kernel -> nonzero)\n", mx_cos);
    }

    // --- (C) injection time-march --------------------------------------------
    FILE* f = std::fopen(outp, "w");
    // amp_u/amp_v/amp_w are the SIGNED modal coefficients (magnitude = amplitude).
    std::fprintf(f, "scenario,t,route,mode_id,amp_u,amp_v,amp_w,KE,divmax\n");

    // Each case injects a checkerboard pattern (ix,iy,iz) into u with amplitude eps.
    struct Case { const char* scen; bool tgv; Route route; double eps; int ix,iy,iz; };
    std::vector<Case> cases = {
        {"quiescent_uinject",    false, Route::BMC, eps, 1,0,0},  // clean invariant-subspace test
        {"quiescent_uinject",    false, Route::C6 , eps, 1,0,0},
        {"quiescent_uinject_xy", false, Route::BMC, eps, 1,1,0},  // 2-direction mode -> 2*gamma_1
        {"quiescent_uinject_xy", false, Route::C6 , eps, 1,1,0},
        {"quiescent_uinject_xyz",false, Route::BMC, eps, 1,1,1},  // 3-direction mode -> 3*gamma_1
        {"quiescent_uinject_xyz",false, Route::C6 , eps, 1,1,1},
        {"tgv_uinject",          true,  Route::BMC, eps, 1,0,0},  // adversarial: injected on live TGV
        {"tgv_uinject",          true,  Route::C6 , eps, 1,0,0},
        {"tgv_baseline",         true,  Route::BMC, 0.0, 1,0,0},  // control: natural cascade, no inject
        {"tgv_baseline",         true,  Route::C6 , 0.0, 1,0,0},
    };

    for (const Case& cs : cases) {
        SimConfig c; c.p = pdeg; c.nu = nu; c.route = cs.route;
        for (int a = 0; a < 3; ++a) c.ax[a] = {N, L, BC::Periodic, true};
        IncompressibleNS ns(c);
        if (!ns.ready) { std::printf("setup failed: %s\n", ns.setup_error); return 1; }
        set_base(ns, N, h, cs.tgv);
        // inject the checkerboard into u (survives projection: div=0)
        for (int k=0;k<N;++k) for (int j=0;j<N;++j) for (int i=0;i<N;++i)
            ns.u.a[ns.u.idx(i,j,k)] += cs.eps * sgn(cs.ix,i)*sgn(cs.iy,j)*sgn(cs.iz,k);

        const double dt = cfl*h/1.0;
        const int nst = std::max(1, (int)std::lround(Tend/dt));
        const int rec = std::max(1, nst/100);           // ~100 samples

        std::printf("\n[C] %-18s route=%-4s  dt=%.4f steps=%d -> t=%.1f\n",
                    cs.scen, route_name(cs.route), dt, nst, Tend);

        for (int n = 0; n <= nst; ++n) {
            if (n % rec == 0 || n == nst) {
                const double t = n*dt;
                const double KE = ns.kinetic_energy();
                const double dv = ns.max_divergence();
                for (int m = 0; m < 7; ++m) {
                    const int px=MODES[m][0], py=MODES[m][1], pz=MODES[m][2];
                    const double au = mode_amp(ns.u, N, px,py,pz);   // signed coefficient
                    const double av = mode_amp(ns.v, N, px,py,pz);
                    const double aw = mode_amp(ns.w, N, px,py,pz);
                    std::fprintf(f, "%s,%.5f,%s,%s,%.10e,%.10e,%.10e,%.10e,%.3e\n",
                                 cs.scen, t, route_name(cs.route), MODE_ID[m],
                                 au, av, aw, KE, dv);
                }
                if (n % (rec*10) == 0 || n == nst) {
                    const double ax_u = std::fabs(mode_amp(ns.u, N, cs.ix,cs.iy,cs.iz));
                    std::printf("    t=%5.2f  amp_inj(u)=%.6e  KE=%.6e  div=%.1e\n",
                                t, ax_u, KE, dv);
                }
            }
            if (n < nst) ns.step(dt);
        }
        std::fflush(f);
    }
    std::fclose(f);
    std::printf("\n wrote %s\n", outp);
    return 0;
}
