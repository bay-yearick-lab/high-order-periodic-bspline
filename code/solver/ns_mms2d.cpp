// ns_mms2d.cpp
//
// ORDER VERIFICATION of the assembled incompressible Navier-Stokes solver on an
// EXACT periodic solution. Symbol-level checks and DNS rankings only mean
// something if the fully assembled solver -- advection, viscous Laplacian,
// pressure div.grad projection, SSP-RK3 time march -- actually converges at the
// operator's design order. This driver measures that directly.
//
// The 2D Taylor-Green vortex is an exact decaying solution of the incompressible
// Navier-Stokes equations on [0,2pi]^2, at integer wavenumber km:
//
//   u = -cos(km x) sin(km y) exp(-2 km^2 nu t)
//   v =  sin(km x) cos(km y) exp(-2 km^2 nu t)
//   p = -(cos 2km x + cos 2km y) exp(-4 km^2 nu t) / 4
//
// (km vortex pairs per period; km=2 keeps the finest-grid errors of the
// highest-order schemes above the round-off plateau, equivalent to running
// km=1 on a grid sequence half as fine), run here as a z-invariant field on
// the 3D torus (w = 0, no z dependence), so the
// same 3D solver (ns.h) is exercised unchanged. The field is initialized at t=0,
// marched to t=1, and compared to the exact field at t=1 in the velocity L2 and
// Linf norms. C++ computes the error; the collector only reads the printed number.
//
// TIME-ERROR CONTROL: SSP-RK3 is third order in time, so to expose spatial order
// up to ten the time step is scaled dt = C h^4 (then dt^3 ~ h^12 is far below the
// spatial error at every grid). The dt-halving check confirms the reported error
// is spatially, not temporally, limited.
//
// FORCED MODE (mode == "forced"): the unforced Taylor-Green error is purely
// viscous because the discrete convective term of that field is an exact
// discrete gradient which the projection annihilates, so no Reynolds number
// can expose the first-derivative ladder on it. The forced run instead
// manufactures a two-mode streamfunction whose convective term is NOT a
// gradient (mms_forcing.h, sympy-generated and sympy-verified) and adds the
// analytic momentum source per RK stage at the correct stage times
// (t, t+dt, t+dt/2). With nu small the convective/projection error dominates
// and the first-derivative order 2*ceil(p/2) governs.
//
// CLI:  ns_mms2d N route Re Tend pdeg dt_coeff km mode
//   N        grid (power of two; the FFT Poisson requires it)
//   route    bmc | bmg | c6 | c8 | fd6 | fd4 | fd2 | drp
//   Re       Reynolds number (nu = 1/Re); default 100 -> nu = 0.01
//   Tend     final time; default 1.0
//   pdeg     spline degree (5,7,9 for BMC5/7/9); ignored by compact routes
//   dt_coeff C in dt = C h^4; default 0.02. Pass <=0 to use dt = |dt_coeff_abs|
//            as a fixed absolute step (used by the dt-halving check).
//   km       integer wavenumber of the Taylor-Green field; default 1.
//   mode     "tg" (default) | "forced" | "check" (evaluate mms_forcing.h at
//            "x y t nu" quadruples read from stdin and print u v su sv; used
//            by the generator-vs-C validation).

#include <cstdio>
#include <cmath>
#include <string>
#include "ns.h"
#include "mms_forcing.h"

static const double PI = 3.14159265358979323846;

static Route parse_route(const std::string& s) {
    if (s == "exp") return Route::EXP;
    if (s == "bmg") return Route::BMG;  if (s == "c6")  return Route::C6;
    if (s == "c8")  return Route::C8;
    if (s == "c10") return Route::C10;
    if (s == "c12") return Route::C12;
    if (s == "sl")  return Route::SL;
    if (s == "opt") return Route::OPT;
    if (s == "cw12a") return Route::CW12A;
    if (s == "cw12b") return Route::CW12B;
    if (s == "drp") return Route::DRP;  if (s == "fd6") return Route::FD6;
    if (s == "fd4") return Route::FD4;  if (s == "fd2") return Route::FD2;
    return Route::BMC;
}

// Pull "--exp-member <path>" out of the argument vector before the positional
// arguments are read (EXP_MEMBER_JSON is the equivalent environment fallback).
static int extract_exp_member(int argc, char** argv) {
    int out = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) != "--exp-member") { argv[out++] = argv[i]; continue; }
        if (i + 1 >= argc) {
            std::printf("--exp-member needs a path argument\n");
            return out;
        }
        exp_member_set_path(argv[i + 1]);
        ++i;
    }
    return out;
}

int main(int argc, char** argv) {
    argc = extract_exp_member(argc, argv);
    const int    N     = (argc > 1) ? std::atoi(argv[1]) : 32;
    std::string  rstr  = (argc > 2) ? argv[2] : "bmc";
    const double Re    = (argc > 3) ? std::atof(argv[3]) : 100.0;
    const double Tend  = (argc > 4) ? std::atof(argv[4]) : 1.0;
    const int    pdeg  = (argc > 5) ? std::atoi(argv[5]) : 5;
    const double dtcf  = (argc > 6) ? std::atof(argv[6]) : 0.02;
    const int    km    = (argc > 7) ? std::atoi(argv[7]) : 1;
    std::string  mode  = (argc > 8) ? argv[8] : "tg";
    const Route  route = parse_route(rstr);
    const double L = 2 * PI, nu = 1.0 / Re;

    if (mode == "check") {                       // generator-vs-C validation
        double xx, yy, tt, nn;
        while (std::scanf("%lf %lf %lf %lf", &xx, &yy, &tt, &nn) == 4) {
            double ue, ve, su, sv;
            mms_forced_eval(xx, yy, tt, nn, &ue, &ve, &su, &sv);
            std::printf("%.17g %.17g %.17g %.17g\n", ue, ve, su, sv);
        }
        return 0;
    }
    const bool forced = (mode == "forced");

    SimConfig c; c.p = pdeg; c.nu = nu; c.route = route;
    for (int a = 0; a < 3; ++a) c.ax[a] = {N, L, BC::Periodic, true};

    IncompressibleNS ns(c);
    if (!ns.ready) { std::printf("setup failed: %s\n", ns.setup_error); return 1; }
    const double h = L / N;

    // Exact 2D Taylor-Green field (z-invariant) written into (fu,fv); w = 0.
    auto exact = [&](double t, Field3D& fu, Field3D& fv) {
        const double e2 = std::exp(-2.0 * km * km * nu * t);
        for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            const double x = km * i * h, y = km * j * h; const size_t id = fu.idx(i, j, k);
            fu.a[id] = -std::cos(x) * std::sin(y) * e2;
            fv.a[id] =  std::sin(x) * std::cos(y) * e2;
        }
    };

    // Exact forced-MMS field (z-invariant) written into (fu,fv); w = 0.
    auto exact_forced = [&](double t, Field3D& fu, Field3D& fv) {
        for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            double ue, ve, su, sv;
            mms_forced_eval(i * h, j * h, t, nu, &ue, &ve, &su, &sv);
            const size_t id = fu.idx(i, j, k);
            fu.a[id] = ue; fv.a[id] = ve;
        }
    };

    // Add the analytic momentum source S(x,y,ts) to (ru,rv), broadcast in z.
    auto add_source = [&](double ts) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            double ue, ve, su, sv;
            mms_forced_eval(i * h, j * h, ts, nu, &ue, &ve, &su, &sv);
            for (int k = 0; k < N; ++k) {
                const size_t id = ns.ru.idx(i, j, k);
                ns.ru.a[id] += su; ns.rv.a[id] += sv;      // Sw = 0
            }
        }
    };

    // One forced SSP-RK3 step from time tn: same convex combinations as
    // ns.step(dt), with the time-dependent source evaluated at the standard
    // Shu-Osher stage times tn, tn+dt, tn+dt/2.
    auto step_forced = [&](double dt_, double tn) {
        ns.u0.a = ns.u.a; ns.v0.a = ns.v.a; ns.w0.a = ns.w.a;
        ns.rhs(); add_source(tn);
        axpy(ns.u, dt_, ns.ru); axpy(ns.v, dt_, ns.rv); axpy(ns.w, dt_, ns.rw);
        ns.project();
        ns.rhs(); add_source(tn + dt_);
        rk_combine(ns.u, ns.u0, 0.75, 0.25, ns.ru, 0.25 * dt_);
        rk_combine(ns.v, ns.v0, 0.75, 0.25, ns.rv, 0.25 * dt_);
        rk_combine(ns.w, ns.w0, 0.75, 0.25, ns.rw, 0.25 * dt_);
        ns.project();
        ns.rhs(); add_source(tn + 0.5 * dt_);
        rk_combine(ns.u, ns.u0, 1.0 / 3.0, 2.0 / 3.0, ns.ru, (2.0 / 3.0) * dt_);
        rk_combine(ns.v, ns.v0, 1.0 / 3.0, 2.0 / 3.0, ns.rv, (2.0 / 3.0) * dt_);
        rk_combine(ns.w, ns.w0, 1.0 / 3.0, 2.0 / 3.0, ns.rw, (2.0 / 3.0) * dt_);
        ns.project();
    };

    // Initialize at t=0 (already divergence-free; projection is a no-op to round-off).
    if (forced) exact_forced(0.0, ns.u, ns.v);
    else        exact(0.0, ns.u, ns.v);
    for (size_t q = 0; q < ns.w.a.size(); ++q) ns.w.a[q] = 0.0;
    ns.project();

    // Time step: dt = C h^4 so temporal error dt^3 ~ h^12 is subdominant. A
    // negative dt_coeff means "use |dt_coeff| as a fixed absolute step" (the
    // dt-halving check), so the reported error can be shown dt-independent.
    double dt;
    if (dtcf > 0.0) dt = dtcf * h * h * h * h;
    else            dt = -dtcf;
    // Land exactly on Tend by an integer number of steps.
    int nst = (int)std::ceil(Tend / dt - 1e-9);
    if (nst < 1) nst = 1;
    dt = Tend / nst;

    if (forced) { for (int n = 0; n < nst; ++n) step_forced(dt, n * dt); }
    else        { for (int n = 0; n < nst; ++n) ns.step(dt); }   // SSP-RK3

    // Error against the exact field at t = Tend, velocity L2 and Linf (mean-square
    // L2 density, matching kinetic_energy()'s convention; grid-independent scale).
    Field3D eu(N, N, N), ev(N, N, N);
    if (forced) exact_forced(Tend, eu, ev);
    else        exact(Tend, eu, ev);
    double s2 = 0.0, linf = 0.0; const size_t M = ns.u.a.size();
    for (size_t q = 0; q < M; ++q) {
        const double du = ns.u.a[q] - eu.a[q];
        const double dv = ns.v.a[q] - ev.a[q];
        const double dw = ns.w.a[q];                 // exact w = 0
        s2 += du * du + dv * dv + dw * dw;
        linf = std::fmax(linf, std::fmax(std::fabs(du), std::fmax(std::fabs(dv), std::fabs(dw))));
    }
    const double l2 = std::sqrt(s2 / (double)M);

    // Machine-readable line for the collector; JSON-friendly fields.
    std::printf("MMS mode=%s route=%s p=%d N=%d km=%d nu=%.6e Tend=%.4f dt=%.6e "
                "steps=%d L2=%.12e Linf=%.12e div=%.3e\n",
                mode.c_str(), rstr.c_str(), pdeg, N, km, nu, Tend, dt, nst,
                l2, linf, ns.max_divergence());
    return 0;
}
