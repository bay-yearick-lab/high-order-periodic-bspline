// tgv_re1600.cpp
//
// FRONTIER DNS #1 -- the Taylor-Green vortex at Re=1600, the international
// High-Order CFD Workshop benchmark. The flow starts from a single smooth mode
//
//   u =  sin x cos y cos z,  v = -cos x sin y cos z,  w = 0   on [0,2pi]^3,
//
// and cascades to small scales; the kinetic-energy dissipation rate eps(t) peaks
// near t~9 once the smallest resolved scales fill in. The peak is acutely
// resolution sensitive, so a fixed coarse grid ranks schemes by how much of the
// cascade they resolve -- exactly the northstar question, on standard hard
// physics, against the best available local schemes.
//
// The SAME incompressible solver (ns.h) is run with the derivative operator
// swapped via SimConfig.route: BMC/BMG (our spline pair), compact Pade C6
// (SLC 1992), DRP (Tam & Webb 1993), and explicit centered FD2/FD4/FD6. Every
// operator -- advection, viscous Laplacian, pressure div.grad projection -- is
// built from the run's own route symbol (poisson.h), so the comparison is fair.
//
// Two dissipation measures are recorded per step:
//   eps_ens  = nu <|omega|^2>          (enstrophy based, via the curl D1; instantaneous)
//   eps_dEdt = -dE/dt                  (energy based; finite-differenced)
// The skew-symmetric convection produces zero net kinetic energy and the projection
// does no work, so -dE/dt equals the physical viscous dissipation nu <u,-D2 u> through
// the direct second derivative to time-integration error (verified: the two match to
// <1%). There is NO spurious numerical dissipation. eps_ens and eps_dEdt are two
// DIFFERENT discrete measures of that dissipation: the curl-squared symbol mu^2 vs the
// direct-D2 symbol lambda. They coincide only where those symbols agree (spectral, or
// full resolution); on a coarse grid lambda > mu^2 at high k, so eps_dEdt exceeds
// eps_ens -- an operator-consistency gap, not an energy leak.

#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>
#include "ns.h"

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;

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

// Route-agnostic initial velocity field; empty keeps the analytic build.
static std::string g_ic_path;
// Land the final step exactly on Tend instead of on a whole multiple of dt.
static bool g_exact_time = false;

// Pull the options "--exp-member <path>", "--load-ic <path>" and "--exact-time"
// out of the argument vector before the positional arguments are read, so route
// "exp" can be pointed at any member JSON, every route can be handed the same
// t=0 field, and the horizon can be landed exactly, all without disturbing the
// existing positional interface. The environment variable EXP_MEMBER_JSON is
// the equivalent fallback for the member file.
static int extract_options(int argc, char** argv) {
    int out = 0;
    for (int i = 0; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--exact-time") { g_exact_time = true; continue; }
        if (arg != "--exp-member" && arg != "--load-ic") { argv[out++] = argv[i]; continue; }
        if (i + 1 >= argc) {
            std::printf("%s needs a path argument\n", arg.c_str());
            return out;
        }
        if (arg == "--exp-member") exp_member_set_path(argv[i + 1]);
        else                       g_ic_path = argv[i + 1];
        ++i;
    }
    return out;
}

int main(int argc, char** argv) {
    argc = extract_options(argc, argv);
    const int    N    = (argc > 1) ? std::atoi(argv[1]) : 64;
    std::string  rstr = (argc > 2) ? argv[2] : "bmc";
    const double Re   = (argc > 3) ? std::atof(argv[3]) : 1600.0;
    const double Tend = (argc > 4) ? std::atof(argv[4]) : 20.0;
    const int    pdeg = (argc > 5) ? std::atoi(argv[5]) : 5;
    const double cfl  = (argc > 6) ? std::atof(argv[6]) : 0.30;
    const Route  route = parse_route(rstr);
    // Optional nonlinear de-aliasing filter. argv[8] = none|twothirds|smooth.
    // Filtered runs write with a distinct suffix so they never clobber default data.
    const std::string dstr = (argc > 8) ? std::string(argv[8]) : std::string("none");
    const int dmode = (dstr == "twothirds") ? 1 : (dstr == "smooth") ? 2 : 0;
    // Optional output tag (argv[9]) appended to filenames for the time-
    // refinement reruns (different cfl), so they never clobber the base matrix.
    std::string tag = (argc > 9) ? std::string(argv[9]) : std::string();
    if (tag == "-") tag.clear();
    // Common-initialization runs carry a "_cic" suffix, ahead of the de-aliasing
    // suffix, so they never clobber the route-IC data files.
    const bool load_ic = !g_ic_path.empty() && g_ic_path != "-";
    const std::string dsfx = std::string(load_ic ? "_cic" : "")
                           + std::string((dmode == 1) ? "_da23" : (dmode == 2) ? "_daexp" : "") + tag;
    const double L = 2 * PI, nu = 1.0 / Re;

    SimConfig c; c.p = pdeg; c.nu = nu; c.route = route; c.dealias = dmode;
    for (int a = 0; a < 3; ++a) c.ax[a] = {N, L, BC::Periodic, true};

    std::printf("=================================================================\n");
    std::printf(" Taylor-Green vortex DNS  (N=%d^3, Re=%.0f, route=%s p=%d, cfl=%.2f)\n",
                N, Re, route_name(route), c.p, cfl);
    std::printf("=================================================================\n");

    IncompressibleNS ns(c);
    if (!ns.ready) { std::printf("setup failed: %s\n", ns.setup_error); return 1; }
    const double h = L / N;

    // Initial condition (projected divergence-free).
    if (load_ic) {
        // Common divergence-free initial field, identical bytes for every route.
        FILE* bfile = std::fopen(g_ic_path.c_str(), "rb");
        if (!bfile) { std::printf("cannot open IC file %s\n", g_ic_path.c_str()); return 1; }
        const size_t nn = ns.u.a.size();
        size_t rd = std::fread(ns.u.a.data(), sizeof(double), nn, bfile);
        rd += std::fread(ns.v.a.data(), sizeof(double), nn, bfile);
        rd += std::fread(ns.w.a.data(), sizeof(double), nn, bfile);
        std::fclose(bfile);
        if (rd != 3*nn) { std::printf("IC file size mismatch (%zu != %zu)\n", rd, 3*nn); return 1; }
        // No standalone ns.project() here on purpose: the loaded field is
        // spectrally divergence-free, so each route's discrete divergence of it
        // is only O(truncation) and is removed by the projection inside the
        // first SSP-RK3 step. Skipping the initial projection keeps the t=0
        // field bit-identical for every route.
        std::printf(" [loaded common IC from %s : t=0 field bit-identical across routes]\n",
                    g_ic_path.c_str());
    } else {
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i*h, y = j*h, z = k*h; const size_t id = ns.u.idx(i,j,k);
        ns.u.a[id] =  std::sin(x)*std::cos(y)*std::cos(z);
        ns.v.a[id] = -std::cos(x)*std::sin(y)*std::cos(z);
        ns.w.a[id] =  0.0;
    }
    ns.project();
    }

    Field3D wx(N,N,N), wy(N,N,N), wz(N,N,N), tA(N,N,N), tB(N,N,N);
    // mean enstrophy <1/2 |omega|^2> over the periodic box.
    auto enstrophy = [&]() {
        ns.D.deriv(ns.w,tA,1,1); ns.D.deriv(ns.v,tB,2,1); for(size_t q=0;q<wx.a.size();++q) wx.a[q]=tA.a[q]-tB.a[q];
        ns.D.deriv(ns.u,tA,2,1); ns.D.deriv(ns.w,tB,0,1); for(size_t q=0;q<wy.a.size();++q) wy.a[q]=tA.a[q]-tB.a[q];
        ns.D.deriv(ns.v,tA,0,1); ns.D.deriv(ns.u,tB,1,1); for(size_t q=0;q<wz.a.size();++q) wz.a[q]=tA.a[q]-tB.a[q];
        double s=0; for(size_t q=0;q<wx.a.size();++q) s+=wx.a[q]*wx.a[q]+wy.a[q]*wy.a[q]+wz.a[q]*wz.a[q];
        return 0.5*s/(double)wx.a.size();                  // <1/2|omega|^2>
    };

    const double umax0 = 1.0;
    const double dt = cfl * h / umax0;
    // Base rule: a whole number of steps of length dt. Exact-time mode takes
    // floor(Tend/dt) whole steps and one final partial step of length
    // Tend - nst dt, so the state lands on Tend to round-off and every route's
    // field dump sits at the same physical time as the reference dump.
    int nst; double dt_last = 0.0;
    if (g_exact_time) {
        nst = (int)std::floor(Tend/dt * (1.0 - 1e-12));
        dt_last = Tend - nst*dt;
        if (dt_last < 1e-10) dt_last = 0.0;                // Tend is an exact multiple
    } else {
        nst = (int)std::lround(Tend/dt);
    }
    const int ntot = nst + ((dt_last > 0.0) ? 1 : 0);
    const int rec = std::max(1, ntot/200);                 // ~200 samples
    std::printf(" nu=%.3e dt=%.4f steps=%d dt_last=%.3e -> t=%.10f\n",
                nu, dt, ntot, dt_last, Tend);

    char path[256];
    std::snprintf(path, sizeof(path), "data/tgv_%s%s_p%d_N%d_re%.0f.csv", rstr.c_str(), dsfx.c_str(), pdeg, N, Re);
    FILE* f = std::fopen(path, "w");
    std::fprintf(f, "t,E,enstrophy,eps_ens,eps_dEdt,divmax\n");

    // Dump the velocity field once near the dissipation peak so the energy
    // spectrum and vorticity slices can be computed offline (Python). The raw
    // dump is N^3 doubles each of u, v, w; it is a transient artifact (gitignored).
    const double t_dump = (argc > 7) ? std::atof(argv[7]) : 9.0;
    bool dumped = false;
    auto dump_field = [&](double t){
        char fp[256];
        std::snprintf(fp, sizeof(fp), "data/tgvfield_%s%s_p%d_N%d.bin", rstr.c_str(), dsfx.c_str(), pdeg, N);
        FILE* b = std::fopen(fp, "wb");
        std::fwrite(ns.u.a.data(), sizeof(double), ns.u.a.size(), b);
        std::fwrite(ns.v.a.data(), sizeof(double), ns.v.a.size(), b);
        std::fwrite(ns.w.a.data(), sizeof(double), ns.w.a.size(), b);
        std::fclose(b);
        std::printf("  [dumped velocity field at t=%.2f -> %s]\n", t, fp);
        if (g_exact_time)
            std::printf(" [dumped field at exactly t=%.10f -> %s]\n", t, fp);
    };

    const double E0 = ns.kinetic_energy();
    double Eprev = E0, tprev = 0.0; bool have_prev = false;
    const char* reason = "completed";
    bool unstable = false;
    double t = 0.0;
    auto t0 = Clock::now();
    for (int n=0; n<=ntot; ++n) {
        t = (n <= nst) ? n*dt : Tend;
        if (g_exact_time && n == ntot) t = Tend;
        if (!dumped && !g_exact_time && t >= t_dump) { dump_field(t); dumped = true; }
        if (n % rec == 0 || n == ntot) {
            const double E = ns.kinetic_energy();
            const double ens = enstrophy();
            const double eps_ens = 2.0*nu*ens;                       // nu <|omega|^2>
            double eps_dEdt = 0.0;
            if (have_prev && t > tprev) eps_dEdt = -(E - Eprev)/(t - tprev);
            const double dv = ns.max_divergence();
            std::fprintf(f, "%.5f,%.10e,%.10e,%.10e,%.10e,%.3e\n", t, E, ens, eps_ens, eps_dEdt, dv);
            std::fflush(f);
            if (n % (rec*10) == 0 || n == ntot)
                std::printf("  t=%5.2f  E/E0=%.5f  eps_ens=%.4e  eps_-dEdt=%.4e  div=%.1e\n",
                            t, E/E0, eps_ens, eps_dEdt, dv);
            Eprev = E; tprev = t; have_prev = true;
            if (g_exact_time) {
                // Instability screen: a nonfinite diagnostic, kinetic energy
                // above the initial value by more than two percent, or a
                // divergence spike all mark the run as unusable.
                if (!std::isfinite(E) || !std::isfinite(dv)) { reason = "nonfinite"; unstable = true; }
                else if (E/E0 > 1.02)                        { reason = "E_growth";  unstable = true; }
                else if (std::fabs(dv) > 1e-3)               { reason = "div_spike"; unstable = true; }
                if (unstable) break;
            }
        }
        // Exact-time mode dumps the landed state, once the final record has
        // confirmed that the run is usable.
        if (g_exact_time && n == ntot && !unstable && !dumped) { dump_field(t); dumped = true; }
        if (n < ntot) ns.step((n < nst) ? dt : dt_last);              // explicit SSP-RK3
    }
    std::fclose(f);
    const double wall = std::chrono::duration<double>(Clock::now()-t0).count();
    std::printf("\n wrote %s  (%.1f s, %.0f M point-steps/s)\n",
                path, wall, (double)N*N*N*ntot/1e6/wall);
    if (g_exact_time)
        std::printf("VERDICT,%s,t=%.5f,reason=%s,loop=%.4f,steps=%d,wall_per_step=%.6f\n",
                    unstable ? "unstable" : "stable", t, reason, wall, ntot,
                    wall/(double)std::max(1, ntot));
    return unstable ? 2 : 0;
}
