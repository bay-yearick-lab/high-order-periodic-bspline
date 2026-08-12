// ns_reconnection.cpp
//
// FREAKY DNS #1: antiparallel vortex-tube RECONNECTION -- the canonical hard DNS
// whose peak-vorticity amplification omega_max(t)/omega_0 is acutely resolution
// sensitive (thin vorticity bridges form). The cost story: our spectral-class
// banded scheme captures the peak at a COARSE grid where matched-order finite
// differences badly under-predict it, matching a fine reference -> same physics,
// far fewer points (the reason to use these schemes for scalable / wall-bound DNS,
// where Fourier pseudospectral cannot go).
//
// This program sets up the flow and reports the resolution-sensitive diagnostics
// (peak |omega|, enstrophy, energy, divergence). Initial condition: two Gaussian
// antiparallel tubes along x at y = +/- b, sinusoidally drawn together near x=0;
// built divergence-free via omega -> (clean div) -> psi = lap^{-1}(-omega) ->
// u = curl(psi).

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "ns.h"

static const double PI = 3.14159265358979323846;

struct ReconnectionCheckpointHeader {
    char magic[8];
    int version;
    int grid;
    int degree;
    int completed_step;
    double reynolds;
    double dt_scale;
    double dt;
    double final_time;
    char route[16];
};

static std::string joined_path(const std::string& directory, const std::string& name) {
    if (directory.empty() || directory == ".") return name;
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

static bool file_exists(const std::string& path) {
    FILE* handle = std::fopen(path.c_str(), "rb");
    if (!handle) return false;
    std::fclose(handle);
    return true;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    std::string rstr = (argc > 2) ? argv[2] : "bmc";
    Route route = (rstr=="fd6")?Route::FD6 : (rstr=="fd4")?Route::FD4 : (rstr=="fd2")?Route::FD2
                : (rstr=="c6")?Route::C6 : (rstr=="c8")?Route::C8
                : (rstr=="c10")?Route::C10 : (rstr=="c12")?Route::C12 : (rstr=="drp")?Route::DRP
                : (rstr=="sl")?Route::SL : (rstr=="opt")?Route::OPT
                : (rstr=="cw12a")?Route::CW12A : (rstr=="cw12b")?Route::CW12B
                : (rstr=="bmg")?Route::BMG : Route::BMC;
    const double Re = (argc > 3) ? std::atof(argv[3]) : 2000.0;
    const int pdeg = (argc > 5) ? std::atoi(argv[5]) : 5;  // spline degree (BMC p: 5,7,9,11 = BMG2,3,4,5)
    // Optional common initial condition. argv[6] = path to a route-agnostic
    // divergence-free t=0 velocity binary; "" or "-" keeps the
    // default per-route analytic build. Loaded-IC runs write with a "_cic" suffix so
    // they never clobber the default route-IC data files.
    const std::string ic_path = (argc > 6) ? std::string(argv[6]) : std::string();
    const bool load_ic = !ic_path.empty() && ic_path != "-";
    // Optional time-step scale (argv[7], default 1.0) for the time-refinement
    // check, and an optional output tag (argv[8]) appended to every output filename
    // so refined runs never clobber the base matrix. dt_scale=1.0 and tag="" keep
    // the default behavior byte-identical.
    const double dt_scale = (argc > 7) ? std::atof(argv[7]) : 1.0;
    std::string tag = (argc > 8) ? std::string(argv[8]) : std::string();
    if (tag == "-") tag.clear();
    const std::string output_dir = (argc > 9) ? std::string(argv[9]) : std::string("data");
    const std::string resume_path = (argc > 10) ? std::string(argv[10]) : std::string();
    const bool resume = !resume_path.empty() && resume_path != "-";
    const int checkpoint_every = (argc > 11) ? std::atoi(argv[11]) : 0;
    const double field_start = (argc > 12) ? std::atof(argv[12]) : -1.0;
    const double field_stop = (argc > 13) ? std::atof(argv[13]) : -1.0;
    const double field_stride = (argc > 14) ? std::atof(argv[14]) : 0.0;
    const std::string sfx = (load_ic ? std::string("_cic") : std::string()) + tag;
    const double L = 2*PI, Gam = 1.0, nu = Gam/Re;
    const double b = 0.9, acore = 0.45, del = 0.45;       // separation, core, perturbation
    SimConfig c; c.p = pdeg; c.nu = nu; c.route = route;  // higher degree -> higher order + resolution
    c.ax[0] = {N, L, BC::Periodic, true};
    c.ax[1] = {N, L, BC::Periodic, true};
    c.ax[2] = {N, L, BC::Periodic, true};
    std::printf("=================================================================\n");
    std::printf(" Antiparallel vortex reconnection DNS  (N=%d, Re=%.0f, route=%s p=%d)\n",
                N, Re, route_name(route), c.p);
    std::printf(" output_dir=%s checkpoint_every=%d resume=%s\n",
                output_dir.c_str(), checkpoint_every, resume ? resume_path.c_str() : "none");
    if (field_stride > 0.0)
        std::printf(" field_schedule=[%.6f, %.6f] stride=%.6f\n",
                    field_start, field_stop, field_stride);
    std::printf("=================================================================\n");

    IncompressibleNS ns(c);
    if (!ns.ready) { std::printf("setup failed: %s\n", ns.setup_error); return 1; }
    const char* why = nullptr; SeparablePoisson lap;
    if (!lap.build(c, why, /*divgrad=*/false)) { std::printf("poisson: %s\n", why); return 1; }
    const double h = L/N;
    Field3D wx(N,N,N), wy(N,N,N), wz(N,N,N), px(N,N,N), py(N,N,N), pz(N,N,N), tA(N,N,N), tB(N,N,N);

    if (load_ic) {
        // Common divergence-free initial field, identical bytes for every route.
        FILE* bfile = std::fopen(ic_path.c_str(), "rb");
        if (!bfile) { std::printf("cannot open IC file %s\n", ic_path.c_str()); return 1; }
        const size_t nn = ns.u.a.size();
        size_t rd = std::fread(ns.u.a.data(), sizeof(double), nn, bfile);
        rd += std::fread(ns.v.a.data(), sizeof(double), nn, bfile);
        rd += std::fread(ns.w.a.data(), sizeof(double), nn, bfile);
        std::fclose(bfile);
        if (rd != 3*nn) { std::printf("IC file size mismatch (%zu != %zu)\n", rd, 3*nn); return 1; }
        // No standalone ns.project() here on purpose: the loaded field is spectrally
        // divergence-free, so each route's discrete divergence of it is only
        // O(truncation) and is removed by the projection inside the first SSP-RK3
        // step. Skipping the initial projection keeps the t=0 field bit-identical
        // for every route (the whole point of the common-initialization test).
        std::printf(" [loaded common IC from %s : t=0 field bit-identical across routes]\n",
                    ic_path.c_str());
    } else {
    // Deposit vorticity along the two perturbed tube tangents (antiparallel).
    for (int k=0;k<N;++k) for (int j=0;j<N;++j) for (int i=0;i<N;++i) {
        const double x=i*h, y=j*h-PI, z=k*h-PI;                 // centered y,z in [-pi,pi)
        double Wx=0,Wy=0,Wz=0;
        for (int s=-1;s<=1;s+=2) {                              // s=+1 upper tube, s=-1 lower
            const double yc = s*(b - del*std::cos(x));          // centerline drawn together near x=0
            const double dyc = s*del*std::sin(x);               // dyc/dx
            double tx=1.0, ty=dyc, tz=0.0, tn=std::sqrt(tx*tx+ty*ty+tz*tz);
            tx/=tn; ty/=tn; tz/=tn;                             // unit tangent
            const double r2=(y-yc)*(y-yc)+z*z;                  // distance^2 from centerline
            const double mag = s*(Gam/(PI*acore*acore))*std::exp(-r2/(acore*acore));
            Wx+=mag*tx; Wy+=mag*ty; Wz+=mag*tz;
        }
        wx.a[wx.idx(i,j,k)]=Wx; wy.a[wy.idx(i,j,k)]=Wy; wz.a[wz.idx(i,j,k)]=Wz;
    }
    // Clean divergence of omega: chi = lap^{-1}(div omega); omega -= grad chi.
    ns.D.divergence(wx,wy,wz,tA,tB); lap.solve(tA,tB,0.0,1.0);
    ns.D.deriv(tB,tA,0,1); for(size_t q=0;q<wx.a.size();++q) wx.a[q]-=tA.a[q];
    ns.D.deriv(tB,tA,1,1); for(size_t q=0;q<wy.a.size();++q) wy.a[q]-=tA.a[q];
    ns.D.deriv(tB,tA,2,1); for(size_t q=0;q<wz.a.size();++q) wz.a[q]-=tA.a[q];
    // psi_i = lap^{-1}(-omega_i)
    for(size_t q=0;q<wx.a.size();++q){ wx.a[q]=-wx.a[q]; wy.a[q]=-wy.a[q]; wz.a[q]=-wz.a[q]; }
    lap.solve(wx,px,0.0,1.0); lap.solve(wy,py,0.0,1.0); lap.solve(wz,pz,0.0,1.0);
    // u = curl(psi)
    ns.D.deriv(pz,tA,1,1); ns.D.deriv(py,tB,2,1); for(size_t q=0;q<ns.u.a.size();++q) ns.u.a[q]=tA.a[q]-tB.a[q];
    ns.D.deriv(px,tA,2,1); ns.D.deriv(pz,tB,0,1); for(size_t q=0;q<ns.v.a.size();++q) ns.v.a[q]=tA.a[q]-tB.a[q];
    ns.D.deriv(py,tA,0,1); ns.D.deriv(px,tB,1,1); for(size_t q=0;q<ns.w.a.size();++q) ns.w.a[q]=tA.a[q]-tB.a[q];
    ns.project();
    }

    // Vorticity diagnostics: peak |omega|, enstrophy = sum|omega|^2 * h^3, energy.
    auto vortdiag = [&](double& peak, double& ens){
        ns.D.deriv(ns.w,tA,1,1); ns.D.deriv(ns.v,tB,2,1); for(size_t q=0;q<wx.a.size();++q) wx.a[q]=tA.a[q]-tB.a[q];
        ns.D.deriv(ns.u,tA,2,1); ns.D.deriv(ns.w,tB,0,1); for(size_t q=0;q<wy.a.size();++q) wy.a[q]=tA.a[q]-tB.a[q];
        ns.D.deriv(ns.v,tA,0,1); ns.D.deriv(ns.u,tB,1,1); for(size_t q=0;q<wz.a.size();++q) wz.a[q]=tA.a[q]-tB.a[q];
        peak=0; ens=0; for(size_t q=0;q<wx.a.size();++q){ double m2=wx.a[q]*wx.a[q]+wy.a[q]*wy.a[q]+wz.a[q]*wz.a[q];
            ens+=m2; peak=std::fmax(peak,std::sqrt(m2)); } ens*=h*h*h;
    };
    double peak0,ens0; vortdiag(peak0,ens0);
    const double E0 = ns.kinetic_energy();
    std::printf("\n init: peak|omega|=%.4f  enstrophy=%.4f  KE=%.4f  div=%.2e\n",
                peak0, ens0, E0, ns.max_divergence());

    const double umax = peak0*acore;                          // ~ velocity scale
    const double dt = dt_scale * 0.4 * h / std::fmax(umax, 0.2);
    const double Tend = (argc > 4) ? std::atof(argv[4]) : 50.0;
    const int nst = (int)std::lround(Tend/dt);
    std::printf(" dt=%.4f steps=%d  (advance to t=%.0f)\n", dt, nst, Tend);

    int start_step = 0;
    if (resume) {
        FILE* checkpoint = std::fopen(resume_path.c_str(), "rb");
        if (!checkpoint) { std::printf("cannot open checkpoint %s\n", resume_path.c_str()); return 1; }
        ReconnectionCheckpointHeader header{};
        const size_t header_read = std::fread(&header, sizeof(header), 1, checkpoint);
        const size_t nn = ns.u.a.size();
        const bool valid = header_read == 1
            && std::memcmp(header.magic, "BMC11CP", 7) == 0
            && header.version == 1 && header.grid == N && header.degree == pdeg
            && std::strcmp(header.route, route_name(route)) == 0
            && std::fabs(header.reynolds - Re) <= 1e-12
            && std::fabs(header.dt_scale - dt_scale) <= 1e-12
            && std::fabs(header.dt - dt) <= 1e-12
            && std::fabs(header.final_time - Tend) <= 1e-12;
        size_t read_count = 0;
        if (valid) {
            read_count += std::fread(ns.u.a.data(), sizeof(double), nn, checkpoint);
            read_count += std::fread(ns.v.a.data(), sizeof(double), nn, checkpoint);
            read_count += std::fread(ns.w.a.data(), sizeof(double), nn, checkpoint);
        }
        std::fclose(checkpoint);
        if (!valid || read_count != 3 * nn || header.completed_step < 0 || header.completed_step > nst) {
            std::printf("checkpoint validation failed for %s\n", resume_path.c_str());
            return 1;
        }
        start_step = header.completed_step;
        std::printf(" resumed checkpoint step=%d t=%.8f path=%s\n",
                    start_step, start_step * dt, resume_path.c_str());
    }

    // CSV trace for the paper figures: peak vorticity and enstrophy vs time. The
    // amplification omega_max(t)/omega_0 is the resolution-sensitive diagnostic.
    char path[256];
    std::snprintf(path, sizeof(path), "recon_%s%s_p%d_N%d_re%.0f.csv", rstr.c_str(), sfx.c_str(), pdeg, N, Re);
    const std::string csv_path = joined_path(output_dir, path);
    FILE* csv = std::fopen(csv_path.c_str(), resume ? "a" : "w");
    if (!csv) { std::printf("cannot open trace %s\n", csv_path.c_str()); return 1; }
    if (!resume) std::fprintf(csv, "t,peak,peak_rel,enstrophy,KErel,divmax\n");

    // Dump the velocity field at t=0 (the antiparallel tubes) and near the peak so
    // the setup schematic and vorticity slices can be rendered offline (Python).
    // Raw N^3 doubles each of u, v, w; transient artifacts (gitignored).
    auto dump_field = [&](const std::string& name){
        const std::string final_path = joined_path(output_dir, name);
        if (file_exists(final_path)) return;
        const std::string temporary_path = final_path + ".partial";
        FILE* b = std::fopen(temporary_path.c_str(), "wb");
        if (!b) { std::printf("cannot open field output %s\n", temporary_path.c_str()); std::exit(1); }
        const size_t nn = ns.u.a.size();
        size_t written = std::fwrite(ns.u.a.data(), sizeof(double), nn, b);
        written += std::fwrite(ns.v.a.data(), sizeof(double), nn, b);
        written += std::fwrite(ns.w.a.data(), sizeof(double), nn, b);
        std::fflush(b);
        std::fclose(b);
        if (written != 3 * nn || std::rename(temporary_path.c_str(), final_path.c_str()) != 0) {
            std::printf("field output failed for %s\n", final_path.c_str());
            std::exit(1);
        }
    };
    if (!resume) {
        char field_name[256];
        std::snprintf(field_name, sizeof(field_name), "field_000_t%.8f.bin", 0.0);
        dump_field(field_name);
    }

    std::vector<int> field_steps;
    std::vector<double> field_targets;
    if (field_stride > 0.0 && field_start >= 0.0 && field_stop >= field_start) {
        for (double target = field_start; target <= field_stop + 0.25 * field_stride; target += field_stride) {
            field_targets.push_back(target);
            field_steps.push_back((int)std::lround(target / dt));
        }
    }

    const std::string field_index_path = joined_path(output_dir, "fields.csv");
    FILE* field_index = std::fopen(field_index_path.c_str(), resume ? "a" : "w");
    if (!field_index) { std::printf("cannot open field index %s\n", field_index_path.c_str()); return 1; }
    if (!resume) std::fprintf(field_index, "field_id,target_time,actual_time,step,path\n");

    const std::string checkpoint_path = joined_path(output_dir, "checkpoint_latest.bin");
    auto write_checkpoint = [&](int completed_step) {
        const std::string temporary_path = checkpoint_path + ".partial";
        FILE* checkpoint = std::fopen(temporary_path.c_str(), "wb");
        if (!checkpoint) { std::printf("cannot open checkpoint output %s\n", temporary_path.c_str()); std::exit(1); }
        ReconnectionCheckpointHeader header{};
        std::memcpy(header.magic, "BMC11CP", 7);
        header.version = 1;
        header.grid = N;
        header.degree = pdeg;
        header.completed_step = completed_step;
        header.reynolds = Re;
        header.dt_scale = dt_scale;
        header.dt = dt;
        header.final_time = Tend;
        std::snprintf(header.route, sizeof(header.route), "%s", route_name(route));
        const size_t nn = ns.u.a.size();
        size_t written = std::fwrite(&header, sizeof(header), 1, checkpoint);
        written += std::fwrite(ns.u.a.data(), sizeof(double), nn, checkpoint);
        written += std::fwrite(ns.v.a.data(), sizeof(double), nn, checkpoint);
        written += std::fwrite(ns.w.a.data(), sizeof(double), nn, checkpoint);
        std::fflush(checkpoint);
        std::fclose(checkpoint);
        if (written != 1 + 3 * nn || std::rename(temporary_path.c_str(), checkpoint_path.c_str()) != 0) {
            std::printf("checkpoint write failed for %s\n", checkpoint_path.c_str());
            std::exit(1);
        }
        std::printf(" checkpoint step=%d t=%.8f path=%s\n",
                    completed_step, completed_step * dt, checkpoint_path.c_str());
        std::fflush(stdout);
    };

    std::printf("\n   t      peak|omega|   enstrophy   KE/KE0    max|div|   (peak0=%.4f)\n", peak0);
    for (int n=start_step;n<=nst;++n){
        for (size_t output_id = 0; output_id < field_steps.size(); ++output_id) {
            if (n == field_steps[output_id]) {
                char field_name[256];
                std::snprintf(field_name, sizeof(field_name), "field_%03zu_t%.8f.bin", output_id + 1, n * dt);
                const std::string final_path = joined_path(output_dir, field_name);
                const bool existed = file_exists(final_path);
                dump_field(field_name);
                if (!existed) {
                    std::fprintf(field_index, "%zu,%.8f,%.8f,%d,%s\n",
                                 output_id + 1, field_targets[output_id], n * dt, n, final_path.c_str());
                    std::fflush(field_index);
                }
            }
        }
        if (n % (nst/50) == 0 || n == nst) {
            double pk,en; vortdiag(pk,en);
            const double ke = ns.kinetic_energy()/E0, dv = ns.max_divergence();
            std::fprintf(csv, "%.5f,%.6f,%.6f,%.6f,%.6f,%.3e\n", n*dt, pk, pk/peak0, en, ke, dv);
            if (n % (nst/25) == 0)
                std::printf("  %5.1f    %.5f     %.5f    %.5f   %.1e\n", n*dt, pk, en, ke, dv);
            std::fflush(csv);
            std::fflush(stdout);
        }
        if (n<nst) {
            ns.step(dt);
            const int completed_step = n + 1;
            const bool first_checkpoint = checkpoint_every > 0 && start_step == 0
                && completed_step == std::min(10, checkpoint_every);
            const bool periodic_checkpoint = checkpoint_every > 0 && completed_step % checkpoint_every == 0;
            if (first_checkpoint || periodic_checkpoint || completed_step == nst)
                write_checkpoint(completed_step);
        }
    }
    std::fclose(field_index);
    std::fclose(csv);
    std::printf("\nwrote %s\nDone.\n", csv_path.c_str());
    return 0;
}
