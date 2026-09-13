// bm_apply.h
//
// The PRODUCTION application path for the B-spline mimetic derivatives: the
// reduce-solve-expand chain. The dense operators in
// bm_operators.h are built only to verify identities and measure order; here the
// same maps are applied without ever materializing an n x n matrix.
//
//   BMG D1 = B M^{-1} K B^{-1}   applied to nodal u as
//       u --> alpha (solve B alpha = u)        banded / cyclic solve, O(n*bw^2) once
//          --> r = K alpha                     banded matvec,         O(n*bw)
//          --> beta (solve M beta = r)         banded / cyclic solve
//          --> y = B beta                      banded matvec
//
//   BMC D1 = B' B^{-1},  BMC D2 = B'' B^{-1}   solve B, then one banded matvec.
//   BMG D2 (periodic) = -B M^{-1} S B^{-1}     same chain with S, negated.
//   BMF G  = Bhat_eta Delta B^{-1}             solve B, Delta (bidiagonal), matvec.
//
// PERIODIC operators are circulant: each is one stencil of length 2*bw+1 and the
// solves use CyclicSolver::set_from_stencil. Nothing is O(n^2) -- storage and
// per-apply cost are O(n*bw), so a single line scales to millions of points.
// BOUNDED operators are banded; they keep dense-backed storage here because the
// bounded path is only exercised for correctness at modest n. B and M are
// factored ONCE and reused for every right-hand side -- every grid line, every
// time step. The bounded wide second derivatives (BMF Dv G and BMG flux form)
// are not pure reduce-solve-expand chains and stay in bm_operators.h.

#ifndef BM_APPLY_H
#define BM_APPLY_H

#include <vector>
#include "bspline.h"
#include "exp_member.h"

// A circulant operator stored as one centered stencil st[o + bw], o in [-bw,bw].
// The matvec splits into a wrap-free interior (the bulk, a stride-1 dot product
// the compiler vectorizes with FMA) and the 2*bw boundary rows that wrap. Keeping
// the modulo out of the hot loop is what makes this fast.
struct Circulant {
    int bw = 0;
    std::vector<double> st;          // length 2*bw+1
    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        const int n = (int)x.size();
        const int w = 2 * bw + 1;
        const double* s = st.data();
        for (int i = bw; i < n - bw; ++i) {              // interior: no wrap
            const double* xp = &x[i - bw];
            double acc = 0.0;
            for (int o = 0; o < w; ++o) acc += s[o] * xp[o];
            y[i] = acc;
        }
        auto wrap_row = [&](int i) {
            double acc = 0.0;
            for (int o = -bw; o <= bw; ++o) acc += s[o + bw] * x[((i + o) % n + n) % n];
            y[i] = acc;
        };
        for (int i = 0; i < bw && i < n; ++i)      wrap_row(i);
        for (int i = (n - bw > bw ? n - bw : bw); i < n; ++i) wrap_row(i);
    }
    std::vector<double> matvec(const std::vector<double>& x) const {
        std::vector<double> y(x.size());
        matvec(x, y);
        return y;
    }
    // Batched matvec: x, y are n x W row-major; W lanes processed together.
    void matvec_block(const double* x, double* y, int n, int W) const {
        const int wd = 2 * bw + 1;
        const double* s = st.data();
        for (int i = bw; i < n - bw; ++i) {              // interior: no wrap
            double* yi = y + static_cast<size_t>(i) * W;
            for (int l = 0; l < W; ++l) yi[l] = 0.0;
            const double* xp = x + static_cast<size_t>(i - bw) * W;
            for (int o = 0; o < wd; ++o) {
                const double so = s[o];
                const double* xo = xp + static_cast<size_t>(o) * W;
                for (int l = 0; l < W; ++l) yi[l] += so * xo[l];
            }
        }
        auto wrap_row = [&](int i) {
            double* yi = y + static_cast<size_t>(i) * W;
            for (int l = 0; l < W; ++l) yi[l] = 0.0;
            for (int o = -bw; o <= bw; ++o) {
                const int j = ((i + o) % n + n) % n;
                const double so = s[o + bw];
                const double* xj = x + static_cast<size_t>(j) * W;
                for (int l = 0; l < W; ++l) yi[l] += so * xj[l];
            }
        };
        for (int i = 0; i < bw && i < n; ++i) wrap_row(i);
        for (int i = (n - bw > bw ? n - bw : bw); i < n; ++i) wrap_row(i);
    }
};

// Banded matvec y = A x touching only the half-bandwidth bw (bounded case).
inline void matvec_banded(const Mat& A, const std::vector<double>& x, int bw,
                          std::vector<double>& y) {
    const int n = A.r;
    for (int i = 0; i < n; ++i) {
        const int jlo = (i - bw > 0) ? i - bw : 0;
        const int jhi = (i + bw < n - 1) ? i + bw : n - 1;
        double s = 0.0;
        for (int j = jlo; j <= jhi; ++j) s += A(i, j) * x[j];
        y[i] = s;
    }
}
inline std::vector<double> matvec_banded(const Mat& A, const std::vector<double>& x, int bw) {
    std::vector<double> y(A.r);
    matvec_banded(A, x, bw, y);
    return y;
}

// Factored, banded application of the mimetic derivatives on one 1D line. Build
// once per (domain, degree, n); apply to as many fields as needed.
struct FastDeriv {
    bool periodic = false;
    int n = 0, p = 0; double h = 0.0;          // grid spacing (for explicit-FD baseline)

    // periodic (circulant stencil) storage -- O(n*bw)
    Circulant cB, cBp, cBpp, cK, cS;
    CyclicSolver Bcyc, Mcyc;

    // world-class local-high-res competitors (periodic): compact Pade C6 (SLC 1992)
    // LHS tridiagonal solve + RHS stencil; DRP (Tam-Webb 1993) explicit 7-point D1.
    CyclicSolver L1cyc, L2cyc;               // compact LHS (tridiagonal) for D1, D2
    std::vector<double> cr1, cr2, drp1;      // compact RHS (offsets -2..2); DRP D1 (-3..3)
    CyclicSolver L1cyc8, L2cyc8;             // 8th-order compact (C8) LHS for D1, D2
    std::vector<double> cr1_8, cr2_8;        // C8 RHS stencils (offsets -3..3)
    CyclicSolver L1cyc10, L2cyc10;           // 10th-order tridiagonal compact LHS
    std::vector<double> cr1_10, cr2_10;      // C10 RHS stencils (offsets -4..4)
    CyclicSolver L1cyc12, L2cyc12;           // 12th-order tridiagonal compact LHS
    std::vector<double> cr1_12, cr2_12;      // C12 RHS stencils (offsets -5..5)
    CyclicSolver L1cycSL, L2cycSL;           // SLC spectral-like PENTADIAGONAL LHS (D1, D2)
    std::vector<double> cr1_sl, cr2_sl;      // SL RHS stencils (offsets -3..3)
    CyclicSolver L1cycOPT, L2cycOPT;         // paper threshold-optimized TRIDIAGONAL LHS (D1, D2)
    std::vector<double> cr1_opt, cr2_opt;    // OPT RHS stencils (11-point, offsets -5..5)
    CyclicSolver L1cycCWA, L2cycCWA;         // certified wide compact A: PENTADIAGONAL LHS (D1, D2)
    std::vector<double> cr1_cwa, cr2_cwa;    // CW12A RHS stencils (15-point, offsets -7..7)
    CyclicSolver L1cycCWB, L2cycCWB;         // certified wide compact B: PENTADIAGONAL LHS (D1, D2)
    std::vector<double> cr1_cwb, cr2_cwb;    // CW12B RHS stencils (11-point, offsets -5..5)

    // Generic exponential-family route (route "exp"): the same collocated
    // reduce-then-sample chain as BMC, but with the left side and both right
    // sides taken from a member file instead of the polynomial kernels.
    //   left side  E:  E[i,j] = beta(|i-j|),                 bandwidth R
    //   D1 = E' E^{-1},  E'[i,j]  = beta'(j-i) / h           (antisymmetric)
    //   D2 = E'' E^{-1}, E''[i,j] = beta''(|i-j|) / h^2      (symmetric)
    // One cyclic solve produces the coefficients; each derivative is then one
    // banded product, so D1 and D2 share a single solve wherever both are
    // wanted at the same point.
    CyclicSolver Ecyc;                       // symmetric cyclic banded left side, bandwidth R
    Circulant cEp, cEpp;                     // D1 and D2 right-side stencils
    int expR = 0;                            // member support radius R

    // bounded (banded, dense-backed) storage
    Mat B, Bp, Bpp, K, S;
    int bw = 0;
    DerivativeSpace ds;
    BandSolver Bband, Mband;

    mutable std::vector<double> s1, s2;     // reusable scratch for the in-place path

    static FastDeriv build(bool periodic, int n, int p, double L) {
        FastDeriv f;
        f.periodic = periodic; f.n = n; f.p = p; f.bw = p; f.h = L / n;
        f.s1.assign(n, 0.0); f.s2.assign(n, 0.0);
        if (periodic) {
            const double h = L / n, inv_h = 1.0 / h;
            const int bwB = (p + 1) / 2;            // value/derivative support radius
            // Circulant stencils computed directly from the cardinal kernels --
            // no n x n matrix. Offset o = j - i; build_periodic uses t = -o.
            f.cB.bw = bwB;   f.cB.st.resize(2 * bwB + 1);
            f.cBp.bw = bwB;  f.cBp.st.resize(2 * bwB + 1);
            f.cBpp.bw = bwB; f.cBpp.st.resize(2 * bwB + 1);
            for (int o = -bwB; o <= bwB; ++o) {
                const double t = -o;
                f.cB.st[o + bwB]   = beta_value(p, t);
                f.cBp.st[o + bwB]  = (p >= 1) ? beta_deriv1(p, t) * inv_h : 0.0;
                f.cBpp.st[o + bwB] = (p >= 2) ? beta_deriv2(p, t) * inv_h * inv_h : 0.0;
            }
            f.cK.bw = p; f.cK.st.resize(2 * p + 1);
            f.cS.bw = p; f.cS.st.resize(2 * p + 1);
            std::vector<double> stM(2 * p + 1, 0.0);
            for (int o = -p; o <= p; ++o) {
                stM[o + p]        = h * beta_overlap(p, o, 0, 0);
                f.cK.st[o + p]    = (p >= 1) ? beta_overlap(p, o, 0, 1) : 0.0;
                f.cS.st[o + p]    = (p >= 1) ? beta_overlap(p, o, 1, 1) * inv_h : 0.0;
            }
            f.Bcyc.set_from_stencil(n, bwB, f.cB.st);
            f.Mcyc.set_from_stencil(n, p, stM);
            // compact Pade C6 (SLC): D1 alpha=1/3,a=14/9,b=1/9 ; D2 alpha=2/11,a=12/11,b=3/11
            { const double a1=14.0/9, b1=1.0/9, al1=1.0/3;
              f.cr1.assign(5, 0.0);
              f.cr1[2+1]= a1*inv_h/2;  f.cr1[2-1]=-a1*inv_h/2;
              f.cr1[2+2]= b1*inv_h/4;  f.cr1[2-2]=-b1*inv_h/4;
              f.L1cyc.set_from_stencil(n, 1, std::vector<double>{al1, 1.0, al1});
              const double a2=12.0/11, b2=3.0/11, al2=2.0/11, ih2=inv_h*inv_h;
              f.cr2.assign(5, 0.0);
              f.cr2[2+1]= a2*ih2;  f.cr2[2-1]= a2*ih2;
              f.cr2[2+2]= b2*ih2/4; f.cr2[2-2]= b2*ih2/4;
              f.cr2[2+0]= -(2*a2 + b2/2)*ih2;
              f.L2cyc.set_from_stencil(n, 1, std::vector<double>{al2, 1.0, al2}); }
            // compact Pade C8 (SLC 1992, 8th-order tridiagonal). D1: alpha=3/8,
            // RHS (a/2,b/4,c/6)=(25/32,1/20,-1/480). D2: alpha=9/38, stencil derived
            // from the 8th-order order conditions (verified to O(theta^10)):
            // a=147/152, b=51/95, c=-23/760 -> RHS off1=a, off2=b/4, off3=c/9.
            { const double al1=3.0/8;
              f.cr1_8.assign(7, 0.0);
              f.cr1_8[3+1]= (25.0/32)*inv_h;  f.cr1_8[3-1]=-(25.0/32)*inv_h;
              f.cr1_8[3+2]= (1.0/20)*inv_h;   f.cr1_8[3-2]=-(1.0/20)*inv_h;
              f.cr1_8[3+3]= (-1.0/480)*inv_h; f.cr1_8[3-3]=-(-1.0/480)*inv_h;
              f.L1cyc8.set_from_stencil(n, 1, std::vector<double>{al1, 1.0, al1});
              const double al2=9.0/38, ih2=inv_h*inv_h;
              const double o1=147.0/152, o2=51.0/380, o3=-23.0/6840;
              f.cr2_8.assign(7, 0.0);
              f.cr2_8[3+1]= o1*ih2; f.cr2_8[3-1]= o1*ih2;
              f.cr2_8[3+2]= o2*ih2; f.cr2_8[3-2]= o2*ih2;
              f.cr2_8[3+3]= o3*ih2; f.cr2_8[3-3]= o3*ih2;
              f.cr2_8[3+0]= -2.0*(o1+o2+o3)*ih2;
              f.L2cyc8.set_from_stencil(n, 1, std::vector<double>{al2, 1.0, al2}); }
            // compact Pade C10/C12: maximal-order TRIDIAGONAL members with minimal
            // RHS radius (4 resp. 5), coefficients exact from the Taylor order
            // conditions (the same solve reproduces the classical C4/C6/C8
            // fractions; observed orders 10.001/11.998 at 50-digit precision).
            // D1: C10 alpha=2/5, a_s = 39/50, 1/15, -1/210, 1/4200;
            //     C12 alpha=5/12, a_s = 7/9, 5/63, -5/672, 1/1512, -1/30240.
            // D2: C10 alpha=8/29, o_s = 1126/1305, 247/1305, -74/9135, 43/146160;
            //     C12 alpha=25/82, o_s = 4595/5904, 605/2583, -155/11808, 5/5904,
            //     -23/688800. Both D2 symbols are nonpositive on (0,pi].
            { const double al1=2.0/5;
              const double a[4] = {39.0/50, 1.0/15, -1.0/210, 1.0/4200};
              f.cr1_10.assign(9, 0.0);
              for (int s2 = 1; s2 <= 4; ++s2) { f.cr1_10[4+s2]= a[s2-1]*inv_h; f.cr1_10[4-s2]=-a[s2-1]*inv_h; }
              f.L1cyc10.set_from_stencil(n, 1, std::vector<double>{al1, 1.0, al1});
              const double al2=8.0/29, ih2=inv_h*inv_h;
              const double o[4] = {1126.0/1305, 247.0/1305, -74.0/9135, 43.0/146160};
              f.cr2_10.assign(9, 0.0);
              double osum = 0.0;
              for (int s2 = 1; s2 <= 4; ++s2) { f.cr2_10[4+s2]= o[s2-1]*ih2; f.cr2_10[4-s2]= o[s2-1]*ih2; osum += o[s2-1]; }
              f.cr2_10[4+0]= -2.0*osum*ih2;
              f.L2cyc10.set_from_stencil(n, 1, std::vector<double>{al2, 1.0, al2}); }
            { const double al1=5.0/12;
              const double a[5] = {7.0/9, 5.0/63, -5.0/672, 1.0/1512, -1.0/30240};
              f.cr1_12.assign(11, 0.0);
              for (int s2 = 1; s2 <= 5; ++s2) { f.cr1_12[5+s2]= a[s2-1]*inv_h; f.cr1_12[5-s2]=-a[s2-1]*inv_h; }
              f.L1cyc12.set_from_stencil(n, 1, std::vector<double>{al1, 1.0, al1});
              const double al2=25.0/82, ih2=inv_h*inv_h;
              const double o[5] = {4595.0/5904, 605.0/2583, -155.0/11808, 5.0/5904, -23.0/688800};
              f.cr2_12.assign(11, 0.0);
              double osum = 0.0;
              for (int s2 = 1; s2 <= 5; ++s2) { f.cr2_12[5+s2]= o[s2-1]*ih2; f.cr2_12[5-s2]= o[s2-1]*ih2; osum += o[s2-1]; }
              f.cr2_12[5+0]= -2.0*osum*ih2;
              f.L2cyc12.set_from_stencil(n, 1, std::vector<double>{al2, 1.0, al2}); }
            // SLC 1992 spectral-like PENTADIAGONAL compact (JCP 103:16-42, eqs.
            // 3.1.6-3.1.7 and 3.1.12), formal order 4, optimized for resolution.
            // D1 LHS: beta(f'_{i±2}) + alpha(f'_{i±1}) + f'_i; RHS 7-point (a,b,c).
            // D2 LHS: same pentadiagonal shape; RHS 7-point (a2,b2,c2). Both LHS are
            // solved with the CyclicSolver at bandwidth 2 (Woodbury). The D2 symbol
            // is nonpositive on (0,pi] (needed for the SAN audit).
            { const double al1=0.5771439, be1=0.0896406;
              const double a1=1.3025166, b1=0.9935500, c1=0.03750245;
              f.cr1_sl.assign(7, 0.0);
              f.cr1_sl[3+1]= a1*inv_h/2; f.cr1_sl[3-1]=-a1*inv_h/2;
              f.cr1_sl[3+2]= b1*inv_h/4; f.cr1_sl[3-2]=-b1*inv_h/4;
              f.cr1_sl[3+3]= c1*inv_h/6; f.cr1_sl[3-3]=-c1*inv_h/6;
              f.L1cycSL.set_from_stencil(n, 2, std::vector<double>{be1, al1, 1.0, al1, be1});
              const double al2=0.50209266, be2=0.05569169, ih2=inv_h*inv_h;
              const double a2=0.21564935, b2=1.7233220, c2=0.17659730;
              f.cr2_sl.assign(7, 0.0);
              f.cr2_sl[3+1]= a2*ih2;   f.cr2_sl[3-1]= a2*ih2;
              f.cr2_sl[3+2]= b2/4*ih2; f.cr2_sl[3-2]= b2/4*ih2;
              f.cr2_sl[3+3]= c2/9*ih2; f.cr2_sl[3-3]= c2/9*ih2;
              f.cr2_sl[3+0]= -2.0*(a2 + b2/4 + c2/9)*ih2;
              f.L2cycSL.set_from_stencil(n, 2, std::vector<double>{be2, al2, 1.0, al2, be2}); }
            // Paper THRESHOLD-OPTIMIZED TRIDIAGONAL compact (route "opt"), formal
            // order 8, threshold-maximal 1% dispersion reach with an 11-point RHS.
            // D1 (frontier cell L1r5q8, verified 1% cut 0.8810*pi): symbol
            // num=sum 2 a_s sin(s w), den=1+2 alpha cos w; the a_s already absorb the
            // pair normalization so the nodal coefficient of u_{i+s} is a_s/h.
            // D2 (matched partner, 1% cut 0.9582*pi): symbol
            // num=a_0+sum 2 a_s cos(s w), den=1+2 alpha cos w, target -w^2, symbol
            // strictly negative on (0,pi] (SAN audit). Both LHS tridiagonal (bw 1).
            { const double al1=0.4742169929451113;
              const double a1[5]={0.7598825426879071, 0.13489507072322338,
                                  -0.023260830361993556, 0.004069991868971751,
                                  -0.00039063351582985587};
              f.cr1_opt.assign(11, 0.0);
              for (int s2 = 1; s2 <= 5; ++s2) { f.cr1_opt[5+s2]= a1[s2-1]*inv_h; f.cr1_opt[5-s2]=-a1[s2-1]*inv_h; }
              f.L1cycOPT.set_from_stencil(n, 1, std::vector<double>{al1, 1.0, al1});
              const double al2=0.4320698009820234, ih2=inv_h*inv_h;
              const double a0=-1.555224078876143;
              const double a2[5]={0.35890831739343104, 0.45913001629633177,
                                  -0.04560561305199756, 0.00559124812826596,
                                  -0.0004119293279594219};
              f.cr2_opt.assign(11, 0.0);
              f.cr2_opt[5+0]= a0*ih2;
              for (int s2 = 1; s2 <= 5; ++s2) { f.cr2_opt[5+s2]= a2[s2-1]*ih2; f.cr2_opt[5-s2]= a2[s2-1]*ih2; }
              f.L2cycOPT.set_from_stencil(n, 1, std::vector<double>{al2, 1.0, al2}); }
            // CERTIFIED WIDE COMPACTS (routes "cw12a"/"cw12b"): the two
            // interval-certified pentadiagonal formal-order-12 first derivatives
            // from the constrained compact frontier sweep (cw12a = cell
            // L2r7q12 floor 0.05, certified 1% cut 0.8616*pi,
            // dense-grid cut 0.9002*pi; cw12b = cell L2r5q12 floor 0.025,
            // certified cut 0.8225*pi). Symbol convention as for "opt":
            // num = sum 2 a_s sin(s w), den = 1 + 2 al1 cos w + 2 al2 cos 2w; the
            // a_s absorb the pair normalization, so the nodal coefficient of
            // u_{i+s} is a_s/h (antisymmetric). D2 partners built by the SAME
            // threshold-maximal null-space scan as the opt partner:
            // same pentadiagonal LHS bandwidth, same RHS
            // radius, same formal order 12; both D2 symbols strictly negative on
            // (0,pi]; cuts 0.9701*pi (A) / 0.9367*pi (B).
            { const double al1=0.40937878082640167, al2=-0.06462037441519262;
              const double a1[7]={0.8359351614953894, 0.06465601447530303,
                                  -0.05254555071859491, 0.011976865033900547,
                                  -0.0025753384161538, 0.00038591694604337513,
                                  -2.8343087156344804e-05};
              f.cr1_cwa.assign(15, 0.0);
              for (int s2 = 1; s2 <= 7; ++s2) { f.cr1_cwa[7+s2]= a1[s2-1]*inv_h; f.cr1_cwa[7-s2]=-a1[s2-1]*inv_h; }
              f.L1cycCWA.set_from_stencil(n, 2, std::vector<double>{al2, al1, 1.0, al1, al2});
              const double bl1=0.1601749806496769, bl2=-0.21726126868097, ih2=inv_h*inv_h;
              const double b0=-2.4227916839619907;
              const double b2[7]={0.9043854478128212, 0.5970618140260426,
                                  -0.3252810784893023, 0.040779787773292164,
                                  -0.006239267401751708, 0.0007337128953475037,
                                  -4.4574635453931686e-05};
              f.cr2_cwa.assign(15, 0.0);
              f.cr2_cwa[7+0]= b0*ih2;
              for (int s2 = 1; s2 <= 7; ++s2) { f.cr2_cwa[7+s2]= b2[s2-1]*ih2; f.cr2_cwa[7-s2]= b2[s2-1]*ih2; }
              f.L2cycCWA.set_from_stencil(n, 2, std::vector<double>{bl2, bl1, 1.0, bl1, bl2}); }
            { const double al1=0.5796083747693036, al2=0.09310954748722111;
              const double a1[5]={0.6474244112956682, 0.25095267402010113,
                                  0.00827175994799237, -0.0003731748662601286,
                                  1.3116508343528986e-05};
              f.cr1_cwb.assign(11, 0.0);
              for (int s2 = 1; s2 <= 5; ++s2) { f.cr1_cwb[5+s2]= a1[s2-1]*inv_h; f.cr1_cwb[5-s2]=-a1[s2-1]*inv_h; }
              f.L1cycCWB.set_from_stencil(n, 2, std::vector<double>{al2, al1, 1.0, al1, al2});
              const double bl1=0.5273008328764947, bl2=0.06895526765898032, ih2=inv_h*inv_h;
              const double b0=-1.2490664485294838;
              const double b2[5]={0.15392188750639246, 0.43804625186979695,
                                  0.03355551188077371, -0.001018485995992867,
                                  2.8059003771746482e-05};
              f.cr2_cwb.assign(11, 0.0);
              f.cr2_cwb[5+0]= b0*ih2;
              for (int s2 = 1; s2 <= 5; ++s2) { f.cr2_cwb[5+s2]= b2[s2-1]*ih2; f.cr2_cwb[5-s2]= b2[s2-1]*ih2; }
              f.L2cycCWB.set_from_stencil(n, 2, std::vector<double>{bl2, bl1, 1.0, bl1, bl2}); }
            // Generic exponential-family member (route "exp"): the left side and
            // both right-side stencils are the lattice samples of one generator
            // E_{m,r}(omega), read from the member JSON. Offset o = j - i samples
            // the kernel at t = -o; beta is even, beta' is odd and beta'' is even,
            // so the one-sided arrays give every entry. Built only when a member
            // file is available, so the other routes never need one.
            { const char* why_exp = "";
              if (exp_member_ready(why_exp)) {
                  const ExpMember& em = exp_member_store();
                  const int R = em.radius;
                  f.expR = R;
                  f.cEp.bw = R;  f.cEp.st.assign(2 * R + 1, 0.0);
                  f.cEpp.bw = R; f.cEpp.st.assign(2 * R + 1, 0.0);
                  std::vector<double> stE(2 * R + 1, 0.0);
                  for (int o = -R; o <= R; ++o) {
                      const int a = (o < 0) ? -o : o;
                      const double sgn = (o < 0) ? -1.0 : 1.0;
                      stE[o + R]        = em.beta[a];
                      f.cEp.st[o + R]   = -sgn * em.dbeta[a] * inv_h;
                      f.cEpp.st[o + R]  = em.ddbeta[a] * inv_h * inv_h;
                  }
                  if (n >= 2 * R + 1) f.Ecyc.set_from_stencil(n, R, stE);
              } }
            // DRP explicit 7-point first derivative (antisymmetric), the PUBLISHED
            // Tam & Webb JCP 107 (1993) coefficients, identical to the Python
            // symbol library (spline_analysis.py). The previous values
            // (0.770882380518, -0.166705904415, 0.020843142770) were Tam's
            // later REVISED set (optimization range eta in [-1.1, 1.1])
            // mislabeled as the 1993 operator (fixed 2026-07-13).
            { const double d1=0.79926643, d2=-0.18941314, d3=0.02651995;
              f.drp1.assign(7, 0.0);
              f.drp1[3+1]= d1*inv_h; f.drp1[3-1]=-d1*inv_h;
              f.drp1[3+2]= d2*inv_h; f.drp1[3-2]=-d2*inv_h;
              f.drp1[3+3]= d3*inv_h; f.drp1[3-3]=-d3*inv_h; }
        } else {
            SplineOperators op = build_clamped(n, p, L);
            std::vector<double> knots = open_uniform_knots(n, p, L);
            CoeffForms cf = assemble_forms_clamped(n, p, knots);
            f.B = to_mat(op.B); f.Bp = to_mat(op.Bp); f.Bpp = to_mat(op.Bpp);
            f.K = cf.K; f.S = cf.S;
            f.ds = build_derivative_space(n, p, knots);
            f.Bband.set_from_dense(op.B, p, p); f.Bband.factor();
            f.Mband.set_from_dense(to_dense(cf.M), p, p); f.Mband.factor();
        }
        return f;
    }

    // --- dispatched primitives (periodic stencil vs bounded band) -----------
    std::vector<double> applyB(const std::vector<double>& x) const {
        return periodic ? cB.matvec(x) : matvec_banded(B, x, bw);
    }
    std::vector<double> applyBp(const std::vector<double>& x) const {
        return periodic ? cBp.matvec(x) : matvec_banded(Bp, x, bw);
    }
    std::vector<double> applyBpp(const std::vector<double>& x) const {
        return periodic ? cBpp.matvec(x) : matvec_banded(Bpp, x, bw);
    }
    std::vector<double> applyK(const std::vector<double>& x) const {
        return periodic ? cK.matvec(x) : matvec_banded(K, x, bw);
    }
    std::vector<double> applyS(const std::vector<double>& x) const {
        return periodic ? cS.matvec(x) : matvec_banded(S, x, bw);
    }
    std::vector<double> solveB(const std::vector<double>& r) const {
        return periodic ? Bcyc.solve(r) : Bband.solve(r);
    }
    std::vector<double> solveM(const std::vector<double>& r) const {
        return periodic ? Mcyc.solve(r) : Mband.solve(r);
    }

    // --- operators ----------------------------------------------------------
    // BMC same-point derivatives: reduce to coefficients, sample the derivative.
    std::vector<double> bmc_d1(const std::vector<double>& u) const { return applyBp(solveB(u)); }
    std::vector<double> bmc_d2(const std::vector<double>& u) const { return applyBpp(solveB(u)); }

    // BMG Galerkin first derivative: reduce-solve-expand through the mass metric.
    std::vector<double> bmg_d1(const std::vector<double>& u) const {
        return applyB(solveM(applyK(solveB(u))));
    }
    // Zero-allocation periodic version with caller-provided scratch (a, b) and
    // output (y). Thread-safe: the factorization is read-only, so many threads
    // can share one FastDeriv and apply it to independent lines in parallel,
    // each with its own buffers. This is the production hot path.
    void bmg_d1(const std::vector<double>& u, std::vector<double>& y,
                std::vector<double>& a, std::vector<double>& b) const {
        a = u;                             // reduce: a = B^{-1} u
        reduceB(a);
        into_K(a, b);                      // b = K a
        reduceM(b);                        // solve: b = M^{-1} b
        expandB(b, y);                     // expand: y = B b
    }
    // Convenience single-thread overload using the member scratch.
    void bmg_d1(const std::vector<double>& u, std::vector<double>& y) const {
        bmg_d1(u, y, s1, s2);
    }
    // BMG periodic direct second derivative: -B M^{-1} S B^{-1}.
    std::vector<double> bmg_d2_periodic(const std::vector<double>& u) const {
        std::vector<double> y = applyB(solveM(applyS(solveB(u))));
        for (double& v : y) v = -v;
        return y;
    }

    // BMF staggered gradient G = Bhat_eta Delta B^{-1}: primal -> derivative grid.
    std::vector<double> bmf_grad(const std::vector<double>& u) const {
        std::vector<double> a = solveB(u);
        std::vector<double> d(ds.m, 0.0);                 // Delta is bidiagonal
        for (int i = 0; i < ds.m; ++i)
            d[i] = ds.Delta(i, i) * a[i] + ds.Delta(i, i + 1) * a[i + 1];
        return matvec_banded(ds.Bhat, d, p);
    }

    // -----------------------------------------------------------------------
    // Zero-allocation, dispatched (periodic stencil / bounded band) primitives
    // and operators, all taking caller-provided scratch. These are the methods
    // the 3D engine calls per line, in parallel, with per-thread buffers.
    // -----------------------------------------------------------------------
    void reduceB(std::vector<double>& x) const {
        if (periodic) Bcyc.solve_inplace(x); else Bband.solve_inplace(x);
    }
    void reduceM(std::vector<double>& x) const {
        if (periodic) Mcyc.solve_inplace(x); else Mband.solve_inplace(x);
    }
    void expandB(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cB.matvec(x, y); else matvec_banded(B, x, bw, y);
    }
    void into_Bp(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cBp.matvec(x, y); else matvec_banded(Bp, x, bw, y);
    }
    void into_Bpp(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cBpp.matvec(x, y); else matvec_banded(Bpp, x, bw, y);
    }
    void into_K(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cK.matvec(x, y); else matvec_banded(K, x, bw, y);
    }
    void into_S(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cS.matvec(x, y); else matvec_banded(S, x, bw, y);
    }

    // BMC same-point derivatives: reduce, then sample. scratch a (size n).
    void bmc_d1(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a) const {
        a = u; reduceB(a); into_Bp(a, y);
    }
    void bmc_d2(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a) const {
        a = u; reduceB(a); into_Bpp(a, y);
    }
    // Exponential-family member derivatives: reduce through the member left
    // side, then sample. Identical chain to BMC with the member stencils.
    void reduceE(std::vector<double>& x) const { Ecyc.solve_inplace(x); }
    void into_Ep(const std::vector<double>& x, std::vector<double>& y) const { cEp.matvec(x, y); }
    void into_Epp(const std::vector<double>& x, std::vector<double>& y) const { cEpp.matvec(x, y); }
    void exp_d1(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a) const {
        a = u; reduceE(a); into_Ep(a, y);
    }
    void exp_d2(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a) const {
        a = u; reduceE(a); into_Epp(a, y);
    }
    // One solve, both derivatives.
    void exp_d1d2(const std::vector<double>& u, std::vector<double>& d1,
                  std::vector<double>& d2, std::vector<double>& a) const {
        a = u; reduceE(a); into_Ep(a, d1); into_Epp(a, d2);
    }
    // BMG periodic direct second derivative: -B M^{-1} S B^{-1}. scratch a, b.
    void bmg_d2(const std::vector<double>& u, std::vector<double>& y,
                std::vector<double>& a, std::vector<double>& b) const {
        a = u; reduceB(a); into_S(a, b); reduceM(b); expandB(b, y);
        for (double& v : y) v = -v;
    }

    // Apply a centered stencil st[-R..R] (packed st[o+R]) to a periodic line:
    // modulo-free interior loop (vectorizes), wrapped edges only. Keeps every
    // per-line route on the same arithmetic footing as the batched BMC kernel,
    // so wall-clock comparisons measure the schemes, not index arithmetic.
    static void stencil_periodic(const std::vector<double>& u, const double* st,
                                 int R, std::vector<double>& t) {
        const int n = (int)u.size();
        t.resize(n);
        const int lo = R, hi = n - R;
        for (int i = lo; i < hi; ++i) {
            double s = st[R] * u[i];
            for (int o = 1; o <= R; ++o) s += st[R+o]*u[i+o] + st[R-o]*u[i-o];
            t[i] = s;
        }
        for (int i = 0; i < lo; ++i) {
            double s = 0.0;
            for (int o = -R; o <= R; ++o) s += st[o+R]*u[((i+o)%n+n)%n];
            t[i] = s;
        }
        for (int i = hi; i < n; ++i) {
            double s = 0.0;
            for (int o = -R; o <= R; ++o) s += st[o+R]*u[((i+o)%n+n)%n];
            t[i] = s;
        }
    }

    // Explicit centered periodic finite differences (2nd/4th/6th order) -- the FD
    // baselines for the cost comparison (no banded solve; cheap but dispersive).
    void fd2_d1(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih = 1.0/h; y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        for (int i = 0; i < n; ++i) y[i] = ih*0.5*(U(i,1)-U(i,-1));
    }
    void fd2_d2(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0/(h*h); y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        for (int i = 0; i < n; ++i) y[i] = ih2*(U(i,1)-2*U(i,0)+U(i,-1));
    }
    void fd4_d1(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih = 1.0/h; y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c1 = 2.0/3, c2 = -1.0/12;
        for (int i = 0; i < n; ++i) y[i] = ih*(c1*(U(i,1)-U(i,-1)) + c2*(U(i,2)-U(i,-2)));
    }
    void fd4_d2(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0/(h*h); y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c0 = -5.0/2, c1 = 4.0/3, c2 = -1.0/12;
        for (int i = 0; i < n; ++i) y[i] = ih2*(c0*U(i,0) + c1*(U(i,1)+U(i,-1)) + c2*(U(i,2)+U(i,-2)));
    }
    // Compact Pade C6 (periodic): solve LHS f' = RHS u (tridiagonal cyclic solve).
    void compact_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1.data(), 2, t);
        y = L1cyc.solve(t);
    }
    void compact_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2.data(), 2, t);
        y = L2cyc.solve(t);
    }
    // Compact Pade C8 (8th-order tridiagonal): same cyclic-solve path, 7-pt RHS.
    void compact8_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_8.data(), 3, t);
        y = L1cyc8.solve(t);
    }
    void compact8_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_8.data(), 3, t);
        y = L2cyc8.solve(t);
    }
    // DRP (Tam-Webb) explicit first derivative; DRP-family diffusion uses E6 d2.
    void compact10_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_10.data(), 4, t);
        y = L1cyc10.solve(t);
    }
    void compact10_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_10.data(), 4, t);
        y = L2cyc10.solve(t);
    }
    void compact12_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_12.data(), 5, t);
        y = L1cyc12.solve(t);
    }
    void compact12_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_12.data(), 5, t);
        y = L2cyc12.solve(t);
    }
    void drp_d1(const std::vector<double>& u, std::vector<double>& y) const {
        stencil_periodic(u, drp1.data(), 3, y);
    }
    // SLC spectral-like pentadiagonal compact: same cyclic-solve path, 7-pt RHS,
    // bandwidth-2 LHS solve.
    void compactSL_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_sl.data(), 3, t);
        y = L1cycSL.solve(t);
    }
    void compactSL_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_sl.data(), 3, t);
        y = L2cycSL.solve(t);
    }
    // Paper threshold-optimized tridiagonal compact: 11-point RHS (radius 5),
    // bandwidth-1 (tridiagonal) LHS cyclic solve.
    void compactOPT_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_opt.data(), 5, t);
        y = L1cycOPT.solve(t);
    }
    void compactOPT_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_opt.data(), 5, t);
        y = L2cycOPT.solve(t);
    }
    // Certified wide compacts: pentadiagonal (bandwidth-2) LHS cyclic
    // solve; 15-point RHS (cw12a, radius 7) / 11-point RHS (cw12b, radius 5).
    void compactCWA_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_cwa.data(), 7, t);
        y = L1cycCWA.solve(t);
    }
    void compactCWA_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_cwa.data(), 7, t);
        y = L2cycCWA.solve(t);
    }
    void compactCWB_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr1_cwb.data(), 5, t);
        y = L1cycCWB.solve(t);
    }
    void compactCWB_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t;
        stencil_periodic(u, cr2_cwb.data(), 5, t);
        y = L2cycCWB.solve(t);
    }
    void fd6_d1(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih = 1.0 / h; y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c1 = 3.0/4, c2 = -3.0/20, c3 = 1.0/60;
        for (int i = 0; i < n; ++i)
            y[i] = ih*(c1*(U(i,1)-U(i,-1)) + c2*(U(i,2)-U(i,-2)) + c3*(U(i,3)-U(i,-3)));
    }
    void fd6_d2(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0 / (h*h); y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c0 = -49.0/18, c1 = 3.0/2, c2 = -3.0/20, c3 = 1.0/90;
        for (int i = 0; i < n; ++i)
            y[i] = ih2*(c0*U(i,0) + c1*(U(i,1)+U(i,-1)) + c2*(U(i,2)+U(i,-2)) + c3*(U(i,3)+U(i,-3)));
    }

    // -----------------------------------------------------------------------
    // Batched periodic BMC apply: W lines at once (in, out, scratch are n x W
    // row-major). The reduce is one circulant block solve; the expand is one
    // block matvec. This is the cache- and SIMD-efficient path for the strided
    // (y, z) directions of a 3D field, where W adjacent lines are contiguous.
    // -----------------------------------------------------------------------
    void bmc_d2_block(const double* in, double* out, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Bcyc.solve_block(sc, W);
        cBpp.matvec_block(sc, out, n, W);
    }
    void bmc_d1_block(const double* in, double* out, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Bcyc.solve_block(sc, W);
        cBp.matvec_block(sc, out, n, W);
    }
    // One coefficient solve feeds both derivatives (advection + diffusion).
    void bmc_d1d2_block(const double* in, double* d1, double* d2, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Bcyc.solve_block(sc, W);
        cBp.matvec_block(sc, d1, n, W);
        cBpp.matvec_block(sc, d2, n, W);
    }
    // Same batched chain through the exponential-family member left side.
    void exp_d1_block(const double* in, double* out, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Ecyc.solve_block(sc, W);
        cEp.matvec_block(sc, out, n, W);
    }
    void exp_d2_block(const double* in, double* out, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Ecyc.solve_block(sc, W);
        cEpp.matvec_block(sc, out, n, W);
    }
    void exp_d1d2_block(const double* in, double* d1, double* d2, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Ecyc.solve_block(sc, W);
        cEp.matvec_block(sc, d1, n, W);
        cEpp.matvec_block(sc, d2, n, W);
    }
};

#endif // BM_APPLY_H
