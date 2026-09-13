// exp_eval.h
//
// Pointwise phi(x), phi'(x) and phi''(x) of ONE exponential-family generator
// E_{m,r}(omega) at arbitrary real x, for the solver's off-grid path. This is
// the C++ twin of code/analysis/exponential_family/evaluator.py and it reads
// the same member JSON, using the "eval_*" block that gate5_evaluator.py adds
// beside the lattice samples. The lattice-sample block is untouched, so a file
// without the "eval_*" block still drives every existing route and only the
// off-grid evaluator refuses to load.
//
// Representation. The generator is supported on [-n/2, n/2] with
// n = m + 2 r. On the cell y = x + n/2 in [J, J+1), with xi = y - J, it is the
// exponential polynomial
//
//   phi(x) = sum_k a_k(J) xi^k
//          + sum_{g,b} xi^b [ C_{g,b}(J) cos(w_g xi) + S_{g,b}(J) sin(w_g xi) ],
//
// where g runs over the distinct pair frequencies and b over 0..mu_g - 1 with
// mu_g the confluent multiplicity. The Python side assembles a_k, C and S once
// in extended precision from the localized Green function, folds the cubic
// Taylor jet of every oscillatory term at xi = 0 into a_k, and writes the
// three flat tables. What is left here is cancellation free: the remainders
//
//   C_2(z) = 1 - cos z,          S_3(z) = z - sin z,
//   C_4(z) = cos z - 1 + z^2/2,  S_5(z) = sin z - z + z^3/6
//
// are evaluated by their series near zero, and dS_5/dz = C_4, dC_4/dz = S_3,
// dS_3/dz = C_2 give the two derivatives from the same four functions:
//
//   phi   = poly(xi)   + sum xi^b [ C C_4 + S S_5 ],
//   phi'  = poly'(xi)  + sum w xi^b [ C S_3 + S C_4 ] + Leibniz terms of xi^b,
//   phi'' = poly''(xi) + sum w^2 xi^b [ C C_2 + S S_3 ] + Leibniz terms.
//
// phi and both derivatives are exactly zero outside the open support.
//
// Block version 2 (field "eval_version", absent means 1) widens the multiset to
// the Gate 7 real pairs +-alpha and quads +-alpha +- i omega and carries the
// Gate 8 polynomial dial psi_hat = P(xi^2) phi_hat. A group is then one root
// rho = alpha + i omega with omega >= 0 (field "eval_group_alpha" beside
// "eval_group_omega"), the jet folded into the cell polynomial runs to the
// absolute order "eval_fold_order" = K, and what is left are the entire
// functions
//
//   R_j(z) = sum_{i >= j} z^i / i!,   R_j = exp for j <= 0,
//
// which vanish to order j at the origin:
//
//   psi(x) = sum_{k<K} a_k(J) xi^k
//          + sum_{g,b} xi^b [ C_{g,b} Re R_{K-b}(rho xi) + S_{g,b} Im R_{K-b}(rho xi) ].
//
// Derivatives follow from d/dxi R_j(rho xi) = rho R_{j-1}(rho xi) with the
// Leibniz rule for xi^b, and R_{j-1} = R_j + z^{j-1}/(j-1)! builds the lower
// indices from the higher one by addition alone. A version 1 file has no alpha
// and no fold order and takes the version 1 branch verbatim, so files written
// before the extension load and evaluate exactly as before.

#ifndef EXP_EVAL_H
#define EXP_EVAL_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "exp_member.h"

// Threshold and term count of the series branch. They match the Python module
// exactly so the two evaluators agree to the last few bits.
static const double EXP_EVAL_SERIES_LIMIT = 4.0;
static const int    EXP_EVAL_SERIES_TERMS = 26;

inline double exp_eval_C2(double z) {
    const double s = std::sin(0.5 * z);
    return 2.0 * s * s;
}

inline double exp_eval_S3(double z) {
    if (std::fabs(z) > EXP_EVAL_SERIES_LIMIT) return z - std::sin(z);
    const double z2 = z * z;
    double term = z2 * z / 6.0, total = term;
    for (int k = 1; k < EXP_EVAL_SERIES_TERMS; ++k) {
        term = -term * z2 / (double)((2 * k + 2) * (2 * k + 3));
        total += term;
    }
    return total;
}

inline double exp_eval_C4(double z) {
    if (std::fabs(z) > EXP_EVAL_SERIES_LIMIT) return 0.5 * z * z - exp_eval_C2(z);
    const double z2 = z * z;
    double term = z2 * z2 / 24.0, total = term;
    for (int k = 2; k < EXP_EVAL_SERIES_TERMS; ++k) {
        term = -term * z2 / (double)((2 * k + 1) * (2 * k + 2));
        total += term;
    }
    return total;
}

inline double exp_eval_S5(double z) {
    if (std::fabs(z) > EXP_EVAL_SERIES_LIMIT) return std::sin(z) - z + z * z * z / 6.0;
    const double z2 = z * z;
    double term = z2 * z2 * z / 120.0, total = term;
    for (int k = 2; k < EXP_EVAL_SERIES_TERMS; ++k) {
        term = -term * z2 / (double)((2 * k + 2) * (2 * k + 3));
        total += term;
    }
    return total;
}

// The derivative chain S_5 -> C_4 -> S_3 -> C_2 -> sin -> cos -> -sin -> -cos,
// indexed from C_4 (base 0) or from S_5 (base 1). Orders 0, 1 and 2 are the
// ones the three channels use; the tail is carried so the chain stays the same
// function as the one the Python module exposes.
inline double exp_eval_chain(double z, int base, int order) {
    const int j = order + (1 - base);
    if (j == 0) return exp_eval_S5(z);
    if (j == 1) return exp_eval_C4(z);
    if (j == 2) return exp_eval_S3(z);
    if (j == 3) return exp_eval_C2(z);
    switch ((j - 4) % 4) {
        case 0:  return  std::sin(z);
        case 1:  return  std::cos(z);
        case 2:  return -std::sin(z);
        default: return -std::cos(z);
    }
}

// Block version 2: R_j(u + i v) = sum_{i >= j} z^i / i!. Below the limit the
// sum is taken from its own first term; above it the equivalent form exp(z)
// minus the truncated exponential series is used. The term count and the limit
// match general_evaluator.py exactly.
static const double EXP_EVAL_R_LIMIT = 4.0;
static const int    EXP_EVAL_R_TERMS = 48;

inline void exp_eval_R(int j, double u, double v, double& re, double& im) {
    if (j <= 0) {
        const double amplitude = std::exp(u);
        re = amplitude * std::cos(v);
        im = amplitude * std::sin(v);
        return;
    }
    double tre = 1.0, tim = 0.0, nre = 0.0, nim = 0.0;
    for (int i = 1; i <= j; ++i) {
        nre = (tre * u - tim * v) / (double)i;
        nim = (tre * v + tim * u) / (double)i;
        tre = nre; tim = nim;
    }
    double sre = tre, sim = tim;
    for (int i = j + 1; i <= j + EXP_EVAL_R_TERMS; ++i) {
        nre = (tre * u - tim * v) / (double)i;
        nim = (tre * v + tim * u) / (double)i;
        tre = nre; tim = nim;
        sre += tre; sim += tim;
    }
    if (u * u + v * v <= EXP_EVAL_R_LIMIT * EXP_EVAL_R_LIMIT) { re = sre; im = sim; return; }
    double pre = 1.0, pim = 0.0, are = 1.0, aim = 0.0;
    for (int i = 1; i < j; ++i) {
        nre = (pre * u - pim * v) / (double)i;
        nim = (pre * v + pim * u) / (double)i;
        pre = nre; pim = nim;
        are += pre; aim += pim;
    }
    const double amplitude = std::exp(u);
    re = amplitude * std::cos(v) - are;
    im = amplitude * std::sin(v) - aim;
}

// Fill R_j, R_{j-1}, ..., R_{j-depth} at u + i v, the lower indices by the
// addition R_{j-1} = R_j + z^{j-1}/(j-1)!.
inline void exp_eval_R_block(int j, double u, double v, int depth,
                             double* re, double* im) {
    exp_eval_R(j, u, v, re[0], im[0]);
    for (int step = 1; step <= depth; ++step) {
        const int index = j - step;
        if (index < 0) { re[step] = re[step - 1]; im[step] = im[step - 1]; continue; }
        double mre = 1.0, mim = 0.0, nre = 0.0, nim = 0.0;
        for (int i = 1; i <= index; ++i) {
            nre = (mre * u - mim * v) / (double)i;
            nim = (mre * v + mim * u) / (double)i;
            mre = nre; mim = nim;
        }
        re[step] = re[step - 1] + mre;
        im[step] = im[step - 1] + mim;
    }
}

struct ExpEval {
    int n = 0;                       // support width, phi lives on [-n/2, n/2]
    int cells = 0;                   // number of unit cells, equal to n
    int deg = 0;                     // degree of the cell polynomial
    int groups = 0;                  // distinct pair frequencies (version 2: roots)
    int terms = 0;                   // sum of the confluent multiplicities
    int version = 1;                 // evaluator block version
    int fold = 0;                    // version 2: absolute fold order K
    std::vector<double> alpha;       // version 2: real part of the root per group
    std::vector<double> omega;       // one frequency per group
    std::vector<int>    mult;        // confluent multiplicity per group
    std::vector<int>    slot;        // first column of the group in cos/sin
    std::vector<double> poly;        // cells * (deg + 1), cell major
    std::vector<double> cosc;        // cells * terms, cell major
    std::vector<double> sinc;        // cells * terms, cell major
    bool loaded = false;

    double derivative_v2(int J, double xi, int order) const {
        double value = 0.0;
        const double* a = &poly[(size_t)J * (size_t)(deg + 1)];
        for (int k = order; k <= deg; ++k) {
            double fall = 1.0;
            for (int t = 0; t < order; ++t) fall *= (double)(k - t);
            value += fall * a[k] * std::pow(xi, (double)(k - order));
        }
        double bre[8], bim[8];
        for (int g = 0; g < groups; ++g) {
            const double u = alpha[g] * xi;
            const double v = omega[g] * xi;
            for (int b = 0; b < mult[g]; ++b) {
                const size_t col = (size_t)J * (size_t)terms + (size_t)(slot[g] + b);
                const double C = cosc[col], S = sinc[col];
                exp_eval_R_block(fold - b, u, v, order, bre, bim);
                const int imax = (order < b) ? order : b;
                for (int i = 0; i <= imax; ++i) {
                    double fall = 1.0;
                    for (int t = 0; t < i; ++t) fall *= (double)(b - t);
                    if (fall == 0.0) continue;
                    const int d = order - i;
                    double pre = 1.0, pim = 0.0;
                    for (int t = 0; t < d; ++t) {
                        const double q = pre * alpha[g] - pim * omega[g];
                        pim = pre * omega[g] + pim * alpha[g];
                        pre = q;
                    }
                    const double gre = pre * bre[d] - pim * bim[d];
                    const double gim = pre * bim[d] + pim * bre[d];
                    double binom = 1.0;
                    for (int t = 0; t < i; ++t) binom = binom * (double)(order - t) / (double)(t + 1);
                    value += binom * fall * std::pow(xi, (double)(b - i)) * (C * gre + S * gim);
                }
            }
        }
        return value;
    }

    double derivative(double x, int order) const {
        const double y = x + 0.5 * (double)n;
        if (!(y > 0.0) || !(y < (double)n)) return 0.0;
        int J = (int)std::floor(y);
        if (J < 0) J = 0;
        if (J > cells - 1) J = cells - 1;
        const double xi = y - (double)J;
        if (version >= 2) return derivative_v2(J, xi, order);

        double value = 0.0;
        const double* a = &poly[(size_t)J * (size_t)(deg + 1)];
        for (int k = order; k <= deg; ++k) {
            double fall = 1.0;
            for (int t = 0; t < order; ++t) fall *= (double)(k - t);
            value += fall * a[k] * std::pow(xi, (double)(k - order));
        }
        for (int g = 0; g < groups; ++g) {
            const double w = omega[g];
            const double z = w * xi;
            for (int b = 0; b < mult[g]; ++b) {
                const size_t col = (size_t)J * (size_t)terms + (size_t)(slot[g] + b);
                const double C = cosc[col], S = sinc[col];
                const int imax = (order < b) ? order : b;
                for (int i = 0; i <= imax; ++i) {
                    double fall = 1.0;
                    for (int t = 0; t < i; ++t) fall *= (double)(b - t);
                    if (fall == 0.0) continue;
                    double binom = 1.0;
                    for (int t = 0; t < i; ++t) binom = binom * (double)(order - t) / (double)(t + 1);
                    const double weight = binom * fall * std::pow(w, (double)(order - i));
                    const double chain = C * exp_eval_chain(z, 0, order - i)
                                       + S * exp_eval_chain(z, 1, order - i);
                    value += weight * std::pow(xi, (double)(b - i)) * chain;
                }
            }
        }
        return value;
    }

    double value(double x) const { return derivative(x, 0); }
    double d1(double x)    const { return derivative(x, 1); }
    double d2(double x)    const { return derivative(x, 2); }
};

// Read the "eval_*" block of a member file. The lattice-sample block is read by
// exp_member_parse and is not duplicated here.
inline bool exp_eval_parse(const char* path, ExpEval& ev, const char*& why) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { why = "exp member file cannot be opened"; return false; }
    std::string text;
    char buf[4096];
    size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, got);
    std::fclose(f);
    if (text.empty()) { why = "exp member file is empty"; return false; }

    if (!exp_json_int(text, "eval_support_n", ev.n) ||
        !exp_json_int(text, "eval_cells", ev.cells) ||
        !exp_json_int(text, "eval_poly_degree", ev.deg) ||
        !exp_json_int(text, "eval_group_count", ev.groups) ||
        !exp_json_int(text, "eval_term_count", ev.terms)) {
        why = "exp member file has no eval_* evaluator block";
        return false;
    }
    if (!exp_json_int(text, "eval_version", ev.version)) ev.version = 1;
    std::vector<double> mults;
    exp_json_array(text, "eval_group_omega", ev.omega);
    exp_json_array(text, "eval_group_multiplicity", mults);
    if (ev.version >= 2) {
        if (!exp_json_array(text, "eval_group_alpha", ev.alpha) ||
            !exp_json_int(text, "eval_fold_order", ev.fold)) {
            why = "exp eval block version 2 needs eval_group_alpha and eval_fold_order";
            return false;
        }
        if ((int)ev.alpha.size() != ev.groups || ev.fold != ev.deg + 1) {
            why = "exp eval block version 2 has an inconsistent group or fold shape";
            return false;
        }
    } else {
        ev.alpha.assign(ev.groups, 0.0);
        ev.fold = 0;
    }
    if (!exp_json_array(text, "eval_poly", ev.poly) ||
        !exp_json_array(text, "eval_cos", ev.cosc) ||
        !exp_json_array(text, "eval_sin", ev.sinc)) {
        why = "exp member file has an incomplete eval_* evaluator block";
        return false;
    }
    if (ev.n < 2 || ev.cells != ev.n || ev.deg < 0) { why = "exp eval block has bad shape"; return false; }
    if ((int)ev.omega.size() != ev.groups || (int)mults.size() != ev.groups) {
        why = "exp eval block frequency arrays do not match the group count";
        return false;
    }
    ev.mult.assign(ev.groups, 0);
    ev.slot.assign(ev.groups, 0);
    int running = 0;
    for (int g = 0; g < ev.groups; ++g) {
        ev.mult[g] = (int)(mults[g] + 0.5);
        ev.slot[g] = running;
        running += ev.mult[g];
    }
    if (running != ev.terms) { why = "exp eval block multiplicities do not sum to the term count"; return false; }
    if ((int)ev.poly.size() != ev.cells * (ev.deg + 1)) { why = "exp eval poly table has the wrong length"; return false; }
    if ((int)ev.cosc.size() != ev.cells * ev.terms) { why = "exp eval cos table has the wrong length"; return false; }
    if ((int)ev.sinc.size() != ev.cells * ev.terms) { why = "exp eval sin table has the wrong length"; return false; }
    ev.loaded = true;
    return true;
}

// Process-wide evaluator, loaded from the same path as the member samples.
inline ExpEval& exp_eval_store() { static ExpEval ev; return ev; }

inline bool exp_eval_ready(const char*& why) {
    ExpEval& ev = exp_eval_store();
    if (ev.loaded) return true;
    std::string path = exp_member_path_store();
    if (path.empty()) {
        const char* env = std::getenv("EXP_MEMBER_JSON");
        if (env) path = env;
    }
    if (path.empty()) {
        why = "the off-grid evaluator needs a member file (--exp-member <path> or EXP_MEMBER_JSON)";
        return false;
    }
    return exp_eval_parse(path.c_str(), ev, why);
}

#endif // EXP_EVAL_H
