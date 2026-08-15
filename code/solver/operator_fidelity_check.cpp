// operator_fidelity_check.cpp
//
// Deterministic production-path checks for the configured periodic derivative
// pair and the one-dimensional pressure-symbol zero set. The output is a
// comma-delimited receipt consumed by build_operator_fidelity_ledger.py.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bm3d.h"

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

double max_abs_difference(const std::vector<double>& a,
                          const std::vector<double>& b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

double max_abs(const std::vector<double>& a) {
    double result = 0.0;
    for (double value : a) result = std::max(result, std::abs(value));
    return result;
}

void print_fidelity(const char* name, bool pass, double value, double limit) {
    std::printf("FIDELITY,%s,%d,%.17e,%.17e\n",
                name, pass ? 1 : 0, value, limit);
}

bool check_config_and_second_derivative() {
    const int n = 64;
    const double length = 2.0 * PI;
    bool pass = true;

    SimConfig bmc1;
    bmc1.p = 1;
    bmc1.route = Route::BMC;
    const char* why = nullptr;
    const bool bmc1_rejected = !bmc1.validate(why);
    print_fidelity("bmc1_rejected", bmc1_rejected,
                   bmc1_rejected ? 0.0 : 1.0, 0.0);
    pass = pass && bmc1_rejected;

    SimConfig bmg1;
    bmg1.p = 1;
    bmg1.route = Route::BMG;
    why = nullptr;
    const bool bmg1_admitted = bmg1.validate(why);
    print_fidelity("bmg1_admitted", bmg1_admitted,
                   bmg1_admitted ? 0.0 : 1.0, 0.0);
    pass = pass && bmg1_admitted;

    SimConfig configured_case;
    configured_case.p = 2;
    configured_case.route = Route::BMG;
    configured_case.ax[0] = {n, length, BC::Periodic, true};
    configured_case.ax[1] = {n, length, BC::Periodic, true};
    configured_case.ax[2] = {n, length, BC::Periodic, false};
    Deriv3D configured(configured_case);
    const bool dispatch = configured.opkind(1) == BMG_D1 &&
                          configured.opkind(2) == BMG_D2;
    print_fidelity("bmg_derivative_pair_dispatch", dispatch,
                   dispatch ? 0.0 : 1.0, 0.0);
    pass = pass && dispatch;

    std::vector<double> input(n);
    for (int j = 0; j < n; ++j) {
        const double x = 2.0 * PI * static_cast<double>(j) / n;
        input[j] = std::sin(17.0 * x) + 0.2 * std::cos(23.0 * x);
    }

    FastDeriv degree2 = FastDeriv::build(true, n, 2, length);
    FastDeriv degree5 = FastDeriv::build(true, n, 5, length);
    std::vector<double> bmg_d1(n), bmg_d2(n), bmc5_d1(n), bmc5_d2(n),
                        bmc2_d2(n), scratch_a(n), scratch_b(n);
    degree2.bmg_d1(input, bmg_d1, scratch_a, scratch_b);
    degree2.bmg_d2(input, bmg_d2, scratch_a, scratch_b);
    degree5.bmc_d1(input, bmc5_d1, scratch_a);
    degree5.bmc_d2(input, bmc5_d2, scratch_a);
    degree2.bmc_d2(input, bmc2_d2, scratch_a);

    Field3D field(n, n, 1), field_d2(n, n, 1);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) field.a[field.idx(i, j, 0)] = input[i];
    }
    configured.deriv(field, field_d2, 0, 2);

    std::vector<double> configured_first_line(n);
    for (int i = 0; i < n; ++i) configured_first_line[i] = field_d2.a[i];

    const double dispatch_error = max_abs_difference(configured_first_line, bmg_d2);
    const double collapse_d1_error = max_abs_difference(bmg_d1, bmc5_d1);
    const double collapse_d2_error = max_abs_difference(bmg_d2, bmc5_d2);
    const double wrong_pair_separation = max_abs_difference(bmg_d2, bmc2_d2) /
        std::max(max_abs(bmg_d2), 1.0e-300);

    print_fidelity("configured_bmg_d2_matches_native", dispatch_error < 2.0e-11,
                   dispatch_error, 2.0e-11);
    print_fidelity("bmg2_bmc5_d1_collapse", collapse_d1_error < 2.0e-11,
                   collapse_d1_error, 2.0e-11);
    print_fidelity("bmg2_bmc5_d2_collapse", collapse_d2_error < 2.0e-10,
                   collapse_d2_error, 2.0e-10);
    print_fidelity("bmg2_not_bmc2_d2", wrong_pair_separation > 1.0e-3,
                   wrong_pair_separation, 1.0e-3);
    pass = pass && dispatch_error < 2.0e-11;
    pass = pass && collapse_d1_error < 2.0e-11;
    pass = pass && collapse_d2_error < 2.0e-10;
    pass = pass && wrong_pair_separation > 1.0e-3;
    return pass;
}

struct CensusCase {
    std::string label;
    Route route;
    int degree;
    int n;
};

std::vector<CensusCase> census_cases() {
    std::vector<CensusCase> cases;
    const std::vector<std::pair<const char*, Route>> controls = {
        {"E2", Route::FD2}, {"E4", Route::FD4}, {"E6", Route::FD6},
        {"C6", Route::C6}, {"C8", Route::C8}, {"C10", Route::C10},
        {"C12", Route::C12}, {"DRP", Route::DRP}, {"SL", Route::SL},
        {"OPT", Route::OPT}, {"CW12A", Route::CW12A},
        {"CW12B", Route::CW12B},
    };
    for (int n : {64, 128, 256}) {
        for (int degree = 2; degree <= 13; ++degree) {
            cases.push_back({"BMC" + std::to_string(degree),
                             Route::BMC, degree, n});
        }
        for (int degree = 1; degree <= 5; ++degree) {
            cases.push_back({"BMG" + std::to_string(degree),
                             Route::BMG, degree, n});
        }
        for (const auto& item : controls) {
            cases.push_back({item.first, item.second, 5, n});
        }
    }
    return cases;
}

void apply_first_derivative(const FastDeriv& derivative, Route route,
                            const std::vector<double>& input,
                            std::vector<double>& output,
                            std::vector<double>& scratch_a,
                            std::vector<double>& scratch_b) {
    switch (route) {
        case Route::BMC: derivative.bmc_d1(input, output, scratch_a); break;
        case Route::BMG: derivative.bmg_d1(input, output, scratch_a, scratch_b); break;
        case Route::FD2: derivative.fd2_d1(input, output); break;
        case Route::FD4: derivative.fd4_d1(input, output); break;
        case Route::FD6: derivative.fd6_d1(input, output); break;
        case Route::C6: derivative.compact_d1(input, output); break;
        case Route::C8: derivative.compact8_d1(input, output); break;
        case Route::C10: derivative.compact10_d1(input, output); break;
        case Route::C12: derivative.compact12_d1(input, output); break;
        case Route::DRP: derivative.drp_d1(input, output); break;
        case Route::SL: derivative.compactSL_d1(input, output); break;
        case Route::OPT: derivative.compactOPT_d1(input, output); break;
        case Route::CW12A: derivative.compactCWA_d1(input, output); break;
        case Route::CW12B: derivative.compactCWB_d1(input, output); break;
    }
}

bool run_census_case(const CensusCase& test) {
    const double length = 2.0 * PI;
    FastDeriv derivative = FastDeriv::build(true, test.n, test.degree, length);
    std::vector<double> cosine(test.n), sine(test.n),
                        dcosine(test.n), dsine(test.n),
                        scratch_a(test.n), scratch_b(test.n);
    std::vector<double> amplitude(test.n, 0.0);

    for (int mode = 0; mode < test.n; ++mode) {
        for (int j = 0; j < test.n; ++j) {
            const double phase = 2.0 * PI * mode * j / test.n;
            cosine[j] = std::cos(phase);
            sine[j] = std::sin(phase);
        }
        apply_first_derivative(derivative, test.route, cosine, dcosine,
                               scratch_a, scratch_b);
        apply_first_derivative(derivative, test.route, sine, dsine,
                               scratch_a, scratch_b);
        double numerator = 0.0;
        double denominator = 0.0;
        for (int j = 0; j < test.n; ++j) {
            numerator += dcosine[j] * dcosine[j] + dsine[j] * dsine[j];
            denominator += cosine[j] * cosine[j] + sine[j] * sine[j];
        }
        amplitude[mode] = std::sqrt(numerator / std::max(denominator, 1.0e-300));
    }

    const double maximum = *std::max_element(amplitude.begin(), amplitude.end());
    const double cut = 1.0e-10 * maximum;
    std::vector<int> zeros;
    double minimum_gap = 1.0;
    for (int mode = 0; mode < test.n; ++mode) {
        if (amplitude[mode] <= cut) {
            zeros.push_back(mode);
        } else {
            minimum_gap = std::min(minimum_gap, amplitude[mode] / maximum);
        }
    }
    const bool pass = zeros.size() == 2 && zeros[0] == 0 &&
                      zeros[1] == test.n / 2;
    const int zero0 = zeros.empty() ? -1 : zeros[0];
    const int zero1 = zeros.size() < 2 ? -1 : zeros[1];
    std::printf("CENSUS,%s,%d,%d,%d,%d,%d,%.17e,%.17e,%.17e,%d\n",
                test.label.c_str(), test.degree, test.n,
                static_cast<int>(zeros.size()), zero0, zero1,
                cut, maximum, minimum_gap, pass ? 1 : 0);
    return pass;
}

}  // namespace

int main() {
    bool pass = check_config_and_second_derivative();
    for (const CensusCase& test : census_cases()) {
        pass = run_census_case(test) && pass;
    }
    std::printf("SUMMARY,%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
