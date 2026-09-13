// exp_eval_check.cpp -- verify the C++ off-grid evaluator of the exponential
// family against the Python one, point by point.
//
// The Python producer (code/analysis/exponential_family/gate5_evaluator.py)
// writes a table of 10,000 rows
//
//     x, phi(x), phi'(x), phi''(x)
//
// computed by code/analysis/exponential_family/evaluator.py for the member
// named on the command line, together with the "eval_*" block of that member
// file. This program rebuilds phi, phi' and phi'' from the JSON block alone and
// prints the largest deviation in each channel. Requirement of Gate 5 section
// 5.0 check (vi): at most 1e-14.
//
// The same program closes check (vi) of Gate 9 section E4 for a Gate 8 member,
// whose evaluator block is version 2 and whose producer is
// code/analysis/exponential_family/gate9_evaluator.py; the block version is
// read from the file and printed in the banner.
//
//   ./exp_eval_check [--exp-member <member.json>] [--table <table.csv>]
//
// The defaults are the champion member and the Gate 5 table, both relative to
// code/solver, so the program runs with no arguments from its own directory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "exp_eval.h"

static const double TOL = 1e-14;
static const char* DEFAULT_MEMBER = "../../results/exponential_family/members/E62_champion.json";
static const char* DEFAULT_TABLE  = "../../results/exponential_family/gate5/exp_eval_table.csv";

int main(int argc, char** argv) {
    const char* table_path = DEFAULT_TABLE;
    const char* member_path = DEFAULT_MEMBER;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--exp-member") member_path = argv[i + 1];
        if (std::string(argv[i]) == "--table") table_path = argv[i + 1];
    }
    exp_member_set_path(member_path);

    const char* why = "";
    if (!exp_eval_ready(why)) {
        std::printf("EXPEVAL: cannot load evaluator block (%s)\n", why);
        return EXIT_FAILURE;
    }
    const ExpEval& ev = exp_eval_store();
    std::printf("EXPEVAL: member %s  block version=%d n=%d cells=%d poly degree=%d "
                "groups=%d terms=%d fold=%d\n",
                member_path, ev.version, ev.n, ev.cells, ev.deg, ev.groups, ev.terms, ev.fold);

    FILE* f = std::fopen(table_path, "r");
    if (!f) {
        std::printf("EXPEVAL: cannot open table %s\n", table_path);
        return EXIT_FAILURE;
    }
    char line[512];
    long rows = 0;
    double worst[3] = {0.0, 0.0, 0.0};
    double worst_x[3] = {0.0, 0.0, 0.0};
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == 'x' || line[0] == '\n') continue;
        double x = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0;
        if (std::sscanf(line, "%lf,%lf,%lf,%lf", &x, &v0, &v1, &v2) != 4) continue;
        const double ref[3] = {v0, v1, v2};
        for (int q = 0; q < 3; ++q) {
            const double d = std::fabs(ev.derivative(x, q) - ref[q]);
            if (d > worst[q]) { worst[q] = d; worst_x[q] = x; }
        }
        ++rows;
    }
    std::fclose(f);

    if (rows == 0) {
        std::printf("EXPEVAL: table %s holds no usable rows\n", table_path);
        return EXIT_FAILURE;
    }
    const double overall = std::fmax(worst[0], std::fmax(worst[1], worst[2]));
    std::printf("EXPEVAL: rows %ld\n", rows);
    for (int q = 0; q < 3; ++q)
        std::printf("EXPEVAL: max|C++ - Python| d%d %.6e at x = %+.15f\n", q, worst[q], worst_x[q]);
    std::printf("EXPEVAL: max deviation %.6e (tolerance %.0e) -> %s\n",
                overall, TOL, overall <= TOL ? "PASS" : "FAIL");
    return overall <= TOL ? EXIT_SUCCESS : EXIT_FAILURE;
}
