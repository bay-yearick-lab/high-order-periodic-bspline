// exp_member.h
//
// Runtime description of ONE exponential-family generator E_{m,r}(omega), read
// from a JSON member file. The file carries the lattice samples
//
//     beta(n),  beta'(n),  beta''(n),      n = 0, 1, ..., R,
//
// of a generator supported on [-(p+1)/2, (p+1)/2] with R = floor((p+1)/2).
// beta is even, beta' is odd and beta'' is even, so the one-sided block
// determines every entry of the periodic route: the left side is the symmetric
// cyclic banded matrix with entries beta(|n|), the first-derivative right side
// is the antisymmetric stencil (1/h) beta'(n) and the second-derivative right
// side is the symmetric stencil (1/h^2) beta''(|n|).
//
// The file path comes from the command line (--exp-member <path>) or from the
// environment variable EXP_MEMBER_JSON. It is loaded once and shared by every
// axis, because the route is the same along all three directions.
//
// The parser handles exactly this flat schema (string, integers, doubles, flat
// arrays of doubles). It allocates no exceptions, matching the solver build.

#ifndef EXP_MEMBER_H
#define EXP_MEMBER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ExpMember {
    std::string name;
    int    m = 0, r = 0, p = 0, radius = 0;
    double cond = 0.0;
    std::vector<double> omegas;
    std::vector<double> beta, dbeta, ddbeta;   // one-sided samples n = 0..radius
    bool   loaded = false;
};

// --- minimal flat-object JSON scanning ------------------------------------
// The value of "key" starts just past the first ':' following the quoted key.
// Keys are distinct as quoted tokens ("beta" never matches inside "dbeta"),
// so a plain substring search is unambiguous for this schema.
inline const char* exp_json_value(const std::string& s, const char* key) {
    const std::string pat = std::string("\"") + key + "\"";
    const size_t k = s.find(pat);
    if (k == std::string::npos) return nullptr;
    const size_t c = s.find(':', k + pat.size());
    if (c == std::string::npos) return nullptr;
    return s.c_str() + c + 1;
}
inline bool exp_json_double(const std::string& s, const char* key, double& out) {
    const char* v = exp_json_value(s, key);
    if (!v) return false;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v) return false;
    out = x;
    return true;
}
inline bool exp_json_int(const std::string& s, const char* key, int& out) {
    double x = 0.0;
    if (!exp_json_double(s, key, x)) return false;
    out = (int)x;
    return true;
}
inline bool exp_json_string(const std::string& s, const char* key, std::string& out) {
    const char* v = exp_json_value(s, key);
    if (!v) return false;
    while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') ++v;
    if (*v != '"') return false;
    ++v;
    const char* e = std::strchr(v, '"');
    if (!e) return false;
    out.assign(v, (size_t)(e - v));
    return true;
}
inline bool exp_json_array(const std::string& s, const char* key, std::vector<double>& out) {
    const char* v = exp_json_value(s, key);
    if (!v) return false;
    while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') ++v;
    if (*v != '[') return false;
    ++v;
    out.clear();
    for (;;) {
        while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r' || *v == ',') ++v;
        if (*v == ']') return true;
        if (*v == '\0') return false;
        char* end = nullptr;
        const double x = std::strtod(v, &end);
        if (end == v) return false;
        out.push_back(x);
        v = end;
    }
}

// --- process-wide member store --------------------------------------------
inline std::string& exp_member_path_store() { static std::string path; return path; }
inline void exp_member_set_path(const char* p) { exp_member_path_store() = (p ? p : ""); }

inline bool exp_member_parse(const char* path, ExpMember& mem, const char*& why) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { why = "exp member file cannot be opened"; return false; }
    std::string text;
    char buf[4096];
    size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, got);
    std::fclose(f);
    if (text.empty()) { why = "exp member file is empty"; return false; }

    exp_json_string(text, "name", mem.name);
    exp_json_int(text, "m", mem.m);
    exp_json_int(text, "r", mem.r);
    exp_json_int(text, "p", mem.p);
    exp_json_double(text, "cond", mem.cond);
    exp_json_array(text, "omegas", mem.omegas);
    if (!exp_json_int(text, "radius", mem.radius)) { why = "exp member file has no radius"; return false; }
    if (!exp_json_array(text, "beta", mem.beta))     { why = "exp member file has no beta array"; return false; }
    if (!exp_json_array(text, "dbeta", mem.dbeta))   { why = "exp member file has no dbeta array"; return false; }
    if (!exp_json_array(text, "ddbeta", mem.ddbeta)) { why = "exp member file has no ddbeta array"; return false; }
    if (mem.radius < 1) { why = "exp member radius must be >= 1"; return false; }
    const size_t want = (size_t)mem.radius + 1;
    if (mem.beta.size() != want || mem.dbeta.size() != want || mem.ddbeta.size() != want) {
        why = "exp member sample arrays must each hold radius+1 one-sided values";
        return false;
    }
    if (mem.beta[0] <= 0.0) { why = "exp member beta(0) must be positive"; return false; }
    mem.loaded = true;
    return true;
}

inline ExpMember& exp_member_store() { static ExpMember mem; return mem; }

// Load on first use from the configured path, else from EXP_MEMBER_JSON.
// Returns false with a reason if no usable member file is available.
inline bool exp_member_ready(const char*& why) {
    ExpMember& mem = exp_member_store();
    if (mem.loaded) return true;
    std::string path = exp_member_path_store();
    if (path.empty()) {
        const char* env = std::getenv("EXP_MEMBER_JSON");
        if (env) path = env;
    }
    if (path.empty()) {
        why = "route exp needs a member file (--exp-member <path> or EXP_MEMBER_JSON)";
        return false;
    }
    return exp_member_parse(path.c_str(), mem, why);
}

// The loaded member. Aborts only if no member file is available, which the
// setup gates (SeparablePoisson::build and SimConfig::validate) reject first,
// so a driver reports a setup error instead of reaching this path.
inline const ExpMember& exp_member() {
    const char* why = "";
    if (!exp_member_ready(why)) {
        std::fprintf(stderr, "exp member unavailable: %s\n", why);
        std::abort();
    }
    return exp_member_store();
}

#endif // EXP_MEMBER_H
