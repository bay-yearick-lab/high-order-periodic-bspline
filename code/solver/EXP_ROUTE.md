# Route `exp`: a generic exponential-family member

`exp` is one route in `Route` (config.h) that carries no built-in coefficients.
Its left side and both right-side stencils are read at run time from a JSON
member file, so a single binary fields every member of the exponential family
E_{m,r}(omega_1, ..., omega_r) without a rebuild.

## What the route is

A member has degree `p = m + 2 r - 1` and a generator supported on
`[-(p+1)/2, (p+1)/2]`, so the lattice samples

```
beta(n),  beta'(n),  beta''(n),        n = 0, 1, ..., R,   R = floor((p+1)/2)
```

determine every operator. `beta` is even, `beta'` is odd and `beta''` is even,
so only the one-sided block is stored. On the periodic grid with spacing `h`:

| object | entry at column `i + o` | bandwidth |
| --- | --- | --- |
| left side `E` | `beta(abs(o))` | `R` |
| first-derivative right side `E'` | `beta'(-o) / h = -sign(o) beta'(abs(o)) / h` | `R` |
| second-derivative right side `E''` | `beta''(abs(o)) / h^2` | `R` |

and

```
D1 = E' E^{-1},        D2 = E'' E^{-1}.
```

`E` is symmetric, `E'` is antisymmetric and `E''` is symmetric. The offset
convention is the one the polynomial B-spline route already uses in
`bm_apply.h`: offset `o = j - i` samples the kernel at `t = -o`.

The application is the collocated reduce-then-sample chain, identical to the
BMC route and NOT the two-solve pattern of the compact routes: one cyclic solve
of `E` produces the coefficients, then each derivative is one banded product.
`FastDeriv::exp_d1d2` (and `exp_d1d2_block`) take both derivatives off a single
solve. `CyclicSolver::set_from_stencil` handles any bandwidth `R`, so no
tridiagonal or pentadiagonal assumption is baked in anywhere.

## JSON schema

```json
{
  "name": "E62_provisional",
  "m": 6,
  "r": 2,
  "omegas": [1.769951, 2.060426],
  "p": 9,
  "radius": 5,
  "cond": 88.69238393868334,
  "beta":   [ ... radius+1 doubles, beta(0..R) ... ],
  "dbeta":  [ ... radius+1 doubles, beta'(0..R) ... ],
  "ddbeta": [ ... radius+1 doubles, beta''(0..R) ... ]
}
```

`radius`, `beta`, `dbeta` and `ddbeta` are required; the rest is metadata that
the drivers echo. Each of the three arrays must hold exactly `radius + 1`
values, and `beta(0)` must be positive. `cond` is the symbol-level left-side
condition number `max b / min b` over `theta` in `[0, pi]`.

The parser (`exp_member.h`) is a flat-object scanner, so the file must keep the
schema flat: a string, integers, doubles, and arrays of doubles at top level.

## Selecting a member

Both forms are equivalent; the option wins when both are given.

```
./tgv_re1600 --exp-member <path/to/member.json> <N> exp <Re> <Tend> <p> <cfl> ...
EXP_MEMBER_JSON=<path/to/member.json> ./tgv_re1600 <N> exp <Re> <Tend> <p> <cfl> ...
```

`--exp-member <path>` is stripped from `argv` before the positional arguments
are read, so it may appear anywhere on the command line. The same option is
accepted by `ns_mms2d` and by `c_exp_symbol_check`.

Route `exp` with no member file, an unreadable or malformed file, or
`N < 2 R + 1` on an active axis is refused by `SeparablePoisson::build`, which
is the setup gate every driver already reports through (`setup failed: ...`,
exit status 1). `SimConfig::validate` carries the same checks plus the
periodic-axis requirement, for callers that use it.

The positional `p` argument still sets the degree of the polynomial B-spline
operators the solver builds alongside; route `exp` takes its own radius from
the member file and ignores it.

## Symbol convention

With `theta` the grid angle and

```
b(theta) = beta(0)   + 2 sum_{n=1}^{R} beta(n)   cos(n theta)
a(theta) =             2 sum_{n=1}^{R} beta'(n)  sin(n theta)
c(theta) = beta''(0) + 2 sum_{n=1}^{R} beta''(n) cos(n theta)
```

the applied operators have eigenvalues

```
D1:  -(i/h)  a(theta) / b(theta)
D2:   (1/h^2) c(theta) / b(theta)
```

on the grid mode `exp(i n theta)`. This is the APPLIED-operator convention: the
D2 eigenvalue is the one `poisson.h` stores, it is nonpositive, and it matches
the existing BMC branch of `SeparablePoisson::eig_axis` sign for sign and
normalization for normalization. `sum_n beta''(n) = 0` rewrites the
numerator as `-4 sum_{n>0} beta''(n) sin^2(n theta / 2)`, which is nonpositive
for every theta when the samples `beta''(n)`, `n > 0`, are nonnegative (the
polynomial rungs and the Gate 1 members). A polynomial-dial member may carry a
negative `beta''(n)` (X9_m6_C_SLC1_SLC1992_6IIq2 has `beta''(4) < 0`), and its
nonpositivity rests on the per-member positivity check of `kappa_2 b` in
`gate8_family.certify_positivity`, which `c_exp_symbol_check` confirms on the
discrete modes of N = 128 and 127.

Two neighbouring conventions negate this quantity, so keep them apart:

* `family.py` reports `kappa_2` with the nodal second derivative having symbol
  `-(1/h^2) kappa_2`, so `kappa_2 = -c/b`.
* `spline_analysis._second_derivative_symbol_from_entries` carries the same
  leading minus, so `ksecond_exp_member` returns `kappa_2`, not the applied
  eigenvalue.

The pressure projection uses `eig_axis_divgrad`, which composes the route's own
D1 with itself, so the Perot projection stays exactly discrete-divergence-free
for `exp` exactly as it does for every other route.

## Regenerating a member file

```
.venv/bin/python code/analysis/exponential_family/write_member.py \
    --m 6 --omega 1.769951 --omega 2.060426 \
    --name E62_provisional \
    --out results/exponential_family/members/E62_provisional.json
```

`--omega` is repeated once per frequency (`r` of them); omit it entirely for a
pure polynomial core `r = 0`. The samples come from
`ExponentialGenerator(m, omegas).integer_samples()` and are written at full
double precision.

## Verification

```
cd code/solver && make c_exp_symbol_check
./c_exp_symbol_check --exp-member ../../results/exponential_family/members/E62_provisional.json
```

The check builds the route at `N = 128` and `N = 127`, applies the production
`exp_d1` and `exp_d2` kernels to every discrete mode, and compares against the
trigonometric polynomials above. The deviation is measured on the dimensionless
symbols `h lambda_1` and `h^2 lambda_2` (the modified wavenumbers themselves),
because comparing the dimensional eigenvalues would just scale the same
floating-point noise by `1/h^2`. It also asserts that the D1 eigenvalue is
purely imaginary and that the D2 symbol is nonpositive. Required: at most
`1e-12`.

The Python side is covered by `code/tests/test_exp_route.py`, which compares the
assembled operators against `family.py`'s alias-sum symbols and checks the
parity of both matrices.

## Python labels

`spline_analysis.register_exp_member(name, path)` binds a short label, and the
literal label `EXP:<path>` works with no registration. Both are accepted by
`first_derivative_matrix_by_label`, `first_derivative_symbol_by_label` and by
`experiment_d1.unique_scheme_entries` /
`unique_second_derivative_scheme_entries`. The `EXP:` test runs before the
explicit centered `E<order>` test, so labels such as `E6` keep their meaning.

## The off-grid evaluator block (`eval_*`)

`code/solver/exp_eval.h` evaluates `phi(x)`, `phi'(x)` and `phi''(x)` of the
member at arbitrary real `x`, for the off-grid path. It reads a second, purely
additive block of the same member file. Nothing above changes: a file without
the block still drives every operator route, and only `exp_eval.h` refuses to
load. The producer is
`code/analysis/exponential_family/gate5_evaluator.py`, which appends the block
to an existing member file without touching any field already there.

The block carries a version. Version 1 covers the Gate 0 multiset
`(0^m, +-i omega_j)` and is described first; version 2 widens it to the Gate 7
real pairs and quads and to the Gate 8 polynomial dial, and is described in the
subsection that follows. The field `eval_version` is absent from a version 1
file and its absence means version 1, so every file written before the
extension loads and evaluates exactly as before.

### Version 1: the Gate 0 multiset

The generator is supported on `[-n/2, n/2]` with `n = m + 2 r`. On the cell
`y = x + n/2` in `[J, J+1)`, with `xi = y - J`, it is the exponential
polynomial

```
phi(x) = sum_k a_k(J) xi^k
       + sum_{g,b} xi^b [ C_{g,b}(J) cos(w_g xi) + S_{g,b}(J) sin(w_g xi) ],
```

with `g` over the distinct pair frequencies and `b = 0..mu_g - 1`, `mu_g` the
confluent multiplicity of that frequency. The Python side assembles the tables
once in extended precision from the localized causal Green function, folds the
cubic Taylor jet of each oscillatory term at `xi = 0` into `a_k`, and multiplies
the normalization in, so the C++ side needs no further scaling.

| field | type | meaning |
| --- | --- | --- |
| `eval_support_n` | int | `n = m + 2 r`, the support width |
| `eval_cells` | int | number of unit cells, equal to `n` |
| `eval_poly_degree` | int | degree of the cell polynomial, `max(m-1, mu_max+2)` |
| `eval_group_count` | int | number of distinct pair frequencies `G` |
| `eval_term_count` | int | `T = sum_g mu_g`, the number of oscillatory columns |
| `eval_group_omega` | `G` doubles | one frequency per group |
| `eval_group_multiplicity` | `G` doubles | confluent multiplicity per group, integral values |
| `eval_normalization` | double | `prod_j 1 / sinc(omega_j/2)^2`, already folded into the tables |
| `eval_conditioning` | double | largest assembly term over `max abs phi`, a diagnostic |
| `eval_poly` | `n (deg+1)` doubles | `a_k(J)`, cell major, ascending powers |
| `eval_cos` | `n T` doubles | `C_{g,b}(J)`, cell major, groups in order, `b` fastest |
| `eval_sin` | `n T` doubles | `S_{g,b}(J)`, same layout |

What is left in double precision is cancellation free: the remainders
`C_2(z) = 1 - cos z`, `S_3(z) = z - sin z`, `C_4(z) = cos z - 1 + z^2/2` and
`S_5(z) = sin z - z + z^3/6` are taken from their series near zero, and
`dS_5/dz = C_4`, `dC_4/dz = S_3`, `dS_3/dz = C_2` give both derivatives from
the same four functions. Outside the open support all three channels are
exactly zero.

### Version 2: real pairs, quads and the polynomial dial

A Gate 7 member adds real pairs `+-alpha` and quads `+-alpha +- i omega` to the
multiset, and a Gate 8 member multiplies the transform by
`P(xi^2) = 1 + sum_l p_l xi^{2l}`, that is applies `psi = P(-D^2) phi` on the
`x` side. Support and cell count are unchanged: `n` is the number of exponents
and `psi` lives on `[-n/2, n/2]`. The producer is
`code/analysis/exponential_family/gate9_evaluator.py`, which appends the block
the same way, and the Python evaluator is
`code/analysis/exponential_family/general_evaluator.py`.

A group is now one root `rho = alpha + i omega` of the multiset with
`omega >= 0`, other than the zero root, so `eval_group_alpha` stands beside
`eval_group_omega`; a root off the real axis stands for itself and its
conjugate, and a real root stands alone. The jet folded into the cell
polynomial runs to the absolute order `eval_fold_order = K`, so the polynomial
degree is `K - 1` and what is left are the entire functions

```
R_j(z) = sum_{i >= j} z^i / i!,   R_j = exp for j <= 0,
```

which vanish to order `j` at the origin:

```
psi(x) = sum_{k<K} a_k(J) xi^k
       + sum_{g,b} xi^b [ C_{g,b}(J) Re R_{K-b}(rho_g xi)
                        + S_{g,b}(J) Im R_{K-b}(rho_g xi) ].
```

Since every remainder is `O(xi^K)`, the `a_k` are exactly the one-sided Taylor
coefficients of `psi` at the cell anchor, which is why the evaluation carries no
cancellation even in the end cells where `psi` is tiny. Derivatives follow from
`d/dxi R_j(rho xi) = rho R_{j-1}(rho xi)` and the Leibniz rule for the factor
`xi^b`, and `R_{j-1} = R_j + z^{j-1}/(j-1)!` builds the lower indices from the
higher one by addition alone. `R_j` is taken from its own series below modulus
`4` and from `exp(z)` minus the truncated exponential series above it, with the
term count and the limit shared by the Python and the C++ side.

Version 2 keeps every version 1 field with the same meaning, except that
`eval_group_count` counts roots rather than pair frequencies and
`eval_poly_degree` is `K - 1` rather than `max(m-1, mu_max+2)`. The fields it
adds are

| field | type | meaning |
| --- | --- | --- |
| `eval_version` | int | `2`; absent means version 1 |
| `eval_fold_order` | int | `K`, the absolute order of the folded jet, equal to `eval_poly_degree + 1` |
| `eval_group_alpha` | `G` doubles | real part of the root of each group, `0` for an imaginary pair |
| `eval_representation_conditioning` | double | largest stored coefficient over `max abs psi`, a diagnostic |

`eval_group_omega` holds the imaginary part of the same root,
`eval_group_multiplicity` its confluent multiplicity, and `eval_cos` and
`eval_sin` hold `C_{g,b}` and `S_{g,b}` in the version 1 layout. The dial `P` is
applied before the tables are written, so the C++ side needs no knowledge of it;
the member fields `polynomial` and `smoothness` describe it for the reader.

The cell tables of a version 2 block are assembled in decimal arithmetic at 50
digits, because the localization sum and the `2 q` differentiations of the dial
cancel against themselves, and only the finished coefficients are rounded to
double.

### Verification

```
cd code/solver && make exp_eval_check
./exp_eval_check --exp-member ../../results/exponential_family/members/E62_champion.json \
                 --table ../../results/exponential_family/gate5/exp_eval_table.csv
```

The table holds 10,000 rows `x, phi, phi', phi''` written by the Python
evaluator. Required: at most `1e-14` in every channel. This is check (vi) of
Gate 5 section 5.0; the other five checks and this one are recorded together in
`results/exponential_family/gate5/evaluator_checks.json`.

The same program reads a version 2 block and prints the version in its banner:

```
./exp_eval_check --exp-member ../../results/exponential_family/gate8/members/X8_n10_C_SLC1_pert5.json \
                 --table ../../results/exponential_family/gate9/e4_eval_table_pert5.csv
```

That is check (vi) of Gate 9 section E4, whose six checks are recorded in
`results/exponential_family/gate9/e4_evaluator_checks.json` and whose check (v)
is the regression that the version 1 path and the Gate 5 receipt are unchanged.
