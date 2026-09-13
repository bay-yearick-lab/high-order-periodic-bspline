# Periodic B-spline mimetic Navier-Stokes solver (C++17, header-only operators,
# no external numerical libraries, OpenMP across independent 1D line solves).
#
#   make            compile every driver
#   make test       compile and run the compressible verification suite
#   make run        compile and run the symbol-level comparisons
#   make clean      remove executables
#
# Requires g++ >= 9 (or clang++ with libomp). Override CXX/CXXFLAGS as needed,
# e.g.  make CXXFLAGS="-std=c++17 -O3"  for a build without OpenMP.

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -funroll-loops -Wall -Wextra \
            -fno-exceptions -fno-rtti -fopenmp

HEADERS := config.h linalg.h bspline.h bm_apply.h bm_operators.h bm3d.h poisson.h ns.h mms_forcing.h \
           exp_member.h exp_eval.h compressible.h

# Symbol-level comparisons, evaluator checks, and the 3D incompressible and
# compressible drivers.
PROGRAMS := fair_compare caa_dispersion tgv_re1600 ns_dns3d ns_reconnection offgrid_turbulence ns_mms2d \
            nullspace_probe operator_fidelity_check c_sl_symbol_check c_exp_symbol_check exp_eval_check \
            c_opt_symbol_check c_cw12_symbol_check c8_symbol_check c1012_symbol_check drp_symbol_check \
            mms_compressible

.PHONY: all run test clean
all: $(PROGRAMS)

%: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: fair_compare caa_dispersion
	./fair_compare
	./caa_dispersion

test: mms_compressible
	./mms_compressible

clean:
	rm -f $(PROGRAMS)
