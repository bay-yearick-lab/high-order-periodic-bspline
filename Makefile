# One-command entry points. Python targets need uv (https://docs.astral.sh/uv/);
# C++ targets need g++ >= 9 with OpenMP.

.PHONY: verify test bench figures solver solver-test clean

verify:            ## headline identities, convergence orders and receipts (seconds)
	uv run python tools/verify.py

test:              ## 38 validation tests + 306 regression tests (about one minute)
	uv run pytest

bench:             ## matrix-free banded line derivative, accuracy versus cost
	uv run python benchmarks/bench_line_derivative.py
	uv run python benchmarks/tgv_step_timing.py

figures:           ## regenerate assets/*.png from results/
	uv run python tools/make_readme_assets.py

solver:            ## build every C++ driver
	$(MAKE) -C code/solver all

solver-test:       ## compressible conservation, isentropic vortex and MMS orders (about one minute)
	$(MAKE) -C code/solver test

clean:
	$(MAKE) -C code/solver clean
	rm -rf figures_paper .pytest_cache
