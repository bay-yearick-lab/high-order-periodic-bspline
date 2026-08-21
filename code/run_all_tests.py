"""Run the periodic-paper test and validator suite.

Discovers and runs both the regression tests in ``code/tests/`` and the
cross-cutting matrix-level validators ``check_*.py`` in ``code/``.

Each component is bounded and short; the full run is intended to fit
inside a few minutes on a laptop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
TESTS_DIR = CODE_DIR / "tests"


def _run_command(label: str, args: list[str]) -> bool:
    print(f"\n=== {label} ===")
    start = time.perf_counter()
    result = subprocess.run(args, cwd=CODE_DIR)
    wall = time.perf_counter() - start
    ok = result.returncode == 0
    print(f"--- {label}: {'pass' if ok else 'FAIL'} in {wall:.2f}s ---")
    return ok


def _run_unit_tests() -> bool:
    sys.path.insert(0, str(CODE_DIR))
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py", top_level_dir=str(CODE_DIR))
    runner = unittest.TextTestRunner(verbosity=2)
    print("\n=== unit tests in code/tests ===")
    start = time.perf_counter()
    result = runner.run(suite)
    wall = time.perf_counter() - start
    print(f"--- unit tests: {'pass' if result.wasSuccessful() else 'FAIL'} in {wall:.2f}s ---")
    return result.wasSuccessful()


def main() -> int:
    failures: list[str] = []
    if not _run_unit_tests():
        failures.append("unit tests")
    validators = [
        ("collapse identity", "check_collapse_identity.py"),
        ("SBP first/second derivative", "check_sbp_identity.py"),
        ("Morinishi orthogonality", "check_morinishi.py"),
        ("gradient-divergence adjointness", "check_pressure_work.py"),
    ]
    python = os.environ.get("PYTHON", sys.executable)
    for label, script in validators:
        if not _run_command(label, [python, script]):
            failures.append(label)
    if failures:
        print(f"\nFAIL: {len(failures)} component(s) failed: {', '.join(failures)}")
        return 1
    print("\nAll periodic-paper tests and validators pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
