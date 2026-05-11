"""Project path helpers shared across the periodic paper drivers.

Centralizes the figure directory, results directory, and code-relative
imports so that section drivers (``s2_*.py`` through ``s6_*.py``),
appendix drivers (``appA_*.py`` and friends), and the cross-cutting
validators (``check_*.py``) can use a single canonical layout.
"""

from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PROJECT_ROOT / "figures_paper"
RESULTS_DIR = CODE_DIR / "results"
TESTS_DIR = CODE_DIR / "tests"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def figure_path(name: str) -> Path:
    ensure_dirs()
    if not name.endswith(".pdf"):
        name = name + ".pdf"
    return FIGURES_DIR / name


def results_path(name: str) -> Path:
    ensure_dirs()
    return RESULTS_DIR / name


__all__ = [
    "CODE_DIR",
    "PROJECT_ROOT",
    "PAPER_DIR",
    "FIGURES_DIR",
    "RESULTS_DIR",
    "TESTS_DIR",
    "ensure_dirs",
    "figure_path",
    "results_path",
]
