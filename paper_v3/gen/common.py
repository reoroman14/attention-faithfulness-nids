# Shared setup for the paper_v3 generators (Task 42). Read-only on results/ and on paper/: the old generators in
# paper/ are imported as modules and their output directory is redirected to paper_v3/generated, so nothing in paper/
# is written. Every number is read from a result file.
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
OLD = ROOT / "paper"
GEN = V3 / "generated"
RES = ROOT / "results"
S9 = RES / "stage9"
A8X, A6X, SB = S9 / "a8x", S9 / "a6x", S9 / "adaptive"

for p in (str(OLD), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import data_v2 as D  # noqa: E402

YR = {2017: "Seventeen", 2018: "Eighteen", "2017": "Seventeen", "2018": "Eighteen"}
VN = {"V0": "Vzero", "V1": "Vone", "V2": "Vtwo"}
SEEDN = {"seed 0": "SeedZero", "seed 1": "SeedOne", "seed 2": "SeedTwo"}
SETN = {"lam1": "One", "lam100": "Hundred", "norm_w0.5": "Norm"}
CTLN = {"c0": "Czero", "c1": "Cone", "c2": "Ctwo", "a1": "Aone", "a2": "Atwo", "b": "B"}
CONTROLS = ["c0", "c1", "c2", "a1", "a2"]
CTL_TYPE = {"c0": "exact", "c1": "exact", "c2": "exact", "a1": "stand-in", "a2": "stand-in"}


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def fmt(v, d=3):
    # Point value: no sign for positives (as in paper/make_numbers_tex.py), "-0.000" shown as "0.000".
    t = f"{v:.{d}f}"
    return t[1:] if t.startswith("-") and float(t) == 0 else t


def fmts(v, d=3):
    # Signed point value (differences): +0.075, -0.037, 0.000.
    t = fmt(v, d)
    return t if t.startswith("-") or float(t) == 0 else "+" + t


def bound(v, d=3):
    # Interval bound: keeps the sign of a slightly negative bound that rounds to zero (as in the old generators).
    return f"{v:.{d}f}"


def bounds(v, d=3):
    # Signed interval bound: +0.061, -0.023, and -0.000 / +0.000 when a bound rounds to zero (sign kept).
    t = bound(v, d)
    return t if t.startswith("-") else "+" + t


def old_tables():
    # paper/make_tables_v2.py with its output redirected to paper_v3/generated.
    import make_tables_v2 as mt
    mt.OUT = GEN
    return mt


def _pure_python_fonttools():
    # On this machine an Application Control policy blocks the compiled fontTools extensions (.pyd) that matplotlib's
    # PDF backend imports; fontTools ships the same modules as .py sources. This finder loads the .py source whenever a
    # fontTools submodule has one, in this process only (nothing on disk is changed).
    import importlib.machinery
    import importlib.util

    class Finder:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if not name.startswith("fontTools.") or not path:
                return None
            for d in path:
                src = Path(d) / (name.rsplit(".", 1)[1] + ".py")
                if src.exists() and any(Path(d).glob(name.rsplit(".", 1)[1] + ".*.pyd")):
                    return importlib.util.spec_from_file_location(name, src)
            return None
    if not any(type(f).__name__ == "Finder" for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())


def old_figures():
    _pure_python_fonttools()
    import make_figures as mf
    import make_figures_v2 as mf2
    mf.OUT = GEN
    mf2.OUT = GEN
    return mf, mf2
