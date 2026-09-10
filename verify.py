"""Check every certificate in this repository, from the cell heights up.

    python verify.py
    python verify.py certificates/sidon-8400.json

No dependencies. Python 3.9 or newer, standard library only.

WHAT IS BEING CLAIMED. Let C be the largest constant with

    max over |t| <= 1/2 of  integral f(t-x) f(x) dx  >=  C * (integral of f over [-1/4, 1/4])^2

for every non-negative f. Because C is the LARGEST such constant, any single f
you can write down gives an UPPER bound on C -- its own ratio -- and no f can
ever give a lower one. Every certificate here is one f and its ratio.

WHY THE DISCRETISATION IS EXACT, NOT AN APPROXIMATION. Since f >= 0, truncating
it to [-1/4, 1/4] leaves the right-hand side alone and can only shrink the left,
so the support may be assumed to sit inside that interval; then f*f is supported
in [-1/2, 1/2] and the max over t is unconstrained. Cut the interval into n
cells of width h = 1/(2n) with heights a_i >= 0. The convolution of two cell
indicators is a hat of height h, so f*f is piecewise linear with EVERY
breakpoint on the grid, and a piecewise-linear function attains its maximum at a
breakpoint. Hence

    F(a) = 2n * max_k b_k / (sum_i a_i)^2,    b_k = sum_{i+j=k} a_i a_j

exactly. A step function is a function, so F(a) is a rigorous upper bound on C
-- what the discretisation costs is expressive power, not rigour.

WHAT THIS SCRIPT DOES NOT DO. It does not prove the bound is tight, and it says
nothing about the lower side of C, which needs a proof rather than a witness.
It recomputes F from the heights and reports whether the stored claim matches.

RUNTIME. The check is O(n^2) in pure Python: about 10 s at n=8400 and instant
at n=512. That is the price of having no dependencies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: DeepMind's own problem page, read 2026-09-10:
#: https://google-deepmind.github.io/alphaevolve_repository_of_problems/problems/2.html
#: Values are quoted at the precision that page prints them.
UPPER_BOUNDS = [
    (1.57059, "pi/2, Schinzel-Schmidt", "2002"),
    (1.50992, "Matolcsi-Vinuesa", "8 Jul 2009"),
    (1.5053, "Georgiev-Gomez-Serrano-Tao-Wagner", "14 May 2025"),
    (1.5032, "Georgiev-Gomez-Serrano-Tao-Wagner", "20 Dec 2025"),
    (1.5029, "Yuksekgonul et al.", "Jan 2026"),
]

#: The best proved lower bound, from the same page. C is somewhere above this;
#: nothing in this repository speaks to that side.
LOWER_BOUND = (1.2748, "Matolcsi-Vinuesa", "8 Jul 2009")


def ratio(a: list[float]) -> tuple[float, float, int]:
    """(F, max_k b_k, argmax k) recomputed from the heights and nothing else."""
    n = len(a)
    best, best_k = -1.0, -1
    # b_k = sum_{i+j=k} a_i a_j, for k = 0 .. 2n-2
    for k in range(2 * n - 1):
        lo = max(0, k - n + 1)
        hi = min(k, n - 1)
        b = 0.0
        for i in range(lo, hi + 1):
            b += a[i] * a[k - i]
        if b > best:
            best, best_k = b, k
    total = sum(a)
    if total <= 0:
        raise ValueError("the weights sum to zero")
    return 2 * n * best / (total * total), best, best_k


def check(path: Path) -> tuple[bool, str, dict, float]:
    cert = json.loads(path.read_text(encoding="utf-8"))
    a = cert["weights"]
    n = cert["n"]

    if len(a) != n:
        return False, f"{len(a)} weights for a claimed n of {n}", cert, 0.0
    negative = sum(1 for x in a if x < 0)
    if negative:
        return False, f"{negative} negative weights; f must be non-negative", cert, 0.0

    got, peak, k = ratio([float(x) for x in a])
    claimed = float(cert["F"])
    # The stored value came from a different program in a different language;
    # agreement to the last bit is not expected and is not required. What is
    # required is that the recomputed value is the one being claimed.
    if abs(got - claimed) > 1e-12:
        return False, f"stored F {claimed!r} but the weights give {got!r}", cert, got

    return True, "", cert, got


def main() -> int:
    argv = sys.argv[1:]
    paths = [Path(x) for x in argv] if argv else sorted(
        (HERE / "certificates").glob("*.json"),
        key=lambda p: json.loads(p.read_text(encoding="utf-8"))["n"])
    if not paths:
        print("no certificates found")
        return 1

    print(f"{'n':>6} {'non-zero':>9} {'F (recomputed)':>20}  verdict")
    ok, best = True, None
    for path in paths:
        try:
            passed, why, cert, got = check(path)
        except Exception as error:
            print(f"{path.name}: UNREADABLE - {error}")
            ok = False
            continue
        nz = sum(1 for x in cert["weights"] if x != 0)
        if passed:
            print(f"{cert['n']:>6} {nz:>9} {got:>20.15f}  verified")
            if best is None or got < best[0]:
                best = (got, cert["n"])
        else:
            print(f"{cert['n']:>6} {nz:>9} {'':>20}  FAILED - {why}")
            ok = False

    if best is None:
        print("\nnothing verified")
        return 1

    value, n = best
    print(f"\nbest upper bound here: C <= {value:.15f}   (n = {n})")
    print(f"\nagainst the published upper bounds (smaller is better):")
    for bound, who, when in UPPER_BOUNDS:
        verdict = "BEATEN by this work" if value < bound else "not beaten"
        print(f"  {bound:<9} {who:<38} {when:<12} {verdict}")
    lb, who, when = LOWER_BOUND
    print(f"\n  the best proved LOWER bound is {lb} ({who}, {when}),")
    print(f"  so C lies in [{lb}, {value:.6f}] as far as this repository can say.")
    print("\nall certificates verified" if ok else "\nSOMETHING DID NOT VERIFY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
