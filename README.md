# An upper bound of 1.5038012 for the Sidon autocorrelation constant

**C ≤ 1.503801221935561**, with the witness that proves it and a checker that
needs nothing but Python.

This is [problem 2](https://google-deepmind.github.io/alphaevolve_repository_of_problems/problems/2.html)
of DeepMind's *AlphaEvolve repository of problems*. It beats the sixteen-year
human record and AlphaEvolve's May 2025 result. **It does not beat AlphaEvolve's
December 2025 result, and it does not approach the current record** — both of
those are still ahead, and the table below says by how much.

## Where this sits

Upper bounds for `C`, quoted from the problem page as that page prints them
(read 2026-09-10):

| bound | source | date | |
|---|---|---|---|
| 2.00000 | the constant function | — | beaten |
| 1.57059 | `π/2`, Schinzel–Schmidt | 2002 | beaten |
| 1.50992 | Matolcsi–Vinuesa | 8 Jul 2009 | beaten |
| 1.5053 | Georgiev–Gómez-Serrano–Tao–Wagner | 14 May 2025 | beaten |
| **1.5038012** | **this repository** | **Sep 2026** | |
| 1.5032 | Georgiev–Gómez-Serrano–Tao–Wagner | 20 Dec 2025 | **still ahead** |
| 1.5029 | Yuksekgonul et al. | Jan 2026 | **still ahead, current record** |

The best proved *lower* bound is `1.2748` (Matolcsi–Vinuesa, 2009), so `C` lies
somewhere in `[1.2748, 1.5038]` as far as this repository can say. Only the
upper side is a search problem; the lower side needs a proof.

## Check it yourself

No dependencies. Python 3.9+, standard library only.

```bash
python verify.py
```

```
     n  non-zero       F (recomputed)  verdict
   512       374    1.505062018958748  verified
  4096      2799    1.503972214391172  verified
  8400      5608    1.503801221935561  verified

best upper bound here: C <= 1.503801221935561   (n = 8400)
```

`verify.py` reads the cell heights and recomputes everything else — the
convolution, its maximum, the sum, the ratio. It checks that no height is
negative, because the bound is only valid for non-negative `f`. The check is
`O(n²)` in pure Python: about ten seconds at `n = 8400`, instant at `n = 512`.
That is the price of having nothing to install.

## Why the discretisation is exact rather than approximate

Since `f ≥ 0`, truncating it to `[-1/4, 1/4]` leaves the right-hand side of the
inequality alone and can only shrink the left, so the support may be assumed to
lie inside that interval; then `f*f` is supported in `[-1/2, 1/2]` and the
maximum over `t` is unconstrained. Cut the interval into `n` cells of width
`h = 1/(2n)` with heights `a_i ≥ 0`. The convolution of two cell indicators is a
hat of height `h`, so `f*f` is piecewise linear with **every breakpoint on the
grid**, and a piecewise-linear function attains its maximum at a breakpoint.
Hence

    F(a) = 2n · max_k b_k / (Σ_i a_i)²,    b_k = Σ_{i+j=k} a_i a_j

**exactly.** A step function is a function, so every `F(a)` here is a rigorous
upper bound on `C`. What the discretisation costs is expressive power, not
rigour — which is why a finer grid gives a better bound and never an invalid one.

The ladder shows that directly:

| n | F | non-zero cells |
|---|---|---|
| 512 | 1.505062018958748 | 374 |
| 4096 | 1.503972214391172 | 2799 |
| 8400 | 1.503801221935561 | 5608 |

Worth noting: **the `n = 512` row already beats the May 2025 result**, which was
found at `n = 600`. The gain from there to `n = 8400` is real but small —
0.00126 across a sixteenfold refinement — and the curve is clearly flattening,
so more cells alone will not close the remaining 0.0006 to the December 2025
bound.

## How it was found

`src/main.cu` is a memetic search in CUDA, run on a single RTX 3060 Ti. The
pieces that matter:

- **L^p-annealed projected Adam**, fused into one kernel with `a` resident in
  shared memory, driving `‖b‖_p → ‖b‖_∞`. The anneal must be long and end high:
  the constant function is an exact **saddle point** of the smoothed objective,
  so a short anneal stalls above 2.0 and never escapes.
- **A Frank–Wolfe step** solved by Chambolle–Pock. This is the Matolcsi–Vinuesa
  iteration that sits behind every record since 2009. There is no LP solver on a
  GPU, but each iteration is one convolution plus one correlation — kernels the
  search already has.
- **A multiresolution ladder** `n → 2n`. Cell-splitting preserves `F` bit for
  bit, so coarse levels are free structure-finding.
- **Structural-prior operators.** Every published record shares one shape: an
  endpoint atom, a wide hard support gap, a ragged body, roughly a third exact
  zeros, strong left/right asymmetry. Sampling that family directly beats hoping
  generic mutation stumbles into it.

One piece of numerical care decides whether the last digits mean anything:
`(b_k/M)^(p-1)` is computed as `exp((p-1)·log1p((b_k-M)/M))`, never as
`exp(p·log(b_k/M))`. The second form loses every significant digit for peaks
within `1e-6` of the maximum, which is exactly the regime a high-`p` search
lives in.

## Checked three ways

The number was not taken from the search that produced it:

1. the fp32 search kernel, with fp64 accumulation for ranking;
2. `hostF()` in `main.cu` — plain fp64, a different loop order;
3. `tools/probe.cpp --verify` — long double, OpenMP across 12 cores, with an
   explicit round-off bound and a non-negativity check.

and then, for this repository, a fourth: `verify.py`, written independently in
Python against the problem statement rather than against the search. It agrees
with the stored value to `1.6e-15`, which is floating-point noise at this
magnitude.

## What this is not

- **Not the record.** Two published bounds are still ahead. The table above
  names them.
- **Not a lower bound.** No witness can give one. `C` itself is unknown.
- **Not a claim that the method is exhausted.** The ladder was flattening, not
  flat; whether the remaining 0.0006 needs a finer grid, a longer run, or a
  different idea is not settled here.

## What is here

```
verify.py            the checker. stdlib only, recomputes from the heights
certificates/        three certificates: n = 512, 4096, 8400
ladder.csv           the table above, as data
src/main.cu          the memetic search driver, self-tests, benchmark
src/kernels.cuh      the device kernels
tools/probe.cpp      the long-double CPU verifier
```

Built for an RTX 3060 Ti (sm_86) with CUDA 13.3 and MSVC 14.44. The `--arch`
flag and the MSVC path are the only machine-specific parts.

## Related

Two more results from the same effort, each with its own certificates and
checker:

- [autocorrelation-witness-ladder](https://github.com/nanegate-dev/autocorrelation-witness-ladder)
  — a certified lower bound of `0.409821093` for the minimum-autocorrelation
  constant of AlphaEvolve problem 6.6, against a published `0.37`. That one is
  ahead of the field.
- [labs-optimum-ladder](https://github.com/nanegate-dev/labs-optimum-ladder)
  — low-autocorrelation binary sequences, where the optimum is *proven* at small
  lengths and an autonomous search reaches it at seven of seven.

The three are worth reading together: one goes past the best known bound, one
reaches an answer already proven, and this one closes most of the gap to the
front without reaching it.

## Licence

MIT.
