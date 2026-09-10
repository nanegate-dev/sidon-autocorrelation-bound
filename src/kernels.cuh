// ---------------------------------------------------------------------------
//  AlphaEvolve Problem 2 -- "An autocorrelation problem related to Sidon sets"
//
//  Minimise    F(a) = 2*n*max_k b_k / (sum_i a_i)^2 ,  b_k = sum_{i+j=k} a_i a_j
//  over a in R^n, a_i >= 0.   F(a) is a rigorous UPPER BOUND on the constant C.
//
//  Derivation (support [-1/4,1/4], cell width h = 1/(2n), f = sum a_i 1_{I_i}):
//      f*f is piecewise linear with breakpoints exactly at the grid nodes,
//      and (f*f)(-1/2 + m*h) = h * b_{m-1}.  Hence  max f*f = h * max_k b_k
//      and  (int f)^2 = h^2 (sum a_i)^2,  giving the ratio above since 1/h = 2n.
//
//  This header holds every device kernel.
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #call, __FILE__,  \
                   __LINE__, cudaGetErrorString(_e));                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static constexpr int BD    = 256;  // threads per block (one block == one individual)
static constexpr int TK    = 8;    // register tile along k (autoconvolution)
static constexpr int TI    = 8;    // register tile along i (correlation / gradient)
static constexpr int MAXTL = 8;    // max i-tiles per thread => n <= MAXTL*TI*BD = 16384

// --------------------------- small numeric helpers -------------------------
//  IMPORTANT.  In the high-exponent regime we need (b_k/M)^(p-1) for b_k that
//  are within ~1e-6 of the maximum M.  Writing that as exp(p*log(b/M)) loses
//  every significant digit, because log of a number near 1 is a catastrophic
//  cancellation.  We therefore always work with  d = (b-M)/M <= 0  (which is
//  computed exactly, being a difference of nearby floats) and use log1p:
//        (b/M)^q = exp(q * log1p(d)).
//  This is what lets the search resolve near-degenerate peaks at p ~ 1e3-1e5.
__device__ __forceinline__ float  d_powm1(float d, float q)  { return __expf(q * log1pf(d)); }
__device__ __forceinline__ double d_powm1(double d, double q){ return exp(q * log1p(d)); }
__device__ __forceinline__ float  d_sqrt_(float x)          { return sqrtf(x); }
__device__ __forceinline__ double d_sqrt_(double x)         { return sqrt(x); }
__device__ __forceinline__ float  d_exp_(float x)           { return __expf(x); }
__device__ __forceinline__ double d_exp_(double x)          { return exp(x); }

// --------------------------- block reductions ------------------------------
template <typename T>
__device__ __forceinline__ T warpReduceMax(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    T o = __shfl_down_sync(0xffffffffu, v, off);
    v = o > v ? o : v;
  }
  return v;
}
template <typename T>
__device__ __forceinline__ T warpReduceSum(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}
// scratch: >= 32 slots.  Result is broadcast to every thread. Ends synchronised.
template <typename T>
__device__ __forceinline__ T blockMax(T v, T* scratch) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warpReduceMax(v);
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  T r = scratch[0];
  for (int i = 1; i < nw; ++i) r = scratch[i] > r ? scratch[i] : r;
  __syncthreads();
  return r;
}
template <typename T>
__device__ __forceinline__ T blockSum(T v, T* scratch) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warpReduceSum(v);
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  T r = T(0);
  for (int i = 0; i < nw; ++i) r += scratch[i];
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------------------
//  autoconvolution  b_k = sum_i a_i * a_{k-i}   (k = 0 .. 2n-2)
//  Each thread owns TK consecutive k's and slides a TK-wide window of `a`
//  through registers => 2 shared reads per TK FMAs (~1 byte/FLOP).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void crossconv(const T* __restrict__ sA, const T* __restrict__ sG,
                                          T* __restrict__ sB, int n) {
  const int nb    = 2 * n - 1;
  const int ntile = (nb + TK - 1) / TK;
  for (int q = threadIdx.x; q < ntile; q += BD) {
    const int k0 = q * TK;
    T acc[TK];
#pragma unroll
    for (int t = 0; t < TK; ++t) acc[t] = T(0);

    const int ilo = (k0 - (n - 1)) > 0 ? (k0 - (n - 1)) : 0;
    const int ihi = (k0 + TK - 1) < (n - 1) ? (k0 + TK - 1) : (n - 1);

    T w[TK];
#pragma unroll
    for (int t = 0; t < TK; ++t) {
      const int j = k0 + t - ilo;
      w[t] = (j >= 0 && j < n) ? sG[j] : T(0);
    }
    for (int i = ilo; i <= ihi; ++i) {
      const T x = sA[i];
#pragma unroll
      for (int t = 0; t < TK; ++t) acc[t] += x * w[t];
#pragma unroll
      for (int t = TK - 1; t > 0; --t) w[t] = w[t - 1];
      const int j = k0 - (i + 1);
      w[0] = (j >= 0 && j < n) ? sG[j] : T(0);
    }
#pragma unroll
    for (int t = 0; t < TK; ++t)
      if (k0 + t < nb) sB[k0 + t] = acc[t];
  }
}

template <typename T>
__device__ __forceinline__ void autoconv(const T* __restrict__ sA, T* __restrict__ sB, int n) {
  crossconv(sA, sA, sB, n);
}

// ---------------------------------------------------------------------------
//  correlation tile:  G_{i0+s} = sum_{j=0}^{n-1} c_{i0+s+j} * a_j
//  Since i <= n-1 and j <= n-1 we always have i+j <= 2n-2: no bounds checks.
//  sC must be zero-padded up to index 2n-2+TI.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void corrTile(const T* __restrict__ sA, const T* __restrict__ sC,
                                         int n, int i0, T* g) {
#pragma unroll
  for (int s = 0; s < TI; ++s) g[s] = T(0);
  T w[TI];
#pragma unroll
  for (int s = 0; s < TI; ++s) w[s] = sC[i0 + s];
  for (int j = 0; j < n; ++j) {
    const T x = sA[j];
#pragma unroll
    for (int s = 0; s < TI; ++s) g[s] += x * w[s];
#pragma unroll
    for (int s = 0; s < TI - 1; ++s) w[s] = w[s + 1];
    w[TI - 1] = sC[i0 + TI + j];
  }
}

// shared-memory footprint of the fused local-search kernel, in bytes
template <typename T>
inline size_t localSearchSmem(int n) { return sizeof(T) * (size_t)(n + 2 * n + TI + 8 + 64); }

// ---------------------------------------------------------------------------
//  Fused local search.  The entire iteration loop lives inside one launch with
//  `a` resident in shared memory (zero global traffic for the O(n^2) part).
//
//  Smoothed objective  J_p(a) = 2n * ||b||_p / (sum a)^2.
//  With sum a == 1 the descent direction collapses to
//        g_i = G_i / R - M ,
//  where  M = max b,  u = b/M,  R = sum_k u_k^p,  c_k = u_k^{p-1},  G = corr(c,a).
//  (The overall positive factor R^{1/p} is irrelevant to Adam / mirror descent.)
//
//  mode 0 = projected Adam  (can reach exact zeros -> finds support gaps)
//  mode 1 = exponentiated gradient / mirror descent on the simplex (stays > 0)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(BD) void kLocalSearch(
    T* __restrict__ pop, T* __restrict__ mom, T* __restrict__ vel,
    const int* __restrict__ skip, const int* __restrict__ modeArr,
    int n, int iters, T pStart, T pEnd, T lrStart, T lrEnd,
    T beta1, T beta2, int modeDefault) {
  extern __shared__ __align__(16) char smemRaw[];
  T* sA = reinterpret_cast<T*>(smemRaw);
  T* sB = sA + n;                 // 2n + TI + 8 slots (zero padded tail)
  T* sR = sB + 2 * n + TI + 8;    // 64 reduction slots

  const int pid = blockIdx.x;
  if (skip && skip[pid]) return;
  const int mode = modeArr ? modeArr[pid] : modeDefault;

  T* a = pop + (size_t)pid * n;
  T* m = mom + (size_t)pid * n;
  T* v = vel + (size_t)pid * n;

  for (int i = threadIdx.x; i < n; i += BD) { sA[i] = a[i]; m[i] = T(0); v[i] = T(0); }
  for (int k = threadIdx.x; k < 2 * n + TI + 8; k += BD) sB[k] = T(0);
  __syncthreads();

  const int nb     = 2 * n - 1;
  const int ntileI = (n + TI - 1) / TI;
  const T lnPr = T(log(double(pEnd)  / double(pStart)));
  const T lnLr = T(log(double(lrEnd) / double(lrStart)));
  T b1t = T(1), b2t = T(1);

  T newA[MAXTL][TI];

  for (int it = 0; it < iters; ++it) {
    const T frac = iters > 1 ? T(it) / T(iters - 1) : T(0);
    const T pexp = pStart * d_exp_(lnPr * frac);
    const T lr   = lrStart * d_exp_(lnLr * frac);

    autoconv(sA, sB, n);
    __syncthreads();

    T loc = T(0);
    for (int k = threadIdx.x; k < nb; k += BD) loc = sB[k] > loc ? sB[k] : loc;
    const T M = blockMax(loc, sR);
    if (!(M > T(0))) break;

    // c_k <- u_k^{p-1}  (in place);  R = sum_k u_k^p ,  u_k = b_k / M
    T rloc = T(0);
    for (int k = threadIdx.x; k < nb; k += BD) {
      const T d = (sB[k] - M) / M;                 // exact-ish, in [-1, 0]
      const T c = d > T(-1) ? d_powm1(d, pexp - T(1)) : T(0);
      rloc += c * (T(1) + d);
      sB[k] = c;
    }
    __syncthreads();
    const T R = blockSum(rloc, sR);

    b1t *= beta1; b2t *= beta2;
    const T bc1 = T(1) / (T(1) - b1t), bc2 = T(1) / (T(1) - b2t);

    int tl = 0;
    for (int q = threadIdx.x; q < ntileI; q += BD, ++tl) {
      const int i0 = q * TI;
      T g[TI];
      corrTile(sA, sB, n, i0, g);
#pragma unroll
      for (int s = 0; s < TI; ++s) {
        const int i = i0 + s;
        if (i >= n) { newA[tl][s] = T(0); continue; }
        const T gi = g[s] / R - M;
        T x;
        if (mode == 0) {
          const T mm = beta1 * m[i] + (T(1) - beta1) * gi;
          const T vv = beta2 * v[i] + (T(1) - beta2) * gi * gi;
          m[i] = mm; v[i] = vv;
          const T step = (lr / T(n)) * (mm * bc1) / (d_sqrt_(vv * bc2) + T(1e-30));
          x = sA[i] - step;
          if (x < T(0)) x = T(0);
        } else {
          T e = -lr * (gi / M);
          if (e >  T(4)) e =  T(4);
          if (e < -T(4)) e = -T(4);
          x = sA[i] * d_exp_(e);
        }
        newA[tl][s] = x;
      }
    }
    __syncthreads();

    tl = 0;
    T sloc = T(0);
    for (int q = threadIdx.x; q < ntileI; q += BD, ++tl) {
      const int i0 = q * TI;
#pragma unroll
      for (int s = 0; s < TI; ++s) {
        const int i = i0 + s;
        if (i < n) { sA[i] = newA[tl][s]; sloc += newA[tl][s]; }
      }
    }
    __syncthreads();
    T S = blockSum(sloc, sR);
    if (!(S > T(0))) {
      for (int i = threadIdx.x; i < n; i += BD) sA[i] = T(1) / T(n);
      __syncthreads();
      S = T(1);
    }
    const T inv = T(1) / S;
    for (int i = threadIdx.x; i < n; i += BD) sA[i] *= inv;
    // zero the c-scratch tail again for the next autoconv pass
    for (int k = nb + threadIdx.x; k < 2 * n + TI + 8; k += BD) sB[k] = T(0);
    __syncthreads();
  }

  for (int i = threadIdx.x; i < n; i += BD) a[i] = sA[i];
}

// ---------------------------------------------------------------------------
//  Exact objective evaluation.  Input may be float; accumulation is always in
//  `Acc` (use double) so that ranking is never polluted by fp32 round-off.
//  Writes F = 2*n*max_k b_k / (sum a)^2, plus the number of near-active peaks.
//
//  `b` is NEVER materialised: each thread reduces its own TK-tile straight into
//  a running max.  Shared use is only sizeof(Acc)*(n+64), so fp64 evaluation
//  scales to n = 12288 on sm_86 with the 100 KB opt-in.  Two passes over k are
//  used (max, then active count) -- evaluation is rare, so recomputing is free.
// ---------------------------------------------------------------------------
template <typename T, typename Acc>
__global__ __launch_bounds__(BD) void kEval(const T* __restrict__ pop, int n,
                                            Acc* __restrict__ out,
                                            int* __restrict__ nActiveOut,
                                            Acc activeTol) {
  extern __shared__ __align__(16) char smemRaw[];
  Acc* sA = reinterpret_cast<Acc*>(smemRaw);
  Acc* sR = sA + n;

  const int pid = blockIdx.x;
  const T* a = pop + (size_t)pid * n;

  Acc sloc = Acc(0);
  for (int i = threadIdx.x; i < n; i += BD) { const Acc x = Acc(a[i]); sA[i] = x; sloc += x; }
  __syncthreads();
  const Acc S = blockSum(sloc, sR);

  const int nb    = 2 * n - 1;
  const int ntile = (nb + TK - 1) / TK;

  Acc loc = Acc(0);
  for (int q = threadIdx.x; q < ntile; q += BD) {
    const int k0 = q * TK;
    Acc acc[TK];
#pragma unroll
    for (int t = 0; t < TK; ++t) acc[t] = Acc(0);
    const int ilo = (k0 - (n - 1)) > 0 ? (k0 - (n - 1)) : 0;
    const int ihi = (k0 + TK - 1) < (n - 1) ? (k0 + TK - 1) : (n - 1);
    Acc w[TK];
#pragma unroll
    for (int t = 0; t < TK; ++t) { const int j = k0 + t - ilo; w[t] = (j >= 0 && j < n) ? sA[j] : Acc(0); }
    for (int i = ilo; i <= ihi; ++i) {
      const Acc x = sA[i];
#pragma unroll
      for (int t = 0; t < TK; ++t) acc[t] += x * w[t];
#pragma unroll
      for (int t = TK - 1; t > 0; --t) w[t] = w[t - 1];
      const int j = k0 - (i + 1);
      w[0] = (j >= 0 && j < n) ? sA[j] : Acc(0);
    }
#pragma unroll
    for (int t = 0; t < TK; ++t)
      if (k0 + t < nb && acc[t] > loc) loc = acc[t];
  }
  const Acc M = blockMax(loc, sR);

  if (nActiveOut) {
    const Acc thr = M * (Acc(1) - activeTol);
    int cloc = 0;
    for (int q = threadIdx.x; q < ntile; q += BD) {
      const int k0 = q * TK;
      Acc acc[TK];
#pragma unroll
      for (int t = 0; t < TK; ++t) acc[t] = Acc(0);
      const int ilo = (k0 - (n - 1)) > 0 ? (k0 - (n - 1)) : 0;
      const int ihi = (k0 + TK - 1) < (n - 1) ? (k0 + TK - 1) : (n - 1);
      Acc w[TK];
#pragma unroll
      for (int t = 0; t < TK; ++t) { const int j = k0 + t - ilo; w[t] = (j >= 0 && j < n) ? sA[j] : Acc(0); }
      for (int i = ilo; i <= ihi; ++i) {
        const Acc x = sA[i];
#pragma unroll
        for (int t = 0; t < TK; ++t) acc[t] += x * w[t];
#pragma unroll
        for (int t = TK - 1; t > 0; --t) w[t] = w[t - 1];
        const int j = k0 - (i + 1);
        w[0] = (j >= 0 && j < n) ? sA[j] : Acc(0);
      }
#pragma unroll
      for (int t = 0; t < TK; ++t)
        if (k0 + t < nb && acc[t] >= thr) ++cloc;
    }
    const Acc c = blockSum(Acc(cloc), sR);
    if (threadIdx.x == 0) nActiveOut[pid] = (int)(c + Acc(0.5));
  }
  if (threadIdx.x == 0) out[pid] = (S > Acc(0)) ? (Acc(2) * Acc(n) * M / (S * S)) : Acc(1e30);
}

template <typename Acc>
inline size_t evalSmem(int n) { return sizeof(Acc) * (size_t)(n + 64); }

// ---------------------------------------------------------------------------
//  FRANK-WOLFE / LP STEP  (the Matolcsi-Vinuesa "good direction" iteration --
//  the workhorse behind every record on this problem since 2009).
//
//  Given a >= 0 with sum a = 1 and M = max_k (a*a)_k, solve the LP
//        maximise  sum_j g_j     subject to  (a*g)_k <= M  for all k,  g >= 0
//  and move  a <- (1-t)a + t*g  along an exactly line-searched t.  a itself is
//  feasible, so the LP optimum is >= 1; whenever it is > 1 the move strictly
//  decreases F, because the mass grows faster than the convolution ceiling.
//
//  There is no LP solver on a GPU, so we solve the saddle point
//        max_{g>=0} min_{lambda>=0}  1^T g - lambda^T (A g - M 1)
//  with Chambolle-Pock (PDHG).  Every iteration is exactly one convolution
//  (A g = a * g) plus one correlation (A^T lambda), i.e. precisely the two
//  kernels we already have.  ||A||_2 = max_w |ahat(w)| = sum a = 1, so the step
//  sizes tau = 0.9/n and sigma = 0.9n satisfy tau*sigma*||A||^2 < 1 while also
//  matching the natural scales (g_j ~ 1/n, lambda_k ~ 1/(nM)).
//
//  This complements the L^p local search: Adam does careful local descent,
//  Frank-Wolfe takes globally-optimal linearised jumps.
//
//  Shared: sA[n] + sG[n] + sGbar[n] + sLam[2n+TI+8] + 64  =  ~5n.
// ---------------------------------------------------------------------------
template <typename T>
inline size_t fwSmem(int n) { return sizeof(T) * (size_t)(5 * n + TI + 80); }

template <typename T>
__global__ __launch_bounds__(BD) void kFrankWolfe(
    T* __restrict__ pop, const int* __restrict__ skip,
    int n, int rounds, int pdhgIters, int nT, T tMax) {
  extern __shared__ __align__(16) char smemRaw[];
  T* sA   = reinterpret_cast<T*>(smemRaw);
  T* sG   = sA + n;
  T* sGb  = sG + n;
  T* sLam = sGb + n;                    // 2n + TI + 8 slots, tail kept at zero
  T* sR   = sLam + 2 * n + TI + 8;      // 64 reduction slots

  const int pid = blockIdx.x;
  if (skip && skip[pid]) return;
  T* a = pop + (size_t)pid * n;

  const int nb     = 2 * n - 1;
  const int ntileK = (nb + TK - 1) / TK;
  const int ntileI = (n + TI - 1) / TI;

  for (int i = threadIdx.x; i < n; i += BD) sA[i] = a[i];
  for (int k = threadIdx.x; k < 2 * n + TI + 8; k += BD) sLam[k] = T(0);
  __syncthreads();
  {
    T s = T(0);
    for (int i = threadIdx.x; i < n; i += BD) s += sA[i];
    s = blockSum(s, sR);
    if (!(s > T(0))) return;
    const T inv = T(1) / s;
    for (int i = threadIdx.x; i < n; i += BD) sA[i] *= inv;
    __syncthreads();
  }

  const T tau = T(0.9) / T(n), sig = T(0.9) * T(n);

  for (int r = 0; r < rounds; ++r) {
    crossconv(sA, sA, sLam, n);
    __syncthreads();
    T loc = T(0);
    for (int k = threadIdx.x; k < nb; k += BD) loc = sLam[k] > loc ? sLam[k] : loc;
    const T M = blockMax(loc, sR);
    if (!(M > T(0))) break;

    // ---------------- PDHG ----------------
    for (int i = threadIdx.x; i < n; i += BD) { sG[i] = sA[i]; sGb[i] = sA[i]; }
    for (int k = threadIdx.x; k < nb; k += BD) sLam[k] = T(0);
    __syncthreads();

    for (int it = 0; it < pdhgIters; ++it) {
      // lambda <- [lambda + sigma*(a*gbar - M)]_+   (conv fused into the update)
      for (int q = threadIdx.x; q < ntileK; q += BD) {
        const int k0 = q * TK;
        T acc[TK];
#pragma unroll
        for (int t = 0; t < TK; ++t) acc[t] = T(0);
        const int ilo = (k0 - (n - 1)) > 0 ? (k0 - (n - 1)) : 0;
        const int ihi = (k0 + TK - 1) < (n - 1) ? (k0 + TK - 1) : (n - 1);
        T w[TK];
#pragma unroll
        for (int t = 0; t < TK; ++t) { const int j = k0 + t - ilo; w[t] = (j >= 0 && j < n) ? sGb[j] : T(0); }
        for (int i = ilo; i <= ihi; ++i) {
          const T x = sA[i];
#pragma unroll
          for (int t = 0; t < TK; ++t) acc[t] += x * w[t];
#pragma unroll
          for (int t = TK - 1; t > 0; --t) w[t] = w[t - 1];
          const int j = k0 - (i + 1);
          w[0] = (j >= 0 && j < n) ? sGb[j] : T(0);
        }
#pragma unroll
        for (int t = 0; t < TK; ++t) {
          const int k = k0 + t;
          if (k < nb) { const T l = sLam[k] + sig * (acc[t] - M); sLam[k] = l > T(0) ? l : T(0); }
        }
      }
      __syncthreads();

      // g_new <- [g + tau*(1 - A^T lambda)]_+   (parked in sGb, freeing gbar)
      for (int q = threadIdx.x; q < ntileI; q += BD) {
        const int i0 = q * TI;
        T c[TI];
        corrTile(sA, sLam, n, i0, c);
#pragma unroll
        for (int s = 0; s < TI; ++s) {
          const int i = i0 + s;
          if (i < n) { const T gn = sG[i] + tau * (T(1) - c[s]); sGb[i] = gn > T(0) ? gn : T(0); }
        }
      }
      __syncthreads();
      // gbar <- 2*g_new - g_old ; g <- g_new
      for (int i = threadIdx.x; i < n; i += BD) {
        const T gn = sGb[i], go = sG[i];
        sG[i]  = gn;
        sGb[i] = T(2) * gn - go;
      }
      __syncthreads();
    }

    // ---------------- normalise g, then exact line search ----------------
    T sg = T(0);
    for (int i = threadIdx.x; i < n; i += BD) sg += sG[i];
    sg = blockSum(sg, sR);
    if (!(sg > T(0))) break;
    { const T inv = T(1) / sg; for (int i = threadIdx.x; i < n; i += BD) sG[i] *= inv; }
    __syncthreads();

    // sum a = sum g = 1  =>  sum h = 1 for every t, so F = 2n*max(h*h)
    T bestT = T(0), bestM = M;
    for (int s = 1; s <= nT; ++s) {
      const T t = tMax * T(s) / T(nT);
      for (int i = threadIdx.x; i < n; i += BD) sGb[i] = (T(1) - t) * sA[i] + t * sG[i];
      __syncthreads();
      crossconv(sGb, sGb, sLam, n);
      __syncthreads();
      T lc = T(0);
      for (int k = threadIdx.x; k < nb; k += BD) lc = sLam[k] > lc ? sLam[k] : lc;
      const T mm = blockMax(lc, sR);
      if (mm < bestM) { bestM = mm; bestT = t; }
    }
    if (!(bestT > T(0))) break;                       // no improving step left
    for (int i = threadIdx.x; i < n; i += BD) sA[i] = (T(1) - bestT) * sA[i] + bestT * sG[i];
    __syncthreads();
  }

  for (int i = threadIdx.x; i < n; i += BD) a[i] = sA[i];
}

// ---------------------------------------------------------------------------
//  Multiresolution refinement:  a'_{2i} = a'_{2i+1} = a_i  (n -> 2n).
//  The underlying step function is unchanged, so F is preserved EXACTLY.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kRefine(const T* __restrict__ src, int n, T* __restrict__ dst) {
  const int pid = blockIdx.y;
  const T* s = src + (size_t)pid * n;
  T* d = dst + (size_t)pid * 2 * n;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const T x = s[i];
    d[2 * i] = x; d[2 * i + 1] = x;
  }
}

// coarsening (2n -> n) by pairwise averaging, used by some variation operators
template <typename T>
__global__ void kCoarsen(const T* __restrict__ src, int n2, T* __restrict__ dst) {
  const int pid = blockIdx.y;
  const T* s = src + (size_t)pid * n2;
  T* d = dst + (size_t)pid * (n2 / 2);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2 / 2; i += gridDim.x * blockDim.x)
    d[i] = T(0.5) * (s[2 * i] + s[2 * i + 1]);
}

// ---------------------------------------------------------------------------
//  Variation operators.  One block per offspring.  The host decides parents
//  and operator ids so that rates stay easy to tune and to log.
//
//  op 0  blend crossover           op 5  sparsify (threshold small entries)
//  op 1  segment (2-point) xover   op 6  symmetrise / reflect
//  op 2  log-normal multiplicative op 7  shift
//  op 3  smooth low-freq (Fourier) op 8  block resample from the other parent
//  op 4  additive gaussian         op 9  fresh random restart
//  op 10 spike injection
//
//  Ops 11-13 encode the STRUCTURAL PRIOR that every published record shares
//  (Matolcsi-Vinuesa 2009 through TTT-Discover 2026): a near-atomic spike at
//  one endpoint of [-1/4,1/4], a wide hard support gap covering several per
//  cent of the interval, a ragged non-smooth body, ~30 % exact zeros, and a
//  pronounced left/right mass asymmetry (~1/sqrt(2)).  Sampling that family
//  directly is far more effective than hoping generic mutation stumbles on it.
//
//  op 11 structured seed (atom + gap + ragged body + asymmetry)
//  op 12 gap surgery    (carve / fill a hard support gap)
//  op 13 endpoint atom  (grow or shrink the boundary spike)
//  op 20 clone (untouched elite copy)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(BD) void kVariation(
    const T* __restrict__ pop, T* __restrict__ out, int n, int nOff,
    const int* __restrict__ p1, const int* __restrict__ p2,
    const int* __restrict__ op, const float* __restrict__ strength,
    unsigned long long seed, unsigned long long gen) {
  extern __shared__ __align__(16) char smemRaw[];
  T* sA = reinterpret_cast<T*>(smemRaw);
  T* sX = sA + n;
  float* sPar = reinterpret_cast<float*>(sX + n);   // [0,64) shared randomness,
  T*     sRd  = reinterpret_cast<T*>(sPar + 64);    // [64,..) reduction scratch

  const int j = blockIdx.x;
  if (j >= nOff) return;
  const int o = op[j];
  const T* A = pop + (size_t)p1[j] * n;
  const T* B = pop + (size_t)p2[j] * n;
  T* O = out + (size_t)j * n;
  const float sg = strength[j];

  curandStatePhilox4_32_10_t st;
  curand_init(seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)j,
              (unsigned long long)threadIdx.x, gen * 4096ULL, &st);

  if (threadIdx.x == 0) {
    curandStatePhilox4_32_10_t s0;
    curand_init(seed ^ 0xD1B54A32D192ED03ULL, (unsigned long long)j, gen * 4096ULL, &s0);
    for (int i = 0; i < 64; ++i) sPar[i] = curand_uniform(&s0);
  }
  for (int i = threadIdx.x; i < n; i += BD) { sA[i] = A[i]; sX[i] = B[i]; }
  __syncthreads();

  switch (o) {
    case 0: {  // blend crossover, lambda in [-0.3, 1.3]
      const T lam = T(-0.3f + 1.6f * sPar[0]);
      for (int i = threadIdx.x; i < n; i += BD) {
        T x = lam * sA[i] + (T(1) - lam) * sX[i];
        O[i] = x > T(0) ? x : T(0);
      }
      break;
    }
    case 1: {  // two-point segment crossover
      int c1 = (int)(sPar[1] * n), c2 = (int)(sPar[2] * n);
      if (c1 > c2) { const int t = c1; c1 = c2; c2 = t; }
      for (int i = threadIdx.x; i < n; i += BD) O[i] = (i >= c1 && i < c2) ? sX[i] : sA[i];
      break;
    }
    case 2: {  // log-normal multiplicative noise
      const T s = T(sg);
      for (int i = threadIdx.x; i < n; i += BD) {
        const float z = curand_normal(&st);
        O[i] = sA[i] * d_exp_(T(s * z));
      }
      break;
    }
    case 3: {  // smooth low-frequency perturbation (few random Fourier modes)
      const int nf = 1 + (int)(sPar[3] * 5.0f);
      for (int i = threadIdx.x; i < n; i += BD) {
        float m = 0.f;
        for (int f = 0; f < nf; ++f) {
          const float freq = 1.0f + sPar[4 + f] * 12.0f;
          const float ph   = 6.2831853f * sPar[12 + f];
          m += __cosf(6.2831853f * freq * (float)i / (float)n + ph) / (float)nf;
        }
        T x = sA[i] * T(1.0f + sg * 3.0f * m);
        O[i] = x > T(0) ? x : T(0);
      }
      break;
    }
    case 4: {  // additive gaussian noise scaled by the mean height
      T mean = T(0);
      for (int i = threadIdx.x; i < n; i += BD) mean += sA[i];
      mean = blockSum(mean, sRd) / T(n);
      for (int i = threadIdx.x; i < n; i += BD) {
        T x = sA[i] + T(sg) * mean * T(curand_normal(&st));
        O[i] = x > T(0) ? x : T(0);
      }
      break;
    }
    case 5: {  // sparsify: kill entries below a fraction of the mean
      T mean = T(0);
      for (int i = threadIdx.x; i < n; i += BD) mean += sA[i];
      mean = blockSum(mean, sRd) / T(n);
      const T thr = mean * T(0.05f + 0.9f * sPar[20]);
      for (int i = threadIdx.x; i < n; i += BD) O[i] = sA[i] < thr ? T(0) : sA[i];
      break;
    }
    case 6: {  // symmetrise (or hard reflect)
      const bool hard = sPar[21] < 0.5f;
      for (int i = threadIdx.x; i < n; i += BD)
        O[i] = hard ? sA[n - 1 - i] : T(0.5) * (sA[i] + sA[n - 1 - i]);
      break;
    }
    case 7: {  // shift by up to +-n/8, zero fill
      const int sh = (int)((sPar[22] - 0.5f) * (float)n * 0.25f);
      for (int i = threadIdx.x; i < n; i += BD) {
        const int k = i - sh;
        O[i] = (k >= 0 && k < n) ? sA[k] : T(0);
      }
      break;
    }
    case 8: {  // block resample: splice a random block of B into A
      const int len = 1 + (int)(sPar[23] * 0.5f * (float)n);
      const int st0 = (int)(sPar[24] * (float)(n - len > 0 ? n - len : 1));
      for (int i = threadIdx.x; i < n; i += BD)
        O[i] = (i >= st0 && i < st0 + len) ? sX[i] : sA[i];
      break;
    }
    case 9: {  // fresh random restart: smooth positive random function
      const int nf = 2 + (int)(sPar[25] * 6.0f);
      for (int i = threadIdx.x; i < n; i += BD) {
        float m = 0.f;
        for (int f = 0; f < nf; ++f) {
          const float freq = 1.0f + sPar[4 + (f % 8)] * 10.0f;
          const float ph   = 6.2831853f * sPar[12 + (f % 8)];
          m += __cosf(6.2831853f * freq * (float)i / (float)n + ph);
        }
        float base = 0.3f + 0.7f * sPar[26];
        float val  = base + m / (float)nf + 0.4f * curand_normal(&st);
        O[i] = val > 0.f ? T(val) : T(0);
      }
      break;
    }
    case 10: {  // spike injection at a random location
      const int loc = (int)(sPar[27] * (float)n);
      const int wdt = 1 + (int)(sPar[28] * (float)n * 0.03f);
      T mean = T(0);
      for (int i = threadIdx.x; i < n; i += BD) mean += sA[i];
      mean = blockSum(mean, sRd) / T(n);
      const T amp = mean * T(1.0f + 6.0f * sPar[29]);
      for (int i = threadIdx.x; i < n; i += BD) {
        T x = sA[i];
        if (i >= loc - wdt && i <= loc + wdt) x += amp;
        O[i] = x;
      }
      break;
    }
    case 11: {
      // Structured seed drawn from the family every published record lives in:
      //   heavy ragged plateau -> deep depression -> second hump -> moderate
      //   ragged band -> WIDE HARD GAP -> terminal spike, with a near-atom at
      //   the starting endpoint.  Randomising the breakpoints samples the
      //   family rather than one member of it.
      const bool  flip  = sPar[30] < 0.5f;                 // which end holds the atom
      const float atom  = 30.f + 250.f * sPar[31];         // endpoint atom, x mean density
      const float pl1   = 0.28f + 0.16f * sPar[32];        // end of the heavy plateau
      const float pl2   = pl1  + 0.10f + 0.14f * sPar[33]; // end of the depression
      const float hump  = pl2  + 0.02f + 0.08f * sPar[34]; // end of the second hump
      const float gap0  = 0.70f + 0.16f * sPar[35];        // start of the dead gap
      const float gap1  = gap0 + 0.05f + 0.15f * sPar[36];
      const float tail  = 3.0f + 14.0f * sPar[37];         // terminal spike density
      const int   nf    = 3 + (int)(sPar[38] * 6.f);
      for (int i = threadIdx.x; i < n; i += BD) {
        const int   d = flip ? (n - 1 - i) : i;
        const float x = ((float)d + 0.5f) / (float)n;
        float rough = 0.f;
        for (int f = 0; f < nf; ++f)
          rough += __cosf(6.2831853f * (2.f + sPar[40 + f] * 40.f) * x + 6.2831853f * sPar[48 + f]);
        rough /= (float)nf;
        float v;
        if      (x < pl1)  v = 4.2f * (1.f + 0.45f * rough);
        else if (x < pl2)  v = 0.7f * (1.f + 0.80f * rough);
        else if (x < hump) v = 5.2f * (1.f + 0.30f * rough);
        else if (x < gap0) v = 2.0f * (1.f + 0.70f * rough);
        else if (x < gap1) v = 0.f;
        else               v = tail * (1.f + 0.20f * rough);
        if      (d == 0) v += atom;
        else if (d <  4) v += atom * 0.10f / (float)d;
        O[i] = v > 0.f ? T(v) : T(0);
      }
      break;
    }
    case 12: {  // gap surgery: carve out (or fill in) a hard support gap
      T mean = T(0);
      for (int i = threadIdx.x; i < n; i += BD) mean += sA[i];
      mean = blockSum(mean, sRd) / T(n);
      const int  len   = 1 + (int)((0.015f + 0.13f * sPar[30]) * (float)n);
      const int  st0   = (int)(sPar[31] * (float)(n > len ? n - len : 1));
      const bool carve = sPar[32] < 0.7f;
      const T    fill  = mean * T(0.2f + 1.8f * sPar[33]);
      for (int i = threadIdx.x; i < n; i += BD) {
        T x = sA[i];
        if (i >= st0 && i < st0 + len) x = carve ? T(0) : (x + fill);
        O[i] = x;
      }
      break;
    }
    case 13: {  // endpoint atom: grow or shrink the boundary spike
      T mean = T(0);
      for (int i = threadIdx.x; i < n; i += BD) mean += sA[i];
      mean = blockSum(mean, sRd) / T(n);
      const bool left = sPar[30] < 0.5f;
      const int  w    = 1 + (int)(sPar[31] * 0.012f * (float)n);
      const T    add  = mean * T(-20.f + 260.f * sPar[32]);
      for (int i = threadIdx.x; i < n; i += BD) {
        const int d = left ? i : (n - 1 - i);
        T x = sA[i];
        if (d < w) { x += add / T(1 + d); if (x < T(0)) x = T(0); }
        O[i] = x;
      }
      break;
    }
    default: {  // clone
      for (int i = threadIdx.x; i < n; i += BD) O[i] = sA[i];
      break;
    }
  }
  __syncthreads();

  // renormalise sum to 1 (objective is scale invariant; keeps magnitudes sane)
  T sloc = T(0);
  for (int i = threadIdx.x; i < n; i += BD) sloc += O[i];
  const T S = blockSum(sloc, sRd);
  if (S > T(0)) {
    const T inv = T(1) / S;
    for (int i = threadIdx.x; i < n; i += BD) O[i] *= inv;
  } else {
    for (int i = threadIdx.x; i < n; i += BD) O[i] = T(1) / T(n);
  }
}

// ---------------------------------------------------------------------------
//  Support signature for the tabu / archive machinery: a 64-bit hash of the
//  zero-pattern of `a` (indices with a_i <= tol * mean), plus the count of
//  non-zero entries.  Two candidates with the same signature sit in the same
//  combinatorial cell of the search space.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(BD) void kSupportSig(const T* __restrict__ pop, int n,
                                                  float tol,
                                                  unsigned long long* __restrict__ sig,
                                                  int* __restrict__ nnz) {
  extern __shared__ __align__(16) char smemRaw[];
  T* sR = reinterpret_cast<T*>(smemRaw);
  const int pid = blockIdx.x;
  const T* a = pop + (size_t)pid * n;

  T sloc = T(0);
  for (int i = threadIdx.x; i < n; i += BD) sloc += a[i];
  const T mean = blockSum(sloc, sR) / T(n);
  const T thr  = mean * T(tol);

  unsigned long long h = 1469598103934665603ULL;
  int c = 0;
  for (int i = threadIdx.x; i < n; i += BD) {
    if (a[i] > thr) {
      ++c;
      unsigned long long x = (unsigned long long)i * 0x9E3779B97F4A7C15ULL;
      x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 32;
      h ^= x;                                     // xor: order independent
    }
  }
  // xor-reduce h and sum-reduce c across the block
  unsigned long long hh = h;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) hh ^= __shfl_down_sync(0xffffffffu, hh, off);
  unsigned long long* sH = reinterpret_cast<unsigned long long*>(sR + 64);
  if ((threadIdx.x & 31) == 0) sH[threadIdx.x >> 5] = hh;
  __syncthreads();
  const T cc = blockSum(T(c), sR);
  if (threadIdx.x == 0) {
    unsigned long long r = 0;
    const int nw = (BD + 31) >> 5;
    for (int i = 0; i < nw; ++i) r ^= sH[i];
    sig[pid] = r;
    nnz[pid] = (int)(cc + T(0.5));
  }
}
