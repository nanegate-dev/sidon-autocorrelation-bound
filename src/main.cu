// ---------------------------------------------------------------------------
//  sidon.exe -- GPU memetic evolutionary search for AlphaEvolve Problem 2
//
//  Minimise  F(a) = 2*n*max_k b_k / (sum_i a_i)^2,  b_k = sum_{i+j=k} a_i a_j,
//  a_i >= 0.  Every value of F produced here is a rigorous upper bound on the
//  constant C of the autocorrelation / Sidon-set problem.
//
//  Architecture
//    * fully GPU resident: population, variation, local search, evaluation
//    * memetic: every individual is polished by L^p-annealed projected Adam
//      (or mirror descent) before it is ranked
//    * multiresolution ladder n -> 2n (refinement preserves F exactly)
//    * island model with per-island exponent / step / mode regimes + migration
//    * tabu list over SUPPORT PATTERNS (the discrete skeleton of the problem)
//      plus an elite archive with distance-based rejection
//    * adaptive operator selection (bandit-style credit assignment)
//    * final fp64 polish with a very large exponent
// ---------------------------------------------------------------------------
#include "kernels.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;
static double nowSec(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ------------------------------- CLI ---------------------------------------
struct Args {
  std::vector<int> ladder{64, 128, 256, 512, 1024, 2048};
  int    pop        = 256;
  int    islands    = 8;
  int    elite      = 2;      // elites per island
  int    lsIters    = 4000;   // WARM local-search iterations (converged parents)
  int    coldIters  = 20000;  // COLD local-search iterations (restarts, restructures)
  double pMax       = 20000;  // top of the fp32 exponent anneal
  int    fwRounds   = 6;      // Frank-Wolfe outer rounds per generation (0 = off)
  int    fwIters    = 500;    // PDHG iterations per Frank-Wolfe round
  int    gensPerLvl = 0;      // 0 => governed purely by the time budget
  double seconds    = 600.0;  // total wall-clock budget
  double lvlFrac    = 0.0;    // 0 => auto (later levels get more time)
  unsigned long long seed = 20260905ULL;
  std::string out    = "best";
  std::string resume;
  int    polishIters = 40000;
  int    polishTop   = 8;
  bool   selftest    = false;
  bool   bench       = false;
  int    verbose     = 1;
};

static void usage() {
  std::printf(
      "sidon.exe [options]\n"
      "  --ladder 64,128,...   multiresolution ladder (default 64..2048)\n"
      "  --pop N               population size (default 256)\n"
      "  --islands N           number of islands (default 8)\n"
      "  --elite N             elites kept per island (default 2)\n"
      "  --ls-iters N          local-search steps per generation (default 300)\n"
      "  --gens N              generations per level (0 = time driven)\n"
      "  --seconds S           total wall-clock budget (default 600)\n"
      "  --seed S              RNG seed\n"
      "  --out PREFIX          output prefix (default 'best')\n"
      "  --resume FILE         seed the run from a saved vector\n"
      "  --polish-iters N      fp64 polish iterations (default 40000)\n"
      "  --polish-top N        how many elites to polish (default 8)\n"
      "  --selftest            verify the objective against hand values\n"
      "  --bench               micro-benchmark the kernels and exit\n"
      "  --quiet\n");
}

static std::vector<int> parseList(const std::string& s) {
  std::vector<int> v; std::string cur;
  for (char c : s) {
    if (c == ',') { if (!cur.empty()) v.push_back(std::stoi(cur)); cur.clear(); }
    else cur.push_back(c);
  }
  if (!cur.empty()) v.push_back(std::stoi(cur));
  return v;
}

// ------------------------- host reference objective -------------------------
static double hostF(const std::vector<double>& a) {
  const int n = (int)a.size();
  std::vector<double> b(2 * n - 1, 0.0);
  for (int i = 0; i < n; ++i) {
    if (a[i] == 0.0) continue;
    for (int j = 0; j < n; ++j) b[i + j] += a[i] * a[j];
  }
  double M = 0.0; for (double x : b) M = std::max(M, x);
  double S = 0.0; for (double x : a) S += x;
  return 2.0 * n * M / (S * S);
}

// ------------------------------- GPU state ---------------------------------
struct Gpu {
  int n = 0, P = 0;
  float  *pop = nullptr, *nxt = nullptr, *mom = nullptr, *vel = nullptr;
  double *fit = nullptr;
  int    *nact = nullptr, *modeArr = nullptr, *skip = nullptr;
  int    *p1 = nullptr, *p2 = nullptr, *op = nullptr;
  float  *strength = nullptr;
  unsigned long long* sig = nullptr;
  int    *nnz = nullptr;

  void alloc(int n_, int P_) {
    free();
    n = n_; P = P_;
    const size_t N = (size_t)n * P;
    CUDA_OK(cudaMalloc(&pop, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&nxt, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&mom, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&vel, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&fit, P * sizeof(double)));
    CUDA_OK(cudaMalloc(&nact, P * sizeof(int)));
    CUDA_OK(cudaMalloc(&modeArr, P * sizeof(int)));
    CUDA_OK(cudaMalloc(&skip, P * sizeof(int)));
    CUDA_OK(cudaMalloc(&p1, P * sizeof(int)));
    CUDA_OK(cudaMalloc(&p2, P * sizeof(int)));
    CUDA_OK(cudaMalloc(&op, P * sizeof(int)));
    CUDA_OK(cudaMalloc(&strength, P * sizeof(float)));
    CUDA_OK(cudaMalloc(&sig, P * sizeof(unsigned long long)));
    CUDA_OK(cudaMalloc(&nnz, P * sizeof(int)));
    CUDA_OK(cudaMemset(skip, 0, P * sizeof(int)));
  }
  void free() {
    if (pop) { cudaFree(pop); cudaFree(nxt); cudaFree(mom); cudaFree(vel); cudaFree(fit);
               cudaFree(nact); cudaFree(modeArr); cudaFree(skip); cudaFree(p1); cudaFree(p2);
               cudaFree(op); cudaFree(strength); cudaFree(sig); cudaFree(nnz); }
    pop = nullptr;
  }
};

// ---------------------------- kernel launch helpers -------------------------
static void setSmemLimit(const void* fn, size_t bytes) {
  if (bytes > 48 * 1024)
    CUDA_OK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

static void runLocalSearch(Gpu& g, int iters, float pA, float pB, float lrA, float lrB,
                           const int* modeArr, const int* skip = nullptr) {
  const size_t sm = localSearchSmem<float>(g.n);
  setSmemLimit((const void*)kLocalSearch<float>, sm);
  kLocalSearch<float><<<g.P, BD, sm>>>(g.pop, g.mom, g.vel, skip, modeArr,
                                       g.n, iters, pA, pB, lrA, lrB, 0.9f, 0.999f, 0);
  CUDA_OK(cudaGetLastError());
}

static void runEval(Gpu& g, double activeTol = 1e-4) {
  const size_t sm = evalSmem<double>(g.n);
  setSmemLimit((const void*)kEval<float, double>, sm);
  kEval<float, double><<<g.P, BD, sm>>>(g.pop, g.n, g.fit, g.nact, activeTol);
  CUDA_OK(cudaGetLastError());
}

// Frank-Wolfe / LP step.  Returns false if n is too large for the 5n shared
// footprint on this device (the caller then simply skips the step).
static bool runFrankWolfe(Gpu& g, int rounds, int pdhgIters, int nT, float tMax,
                          int smemOptin, const int* skip = nullptr) {
  const size_t sm = fwSmem<float>(g.n);
  if ((int)sm > smemOptin) return false;
  setSmemLimit((const void*)kFrankWolfe<float>, sm);
  kFrankWolfe<float><<<g.P, BD, sm>>>(g.pop, skip, g.n, rounds, pdhgIters, nT, tMax);
  CUDA_OK(cudaGetLastError());
  return true;
}

static void runSig(Gpu& g, float tol = 0.02f) {
  const size_t sm = sizeof(float) * 64 + sizeof(unsigned long long) * 16;
  kSupportSig<float><<<g.P, BD, sm>>>(g.pop, g.n, tol, g.sig, g.nnz);
  CUDA_OK(cudaGetLastError());
}

// ============================== self test ==================================
static int selftest() {
  std::printf("--- self test ---\n");
  int bad = 0;
  auto check = [&](const char* name, double got, double want, double tol) {
    const bool ok = std::fabs(got - want) <= tol * std::max(1.0, std::fabs(want));
    std::printf("  %-28s got %.12f  want %.12f   %s\n", name, got, want, ok ? "OK" : "FAIL");
    if (!ok) ++bad;
  };

  // (1) constant vector  -> F = 2 exactly, for every n
  for (int n : {4, 8, 37, 128}) {
    std::vector<double> a(n, 1.0);
    char nm[64]; std::snprintf(nm, sizeof nm, "constant n=%d", n);
    check(nm, hostF(a), 2.0, 1e-12);
  }
  // (2) a = (1,2,2,1), n = 4 -> b = (1,4,8,10,8,4,1), M = 10, S = 6 -> 2*4*10/36
  { std::vector<double> a{1, 2, 2, 1}; check("(1,2,2,1) n=4", hostF(a), 80.0 / 36.0, 1e-12); }
  // (3) a = (1,0,0,1), n = 4 -> b = (1,0,0,2,0,0,1), M = 2, S = 2 -> 2*4*2/4 = 4
  { std::vector<double> a{1, 0, 0, 1}; check("(1,0,0,1) n=4", hostF(a), 4.0, 1e-12); }
  // (4) single spike -> F = 2n / 1 ... a=(0,1,0,0): b_2 = 1, S=1 -> 2*4*1/1 = 8
  { std::vector<double> a{0, 1, 0, 0}; check("spike n=4", hostF(a), 8.0, 1e-12); }
  // (5) refinement invariance: a -> (a0,a0,a1,a1,...) must preserve F exactly
  {
    std::mt19937 rng(7); std::uniform_real_distribution<double> U(0, 1);
    for (int rep = 0; rep < 3; ++rep) {
      const int n = 23 + rep * 11;
      std::vector<double> a(n); for (auto& x : a) x = U(rng);
      std::vector<double> a2(2 * n);
      for (int i = 0; i < n; ++i) { a2[2 * i] = a[i]; a2[2 * i + 1] = a[i]; }
      char nm[64]; std::snprintf(nm, sizeof nm, "refine invariance n=%d", n);
      check(nm, hostF(a2), hostF(a), 1e-12);
    }
  }
  // (6) scale invariance
  {
    std::mt19937 rng(11); std::uniform_real_distribution<double> U(0, 1);
    const int n = 40; std::vector<double> a(n); for (auto& x : a) x = U(rng);
    std::vector<double> s = a; for (auto& x : s) x *= 37.5;
    check("scale invariance", hostF(s), hostF(a), 1e-12);
  }
  // (7) GPU kEval must agree with the host reference
  {
    std::mt19937 rng(3); std::uniform_real_distribution<double> U(0, 1);
    const int n = 257, P = 5;
    std::vector<float> h((size_t)n * P);
    std::vector<double> hf(P);
    for (int p = 0; p < P; ++p) {
      std::vector<double> a(n);
      for (int i = 0; i < n; ++i) { a[i] = (p == 0) ? 1.0 : U(rng); h[(size_t)p * n + i] = (float)a[i]; }
      // recompute from the exact float values that the GPU will see
      std::vector<double> af(n);
      for (int i = 0; i < n; ++i) af[i] = (double)h[(size_t)p * n + i];
      hf[p] = hostF(af);
    }
    float* d; CUDA_OK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    double* df; CUDA_OK(cudaMalloc(&df, P * sizeof(double)));
    int* dn; CUDA_OK(cudaMalloc(&dn, P * sizeof(int)));
    const size_t sm = evalSmem<double>(n);
    setSmemLimit((const void*)kEval<float, double>, sm);
    kEval<float, double><<<P, BD, sm>>>(d, n, df, dn, 1e-9);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<double> got(P);
    CUDA_OK(cudaMemcpy(got.data(), df, P * sizeof(double), cudaMemcpyDeviceToHost));
    for (int p = 0; p < P; ++p) {
      char nm[64]; std::snprintf(nm, sizeof nm, "GPU kEval individual %d", p);
      check(nm, got[p], hf[p], 1e-13);
    }
    cudaFree(d); cudaFree(df); cudaFree(dn);
  }
  // (8) the fp32 local search must strictly improve a random start
  {
    const int n = 128, P = 4;
    Gpu g; g.alloc(n, P);
    std::mt19937 rng(5); std::uniform_real_distribution<float> U(0.1f, 1.0f);
    std::vector<float> h((size_t)n * P); for (auto& x : h) x = U(rng);
    CUDA_OK(cudaMemcpy(g.pop, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    runEval(g); CUDA_OK(cudaDeviceSynchronize());
    std::vector<double> f0(P); CUDA_OK(cudaMemcpy(f0.data(), g.fit, P * sizeof(double), cudaMemcpyDeviceToHost));
    runLocalSearch(g, 800, 4.f, 300.f, 0.06f, 0.002f, nullptr);
    runEval(g); CUDA_OK(cudaDeviceSynchronize());
    std::vector<double> f1(P); CUDA_OK(cudaMemcpy(f1.data(), g.fit, P * sizeof(double), cudaMemcpyDeviceToHost));
    for (int p = 0; p < P; ++p) {
      std::printf("  local search ind %d: %.9f -> %.9f  %s\n", p, f0[p], f1[p],
                  f1[p] < f0[p] ? "OK" : "FAIL");
      if (!(f1[p] < f0[p])) ++bad;
    }
    // and the polished vector must still be non-negative and finite
    std::vector<float> hh((size_t)n * P);
    CUDA_OK(cudaMemcpy(hh.data(), g.pop, hh.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool okneg = true;
    for (float x : hh) if (!(x >= 0.f) || !std::isfinite(x)) okneg = false;
    std::printf("  non-negativity preserved: %s\n", okneg ? "OK" : "FAIL");
    if (!okneg) ++bad;
    // GPU value must match the host reference on the polished vector
    std::vector<double> a0(n);
    for (int i = 0; i < n; ++i) a0[i] = (double)hh[i];
    std::printf("  polished GPU %.12f vs host %.12f  %s\n", f1[0], hostF(a0),
                std::fabs(f1[0] - hostF(a0)) < 1e-12 ? "OK" : "FAIL");
    if (std::fabs(f1[0] - hostF(a0)) >= 1e-12) ++bad;
    g.free();
  }
  std::printf("--- %s (%d failures) ---\n", bad ? "FAILED" : "ALL PASSED", bad);
  return bad ? 1 : 0;
}

// ============================== benchmark ==================================
static void bench() {
  std::printf("--- benchmark ---\n");
  for (int n : {128, 256, 512, 1024, 2048, 4096}) {
    const int P = std::max(32, std::min(512, 4 * 1024 * 1024 / n));
    Gpu g; g.alloc(n, P);
    std::vector<float> h((size_t)n * P, 1.0f / n);
    std::mt19937 rng(1); std::uniform_real_distribution<float> U(0.2f, 1.f);
    for (auto& x : h) x = U(rng);
    CUDA_OK(cudaMemcpy(g.pop, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int it = 100;
    runLocalSearch(g, 10, 8.f, 8.f, 0.01f, 0.01f, nullptr);
    CUDA_OK(cudaDeviceSynchronize());
    auto t0 = Clock::now();
    runLocalSearch(g, it, 8.f, 200.f, 0.05f, 0.005f, nullptr);
    CUDA_OK(cudaDeviceSynchronize());
    const double dt = nowSec(t0);
    auto t1 = Clock::now();
    runEval(g); CUDA_OK(cudaDeviceSynchronize());
    const double de = nowSec(t1);
    const double flops = 2.0 * (double)P * it * 2.0 * (double)n * n;
    std::printf("  n=%5d P=%4d  smem=%6zu KB  LS %7.2f ms (%6.2f ms/iter, %5.2f TFLOP/s)"
                "   eval(fp64) %6.2f ms\n",
                n, P, localSearchSmem<float>(n) / 1024, dt * 1e3, dt * 1e3 / it,
                flops / dt / 1e12, de * 1e3);
    g.free();
  }
}

// ============================== main search =================================
struct Best {
  double f = 1e30;
  int n = 0;
  std::vector<double> a;
};

static void saveVec(const std::string& path, const std::vector<double>& a, double f) {
  std::ofstream o(path);
  o.precision(17);
  o << "# AlphaEvolve problem 2 -- autocorrelation / Sidon\n";
  o << "# F = 2*n*max_k b_k / (sum a)^2  is an upper bound on C\n";
  o << "n " << a.size() << "\n";
  o << "F " << std::scientific << f << "\n";
  o << std::defaultfloat;
  for (double x : a) o << x << "\n";
}

static bool loadVec(const std::string& path, std::vector<double>& a) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line; int n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line.rfind("n ", 0) == 0) { n = std::stoi(line.substr(2)); continue; }
    if (line.rfind("F ", 0) == 0) continue;
    a.push_back(std::stod(line));
  }
  if (n && (int)a.size() != n) return false;
  return !a.empty();
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  Args A;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if      (s == "--ladder")       A.ladder = parseList(next());
    else if (s == "--pop")          A.pop = std::stoi(next());
    else if (s == "--islands")      A.islands = std::stoi(next());
    else if (s == "--elite")        A.elite = std::stoi(next());
    else if (s == "--ls-iters")     A.lsIters = std::stoi(next());
    else if (s == "--cold-iters")   A.coldIters = std::stoi(next());
    else if (s == "--pmax")         A.pMax = std::stod(next());
    else if (s == "--fw-rounds")    A.fwRounds = std::stoi(next());
    else if (s == "--fw-iters")     A.fwIters = std::stoi(next());
    else if (s == "--gens")         A.gensPerLvl = std::stoi(next());
    else if (s == "--seconds")      A.seconds = std::stod(next());
    else if (s == "--seed")         A.seed = std::stoull(next());
    else if (s == "--out")          A.out = next();
    else if (s == "--resume")       A.resume = next();
    else if (s == "--polish-iters") A.polishIters = std::stoi(next());
    else if (s == "--polish-top")   A.polishTop = std::stoi(next());
    else if (s == "--selftest")     A.selftest = true;
    else if (s == "--bench")        A.bench = true;
    else if (s == "--quiet")        A.verbose = 0;
    else if (s == "-h" || s == "--help") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); usage(); return 1; }
  }

  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s  sm_%d%d  %d SMs  smemOptin=%d B\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, (int)prop.sharedMemPerBlockOptin);

  if (A.selftest) return selftest();
  if (A.bench)    { bench(); return 0; }

  // sanity: every ladder level must fit in shared memory
  for (int n : A.ladder) {
    if ((int)localSearchSmem<float>(n) > prop.sharedMemPerBlockOptin) {
      std::fprintf(stderr, "n=%d needs %zu B shared, device offers %d\n", n,
                   localSearchSmem<float>(n), (int)prop.sharedMemPerBlockOptin);
      return 1;
    }
    if (n > MAXTL * TI * BD) { std::fprintf(stderr, "n=%d exceeds MAXTL*TI*BD\n", n); return 1; }
  }

  std::mt19937_64 rng(A.seed);
  const auto t0 = Clock::now();

  const int P  = A.pop;
  const int NI = std::max(1, A.islands);
  const int PI = P / NI;
  if (PI * NI != P) { std::fprintf(stderr, "pop must be divisible by islands\n"); return 1; }

  Best best;
  // cumulative search-effort counters, surfaced in the STAT log line
  long long ctEval = 0, ctLsIters = 0, ctFw = 0;
  double    ctFlop = 0.0;
  std::vector<double> seedVec;
  if (!A.resume.empty()) {
    if (loadVec(A.resume, seedVec)) {
      // Adopt the resumed vector as the incumbent immediately.  Without this
      // the run can REPORT a worse value than it was handed, because `best` is
      // only updated after a local search that may perturb the seed.
      best.a = seedVec;
      best.n = (int)seedVec.size();
      best.f = hostF(seedVec);
      std::printf("resumed %zu-vector from %s, F = %.15f  (adopted as incumbent)\n",
                  seedVec.size(), A.resume.c_str(), best.f);
    } else {
      std::fprintf(stderr, "could not read %s, ignoring\n", A.resume.c_str());
    }
  }

  // ---- a resumed vector makes coarser levels pointless --------------------
  // Refinement n -> 2n preserves F exactly, so a level with n < |seed| can
  // never beat the incumbent; it can only burn budget.  Drop those levels
  // (always keeping at least the finest one).
  if (!seedVec.empty()) {
    const int m = (int)seedVec.size();
    size_t drop = 0;
    while (drop + 1 < A.ladder.size() && A.ladder[drop] < m) ++drop;
    if (drop > 0) {
      std::printf("skipping %zu ladder level(s) coarser than the %d-cell seed:", drop, m);
      for (size_t i = 0; i < drop; ++i) std::printf(" %d", A.ladder[i]);
      std::printf("\n");
      A.ladder.erase(A.ladder.begin(), A.ladder.begin() + drop);
    }
  }

  // ---- time split across ladder levels: later (finer) levels get more -----
  std::vector<double> lvlBudget(A.ladder.size());
  {
    double tot = 0;
    for (size_t l = 0; l < A.ladder.size(); ++l) { lvlBudget[l] = std::pow(1.85, (double)l); tot += lvlBudget[l]; }
    for (auto& x : lvlBudget) x = x / tot * A.seconds * 0.82;   // 18 % reserved for polish
  }

  Gpu g;
  std::vector<float>  hpop;
  std::vector<double> hfit(P);
  std::vector<int>    hnact(P), hmode(P), hp1(P), hp2(P), hop(P);
  std::vector<float>  hstr(P);
  std::vector<unsigned long long> hsig(P);
  std::vector<int>    hnnz(P);

  // ---- adaptive operator selection (bandit credit assignment) -------------
  static constexpr int NOPS = 14;
  const char* opName[NOPS] = {"blend", "segment", "lognorm", "smooth", "addnoise",
                              "sparsify", "symm", "shift", "block", "restart", "spike",
                              "structured", "gap", "atom"};
  std::vector<double> opW(NOPS, 1.0), opUse(NOPS, 0.0), opWin(NOPS, 0.0);
  opW[11] = 2.5; opW[12] = 1.8; opW[13] = 1.8;   // favour the structural prior early

  // ---- tabu over support patterns ----------------------------------------
  std::unordered_map<unsigned long long, int>    tabuHits;
  std::unordered_map<unsigned long long, double> cellBest;
  int tabuTenure = 6;

  std::vector<float> prevPop;

  for (size_t lvl = 0; lvl < A.ladder.size(); ++lvl) {
    const int n = A.ladder[lvl];
    g.alloc(n, P);
    hpop.assign((size_t)n * P, 0.f);

    // ---------------- seed the level ----------------
    // seedOp[p] records HOW slot p was seeded, so that generation 0 gives a
    // full cold anneal only to the genuinely random individuals.  Refined
    // carry-overs from the previous level are already converged: annealing
    // them from p=6 with a large step would simply destroy them.
    std::vector<int> seedOp(P, 9);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);
    for (int p = 0; p < P; ++p) {
      float* dst = &hpop[(size_t)p * n];
      if (lvl > 0 && !prevPop.empty() && p < P) {
        // refine the previous level's individual p (exactly preserves F)
        const int np = A.ladder[lvl - 1];
        if (n == 2 * np) {
          for (int i = 0; i < np; ++i) { dst[2 * i] = prevPop[(size_t)p * np + i]; dst[2 * i + 1] = prevPop[(size_t)p * np + i]; }
        } else {  // arbitrary ratio: nearest-neighbour resample
          for (int i = 0; i < n; ++i) dst[i] = prevPop[(size_t)p * np + std::min(np - 1, i * np / n)];
        }
        // keep 20 % of the population fresh to preserve diversity
        if (p >= (int)(0.8 * P)) for (int i = 0; i < n; ++i) dst[i] = 0.f;
        else seedOp[p] = 20;                       // refined carry-over
      }
      bool empty = true;
      for (int i = 0; i < n; ++i) if (dst[i] > 0.f) { empty = false; break; }
      if (empty) {
        seedOp[p] = 9;
        const int nf = 2 + (int)(U01(rng) * 6);
        std::vector<double> fr(nf), ph(nf);
        for (int f = 0; f < nf; ++f) { fr[f] = 1 + U01(rng) * 10; ph[f] = U01(rng) * 6.283185307; }
        if (U01(rng) < 0.6) {
          // structured seed: endpoint atom + plateau + depression + hump +
          // ragged band + wide dead gap + terminal spike (see kernels.cuh op 11)
          const bool   flip = U01(rng) < 0.5;
          const double atom = 30 + 250 * U01(rng);
          const double pl1  = 0.28 + 0.16 * U01(rng);
          const double pl2  = pl1 + 0.10 + 0.14 * U01(rng);
          const double hmp  = pl2 + 0.02 + 0.08 * U01(rng);
          const double gp0  = 0.70 + 0.16 * U01(rng);
          const double gp1  = gp0 + 0.05 + 0.15 * U01(rng);
          const double tail = 3.0 + 14.0 * U01(rng);
          for (int i = 0; i < n; ++i) {
            const int    d = flip ? (n - 1 - i) : i;
            const double x = (d + 0.5) / n;
            double rough = 0;
            for (int f = 0; f < nf; ++f) rough += std::cos(6.283185307 * (2 + 6 * fr[f]) * x + ph[f]);
            rough /= nf;
            double v;
            if      (x < pl1) v = 4.2 * (1 + 0.45 * rough);
            else if (x < pl2) v = 0.7 * (1 + 0.80 * rough);
            else if (x < hmp) v = 5.2 * (1 + 0.30 * rough);
            else if (x < gp0) v = 2.0 * (1 + 0.70 * rough);
            else if (x < gp1) v = 0.0;
            else              v = tail * (1 + 0.20 * rough);
            if      (d == 0) v += atom;
            else if (d <  4) v += atom * 0.10 / d;
            dst[i] = (float)std::max(0.0, v);
          }
        } else {
          const double base = 0.3 + 0.7 * U01(rng);
          const double amp  = 0.2 + 1.2 * U01(rng);
          for (int i = 0; i < n; ++i) {
            double m = 0;
            for (int f = 0; f < nf; ++f) m += std::cos(6.283185307 * fr[f] * i / n + ph[f]);
            double v = base + amp * m / nf + 0.3 * N01(rng);
            dst[i] = (float)std::max(0.0, v);
          }
        }
      }
      // normalise
      double S = 0; for (int i = 0; i < n; ++i) S += dst[i];
      if (S <= 0) { for (int i = 0; i < n; ++i) dst[i] = 1.f / n; S = 1; }
      else        { for (int i = 0; i < n; ++i) dst[i] = (float)(dst[i] / S); }
    }
    // inject the global best-so-far and any resume vector
    auto injectDouble = [&](const std::vector<double>& v, int slot) {
      if (v.empty()) return;
      float* dst = &hpop[(size_t)slot * n];
      const int m = (int)v.size();
      double S = 0;
      for (int i = 0; i < n; ++i) { const double x = v[std::min(m - 1, (int)((long long)i * m / n))]; dst[i] = (float)x; S += x; }
      if (S > 0) for (int i = 0; i < n; ++i) dst[i] = (float)(dst[i] / S);
    };
    injectDouble(best.a, 0);   seedOp[0] = 20;
    injectDouble(seedVec, 1);  if (!seedVec.empty()) seedOp[1] = 20;

    CUDA_OK(cudaMemcpy(g.pop, hpop.data(), hpop.size() * sizeof(float), cudaMemcpyHostToDevice));

    // island regimes: exponent multiplier, step multiplier, descent mode
    for (int p = 0; p < P; ++p) {
      const int isl = p / PI;
      hmode[p] = (isl % 4 == 3) ? 1 : 0;      // one island in four uses mirror descent
    }
    CUDA_OK(cudaMemcpy(g.modeArr, hmode.data(), P * sizeof(int), cudaMemcpyHostToDevice));

    const double tLvlEnd = nowSec(t0) + lvlBudget[lvl];
    int gen = 0, sinceImp = 0; bool fwOk = true;
    double lvlBest = 1e30;

    std::printf("\n=== level %zu: n=%d  budget %.0fs  (smem %zu KB) ===\n", lvl, n,
                lvlBudget[lvl], localSearchSmem<float>(n) / 1024);

    // op that produced each individual, driving the cold/warm split
    std::vector<int> curOp = seedOp;
    std::vector<int> skipCold(P);

    while (nowSec(t0) < tLvlEnd && (A.gensPerLvl == 0 || gen < A.gensPerLvl)) {
      const double warm = std::min(1.0, gen / 40.0);

      // ------------------------------------------------------------------
      //  The single most important tuning lesson from the CPU twin: the
      //  exponent anneal has to be LONG and has to end HIGH.  p: 6 -> 2e4
      //  over ~2e4 steps takes a random start to ~1.513 at n=128, whereas a
      //  short anneal ending at p=300 stalls above 2.0 (the flat function,
      //  which is an exact saddle point of the smoothed objective).
      //
      //  So we split the population: individuals whose structure just
      //  changed get the full COLD anneal; small perturbations of already
      //  converged parents only need a short high-exponent WARM refinement.
      // ------------------------------------------------------------------
      //  THREE anneal tiers, not two.  A cold anneal re-solves the problem from
      //  scratch and therefore lands on a TYPICAL local optimum (mean ~1.52 at
      //  n=256) rather than near the incumbent (~1.507) -- so it is pure
      //  exploration and must stay a minority of the budget, especially at
      //  large n where it is also the most expensive thing we do.
      //    cold   (fresh restarts)          full anneal, p 6 -> pTop
      //    medium (structure changed)       1/3 anneal,  p 40 -> pTop
      //    warm   (small perturbation)      short,       p 400 -> 3*pTop
      const double n2 = (double)n * (double)n;
      const int coldCap = std::max(1, (int)(0.18 * P));
      int nCold = 0, nMed = 0;
      std::vector<int> tier(P, 2);
      for (int p = 0; p < P; ++p) {
        const int o = curOp[p];
        if ((o == 9 || o == 11) && nCold < coldCap) { tier[p] = 0; ++nCold; }
        else if (o == 9 || o == 11 || o == 10 || o == 5 || o == 7 || o == 8 || o == 12 || o == 13)
             { tier[p] = 1; ++nMed; }
      }
      const float pTop = (float)A.pMax;
      auto launchTier = [&](int t, int iters, float pa, float pb, float la, float lb) {
        int active = 0;
        for (int p = 0; p < P; ++p) { skipCold[p] = (tier[p] == t) ? 0 : 1; active += 1 - skipCold[p]; }
        CUDA_OK(cudaMemcpy(g.skip, skipCold.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        runLocalSearch(g, iters, pa, pb, la, lb, g.modeArr, g.skip);
        ctLsIters += (long long)active * iters;
        ctFlop    += 4.0 * n2 * (double)active * (double)iters;  // conv + correlation
      };
      if (nCold)          launchTier(0, A.coldIters,       6.f,  pTop,        0.085f, 0.0015f);
      if (nMed)           launchTier(1, A.coldIters / 3,  40.f,  pTop,        0.040f, 0.0010f);
      if (nCold + nMed < P) launchTier(2, A.lsIters,      400.f, pTop * 3.f,  0.012f, 0.0003f);

      // Frank-Wolfe / LP step on everybody: globally-optimal linearised jumps
      // that the smoothed gradient cannot make on its own.
      if (A.fwRounds > 0) {
        fwOk = runFrankWolfe(g, A.fwRounds, A.fwIters, 12, 0.30f, prop.sharedMemPerBlockOptin);
        if (fwOk) {
          ctFw   += (long long)A.fwRounds * P;
          ctFlop += (double)P * A.fwRounds * (4.0 * n2 * A.fwIters + 2.0 * n2 * 13.0);
        }
      }
      runEval(g);
      ctEval += P;
      ctFlop += 2.0 * n2 * P;
      runSig(g);
      CUDA_OK(cudaMemcpy(hfit.data(), g.fit, P * sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(hnact.data(), g.nact, P * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(hsig.data(), g.sig, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(hnnz.data(), g.nnz, P * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_OK(cudaDeviceSynchronize());

      // ---- track the best, update the tabu structures ----
      int bi = 0;
      for (int p = 1; p < P; ++p) if (hfit[p] < hfit[bi]) bi = p;
      bool improved = false;
      if (hfit[bi] < best.f - 1e-13) {
        best.f = hfit[bi]; best.n = n;
        std::vector<float> tmp(n);
        CUDA_OK(cudaMemcpy(tmp.data(), g.pop + (size_t)bi * n, n * sizeof(float), cudaMemcpyDeviceToHost));
        best.a.assign(n, 0.0);
        for (int i = 0; i < n; ++i) best.a[i] = (double)tmp[i];
        improved = true;
        saveVec(A.out + ".txt", best.a, best.f);
      }
      if (hfit[bi] < lvlBest - 1e-13) { lvlBest = hfit[bi]; sinceImp = 0; }
      else ++sinceImp;

      for (int p = 0; p < P; ++p) {
        auto it = cellBest.find(hsig[p]);
        if (it == cellBest.end() || hfit[p] < it->second - 1e-12) {
          cellBest[hsig[p]] = hfit[p];
          tabuHits[hsig[p]] = 0;
        } else {
          tabuHits[hsig[p]] += 1;                 // revisited without improving
        }
      }

      // ---- selection: rank within each island ----
      std::vector<int> order(P); std::iota(order.begin(), order.end(), 0);
      std::vector<std::vector<int>> isl(NI);
      for (int i = 0; i < NI; ++i) {
        isl[i].resize(PI);
        std::iota(isl[i].begin(), isl[i].end(), i * PI);
        std::sort(isl[i].begin(), isl[i].end(), [&](int a, int b) { return hfit[a] < hfit[b]; });
      }
      // migration every 12 generations: best of island i replaces worst of i+1
      if (gen > 0 && gen % 12 == 0) {
        for (int i = 0; i < NI; ++i) {
          const int src = isl[i][0], dst = isl[(i + 1) % NI][PI - 1];
          CUDA_OK(cudaMemcpy(g.pop + (size_t)dst * n, g.pop + (size_t)src * n,
                             n * sizeof(float), cudaMemcpyDeviceToDevice));
          hfit[dst] = hfit[src]; hsig[dst] = hsig[src];
        }
      }

      // ---- build the offspring plan ----
      const double stagn = std::min(1.0, sinceImp / 25.0);
      std::discrete_distribution<int> opPick(opW.begin(), opW.end());
      std::vector<double> parentFit(P);
      for (int p = 0; p < P; ++p) parentFit[p] = hfit[p];

      for (int i = 0; i < NI; ++i) {
        for (int r = 0; r < PI; ++r) {
          const int slot = i * PI + r;
          if (r < A.elite) {                       // elitism: clone untouched
            hp1[slot] = isl[i][r]; hp2[slot] = isl[i][r];
            hop[slot] = 20; hstr[slot] = 0.f; continue;
          }
          // tournament selection inside the island
          auto tourn = [&]() {
            int a = isl[i][(int)(U01(rng) * PI)], b = isl[i][(int)(U01(rng) * PI)];
            return hfit[a] < hfit[b] ? a : b;
          };
          int pa = tourn(), pb = tourn();
          int o = opPick(rng);
          // tabu: if the parent's support cell has been revisited too often,
          // force a structure-changing (diversifying) operator instead
          // Escape a support cell only if it has been revisited without
          // improving AND it is not competitive with the incumbent: cells
          // already near the best deserve intensification, not diversification.
          const int hits = tabuHits.count(hsig[pa]) ? tabuHits[hsig[pa]] : 0;
          const auto cbIt = cellBest.find(hsig[pa]);
          const double cb = (cbIt == cellBest.end()) ? 1e30 : cbIt->second;
          if (hits > tabuTenure && cb > best.f * (1.0 + 2e-4)) {
            static const int divers[7] = {5, 7, 9, 10, 8, 11, 12};
            o = divers[(int)(U01(rng) * 7)];
          }
          if (stagn > 0.7 && U01(rng) < 0.10) o = 9;   // stagnation -> restarts
          hp1[slot] = pa; hp2[slot] = pb; hop[slot] = o;
          hstr[slot] = (float)((0.03 + 0.35 * U01(rng)) * (0.5 + warm) * (1.0 + 1.5 * stagn));
          opUse[o] += 1.0;
        }
      }
      CUDA_OK(cudaMemcpy(g.p1, hp1.data(), P * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(g.p2, hp2.data(), P * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(g.op, hop.data(), P * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(g.strength, hstr.data(), P * sizeof(float), cudaMemcpyHostToDevice));

      const size_t smv = sizeof(float) * ((size_t)2 * n + 64 + 128);
      setSmemLimit((const void*)kVariation<float>, smv);
      kVariation<float><<<P, BD, smv>>>(g.pop, g.nxt, n, P, g.p1, g.p2, g.op, g.strength,
                                        A.seed, (unsigned long long)(gen + 1000 * (int)lvl));
      CUDA_OK(cudaGetLastError());
      std::swap(g.pop, g.nxt);
      curOp = hop;

      // keep the global best in the gene pool: warm refinement lets elites
      // drift, so re-seed it periodically into the weakest slot of island 0
      if (gen % 10 == 9 && !best.a.empty() && best.n == n) {
        std::vector<float> tmp(n);
        for (int i = 0; i < n; ++i) tmp[i] = (float)best.a[i];
        CUDA_OK(cudaMemcpy(g.pop + (size_t)isl[0][PI - 1] * n, tmp.data(),
                           n * sizeof(float), cudaMemcpyHostToDevice));
        curOp[isl[0][PI - 1]] = 20;
      }

      // ---- credit assignment: did the operator beat its first parent? ----
      if (gen > 0 && gen % 5 == 0) {
        runEval(g);
        CUDA_OK(cudaMemcpy(hnact.data(), g.nact, P * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> childFit(P);
        CUDA_OK(cudaMemcpy(childFit.data(), g.fit, P * sizeof(double), cudaMemcpyDeviceToHost));
        for (int p = 0; p < P; ++p)
          if (hop[p] < NOPS && childFit[p] < parentFit[hp1[p]]) opWin[hop[p]] += 1.0;
        for (int o = 0; o < NOPS; ++o) {
          const double rate = opUse[o] > 20 ? opWin[o] / opUse[o] : 0.25;
          opW[o] = 0.85 * opW[o] + 0.15 * (0.15 + 3.0 * rate);
          opW[o] = std::max(0.05, std::min(4.0, opW[o]));
        }
        for (int o = 0; o < NOPS; ++o) { opUse[o] *= 0.7; opWin[o] *= 0.7; }
      }

      if (A.verbose && (gen % 10 == 0 || improved)) {
        double mean = 0; for (double x : hfit) mean += x; mean /= P;
        std::printf("  n=%4d gen %4d  t=%6.1fs  best %.9f  lvl %.9f  mean %.6f  "
                    "peaks %3d  nnz %4d  cold %2d med %3d  fw%d  cells %zu %s\n",
                    n, gen, nowSec(t0), best.f, lvlBest, mean, hnact[bi], hnnz[bi],
                    nCold, nMed, (int)fwOk, cellBest.size(), improved ? "*" : "");
        // machine-readable progress record for the live monitor
        std::printf("STAT t=%.1f n=%d gen=%d best=%.15f lvl=%.15f mean=%.9f "
                    "evals=%lld lsiters=%lld fw=%lld cells=%zu pflop=%.6f peaks=%d nnz=%d\n",
                    nowSec(t0), n, gen, best.f, lvlBest, mean, ctEval, ctLsIters, ctFw,
                    cellBest.size(), ctFlop / 1e15, hnact[bi], hnnz[bi]);
        std::fflush(stdout);
      }
      ++gen;
    }

    // carry the population to the next level
    std::fprintf(stderr, "[M1]\n");
    prevPop.assign((size_t)n * P, 0.f);
    std::fprintf(stderr, "[M2]\n");
    CUDA_OK(cudaMemcpy(prevPop.data(), g.pop, prevPop.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "[M3]\n");
    g.free();
    std::fprintf(stderr, "[M4]\n");
    std::printf("=== level n=%d done: best %.12f after %d generations ===\n", n, best.f, gen);
  }

  // ------------------------- fp64 polish of the elite ----------------------
  if (!best.a.empty()) {
    const int n = best.n;
    const int K = std::max(1, A.polishTop);
    std::printf("\n=== fp64 polish: n=%d, %d candidates, %d iters ===\n", n, K, A.polishIters);

    double *dp, *dm, *dv, *df; int* dn;
    CUDA_OK(cudaMalloc(&dp, (size_t)n * K * sizeof(double)));
    CUDA_OK(cudaMalloc(&dm, (size_t)n * K * sizeof(double)));
    CUDA_OK(cudaMalloc(&dv, (size_t)n * K * sizeof(double)));
    CUDA_OK(cudaMalloc(&df, K * sizeof(double)));
    CUDA_OK(cudaMalloc(&dn, K * sizeof(int)));

    std::vector<double> hp((size_t)n * K);
    std::mt19937_64 r2(A.seed ^ 0xABCDEF);
    std::normal_distribution<double> N01(0, 1);
    for (int k = 0; k < K; ++k) {
      double S = 0;
      for (int i = 0; i < n; ++i) {
        double x = best.a[i];
        // basin hopping: small, BOUNDED perturbations.  Scaling the amplitude
        // with k made the later candidates useless once polish-top grew.
        if (k) x *= std::exp(0.003 * (1 + (k % 8)) * N01(r2));
        hp[(size_t)k * n + i] = std::max(0.0, x); S += std::max(0.0, x);
      }
      for (int i = 0; i < n; ++i) hp[(size_t)k * n + i] /= S;
    }
    CUDA_OK(cudaMemcpy(dp, hp.data(), hp.size() * sizeof(double), cudaMemcpyHostToDevice));

    const size_t sm = localSearchSmem<double>(n);
    if ((int)sm <= prop.sharedMemPerBlockOptin) {
      setSmemLimit((const void*)kLocalSearch<double>, sm);
      // several passes of increasing exponent and shrinking step
      const int passes = 6;
      for (int q = 0; q < passes; ++q) {
        const double pA = 200.0 * std::pow(4.0, q);
        const double pB = pA * 30.0;
        const double lA = 0.010 * std::pow(0.45, q);
        const double lB = lA * 0.02;
        kLocalSearch<double><<<K, BD, sm>>>(dp, dm, dv, nullptr, nullptr, n,
                                            A.polishIters / passes, pA, pB, lA, lB,
                                            0.9, 0.999, 0);
        CUDA_OK(cudaGetLastError());
        const size_t sme = evalSmem<double>(n);
        setSmemLimit((const void*)kEval<double, double>, sme);
        kEval<double, double><<<K, BD, sme>>>(dp, n, df, dn, 1e-6);
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<double> f(K); CUDA_OK(cudaMemcpy(f.data(), df, K * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int>    na(K); CUDA_OK(cudaMemcpy(na.data(), dn, K * sizeof(int), cudaMemcpyDeviceToHost));
        int bk = 0; for (int k = 1; k < K; ++k) if (f[k] < f[bk]) bk = k;
        std::printf("  polish pass %d (p=%.0f..%.0f): best %.14f  peaks %d\n", q, pA, pB, f[bk], na[bk]);
        if (f[bk] < best.f) {
          best.f = f[bk];
          std::vector<double> tmp(n);
          CUDA_OK(cudaMemcpy(tmp.data(), dp + (size_t)bk * n, n * sizeof(double), cudaMemcpyDeviceToHost));
          best.a = tmp;
          saveVec(A.out + ".txt", best.a, best.f);
        }
        setSmemLimit((const void*)kLocalSearch<double>, sm);
      }
    } else {
      std::printf("  (skipped: fp64 local search needs %zu B shared, device offers %d)\n",
                  sm, (int)prop.sharedMemPerBlockOptin);
    }
    cudaFree(dp); cudaFree(dm); cudaFree(dv); cudaFree(df); cudaFree(dn);
  }

  // ------------------------------- report ---------------------------------
  const double hf = best.a.empty() ? 1e30 : hostF(best.a);
  std::printf("\n==========================================================\n");
  std::printf("  best  F = %.15f   (n = %d)\n", best.f, best.n);
  std::printf("  host recheck (fp64, independent code path) = %.15f\n", hf);
  std::printf("  => C <= %.9f\n", std::max(best.f, hf));
  std::printf("  vector written to %s.txt\n", A.out.c_str());
  std::printf("  wall clock %.1f s\n", nowSec(t0));
  std::printf("==========================================================\n");
  if (!best.a.empty()) saveVec(A.out + ".txt", best.a, std::max(best.f, hf));
  return 0;
}
