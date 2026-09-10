// ---------------------------------------------------------------------------
//  probe.exe -- CPU companion for the GPU search (AlphaEvolve problem 2).
//
//  Three jobs, all on the 12 CPU cores via OpenMP:
//    --shapes            evaluate closed-form candidate profiles
//    --opt               run the SAME L^p-annealed local search on the CPU so
//                        that hyper-parameters can be tuned in seconds, and so
//                        that the GPU implementation has an independent twin
//    --verify FILE       re-evaluate a saved vector in long double, with a
//                        rigorous round-off bound, on a completely separate
//                        code path from the GPU
//
//  Objective:  F(a) = 2*n*max_k b_k / (sum_i a_i)^2 ,  b_k = sum_{i+j=k} a_i a_j
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using std::size_t;

// ------------------------------ objective ----------------------------------
template <typename T>
static void autoconv(const std::vector<T>& a, std::vector<T>& b) {
  const int n = (int)a.size();
  b.assign(2 * n - 1, T(0));
  for (int i = 0; i < n; ++i) {
    const T ai = a[i];
    if (ai == T(0)) continue;
    T* bp = b.data() + i;
    for (int j = 0; j < n; ++j) bp[j] += ai * a[j];
  }
}

template <typename T>
static T objective(const std::vector<T>& a, std::vector<T>& b) {
  const int n = (int)a.size();
  autoconv(a, b);
  T M = T(0); for (T x : b) M = std::max(M, x);
  T S = T(0); for (T x : a) S += x;
  return T(2) * T(n) * M / (S * S);
}

// ------------------------- L^p annealed projected Adam ----------------------
struct LsCfg {
  int    iters = 20000;
  double pA = 6, pB = 4000;
  double lrA = 0.08, lrB = 0.002;
  double b1 = 0.9, b2 = 0.999;
  int    mode = 0;      // 0 = projected Adam, 1 = mirror / exponentiated grad
};

// One local search.  Mirrors kLocalSearch exactly (same gradient, same update).
static double localSearch(std::vector<double>& a, const LsCfg& c) {
  const int n = (int)a.size();
  const int nb = 2 * n - 1;
  std::vector<double> b(nb), cc(nb), m(n, 0.0), v(n, 0.0), g(n);
  double b1t = 1, b2t = 1;
  const double lnP = std::log(c.pB / c.pA), lnL = std::log(c.lrB / c.lrA);

  // normalise
  { double S = 0; for (double x : a) S += x; for (auto& x : a) x /= S; }

  for (int it = 0; it < c.iters; ++it) {
    const double frac = c.iters > 1 ? double(it) / double(c.iters - 1) : 0.0;
    const double p  = c.pA  * std::exp(lnP * frac);
    const double lr = c.lrA * std::exp(lnL * frac);

    autoconv(a, b);
    double M = 0; for (double x : b) M = std::max(M, x);
    if (!(M > 0)) break;

    double R = 0;
    for (int k = 0; k < nb; ++k) {
      const double d = (b[k] - M) / M;                       // in [-1, 0]
      const double q = d > -1.0 ? std::exp((p - 1) * std::log1p(d)) : 0.0;
      cc[k] = q; R += q * (1.0 + d);
    }
    // G_i = sum_j c_{i+j} a_j
    for (int i = 0; i < n; ++i) {
      double s = 0; const double* cp = cc.data() + i;
      for (int j = 0; j < n; ++j) s += cp[j] * a[j];
      g[i] = s / R - M;
    }
    b1t *= c.b1; b2t *= c.b2;
    const double bc1 = 1.0 / (1.0 - b1t), bc2 = 1.0 / (1.0 - b2t);
    double S = 0;
    for (int i = 0; i < n; ++i) {
      double x;
      if (c.mode == 0) {
        m[i] = c.b1 * m[i] + (1 - c.b1) * g[i];
        v[i] = c.b2 * v[i] + (1 - c.b2) * g[i] * g[i];
        const double step = (lr / n) * (m[i] * bc1) / (std::sqrt(v[i] * bc2) + 1e-30);
        x = a[i] - step;
        if (x < 0) x = 0;
      } else {
        double e = -lr * (g[i] / M);
        e = std::max(-4.0, std::min(4.0, e));
        x = a[i] * std::exp(e);
      }
      a[i] = x; S += x;
    }
    if (S > 0) for (int i = 0; i < n; ++i) a[i] /= S;
    else       for (int i = 0; i < n; ++i) a[i] = 1.0 / n;
  }
  std::vector<double> bb;
  return objective(a, bb);
}

// ------------------------------ shapes -------------------------------------
static void shapes(int n) {
  std::vector<double> b;
  auto rep = [&](const char* name, std::vector<double> a) {
    for (auto& x : a) if (x < 0) x = 0;
    std::printf("  %-34s F = %.9f\n", name, objective(a, b));
  };
  const double PI = 3.14159265358979323846;
  std::vector<double> a(n);
  auto fill = [&](auto f) { for (int i = 0; i < n; ++i) a[i] = f((i + 0.5) / n); return a; };

  rep("constant",              fill([&](double) { return 1.0; }));
  rep("arcsine 1/sqrt(x(1-x))", fill([&](double x) { return 1.0 / std::sqrt(x * (1 - x) + 1e-12); }));
  rep("sqrt(x(1-x))  (semicircle)", fill([&](double x) { return std::sqrt(x * (1 - x)); }));
  rep("sin(pi x)",             fill([&](double x) { return std::sin(PI * x); }));
  rep("1/ (x(1-x))^{1/4}",     fill([&](double x) { return std::pow(x * (1 - x) + 1e-12, -0.25); }));
  rep("U shape x^2+(1-x)^2",   fill([&](double x) { return x * x + (1 - x) * (1 - x); }));
  rep("edge spikes + flat",    fill([&](double x) { return 1.0 + 3.0 * (x < 0.03 || x > 0.97 ? 1.0 : 0.0); }));
  rep("cos^2 bump x3",         fill([&](double x) { return 1.0 + 0.6 * std::cos(6 * PI * x); }));
  rep("linear ramp",           fill([&](double x) { return x; }));
  rep("two blocks",            fill([&](double x) { return (x < 0.35 || x > 0.65) ? 1.0 : 0.2; }));
  for (double al : {0.1, 0.2, 0.3, 0.4, 0.45, 0.49}) {
    char nm[64]; std::snprintf(nm, sizeof nm, "(x(1-x))^{-%.2f}", al);
    rep(nm, fill([&](double x) { return std::pow(x * (1 - x) + 1e-14, -al); }));
  }
}

// ------------------------------ verify -------------------------------------
static int verify(const std::string& path) {
  // C stdio: the libstdc++ ifstream path faults at -O3 in this MinGW build.
  FILE* in = std::fopen(path.c_str(), "r");
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
  std::vector<long double> a;
  char line[512]; int nDecl = 0;
  while (std::fgets(line, sizeof line, in)) {
    if (line[0] == '\0' || line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    if (line[0] == 'n' && line[1] == ' ') { nDecl = std::atoi(line + 2); continue; }
    if (line[0] == 'F' && line[1] == ' ') continue;
    a.push_back(std::strtold(line, nullptr));
  }
  std::fclose(in);
  const int n = (int)a.size();
  if (nDecl && nDecl != n) { std::fprintf(stderr, "length mismatch\n"); return 1; }
  bool neg = false; for (auto x : a) if (x < 0) neg = true;

  // long double autoconvolution, parallel over k
  const int nb = 2 * n - 1;
  std::vector<long double> b(nb, 0.0L);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < nb; ++k) {
    const int ilo = std::max(0, k - n + 1), ihi = std::min(k, n - 1);
    long double s = 0.0L;
    for (int i = ilo; i <= ihi; ++i) s += a[i] * a[k - i];
    b[k] = s;
  }
  long double M = 0, S = 0; int kmax = 0;
  for (int k = 0; k < nb; ++k) if (b[k] > M) { M = b[k]; kmax = k; }
  for (auto x : a) S += x;
  const long double F = 2.0L * (long double)n * M / (S * S);

  // round-off bound: |fl(sum of m terms) - exact| <= (m*eps/(1-m*eps)) * sum|terms|
  const long double eps = 1.084202172485504434e-19L;   // 2^-63, x87 long double
  const long double gam = (long double)n * eps / (1 - (long double)n * eps);
  const long double relBound = 3 * gam;                // conv + two sums + divide
  int nActive = 0;
  for (int k = 0; k < nb; ++k) if (b[k] >= M * (1 - 1e-9L)) ++nActive;

  std::printf("verify %s\n", path.c_str());
  std::printf("  n              = %d\n", n);
  std::printf("  all a_i >= 0   = %s\n", neg ? "NO  <-- INVALID" : "yes");
  std::printf("  sum a_i        = %.20Lg\n", S);
  std::printf("  max_k b_k      = %.20Lg   at k = %d of %d\n", M, kmax, nb - 1);
  std::printf("  active peaks   = %d  (within 1e-9 of the max)\n", nActive);
  std::printf("  F              = %.20Lg\n", F);
  std::printf("  round-off bnd  = %.3Lg relative  ->  F <= %.20Lg\n", relBound, F * (1 + relBound));
  std::printf("  ==> C <= %.12Lf   (rigorous modulo the bound above)\n", F * (1 + relBound));
  return neg ? 1 : 0;
}

// -------------------------------- main -------------------------------------
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int  n = 128, restarts = 96, iters = 20000, mode = 0;
  double pA = 6, pB = 4000, lrA = 0.08, lrB = 0.002;
  unsigned seed = 1;
  std::string job, file;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if      (s == "--shapes")   job = "shapes";
    else if (s == "--opt")      job = "opt";
    else if (s == "--verify")   { job = "verify"; file = nx(); }
    else if (s == "--n")        n = std::stoi(nx());
    else if (s == "--restarts") restarts = std::stoi(nx());
    else if (s == "--iters")    iters = std::stoi(nx());
    else if (s == "--pa")       pA = std::stod(nx());
    else if (s == "--pb")       pB = std::stod(nx());
    else if (s == "--lra")      lrA = std::stod(nx());
    else if (s == "--lrb")      lrB = std::stod(nx());
    else if (s == "--mode")     mode = std::stoi(nx());
    else if (s == "--seed")     seed = (unsigned)std::stoul(nx());
    else if (s == "--out")      file = nx();
    else { std::fprintf(stderr, "unknown %s\n", s.c_str()); return 1; }
  }
#ifdef _OPENMP
  std::printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif

  if (job == "shapes") { std::printf("closed-form profiles, n = %d\n", n); shapes(n); return 0; }
  if (job == "verify") return verify(file);
  if (job != "opt")    { std::printf("usage: probe --shapes|--opt|--verify FILE [...]\n"); return 1; }

  LsCfg cfg; cfg.iters = iters; cfg.pA = pA; cfg.pB = pB; cfg.lrA = lrA; cfg.lrB = lrB; cfg.mode = mode;
  std::printf("L^p Adam: n=%d restarts=%d iters=%d p=%.0f..%.0f lr=%.4f..%.5f mode=%d\n",
              n, restarts, iters, pA, pB, lrA, lrB, mode);

  // Every restart writes into its own slice; the winner is picked afterwards.
  // (No omp critical: a non-POD assignment inside one miscompiles at -O3 here.)
  std::vector<double> allF(restarts, 1e30);
  std::vector<double> allA((size_t)restarts * n, 0.0);

#pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < restarts; ++r) {
    std::mt19937_64 rng(seed * 1000003ULL + r);
    std::uniform_real_distribution<double> U(0, 1);
    std::normal_distribution<double> N(0, 1);
    std::vector<double> a(n);
    const int kind = r % 4;
    if (kind == 0) {                       // smooth random Fourier profile
      const int nf = 2 + (int)(U(rng) * 6);
      std::vector<double> fr(nf), ph(nf);
      for (int f = 0; f < nf; ++f) { fr[f] = 1 + U(rng) * 10; ph[f] = U(rng) * 6.283185307; }
      const double base = 0.2 + 0.9 * U(rng), amp = 0.3 + 1.5 * U(rng);
      for (int i = 0; i < n; ++i) {
        double m = 0;
        for (int f = 0; f < nf; ++f) m += std::cos(6.283185307 * fr[f] * i / n + ph[f]);
        a[i] = std::max(0.0, base + amp * m / nf);
      }
    } else if (kind == 1) {                // iid uniform
      for (int i = 0; i < n; ++i) a[i] = U(rng);
    } else if (kind == 2) {                // antisymmetric kick off the flat saddle
      for (int i = 0; i < n; ++i) { const double v = N(rng); a[i] = 1.0 + 0.6 * v; }
      for (int i = 0; i < n / 2; ++i) { const double d = 0.5 * (a[i] - a[n - 1 - i]); a[i] = 1 + d; a[n - 1 - i] = 1 - d; }
      for (int i = 0; i < n; ++i) a[i] = std::max(0.0, a[i]);
    } else {                               // sparse / lacunary
      for (int i = 0; i < n; ++i) a[i] = (U(rng) < 0.5) ? U(rng) : 0.0;
    }
    double S = 0; for (double x : a) S += x;
    if (S <= 0) { for (auto& x : a) x = 1.0 / n; }
    else        { for (auto& x : a) x /= S; }

    allF[r] = localSearch(a, cfg);
    std::copy(a.begin(), a.end(), allA.begin() + (size_t)r * n);
  }

  int bi = 0;
  for (int r = 1; r < restarts; ++r) if (allF[r] < allF[bi]) bi = r;
  const double bestF = allF[bi];
  const std::vector<double> bestA(allA.begin() + (size_t)bi * n,
                                  allA.begin() + (size_t)(bi + 1) * n);

  std::vector<double> srt = allF;
  std::sort(srt.begin(), srt.end());
  std::printf("  best   %.12f\n", srt.front());
  std::printf("  p10    %.12f\n", srt[restarts / 10]);
  std::printf("  median %.12f\n", srt[restarts / 2]);
  std::printf("  worst  %.12f\n", srt.back());
  int nz = 0; for (double x : bestA) if (x <= 1e-12) ++nz;
  std::printf("  best vector: %d/%d zeros\n", nz, n);
  if (!file.empty()) {
    // C stdio on purpose: the libstdc++ ofstream path in this MinGW/OpenMP
    // build faults at -O3, and this is the only place we write a file.
    FILE* o = std::fopen(file.c_str(), "w");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", file.c_str()); return 1; }
    std::fprintf(o, "n %d\nF %.17g\n", n, bestF);
    for (double x : bestA) std::fprintf(o, "%.17g\n", x);
    std::fclose(o);
    std::printf("  wrote %s\n", file.c_str());
  }
  return 0;
}
