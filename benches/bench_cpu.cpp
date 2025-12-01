#ifdef USE_OPENBLAS

#include <algorithm>
#include <aml/aml.hpp>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static void BM_Cpu_Memory_Copy(benchmark::State &state) {
  structs::Matrix<float> A(state.range(0), 1, 1.0f), B(state.range(0), 1, 0.0f);

  for (auto _ : state) {
    aml::cpu::memory::copy(A.start(), B.start(), state.range(0));
    benchmark::ClobberMemory(); // ensure compiler doesn't optimize away
  }

  double gb = static_cast<float>(sizeof(float) * state.range(0) * 2) / 1e9; // *2 for r&w ?
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Cpu_Memory_Copy)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

static void BM_Cpu_Memory_Copy_Double(benchmark::State &state) {
  structs::Matrix<double> A(state.range(0), 1, 1.0f), B(state.range(0), 1, 0.0f);

  for (auto _ : state) {
    aml::cpu::memory::copy(A.start(), B.start(), state.range(0));
    benchmark::ClobberMemory();
  }

  double gb = sizeof(double) * state.range(0) / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Cpu_Memory_Copy_Double)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000});

// swap
static void BM_Cpu_Memory_Swap(benchmark::State &state) {
  structs::Matrix<float> A(state.range(0), 1, 1.0f), B(state.range(0), 1, 0.0f);

  for (auto _ : state) {
    aml::cpu::memory::swap(A.start(), B.start(), state.range(0));
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * state.range(0) / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Cpu_Memory_Swap)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// scale
static void BM_Cpu_Tensor_Scale(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  for (auto _ : state) {
    aml::cpu::tensor::scale(A.start(), 2.0f, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Tensor_Scale)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// update
static void BM_Cpu_Tensor_Update(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  for (auto _ : state) {
    aml::cpu::tensor::update(A.start(), B.start(), 2.0f, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Tensor_Update)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// fixedupdate
static void BM_Cpu_Tensor_Fixedupdate(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  for (auto _ : state) {
    aml::cpu::tensor::fixed_update(A.start(), 2.0f, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Tensor_Fixedupdate)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// sum
static void BM_Cpu_Tensor_Sum(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  for (auto _ : state) {
    aml::cpu::tensor::sum(A.start(), n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Tensor_Sum)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// mean
static void BM_Cpu_Tensor_Mean(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  for (auto _ : state) {
    aml::cpu::tensor::mean(A.start(), n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Tensor_Mean)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// min
static void BM_Cpu_Tensor_Min(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  for (auto _ : state) {
    aml::cpu::tensor::min(A.start(), n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Cpu_Tensor_Min)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// max
static void BM_Cpu_Tensor_Max(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  for (auto _ : state) {
    aml::cpu::tensor::max(A.start(), n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Cpu_Tensor_Max)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// dot
static void BM_Cpu_Tensor_Dot(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  for (auto _ : state) {
    aml::cpu::linalg::dot(A.start(), B.start(), n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Tensor_Max)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// matmul
static void BM_Cpu_Linalg_Matmul(benchmark::State &state) {
  size_t M = state.range(0);
  size_t N = state.range(1);
  size_t K = state.range(2);

  structs::Matrix<float> A(M, K), B(K, N), C(M, N);

  // fill matrices with rand
  aml::cpu::random::uniform(A.start(), M * K, 0.f, 1.f);
  aml::cpu::random::uniform(B.start(), K * N, 0.f, 1.f);

  for (auto _ : state) {
    aml::cpu::linalg::matmul(A.start(), B.start(), C.start(), M, N, K);
  }

  double gflops = 2.0 * M * N * K / 1e9; // 2mnk
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Cpu_Linalg_Matmul)
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({8192, 8192, 8192})
    ->Args({16384, 16384, 16384});

// relu
// stdev
// variance
// covar
// norm/denorm (gbps and gflop)

BENCHMARK_MAIN();

#endif
