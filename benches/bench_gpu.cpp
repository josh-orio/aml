#ifdef USE_CUBLAS

#include "include/gpustats.hpp"
#include "include/statlib.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static void BM_Gpu_Memory_Copy(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);
  auto d_b = mathlib::gpu::memory::load(B.start(), n);

  for (auto _ : state) {
    mathlib::gpu::memory::copy(d_a, d_b, state.range(0));
    benchmark::ClobberMemory();
  }

  double gb = static_cast<float>(sizeof(float) * state.range(0) * 2) / 1e9; // *2 for r&w ?
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Gpu_Memory_Copy)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

static void BM_Gpu_Memory_Copy_Double(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);
  auto d_b = mathlib::gpu::memory::load(B.start(), n);

  for (auto _ : state) {
    mathlib::gpu::memory::copy(d_a, d_b, state.range(0));
    benchmark::ClobberMemory();
  }

  double gb = sizeof(double) * state.range(0) / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Gpu_Memory_Copy_Double)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000});

// swap
static void BM_Gpu_Memory_Swap(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);
  auto d_b = mathlib::gpu::memory::load(B.start(), n);

  for (auto _ : state) {
    mathlib::gpu::memory::swap(d_a, d_b, state.range(0));
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * state.range(0) / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Gpu_Memory_Swap)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// scale
static void BM_Gpu_Tensor_Scale(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::scale(d_a, 2.0f, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Tensor_Scale)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// update
static void BM_Gpu_Tensor_Update(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);
  auto d_b = mathlib::gpu::memory::load(B.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::update(d_a, d_b, 2.0f, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Tensor_Update)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// fixedupdate
static void BM_Gpu_Tensor_Fixedupdate(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::fixed_update(d_a, 2.0f, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Tensor_Fixedupdate)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// sum
static void BM_Gpu_Tensor_Sum(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::sum(d_a, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Tensor_Sum)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// mean
static void BM_Gpu_Tensor_Mean(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::mean(d_a, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Tensor_Mean)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// min
static void BM_Gpu_Tensor_Min(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::min(d_a, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Gpu_Tensor_Min)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// max
static void BM_Gpu_Tensor_Max(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);

  for (auto _ : state) {
    mathlib::gpu::tensor::max(d_a, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
BENCHMARK(BM_Gpu_Tensor_Max)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// dot
static void BM_Gpu_Tensor_Dot(benchmark::State &state) {
  size_t n = state.range(0);
  structs::Matrix<float> A(n, 1, 1.0f), B(n, 1, 0.0f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);
  auto d_b = mathlib::gpu::memory::load(B.start(), n);

  for (auto _ : state) {
    mathlib::gpu::linalg::dot(d_a, d_b, n);
    benchmark::ClobberMemory();
  }

  double gb = sizeof(float) * n / 1e9;
  state.counters["GB/s"] = benchmark::Counter(gb, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

  double gflops = static_cast<float>(2.0f * n) / 1e9;
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Tensor_Max)->Args({100000})->Args({1000000})->Args({10000000})->Args({100000000})->Args({1000000000})->Args({10000000000});

// matmul
static void BM_Gpu_Linalg_Matmul(benchmark::State &state) {
  size_t M = state.range(0);
  size_t N = state.range(1);
  size_t K = state.range(2);

  structs::Matrix<float> A(M, K), B(K, N), C(M, N);

  // fill matrices with rand
  mathlib::gpu::random::uniform(A.start(), M * K, 0.f, 1.f);
  mathlib::gpu::random::uniform(B.start(), K * N, 0.f, 1.f);

  auto d_a = mathlib::gpu::memory::load(A.start(), n);
  auto d_b = mathlib::gpu::memory::load(B.start(), n);
  auto d_c = mathlib::gpu::memory::load(C.start(), n);

  for (auto _ : state) {
    mathlib::gpu::linalg::matmul(d_a, d_b, d_c, M, N, K);
  }

  double gflops = 2.0 * M * N * K / 1e9; // 2mnk
  state.counters["GFLOPs"] = benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Gpu_Linalg_Matmul)
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({8192, 8192, 8192})
    ->Args({16384, 16384, 16384});

// relu
// stdev
// variance
// covar

BENCHMARK_MAIN();

#endif
