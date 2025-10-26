#include <cblas.h>
#ifdef USE_OPENBLAS

#include "cpustats.hpp"
#include <algorithm>
#include <vector>
#include <random>

namespace mathlib {
namespace cpu {

const float one = 1;
const double one_d = 1;

namespace memory {

template <> void copy(float *src, float *dst, size_t n) { cblas_scopy(n, src, 1, dst, 1); }
template <> void copy(double *src, double *dst, size_t n) { cblas_dcopy(n, src, 1, dst, 1); }
template <> void swap(float *a, float *b, size_t n) { cblas_sswap(n, a, 1, b, 1); }
template <> void swap(double *a, double *b, size_t n) { cblas_dswap(n, a, 1, b, 1); }

} // namespace memory

namespace tensor {

template <> void scale(float *a, float alpha, size_t n) { cblas_sscal(n, alpha, a, 1); }
template <> void scale(double *a, double alpha, size_t n) { cblas_dscal(n, alpha, a, 1); }

template <> void update(float *a, float *b, float alpha, size_t n) { cblas_saxpy(n, alpha, b, 1, a, 1); }
template <> void update(double *a, double *b, double alpha, size_t n) { cblas_daxpy(n, alpha, b, 1, a, 1); }

template <> void fixed_update(float *a, float alpha, size_t n) { cblas_saxpy(n, alpha, &one, 0, a, 1); }
template <> void fixed_update(double *a, double alpha, size_t n) { cblas_daxpy(n, alpha, &one_d, 0, a, 1); }

template <> float sum(float *data, size_t n) { return cblas_sdot(n, data, 1, &one, 0); }
template <> double sum(double *data, size_t n) { return cblas_ddot(n, data, 1, &one_d, 0); }

template <> float mean(float *data, size_t n) { return cblas_sdot(n, data, 1, &one, 0) / static_cast<float>(n); }
template <> double mean(double *data, size_t n) { return cblas_ddot(n, data, 1, &one_d, 0)/ static_cast<double>(n); }

// template <> float min(float *data, size_t n) {}
// template <> double min(double *data, size_t n) {}

// template <> float max(float *data, size_t n) {}
// template <> double max(double *data, size_t n) {} // cannot be parallelized afaik

} // namespace tensor

namespace linalg {
template <> float dot(float *a, float *b, size_t n) { return cblas_sdot(n, a, 1, b, 1); }
template <> double dot(double *a, double *b, size_t n) { return cblas_ddot(n, a, 1, b, 1); }

template <> void matmul(float *a, float *b, float *c, size_t m, size_t n, size_t k) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n);
}
template <> void matmul(double *a, double *b, double *c, size_t m, size_t n, size_t k) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n);
}

} // namespace linalg

namespace nn {
void relu(float *data, size_t n) {
  std::transform(data, data + n, data, [](float x) { return std::max(0.0f, x); });
}

} // namespace nn

namespace stats {

template <> float variance(float *data, size_t n, bool sample) {
  std::vector<float> copy(n, 0.f); // have to use vector for heap mem and to keep templates clean
  memory::copy(data, copy.data(), n);
  float m = tensor::mean(copy.data(), n);
  tensor::fixed_update(copy.data(), -1 * m, copy.size());           // x - mean
  float sumsq = linalg::dot(copy.data(), copy.data(), copy.size()); // sum of squares
  return sumsq / (sample ? (n - 1) : n);                            // decide denominator based on smp or pop
}

template <> double variance(double *data, size_t n, bool sample) {
  std::vector<double> copy(n, 0.f); // have to use vector for heap mem and to keep templates clean
  memory::copy(data, copy.data(), n);
  double m = tensor::mean(copy.data(), n);
  tensor::fixed_update(copy.data(), -1 * m, copy.size());            // x - mean
  double sumsq = linalg::dot(copy.data(), copy.data(), copy.size()); // sum of squares
  return sumsq / (sample ? (n - 1) : n);                             // decide denominator based on smp or pop
}

// template <typename T> T std_deviation(T *data, size_t n) { return std::sqrt(variance(data, n)); }
template <> float std_deviation(float *data, size_t n) { return std::sqrt(variance(data, n)); }
template <> double std_deviation(double *data, size_t n) { return std::sqrt(variance(data, n)); }

float covariance(float *data1, float *data2, size_t n, bool sample) {
  float copy1[n], copy2[n];
  cblas_scopy(n, data1, 1, copy1, 1); // create copies as to not modify orig array
  cblas_scopy(n, data2, 1, copy2, 1);
  float m1 = tensor::mean(copy1, n);
  float m2 = tensor::mean(copy2, n);
  cblas_saxpy(n, -1, &m1, 0, copy1, 1);
  cblas_saxpy(n, -1, &m2, 0, copy2, 1);
  float sumsq = cblas_sdot(n, copy1, 1, copy2, 1); // sum all squares
  return sumsq / (sample ? (n - 1) : n);
}

} // namespace stats

namespace random {



// template <> void uniform(double *data, size_t n, double lower, double upper) {
//   std::random_device rd;  // Will be used to obtain a seed for the random number engine
//   std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
//   std::uniform_real_distribution<double> dist(lower, upper);

//   std::generate(data, data + n, [&]() { return dist(gen); });
// }

template <typename T> void uniform(T *data, size_t n, T lower, T upper) {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<T> dist(lower, upper);

  std::generate(data, data + n, [&]() { return dist(gen); });
}

template void uniform<float>(float*, std::size_t, float, float);
template void uniform<double>(double*, std::size_t, double, double);

template <typename T> T normal(T *data, size_t n, T mean, T stdev) {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution dist{mean, stdev};

  std::generate(data, data + n, [&]() { return dist(gen); });
}

} // namespace random

} // namespace cpu
} // namespace mathlib

#endif
