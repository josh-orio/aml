#include <cblas.h>
#ifdef USE_OPENBLAS

#include "cpustats.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace mathlib {
namespace cpu {

const float one_f = 1;
const double one_d = 1;

namespace memory {

template <typename T> void copy(const T *src, T *dst, size_t n) {
  if (n == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_scopy(n, src, 1, dst, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dcopy(n, src, 1, dst, 1);
  }
}

template void copy(const float *src, float *dst, size_t n);
template void copy(const double *src, double *dst, size_t n);

template <typename T> void swap(T *a, T *b, size_t n) {
  if (n == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_sswap(n, a, 1, b, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dswap(n, a, 1, b, 1);
  }
}

template void swap(float *a, float *b, size_t n);
template void swap(double *a, double *b, size_t n);

} // namespace memory

namespace tensor {

template <typename T> void scale(T *a, T alpha, size_t n) {
  if (n == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_sscal(n, alpha, a, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dscal(n, alpha, a, 1);
  }
}

template void scale(float *a, float alpha, size_t n);
template void scale(double *a, double alpha, size_t n);

template <typename T> void update(T *a, T *b, T alpha, size_t n) {
  if (n == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(n, alpha, b, 1, a, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(n, alpha, b, 1, a, 1);
  }
}

template void update(float *a, float *b, float alpha, size_t n);
template void update(double *a, double *b, double alpha, size_t n);

template <typename T> void fixed_update(T *a, T alpha, size_t n) {
  if (n == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(n, alpha, &one_f, 0, a, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(n, alpha, &one_d, 0, a, 1);
  }
}

template void fixed_update(float *a, float alpha, size_t n);
template void fixed_update(double *a, double alpha, size_t n);

template <typename T> T sum(const T *ptr, size_t n) {
  if (n == 0) {
    return T(0);
  }

  if constexpr (std::is_same_v<T, float>) {
    return static_cast<T>(cblas_sdot(static_cast<int>(n), ptr, 1, &one_f, 0));

  } else if constexpr (std::is_same_v<T, double>) {
    return static_cast<T>(cblas_ddot(static_cast<int>(n), ptr, 1, &one_d, 0));
  }
}

template float sum(const float *, std::size_t);
template double sum(const double *, std::size_t);

template <typename T> T mean(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  if constexpr (std::is_same_v<T, float>) {
    return static_cast<T>(cblas_sdot(static_cast<int>(n), data, 1, &one_f, 0)) / static_cast<T>(n);

  } else if constexpr (std::is_same_v<T, double>) {
    return static_cast<T>(cblas_ddot(static_cast<int>(n), data, 1, &one_d, 0)) / static_cast<T>(n);
  }
}

template float mean(const float *, std::size_t);
template double mean(const double *, std::size_t);

template <typename T> T min(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  auto min /*pointer for some reason*/ = std::min_element(data, data + n); // also performance?
  return *min;
}

template float min(const float *, std::size_t);
template double min(const double *, std::size_t);

template <typename T> T max(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  auto max = std::max_element(data, data + n);
  return *max;
}

template float max(const float *, std::size_t);
template double max(const double *, std::size_t);

} // namespace tensor

namespace linalg {

template <typename T> T dot(const T *a, const T *b, size_t n) {
  if (n == 0) {
    return T(0);
  }

  if constexpr (std::is_same_v<T, float>) {
    return cblas_sdot(n, a, 1, b, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    return cblas_ddot(n, a, 1, b, 1);
  }
}

template float dot(const float *a, const float *b, size_t n);
template double dot(const double *a, const double *b, size_t n);

template <typename T> void matmul(const T *a, const T *b, T *c, size_t m, size_t n, size_t k) {
  if (n == 0) {
    return;
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n);
  }
}

template void matmul(const float *a, const float *b, float *c, size_t m, size_t n, size_t k);
template void matmul(const double *a, const double *b, double *c, size_t m, size_t n, size_t k);

} // namespace linalg

namespace nn {

template <typename T> void relu(T *data, size_t n) {
  if (n == 0) {
    return;
  }

  std::transform(data, data + n, data, [](float x) { return std::max(0.0f, x); });
}

template void relu(float *data, size_t n);
template void relu(double *data, size_t n);

} // namespace nn

namespace stats {

template <typename T> T variance(const T *data, size_t n, bool sample) {
  if (n == 0) {
    return T(0);
  }

  std::vector<T> copy(n, static_cast<T>(0.f)); // have to use vector for heap mem and to keep templates clean
  memory::copy(data, copy.data(), n);
  T m = tensor::mean(copy.data(), n);
  tensor::fixed_update(copy.data(), -1 * m, copy.size());       // x - mean
  T sumsq = linalg::dot(copy.data(), copy.data(), copy.size()); // sum of squares
  return sumsq / static_cast<T>(sample ? (n - 1) : n);          // decide denominator based on smp or pop
}

template float variance(const float *data, size_t n, bool sample);
template double variance(const double *data, size_t n, bool sample);

template <typename T> T std_deviation(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  return std::sqrt(variance(data, n));
}

template float std_deviation(const float *data, size_t n);
template double std_deviation(const double *data, size_t n);

template <typename T> T covariance(const T *a, const T *b, size_t n, bool sample) {
  if (n == 0) {
    return T(0);
  }

  std::vector<T> copy_a(n, static_cast<T>(0.f)), copy_b(n, static_cast<T>(0.f));
  memory::copy(a, copy_a.data(), n);
  memory::copy(b, copy_b.data(), n);

  T mean_a(tensor::mean(a, n)), mean_b(tensor::mean(b, n));

  tensor::fixed_update(copy_a.data(), -1 * mean_a, n);
  tensor::fixed_update(copy_b.data(), -1 * mean_b, n);

  T sumsq = linalg::dot(copy_a.data(), copy_b.data(), n);
  return sumsq / static_cast<T>(sample ? (n - 1) : n);
}

template float covariance(const float *a, const float *b, size_t n, bool sample);
template double covariance(const double *a, const double *b, size_t n, bool sample);

} // namespace stats

namespace random {

template <typename T> void uniform(T *data, size_t n, T lower, T upper) {
  std::random_device rd;
  std::mt19937 gen(rd()); // mersenne_twister_engine
  std::uniform_real_distribution<T> dist(lower, upper);

  std::generate(data, data + n, [&]() { return dist(gen); });
}

template void uniform<float>(float *, std::size_t, float, float);
template void uniform<double>(double *, std::size_t, double, double);

template <typename T> void normal(T *data, size_t n, T mean, T stdev) {
  std::random_device rd;
  std::mt19937 gen(rd()); // mersenne_twister_engine
  std::normal_distribution dist{mean, stdev};

  std::generate(data, data + n, [&]() { return dist(gen); });
}

template void normal(float *data, size_t n, float mean, float stdev);
template void normal(double *data, size_t n, double mean, double stdev);

} // namespace random

} // namespace cpu
} // namespace mathlib

#endif
