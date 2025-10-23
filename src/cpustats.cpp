#ifdef USE_OPENBLAS

#include "cpustats.hpp"
#include <algorithm>

namespace mathlib {
namespace cpu {

namespace tensor {}

namespace linalg {}

namespace nn {}

namespace stats {}

namespace random {}

float one = 1;

float mean(float *data, size_t n) {
  float sum = cblas_sdot(n, data, 1, &one, 0);
  return sum / n;
}

float variance(float *data, size_t n, bool sample) {
  float copy[n];
  cblas_scopy(n, data, 1, copy, 1); // create copy as to not modify orig array
  float m = mean(copy, n);
  cblas_saxpy(n, -1, &m, 0, copy, 1);            // xi - xmean
  float sumsq = cblas_sdot(n, copy, 1, copy, 1); // sum all squares
  return sumsq / (sample ? (n - 1) : n);         // decide denominator based on smp or pop
}

float std_deviation(float *data, size_t n) { return std::sqrt(variance(data, n)); }

float covariance(float *data1, float *data2, size_t n, bool sample) {
  float copy1[n], copy2[n];
  cblas_scopy(n, data1, 1, copy1, 1); // create copies as to not modify orig array
  cblas_scopy(n, data2, 1, copy2, 1);
  float m1 = mean(copy1, n);
  float m2 = mean(copy2, n);
  cblas_saxpy(n, -1, &m1, 0, copy1, 1);
  cblas_saxpy(n, -1, &m2, 0, copy2, 1);
  float sumsq = cblas_sdot(n, copy1, 1, copy2, 1); // sum all squares
  return sumsq / (sample ? (n - 1) : n);
}

namespace activations {

void relu(float *data, size_t n) {
  std::transform(data, data + n, data, [](float x) { return std::max(0.0f, x); });
}

} // namespace activations

namespace fn {
void matmul(float *a, float *b, float *c, size_t m, size_t n, size_t k) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n);
  // very simple - wait until nv gemm is implemented to decide on extra option like T
}
} // namespace fn

} // namespace cpu
} // namespace mathlib

#endif
