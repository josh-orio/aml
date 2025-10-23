#ifndef CPUSTATS_HPP
#define CPUSTATS_HPP

#include <cblas.h>
#include <cmath>
#include <cstddef>

namespace mathlib {
namespace cpu {

namespace tensor {}

namespace linalg {}

namespace nn {}

namespace stats {}

namespace random {}

float sum(float *data, size_t n);
float mean(float *data, size_t n);
float variance(float *data, size_t n, bool sample = false); // returns sigma^2, defaults to pop var
float std_deviation(float *data, size_t n);
float covariance(float *data1, float *data2, size_t n, bool sample = false);
void relu(float *data, size_t n);
void matmul(float *a, float *b, float *c, size_t m, size_t n, size_t k);

} // namespace cpu
} // namespace mathlib

#endif
