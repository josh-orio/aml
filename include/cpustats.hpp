#ifndef CPUSTATS_HPP
#define CPUSTATS_HPP

#include <cblas.h>
#include <cmath>
#include <cstddef>

namespace mathlib {
namespace cpu {

namespace memory {

template <typename T> void copy(T *src, T *dst, size_t n);
template <typename T> void swap(T *a, T *b, size_t n);

} // namespace memory

namespace tensor {

template <typename T> void scale(T *a, T alpha, size_t n); // a = (alpha * a)
template <typename T> void update(T *a, T *b, T alpha, size_t n); // a = a + (alpha * b)
template <typename T> void fixed_update(T* a, T alpha, size_t n); // a = a + alpha

template <typename T> T sum(T *data, size_t n);
template <typename T> T mean(T *data, size_t n);
template <typename T> T min(T *data, size_t n);
template <typename T> T max(T *data, size_t n);

} // namespace tensor

namespace linalg {

// float dot(float *a, float *b, size_t n); // one dimensional arrays only
// void matmul(float *a, float *b, float *c, size_t m, size_t n, size_t k);

template <typename T> T dot(T *a, T *b, size_t n);                                 // one dimensional arrays only
template <typename T> void matmul(T *a, T *b, T *c, size_t m, size_t n, size_t k); // wait until nv gemm is implemented to decide on extra option like T

} // namespace linalg

namespace nn {

template <typename T> void relu(T *data, size_t n);
// void leaky_relu();
// void tanh();
// void sigmoid();

} // namespace nn

namespace stats {

template <typename T> T std_deviation(T *data, size_t n);
template <typename T> T variance(T *data, size_t n, bool sample = false); // returns sigma^2, defaults to pop var
template <typename T> T covariance(T *data1, T *data2, size_t n, bool sample = false);

} // namespace stats

namespace random {

template <typename T> void uniform(T *data, size_t n, T lower, T upper);
template <typename T> void  normal(T *data, size_t n, T mean, T stdev);

} // namespace random

} // namespace cpu
} // namespace mathlib

#endif
