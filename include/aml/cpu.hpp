#ifndef CPUMATH_HPP
#define CPUMATH_HPP

#include <algorithm>
#include <cblas.h>
#include <cstddef>
#include <random>
#include <vector>

namespace aml {
namespace cpu {

namespace memory {

template <typename T> void copy(const T *src, T *dst, size_t n);
template <typename T> void swap(T *a, T *b, size_t n);

} // namespace memory

namespace tensor {

template <typename T> void scale(T *a, T alpha, size_t n);        // a = (alpha * a)
template <typename T> void update(T *a, T *b, T alpha, size_t n); // a = a + (alpha * b)
template <typename T> void fixed_update(T *a, T alpha, size_t n); // a = a + alpha

template <typename T> T sum(const T *data, size_t n);
template <typename T> T mean(const T *data, std::size_t n);

template <typename T> T min(const T *data, size_t n);
template <typename T> T max(const T *data, size_t n);

template <typename T> void normalize(T *data, T *mean, T *std, size_t n); // args because no structs
template <typename T> void denormalize(T *data, T mean, T std, size_t n);

} // namespace tensor

namespace linalg {

template <typename T> T dot(const T *a, const T *b, size_t n);                                 // one dimensional arrays only
template <typename T> void matmul(const T *a, const T *b, T *c, size_t m, size_t n, size_t k); // no transpose atm

} // namespace linalg

namespace nn {

template <typename T> void relu(T *data, size_t n);
// void leaky_relu();
// void tanh();
// void sigmoid();

} // namespace nn

namespace stats {

template <typename T> T std_deviation(const T *data, size_t n);
template <typename T> T variance(const T *data, size_t n, bool sample = false); // returns sigma^2, defaults to pop var
template <typename T> T covariance(const T *a, const T *b, size_t n, bool sample = false);

} // namespace stats

namespace random {

template <typename T> void uniform(T *data, size_t n, T lower, T upper);
template <typename T> void normal(T *data, size_t n, T mean, T stdev);
template <typename T> void skew_normal(T *data, size_t n, T loc, T scale, T shape);

} // namespace random

} // namespace cpu
} // namespace aml

#endif
