#ifndef GPUSTATS_HPP
#define GPUSTATS_HPP

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

namespace mathlib {
namespace gpu {

namespace memory {

template <typename T> void copy(const T *src, T *dst, size_t n);
template <typename T> void swap(T *a, T *b, size_t n);

template <typename T> T *load(const T *hostData, size_t count);                     // load: copy host data to GPU memory
template <typename T> void offload(const T *deviceData, T *hostData, size_t count); // offload: copy gpu data back to host memory
template <typename T> void clear(T *deviceData);                                    // clear: free gpu memory

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

} // namespace tensor

namespace linalg {

template <typename T> T dot(const T *a, const T *b, size_t n);                                 // one dimensional arrays only
template <typename T> void matmul(const T *a, const T *b, T *c, size_t m, size_t n, size_t k); // no transpose atm

} // namespace linalg

namespace nn {

template <typename T> void relu(T *data, size_t n);

} // namespace nn

namespace stats {

template <typename T> T std_deviation(const T *data, size_t n);
template <typename T> T variance(const T *data, size_t n, bool sample = false); // returns sigma^2, defaults to pop var
template <typename T> T covariance(const T *a, const T *b, size_t n, bool sample = false);

} // namespace stats

namespace random {

template <typename T> void uniform(T *data, size_t n, T lower, T upper);
template <typename T> void normal(T *data, size_t n, T mean, T stdev);

} // namespace random

} // namespace gpu
} // namespace mathlib

#endif
