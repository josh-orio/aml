#ifndef GPUSTATS_HPP
#define GPUSTATS_HPP

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace mathlib {
namespace gpu {

namespace memory {

template <typename T> void copy(T *src, T *dst, size_t n);
template <typename T> void swap(T *a, T *b, size_t n);

template <typename T> T *load(const T *hostData, size_t count);                     // load: copy host data to GPU memory
template <typename T> void offload(T *hostData, const T *deviceData, size_t count); // offload: copy gpu data back to host memory
template <typename T> void clear(T *deviceData);                                    // clear: free gpu memory

} // namespace memory

namespace tensor {}

namespace linalg {
void matmul(float *a, float *b, float *c, size_t m, size_t n, size_t k);
}

namespace nn {}

namespace stats {}

namespace random {}

} // namespace gpu
} // namespace mathlib

#endif
