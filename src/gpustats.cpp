#ifdef USE_CUBLAS

#include "gpustats.hpp"

namespace mathlib {
namespace gpu {

namespace memory {

template <typename T> void copy(const T *src, T *dst, size_t n) {
  if (n == 0) {
    return;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  if constexpr (std::is_same_v<T, float>) {
    cublasScopy(handle, n, src, 1, dst, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cublasDcopy(handle, n, src, 1, dst, 1);
  }

  cublasDestroy(handle);
}

template void copy(const float *src, float *dst, size_t n);
template void copy(const double *src, double *dst, size_t n);

template <typename T> void swap(T *a, T *b, size_t n) {
  if (n == 0) {
    return;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  if constexpr (std::is_same_v<T, float>) {
    cublasSswap(handle, n, a, 1, b, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cublasDswap(handle, n, a, 1, b, 1);
  }

  cublasDestroy(handle);
}

template void swap(float *a, float *b, size_t n);
template void swap(double *a, double *b, size_t n);

//   template <> void copy(float *src, float *dst, size_t n) { cblas_scopy(n, src, 1, dst, 1); }
// template <> void copy(double *src, double *dst, size_t n) { cblas_dcopy(n, src, 1, dst, 1); }
// template <> void swap(float *a, float *b, size_t n) { cblas_sswap(n, a, 1, b, 1); }
// template <> void swap(double *a, double *b, size_t n) { cblas_dswap(n, a, 1, b, 1); } switch to cublas or cuda call

// --- load: copy host data to GPU memory
template <typename T> T *load(const T *hostData, size_t count) {
  T *deviceData = nullptr;
  size_t bytes = count * sizeof(T);

  cudaError_t err = cudaMalloc(&deviceData, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }

  err = cudaMemcpy(deviceData, hostData, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy (HostToDevice) failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(deviceData);
    return nullptr;
  }

  return deviceData;
}

template float *load(const float *hostData, size_t count);
template double *load(const double *hostData, size_t count);

// --- offload: copy GPU data back to host memory
template <typename T> void offload(const T *deviceData, T *hostData, size_t count) {
  size_t bytes = count * sizeof(T);
  cudaError_t err = cudaMemcpy(hostData, deviceData, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy (DeviceToHost) failed: " << cudaGetErrorString(err) << std::endl;
  }
}

template void offload(const float *, float *, size_t);
template void offload(const double *, double *, size_t);

// --- clear: free GPU memory
template <typename T> void clear(T *deviceData) {
  if (deviceData != nullptr) {
    cudaError_t err = cudaFree(deviceData);
    if (err != cudaSuccess) {
      std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    }
  }
}

template void clear(float *deviceData);
template void clear(double *deviceData);

} // namespace memory

namespace tensor {

template <typename T> void scale(T *a, T alpha, size_t n) {
  if (n == 0) {
    return;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  if constexpr (std::is_same_v<T, float>) {
    cublasSscal(handle, n, &alpha, a, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cublasDscal(handle, n, &alpha, a, 1);
  }

  cublasDestroy(handle);
}

template void scale(float *a, float alpha, size_t n);
template void scale(double *a, double alpha, size_t n);

template <typename T> void update(T *a, T *b, T alpha, size_t n) {
  if (n == 0) {
    return;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  if constexpr (std::is_same_v<T, float>) {
    cublasSaxpy(handle, n, &alpha, b, 1, a, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cublasDaxpy(handle, n, &alpha, b, 1, a, 1);
  }

  cublasDestroy(handle);
}

template void update(float *a, float *b, float alpha, size_t n);
template void update(double *a, double *b, double alpha, size_t n);

template <typename T> void fixed_update(T *a, T alpha, size_t n) {
  if (n == 0) {
    return;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  T one = static_cast<T>(1);
  T *d_one;
  cudaMalloc(&d_one, sizeof(T));
  cudaMemcpy(d_one, &one, sizeof(T), cudaMemcpyHostToDevice);

  if constexpr (std::is_same_v<T, float>) {
    cublasSaxpy(handle, n, &alpha, d_one, 0, a, 1);

  } else if constexpr (std::is_same_v<T, double>) {
    cublasDaxpy(handle, n, &alpha, d_one, 0, a, 1);
  }

  cudaFree(d_one);
  cublasDestroy(handle);
}

template void fixed_update(float *a, float alpha, size_t n);
template void fixed_update(double *a, double alpha, size_t n);

template <typename T> T sum(const T *ptr, size_t n) {
  if (n == 0) {
    return T(0);
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  T one = static_cast<T>(1);
  T *d_one;
  cudaMalloc(&d_one, sizeof(T));
  cudaMemcpy(d_one, &one, sizeof(T), cudaMemcpyHostToDevice);

  if constexpr (std::is_same_v<T, float>) {
    auto len = static_cast<int>(n);
    T result;

    cublasSdot(handle, len, ptr, 1, d_one, 0, &result);

    return result;

  } else if constexpr (std::is_same_v<T, double>) {
    auto len = static_cast<int>(n);
    T result;

    cublasDdot(handle, len, ptr, 1, d_one, 0, &result);

    return result;
  }

  cudaFree(d_one);
  cublasDestroy(handle);
}

template float sum(const float *, std::size_t);
template double sum(const double *, std::size_t);

template <typename T> T mean(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  return sum(data, n) / static_cast<T>(n);
}

template float mean(const float *, std::size_t);
template double mean(const double *, std::size_t);

template <typename T> T min(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  auto iter = thrust::min_element(data, data + n);
  return *iter;
}

template float min(const float *, std::size_t);
template double min(const double *, std::size_t);

template <typename T> T max(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  auto iter = thrust::max_element(data, data + n);
  return *iter;
}

template float max(const float *, std::size_t);
template double max(const double *, std::size_t);

} // namespace tensor

namespace linalg {

template <typename T> T dot(const T *a, const T *b, size_t n) {
  if (n == 0) {
    return T(0);
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  if constexpr (std::is_same_v<T, float>) {
    T dot;
    cublasSdot(handle, n, a, 1, b, 1, &dot);
    return dot;

  } else if constexpr (std::is_same_v<T, double>) {
    T dot;
    cublasDdot(handle, n, a, 1, b, 1, &dot);
    return dot;
  }

  cublasDestroy(handle);
}

template float dot(const float *a, const float *b, size_t n);
template double dot(const double *a, const double *b, size_t n);

template <typename T> void matmul(const T *a, const T *b, T *c, size_t m, size_t n, size_t k) {
  if (n == 0) {
    return;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  T one(1), zero(0);

  if constexpr (std::is_same_v<T, float>) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, b, n, a, k, &zero, c, n);

  } else if constexpr (std::is_same_v<T, double>) {
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, b, n, a, k, &zero, c, n);
  }

  cublasDestroy(handle);
}

template void matmul(const float *, const float *, float *, size_t, size_t, size_t);
template void matmul(const double *, const double *, double *, size_t, size_t, size_t);

} // namespace linalg

namespace nn {

// template <typename T>
// void relu(T* data, size_t n) {
//     if (n == 0) return;
//     thrust::transform(data, data + n, data,
//                       [] __host__ __device__ (T x) { return x > T(0) ? x : T(0); });
// }

// template void relu(float *, size_t );
// template void relu(double *, size_t );

} // namespace nn

namespace stats {

template <typename T> T variance(const T *data, size_t n, bool sample) {
  if (n == 0) {
    return T(0);
  }

  T *copy = nullptr;
  size_t bytes = n * sizeof(T);

  cudaMalloc(&copy, bytes);

  memory::copy(data, copy, n);
  T m = tensor::mean(copy, n);
  tensor::fixed_update(copy, -1 * m, n);               // x - mean
  T sumsq = linalg::dot(copy, copy, n);                // sum of squares
  return sumsq / static_cast<T>(sample ? (n - 1) : n); // decide denominator based on smp or pop
}

template float variance(const float *data, size_t n, bool sample);
template double variance(const double *data, size_t n, bool sample);

template <typename T> T std_deviation(const T *data, size_t n) {
  if (n == 0) {
    return T(0);
  }

  return sqrt(variance(data, n));
}

template float std_deviation(const float *data, size_t n);
template double std_deviation(const double *data, size_t n);

template <typename T> T covariance(const T *a, const T *b, size_t n, bool sample) {
  if (n == 0) {
    return T(0);
  }

  T *copy_a = nullptr, *copy_b = nullptr;
  size_t bytes = n * sizeof(T);

  cudaMalloc(&copy_a, bytes);
  cudaMalloc(&copy_b, bytes);

  memory::copy(a, copy_a, n);
  memory::copy(b, copy_b, n);

  T mean_a(tensor::mean(a, n)), mean_b(tensor::mean(b, n));

  tensor::fixed_update(copy_a, -1 * mean_a, n);
  tensor::fixed_update(copy_b, -1 * mean_b, n);

  T sumsq = linalg::dot(copy_a, copy_b, n);
  return sumsq / static_cast<T>(sample ? (n - 1) : n);
}

template float covariance(const float *a, const float *b, size_t n, bool sample);
template double covariance(const double *a, const double *b, size_t n, bool sample);

} // namespace stats

namespace random {}

} // namespace gpu
} // namespace mathlib

#endif
