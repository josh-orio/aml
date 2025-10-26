#ifdef USE_CUBLAS

#include "gpustats.hpp"

namespace mathlib {
namespace gpu {

const float one = 1;

namespace memory {

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

// --- offload: copy GPU data back to host memory
template <typename T> void offload(T *hostData, const T *deviceData, size_t count) {
  size_t bytes = count * sizeof(T);
  cudaError_t err = cudaMemcpy(hostData, deviceData, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy (DeviceToHost) failed: " << cudaGetErrorString(err) << std::endl;
  }
}

// --- clear: free GPU memory
template <typename T> void clear(T *deviceData) {
  if (deviceData != nullptr) {
    cudaError_t err = cudaFree(deviceData);
    if (err != cudaSuccess) {
      std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    }
  }
}
}

namespace tensor {}

namespace linalg {
void matmul(float *a, float *b, float *c, size_t m, size_t n, size_t k) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, m, b, k, &one, c, m);

  cublasDestroy(handle);
}
// very simple - wait until nv gemm is implemented to decide on extra option like T
} // namespace linalg

namespace nn {}

namespace stats {}

namespace random {}


} // namespace gpu
} // namespace mathlib

#endif
