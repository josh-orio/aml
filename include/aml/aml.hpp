#ifndef AML_HPP
#define AML_HPP

#include <vector>

#ifdef USE_OPENBLAS
#include <aml/cpu.hpp>
#endif

#ifdef USE_CUBLAS
#include <aml/gpu.hpp>
#endif

namespace aml {
namespace structs {

template <typename T> class Matrix {
public:
  Matrix(size_t rows, size_t cols, T init = T()) : _data(rows * cols, init), _rows(rows), _cols(cols) {}

  size_t rows() const noexcept { return _rows; }
  size_t cols() const noexcept { return _cols; }

  // access contiguous array
  T *start() noexcept { return _data.data(); }
  const T *start() const noexcept { return _data.data(); }

  // access to vector instance
  std::vector<T> &data() noexcept { return _data; }
  const std::vector<T> &data() const noexcept { return _data; }

  // element access
  T &at(size_t row, size_t col) {
    if (row >= _rows || col >= _cols)
      throw std::out_of_range("Matrix::at");
    return _data[row * _cols + col];
  }

  const T &at(size_t row, size_t col) const {
    if (row >= _rows || col >= _cols)
      throw std::out_of_range("Matrix::at");
    return _data[row * _cols + col];
  }

private:
  std::vector<T> _data;
  size_t _rows, _cols;
};

} // namespace structs
} // namespace aml

#endif
