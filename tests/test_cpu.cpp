#include "include/statlib.hpp"
#include <gtest/gtest.h>

#ifdef USE_OPENBLAS

TEST(library, CpuMemoryCopy) {
  std::array<float, 5> a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  mathlib::cpu::memory::copy(a.data(), b.data(), a.size());

  EXPECT_EQ(a, b);
  EXPECT_EQ(b.back(), 4.0f);
}

TEST(library, CpuMemoryCopyDouble) {
  std::array<double, 5> a = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::array<double, 5> b = {};

  mathlib::cpu::memory::copy(a.data(), b.data(), a.size());

  EXPECT_EQ(a, b);
  EXPECT_EQ(b.back(), 4.0f);
}

TEST(library, CpuMemorySwap) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  std::array old_a = a, old_b = b; // keep old copies is the only way, i think

  mathlib::cpu::memory::swap(a.data(), b.data(), a.size());

  EXPECT_EQ(a, old_b);
  EXPECT_EQ(b, old_a);
}

TEST(library, CpuTensorMean) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(a.data(), a.size()), 2.0f);

  std::array b = {-1.0f, 0.0f, 1.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(b.data(), b.size()), 0.0f);
}

#endif
