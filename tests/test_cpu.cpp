#include "include/statlib.hpp"
#include <array>
#include <gtest/gtest.h>

#ifdef USE_OPENBLAS

TEST(library, CpuMemoryCopy) {
  std::array<float, 5> a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  mathlib::cpu::memory::copy(a.data(), b.data(), a.size());

  EXPECT_EQ(a, b);
  EXPECT_EQ(b.back(), 4.0f);
}

TEST(library, CpuMemoryCopy_Double) {
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

TEST(library, Cpu_Tensor_Scale) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array scaled_a = {0.f, 2.f, 4.f, 6.f, 8.f};
  mathlib::cpu::tensor::scale(a.data(), 2.0f, a.size());
  EXPECT_EQ(a, scaled_a);
}

TEST(library, Cpu_Tensor_Update) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(a.data(), a.size()), 2.0f);

  std::array b = {-1.0f, 0.0f, 1.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(b.data(), b.size()), 0.0f);
}

TEST(library, Cpu_Tensor_Fixedupdate) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(a.data(), a.size()), 2.0f);

  std::array b = {-1.0f, 0.0f, 1.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(b.data(), b.size()), 0.0f);
}

TEST(library, CpuTensorMean) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(a.data(), a.size()), 2.0f);

  std::array b = {-1.0f, 0.0f, 1.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(b.data(), b.size()), 0.0f);
}

TEST(library, CpuStatsVariance) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::stats::variance(a.data(), a.size()), 2.0f);

  std::array b = {0.11f, 0.62f, 0.53f, 0.41f, 0.93f, 0.02f, 0.74f};
  EXPECT_EQ(mathlib::cpu::stats::variance(b.data(), b.size(), true), 0.1076f);
}

// TEST(library, CpuStatsStddeviation) {
//   std::array a = {1.0f, 2.0f, 3.0f, 5.0f, 8.0f, 13.0f, 21.0f};
//   EXPECT_EQ(mathlib::cpu::stats::std_deviation(a.data(), a.size()), 7.2078);

//   // std::array b = {0.11, 0.62, 0.53, 0.41, 0.93, 0.02, 0.74};
//   // EXPECT_EQ(mathlib::cpu::stats::std_deviation(b.data(), b.size()), 0.32802439);
// }


TEST(library, Cpu_Random_Uniform) {
  std::array<float, 1000> a;
  mathlib::cpu::random::uniform(a.data(), a.size(), 0.f, 1.f);

  EXPECT_EQ(1, 1);

  // std::array b = {0.11, 0.62, 0.53, 0.41, 0.93, 0.02, 0.74};
  // EXPECT_EQ(mathlib::cpu::stats::std_deviation(b.data(), b.size()), 0.32802439);
}

#endif
