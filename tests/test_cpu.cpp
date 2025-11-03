#include <statlib.hpp>
#include <array>
#include <gtest/gtest.h>

#ifdef USE_OPENBLAS

TEST(library, Cpu_Memory_Copy) {
  std::array<float, 5> a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  mathlib::cpu::memory::copy(a.data(), b.data(), a.size());

  EXPECT_EQ(a, b);
  EXPECT_EQ(b.back(), 4.0f);
}

TEST(library, Cpu_Memory_Copy_Double) {
  std::array<double, 5> a = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::array<double, 5> b = {};

  mathlib::cpu::memory::copy(a.data(), b.data(), a.size());

  EXPECT_EQ(a, b);
  EXPECT_EQ(b.back(), 4.0f);
}

TEST(library, Cpu_Memory_Swap) {
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
  std::array b = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array updated_a = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  mathlib::cpu::tensor::update(a.data(), b.data(), 1.0f, a.size());

  EXPECT_EQ(a, updated_a);
}

TEST(library, Cpu_Tensor_Fixedupdate) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array updated_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  mathlib::cpu::tensor::fixed_update(a.data(), 1.0f, a.size());

  EXPECT_EQ(a, updated_a);
}

TEST(library, Cpu_Tensor_Sum) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::tensor::sum(a.data(), a.size()), 10.0f);

  std::array b = {-1.0f, 0.0f, 1.0f};
  EXPECT_EQ(mathlib::cpu::tensor::sum(b.data(), b.size()), 0.0f);
}

TEST(library, Cpu_Tensor_Mean) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(a.data(), a.size()), 2.0f);

  std::array b = {-1.0f, 0.0f, 1.0f};
  EXPECT_EQ(mathlib::cpu::tensor::mean(b.data(), b.size()), 0.0f);
}

TEST(library, Cpu_Tensor_Min) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array b = {-1.0f, 0.0f, 1.0f};

  EXPECT_EQ(mathlib::cpu::tensor::min(a.data(), a.size()), 0.0f);
  EXPECT_EQ(mathlib::cpu::tensor::min(b.data(), b.size()), -1.0f);
}

TEST(library, Cpu_Tensor_Max) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array b = {-1.0f, 0.0f, 1.0f};

  EXPECT_EQ(mathlib::cpu::tensor::max(a.data(), a.size()), 4.0f);
  EXPECT_EQ(mathlib::cpu::tensor::max(b.data(), b.size()), 1.0f);
}

TEST(library, Cpu_Tensor_Normalize) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.f, 6.f};
  std::array expected_a = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  float mean, std;

  mathlib::cpu::tensor::normalize(a.data(), &mean, &std, a.size());

  EXPECT_EQ(mean, 3.0f);
  EXPECT_EQ(std, 2.0f);
  EXPECT_EQ(a, expected_a);
}

TEST(library, Cpu_Tensor_Denormalize) {
  std::array a = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  std::array expected_a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.f, 6.f};
  float mean(3.0f), std(2.0f);

  mathlib::cpu::tensor::denormalize(a.data(), mean, std, a.size());

  EXPECT_EQ(a, expected_a);
}

TEST(library, Cpu_Linalg_Dot) {
  std::array a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::array b = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  EXPECT_EQ(mathlib::cpu::linalg::dot(a.data(), b.data(), a.size()), 35.f);
}

TEST(library, Cpu_Linalg_Matmul) {
  /*
        B 1 2
          3 4
          5 6
          ---
  A 1 2 3|2228
    4 5 6|4964
  */

  std::vector a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector b = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector c(4, 0.f);
  std::vector expected_c = {22.f, 28.f, 49.f, 64.f};

  mathlib::cpu::linalg::matmul(a.data(), b.data(), c.data(), 2, 2, 3);
  // should do a test using Matrix class too

  EXPECT_EQ(c, expected_c);
}

TEST(library, Cpu_Nn_Relu) {
  std::array a = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  std::array relu_a = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
  mathlib::cpu::nn::relu(a.data(), a.size());
  EXPECT_EQ(a, relu_a);
}

TEST(library, Cpu_Stats_Stddeviation) {
  std::array a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  EXPECT_EQ(mathlib::cpu::stats::std_deviation(a.data(), a.size()), 2.0f);
}

TEST(library, Cpu_Stats_Variance) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(mathlib::cpu::stats::variance(a.data(), a.size()), 2.0f);

  std::array b = {0.11f, 0.62f, 0.53f, 0.41f, 0.93f, 0.02f, 0.74f};
  EXPECT_EQ(mathlib::cpu::stats::variance(b.data(), b.size(), true), 0.1076f);
}

TEST(library, Cpu_Stats_Covariance) {
  std::array a = {10.0f, 34.0f, 23.0f, 54.0f, 9.0f};
  std::array b = {4.f, 5.f, 11.f, 15.f, 20.f};

  EXPECT_EQ(mathlib::cpu::stats::covariance(a.data(), b.data(), a.size()), 4.6f);
  EXPECT_EQ(mathlib::cpu::stats::covariance(a.data(), b.data(), a.size(), true), 5.75);
}

// TEST(library, Cpu_Random_Uniform) {
//   std::array<float, 1000> a;
//   mathlib::cpu::random::uniform(a.data(), a.size(), 0.f, 1.f);

//   // how do i test this?
// }

#endif
