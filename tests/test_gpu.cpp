#include <aml.hpp>
#include <gtest/gtest.h>

#ifdef USE_CUBLAS

TEST(library, Gpu_Memory_Copy) {
  std::array<float, 5> a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  auto d_b = aml::gpu::memory::load(b.data(), b.size());

  aml::gpu::memory::copy(d_a, d_b, a.size());

  aml::gpu::memory::offload(d_b, b.data(), b.size());

  EXPECT_EQ(a, b);
  EXPECT_EQ(b.back(), 4.0f);
}

TEST(library, Gpu_Memory_Swap) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  std::array old_a = a, old_b = b; // keep old copies is the only way, i think

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  auto d_b = aml::gpu::memory::load(b.data(), b.size());

  aml::gpu::memory::swap(d_a, d_b, a.size());

  aml::gpu::memory::offload(d_a, a.data(), a.size());
  aml::gpu::memory::offload(d_b, b.data(), b.size());

  EXPECT_EQ(a, old_b);
  EXPECT_EQ(b, old_a);
}

TEST(library, Gpu_Memory_Load_Offload) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 5> b = {};

  auto dev_mem = aml::gpu::memory::load(a.data(), a.size());
  aml::gpu::memory::offload(dev_mem, b.data(), b.size());

  EXPECT_EQ(a, b);
}

TEST(library, Gpu_Tensor_Scale) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array scaled_a = {0.f, 2.f, 4.f, 6.f, 8.f};

  //   aml::Gpu::tensor::scale(a.data(), 2.0f, a.size());
  auto d_a = aml::gpu::memory::load(a.data(), a.size());

  aml::gpu::tensor::scale(d_a, 2.0f, a.size());

  aml::gpu::memory::offload(d_a, a.data(), a.size());

  EXPECT_EQ(a, scaled_a);
}

TEST(library, Gpu_Tensor_Update) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array b = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array updated_a = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  auto d_b = aml::gpu::memory::load(b.data(), b.size());

  aml::gpu::tensor::update(d_a, d_b, 1.0f, a.size());

  aml::gpu::memory::offload(d_a, a.data(), a.size());

  EXPECT_EQ(a, updated_a);
}

TEST(library, Gpu_Tensor_Fixedupdate) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array updated_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  auto d_a = aml::gpu::memory::load(a.data(), a.size());

  aml::gpu::tensor::fixed_update(d_a, 1.0f, a.size());

  aml::gpu::memory::offload(d_a, a.data(), a.size());

  EXPECT_EQ(a, updated_a);
}

TEST(library, Gpu_Tensor_Sum) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  EXPECT_EQ(aml::gpu::tensor::sum(d_a, a.size()), 10.0f);
}

TEST(library, Gpu_Tensor_Mean) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  EXPECT_EQ(aml::gpu::tensor::mean(d_a, a.size()), 2.0f);
}

TEST(library, Gpu_Tensor_Min) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array b = {-1.0f, 0.0f, 1.0f};

  EXPECT_EQ(aml::gpu::tensor::min(a.data(), a.size()), 0.0f);
  EXPECT_EQ(aml::gpu::tensor::min(b.data(), b.size()), -1.0f);
}

TEST(library, Gpu_Tensor_Max) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::array b = {-1.0f, 0.0f, 1.0f};

  EXPECT_EQ(aml::gpu::tensor::max(a.data(), a.size()), 4.0f);
  EXPECT_EQ(aml::gpu::tensor::max(b.data(), b.size()), 1.0f);
}

TEST(library, Gpu_Tensor_Normalize) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.f, 6.f};
  std::array expected_a = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  float mean, std;

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  aml::gpu::tensor::normalize(d_a, &mean, &std, a.size());
  aml::gpu::memory::offload(d_a, a.data(), a.size());

  EXPECT_EQ(mean, 3.0f);
  EXPECT_EQ(std, 2.0f);
  EXPECT_EQ(a, expected_a);
}

TEST(library, Gpu_Tensor_Denormalize) {
  std::array a = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  std::array expected_a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.f, 6.f};
  float mean(3.0f), std(2.0f);

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  aml::gpu::tensor::denormalize(d_a, mean, std, a.size());
  aml::gpu::memory::offload(d_a, a.data(), a.size());

  EXPECT_EQ(a, expected_a);
}

TEST(library, Gpu_Linalg_Dot) {
  std::array a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::array b = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  auto d_b = aml::gpu::memory::load(b.data(), b.size());

  EXPECT_EQ(aml::gpu::linalg::dot(d_a, d_b, a.size()), 35.f);
}

TEST(library, Gpu_Linalg_Matmul) {
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

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  auto d_b = aml::gpu::memory::load(b.data(), b.size());
  auto d_c = aml::gpu::memory::load(c.data(), c.size());

  aml::gpu::linalg::matmul(d_a, d_b, d_c, 2, 2, 3);

  aml::gpu::memory::offload(d_c, c.data(), c.size());

  EXPECT_EQ(c, expected_c);
}

TEST(library, Gpu_Nn_Relu) {
  std::array a = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  std::array relu_a = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

  auto d_a = aml::gpu::memory::load(a.data(), a.size());

  aml::gpu::nn::relu(d_a, a.size());

  aml::gpu::memory::offload(d_a, a.data(), a.size());

  EXPECT_EQ(a, relu_a);
}

TEST(library, Gpu_Stats_Stddeviation) {
  std::array a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  EXPECT_EQ(aml::gpu::stats::std_deviation(d_a, a.size()), 2.0f);
}

TEST(library, Gpu_Stats_Variance) {
  std::array a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  EXPECT_EQ(aml::gpu::stats::variance(d_a, a.size()), 2.0f);
}

TEST(library, Gpu_Stats_Covariance) {
  std::array a = {10.0f, 34.0f, 23.0f, 54.0f, 9.0f};
  std::array b = {4.f, 5.f, 11.f, 15.f, 20.f};

  auto d_a = aml::gpu::memory::load(a.data(), a.size());
  auto d_b = aml::gpu::memory::load(b.data(), b.size());

  EXPECT_EQ(aml::gpu::stats::covariance(d_a, d_b, a.size()), 4.6f);
  EXPECT_EQ(aml::gpu::stats::covariance(d_a, d_b, a.size(), true), 5.75);
}

#endif
