#include <aml/aml.hpp>
#include <array>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::array a = {0.f, 1.f, 2.f, 3.f, 4.f};
  float mean_a = aml::cpu::tensor::mean(a.data(), a.size());

  size_t m = 8192, n = 8192, k = 48192;

  aml::structs::Matrix<float> mfa(m, k, 1.0f);
  aml::structs::Matrix<float> mfb(k, n, 1.0f);
  aml::structs::Matrix<float> mfc(m, n, 0.0f);

  auto start = std::chrono::high_resolution_clock::now();
  aml::cpu::linalg::matmul(mfa.start(), mfb.start(), mfc.start(), m, n, k);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "matmul took " << elapsed.count() << " seconds.\n";

  return 0;
}
