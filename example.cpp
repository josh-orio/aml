#include "include/cpustats.hpp"
#include "include/statlib.hpp"
#include <array>
#include <iostream>
#include <random>
#include <vector>

int main() {
  // std::array a = {0.f,1.f,2.f,3.f,4.f};
  // float mean_a = mathlib::cpu::tensor::mean(a.data(), a.size());

  const size_t m = 16, n = 16, k = 256;

  std::array<std::array<float, k>, m> a;
  std::array<std::array<float, n>, k> b;
  std::array<std::array<float, n>, m> c;

  mathlib::cpu::random::uniform(a.data()->data(), k * m, 0.f, 1.f);
  mathlib::cpu::random::uniform(b.data()->data(), n * k, 0.f, 1.f);

  mathlib::cpu::linalg::matmul(a.data()->data(), b.data()->data(), c.data()->data(), m, n, k);

  return 0;
}
