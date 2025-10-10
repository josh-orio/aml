#include "include/cpustats.hpp"
#include "include/statlib.hpp"
#include <iostream>
#include <vector>

int main() {
  float a[] = {0, 1, 2, 3, 4};
  int n = 5;

  std::cout << mean(a, n) << std::endl;

  std::vector<float> data;
  data = {0, 1, 2, 3, 4};

  std::cout << mean(data.data(), data.size()) << std::endl;

  return 0;
}
