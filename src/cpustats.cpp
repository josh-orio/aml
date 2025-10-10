#include "cpustats.hpp"

float one = 1;

float mean(float *data, size_t n) { return cblas_sdot(n, data, 1, &one, 0) / n; }
