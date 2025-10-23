## Library

This document explains the design of the library.

The library had a CPU and GPU backend, both of which are accessible through namespaces ```library::cpu``` and ```library::gpu``` respectively.

Both backends are available at runtime, given that both BLAS dependencies are present. This gives the programmer the choice over what hardware a workload is run on.

Operations are split into the following categories:

| Group  | Description                                                          | Docs                   |
| ------ | -------------------------------------------------------------------- | ---------------------- |
| Tensor | Elementwise and reduction ops (mean, sum, max, etc.)                 | [tensor.md](tensor.md) |
| Linalg | Linear algebra ops (matmul, dot, inv, svd, etc.)                     | [linalg.md](linalg.md) |
| NN     | Neural-network-specific functions (ReLU, softmax, etc.)              | [nn.md](nn.md)         |
| Stats  | Statistical ops (variance, stddev, distribution sampling)            | [stats.md](stats.md)   |
| Random | Random number generation, distributions (uniform, normal, bernoulli) | [random.md](random.md) |
