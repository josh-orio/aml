## Library

This document explains the design of the library.

The library has a CPU and GPU backend, both of which are accessible through namespaces ```library::cpu``` and ```library::gpu``` respectively.

Both backends are available at runtime, given that both BLAS dependencies are present. This gives the programmer the choice over what hardware a workload is run on.

Operations are split into the following categories:

| Group  | Description                                                          | Docs                   |
| ------ | -------------------------------------------------------------------- | ---------------------- |
| Memory | Non-mathematical memory ops (swap, copy)                             | [memory.md](memory.md) |
| Tensor | Elementwise and reduction ops (mean, sum, max, etc.)                 | [tensor.md](tensor.md) |
| Linalg | Linear algebra ops (matmul, dot, inv, svd, etc.)                     | [linalg.md](linalg.md) |
| NN     | Neural-network-specific functions (ReLU, softmax, etc.)              | [nn.md](nn.md)         |
| Stats  | Statistical ops (variance, stddev, distribution sampling)            | [stats.md](stats.md)   |
| Random | Random number generation, distributions (uniform, normal, bernoulli) | [random.md](random.md) |

Documentation for each of the groups is available [here](groups/).

### Dependencies

This library requires at least one of OpenBLAS and/or cuBLAS, installation guides are avilable for both [here](dependencies/).

### API Design

This library uses a stateless interface, no structs or internal systems are implemented, every function is a wrapper to BLAS functions, usually with a template to the appropriate BLAS routine.

The simplistic design of the interface is designed not to get in the way of performance, this means that there is no runtime checking of function parameters or try/catch blocks. As you use the library, you should be very careful that you only pass host memory pointers to functions in the ```cpu``` namespace functions and only pass device memory pointers to ```gpu``` namespace functions (with the obvious exception of transfer functions).

> [!CAUTION]
> Ensure host/device pointers are not used interchangably, they will crash your program.
