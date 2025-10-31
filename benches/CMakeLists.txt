include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.zip
)
FetchContent_MakeAvailable(googlebenchmark)

# define benching executable + link
add_executable(bench_example bench_cpu.cpp bench_gpu.cpp)
target_link_libraries(bench_example PRIVATE
    statlib benchmark::benchmark pthread)

# use higher optimization for benchmarks (idk how much this does)
target_compile_options(bench_example PRIVATE -O3)
