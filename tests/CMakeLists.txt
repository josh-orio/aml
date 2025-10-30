include(FetchContent)

# download and configure GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# define the test executable
add_executable(test_all test_cpu.cpp test_gpu.cpp)
target_link_libraries(test_all
  PRIVATE
    statlib
    GTest::gtest_main
)

# discover tests
include(GoogleTest)
gtest_discover_tests(test_all)
