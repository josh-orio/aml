cmake_minimum_required(VERSION 3.28.0)
project(statlib LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

file(GLOB library_SOURCES "src/*.cpp")
file(GLOB library_INCLUDE "include/*.hpp")

# message("library sources: ${library_SOURCES}")
# message("library includes: ${library_INCLUDE}")

add_library(statlib ${library_SOURCES})
target_include_directories(statlib PRIVATE include)
target_compile_options(statlib PRIVATE -Wall -Wextra -Wuninitialized -Wno-sign-compare)

# cuBLAS detection & linking to library
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message(STATUS "Using cuBLAS from CUDA Toolkit")
    target_compile_definitions(statlib PUBLIC USE_CUBLAS)
    target_link_libraries(statlib PUBLIC CUDA::cublas CUDA::cudart)
else()
    message(STATUS "CUDAToolkit not found. Building without cuBLAS.")
endif()

# OpenBLAS detection & linking to library
find_package(OpenBLAS)
if (OpenBLAS_FOUND)
    message(STATUS "Using OpenBLAS")
    target_compile_definitions(statlib PUBLIC USE_OPENBLAS)
    target_link_libraries(statlib PUBLIC OpenBLAS::OpenBLAS)
else()
    message(STATUS "OpenBLAS not found. Building without it.")
endif()

set_target_properties(statlib
    PROPERTIES
    CMAKE_CXX_STANDARD 20
    CMAKE_CXX_STANDARD_REQUIRED ON
    CMAKE_CXX_EXTENSIONS OFF
)

include(GNUInstallDirs)

target_include_directories(statlib
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS statlib
    EXPORT libraryTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${library_INCLUDE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/statlib)

install(EXPORT libraryTargets
    FILE libraryTargets.cmake
    NAMESPACE statlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statlib
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/statlibConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statlib
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/statlibConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statlib
)

add_executable(test_program test.cpp)
target_link_libraries(test_program PRIVATE statlib)
target_compile_options(test_program PRIVATE -Wall -Wextra -Wuninitialized)
