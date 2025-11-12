cmake_minimum_required(VERSION 3.28.0)
project(aml LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

file(GLOB library_SOURCES "src/*.c*")
file(GLOB library_INCLUDE "include/*.hpp")

add_library(aml ${library_SOURCES})
target_include_directories(aml
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(aml PRIVATE -Wall -Wextra -Wuninitialized -Wno-sign-compare)

# cuBLAS detection & linking to library
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
   enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    message(STATUS "Using cuBLAS from CUDA Toolkit")
    target_compile_definitions(aml PUBLIC USE_CUBLAS)
    target_link_libraries(aml PUBLIC CUDA::cublas CUDA::cudart)

    target_compile_options(aml PRIVATE  # enables device lambdas
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

else()
    message(STATUS "CUDAToolkit not found. Building without cuBLAS.")
endif()

# OpenBLAS detection & linking to library
find_package(OpenBLAS)
if (OpenBLAS_FOUND)
    message(STATUS "Using OpenBLAS")
    target_compile_definitions(aml PUBLIC USE_OPENBLAS)
    target_link_libraries(aml PUBLIC OpenBLAS::OpenBLAS)
else()
    message(STATUS "OpenBLAS not found. Building without it.")
endif()

set_target_properties(aml
    PROPERTIES
    CMAKE_CXX_STANDARD 20
    CMAKE_CXX_STANDARD_REQUIRED ON
    CMAKE_CXX_EXTENSIONS OFF
)

include(GNUInstallDirs)

target_include_directories(aml
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS aml
    EXPORT libraryTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${library_INCLUDE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aml)

install(EXPORT libraryTargets
    FILE libraryTargets.cmake
    NAMESPACE aml::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aml
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/amlConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aml
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/amlConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aml
)

add_executable(example_program example.cpp)
target_link_libraries(example_program PRIVATE aml)
target_compile_options(example_program PRIVATE -Wall -Wextra -Wuninitialized)

enable_testing() # enabling ctest - 'make test'
add_subdirectory(tests) # include test dir

option(BUILD_BENCHES "Build benchmark targets" OFF) # cmake -DBUILD_BENCHES=ON/OFF ..
if(BUILD_BENCHES)
    add_subdirectory(benches)
endif()
