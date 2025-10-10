#ifndef STATLIB_HPP
#define STATLIB_HPP

#ifdef USE_OPENBLAS
// #error "OPENBLAS"
#include "cpustats.hpp"

#elif defined(USE_CUBLAS)
// #error "CUBLAS"
#include "gpustats.hpp"

#else
#error "Please define either USE_OPENBLAS or USE_CUBLAS"

#endif

#endif
