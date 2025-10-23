#ifndef STATLIB_HPP
#define STATLIB_HPP

#ifdef USE_OPENBLAS
// #error "OPENBLAS"
#include "cpustats.hpp"
#endif

#ifdef USE_CUBLAS
// #error "CUBLAS"
#include "gpustats.hpp"
#endif

#endif
