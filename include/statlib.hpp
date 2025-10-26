#ifndef STATLIB_HPP
#define STATLIB_HPP

#ifdef USE_OPENBLAS
#include "cpustats.hpp"
#endif

#ifdef USE_CUBLAS
#include "gpustats.hpp"
#endif

#endif
