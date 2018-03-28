/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <cmath>
#include <fstream>

#define stringify(x) #x

#define check(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] - '%s' failed\n", __FILE__, __LINE__, \
            stringify(x)); \
        return false; \
    }

#define check_eq(x, y, eps) \
    if (std::abs((x) - (y)) > eps) { \
        printf( \
            "ASSERT [%s:%d] - expected %f, got %f\n", __FILE__, __LINE__, \
            static_cast<double>(y), static_cast<double>(x)); \
        return false; \
    }

#ifdef DEBUG
#define kassert(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] - '%s' failed\n", __FILE__, __LINE__, \
            stringify(x)); \
        exit(-1); \
    }
#else
#define kassert(x) ;
#endif

namespace keras {

bool read_uint(std::ifstream* file, unsigned& i);
bool read_float(std::ifstream* file, float& f);
bool read_floats(std::ifstream* file, float* f, size_t n);

} // namespace keras
