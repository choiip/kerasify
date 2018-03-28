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
            "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, stringify(x)); \
        return false; \
    }

#define check_eq(x, y, eps) \
    { \
        auto x_ = static_cast<double>(x); \
        auto y_ = static_cast<double>(y); \
        if (std::abs(x_ - y_) > eps) { \
            printf( \
                "ASSERT [%s:%d] %f isn't equal to %f ('%s' != '%s')\n", \
                __FILE__, __LINE__, x_, y_, stringify(x), stringify(y)); \
            return false; \
        } \
    }

#ifdef DEBUG
#define kassert(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, stringify(x)); \
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
