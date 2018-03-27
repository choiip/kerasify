/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <fstream>

#define check(x) \
    if (!(x)) { \
        printf("CHECK: %s(%d)\n", __FILE__, __LINE__); \
        return false; \
    }

#define check_eq(x, y, eps) \
    if (fabs(x - y) > eps) { \
        printf("CHECK: Expected %f, got %f\n", y, x); \
        return false; \
    }

#ifdef DEBUG
#define kassert(x) \
    if (!(x)) { \
        printf("%s(%d)\n", __FILE__, __LINE__); \
        exit(-1); \
    }
#else
#define kassert(x) ;
#endif

bool read_uint(std::ifstream* file, unsigned& i);

bool read_float(std::ifstream* file, float& f);

bool read_floats(std::ifstream* file, float* f, size_t n);
