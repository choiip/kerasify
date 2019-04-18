/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/utility.h"
#include <iostream>

void _asserted_eq(
    float x, float y, float eps,
    const char* file, int lineno, const char* xs, const char* ys
) {
    if (std::abs(x - y) <= eps)
        return;

    std::cout << "ASSERT [" << file << ":" << lineno << "] "
              << x << " isn't equal to " << y
              << " ('" << xs << "' != '" << ys << "')"
              << std::endl;
    std::exit(-1);
}

void _asserted(bool x, const char* file, int lineno, const char* xs) {
    if (x)
        return;

    std::cout << "ASSERT [" << file << ":" << lineno << "] '"
              << xs << "' failed"
              << std::endl;
    std::exit(-1);
}
