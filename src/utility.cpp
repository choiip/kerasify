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
    std::string_view file, int lineno, std::string_view xs, std::string_view ys
) {
    if (std::abs(x - y) <= eps)
        return;

    std::cout << "ASSERT [" << file << ":" << lineno << "] "
              << x << " isn't equal to " << y
              << " ('" << xs << "' != '" << ys << "')"
              << std::endl;
    std::exit(-1);
}

void _asserted(bool x, std::string_view file, int lineno, std::string_view xs) {
    if (x)
        return;

    std::cout << "ASSERT [" << file << ":" << lineno << "] '"
              << xs << "' failed"
              << std::endl;
    std::exit(-1);
}
