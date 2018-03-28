/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/utility.h"

namespace keras {

bool read_uint(std::ifstream* file, unsigned& i)
{
    check(file);

    file->read(reinterpret_cast<char*>(&i), sizeof(unsigned));
    check(file->gcount() == sizeof(unsigned));
    return true;
}

bool read_float(std::ifstream* file, float& f)
{
    check(file);

    file->read(reinterpret_cast<char*>(&f), sizeof(float));
    check(file->gcount() == sizeof(float));
    return true;
}

bool read_floats(std::ifstream* file, float* f, size_t n)
{
    check(file);
    check(f);

    file->read(reinterpret_cast<char*>(f), sizeof(float) * n);
    check(static_cast<size_t>(file->gcount()) == sizeof(float) * n);
    return true;
}

} // namespace keras
