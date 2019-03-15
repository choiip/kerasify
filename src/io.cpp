/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/io.h"

namespace keras {

Stream::Stream(std::string const& filename): _Base{filename, std::ios::binary} {
    if (!is_open())
        throw std::ios_base::failure("Cannot open " + filename);
}

Stream& Stream::read(char* ptr, size_t count) {
    _Base::read(ptr, static_cast<ptrdiff_t>(count));
    if (!*this)
        throw std::ios_base::failure("File read failure");
    return *this;
}

} // namespace keras
