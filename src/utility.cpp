/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/utility.h"

namespace keras {

unsigned Stream::to_uint() noexcept {
    unsigned out;
    *this >> out;
    return out;
}

Stream& Stream::operator>>(unsigned& out) noexcept {
    read(reinterpret_cast<char*>(&out), sizeof(unsigned));
    kassert(gcount() == sizeof(unsigned));
    return *this;
}

Stream& Stream::operator>>(float& out) noexcept {
    read(reinterpret_cast<char*>(&out), sizeof(float));
    kassert(gcount() == sizeof(float));
    return *this;
}

Stream& Stream::operator>>(std::vector<float>& out) noexcept {
    auto size = cast(sizeof(float) * out.size());
    read(reinterpret_cast<char*>(out.data()), size);
    kassert(gcount() == size);
    return *this;
}

} // namespace keras
