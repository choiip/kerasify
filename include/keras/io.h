/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace keras {

class Stream: public std::ifstream {
    using _Base = std::ifstream;

    Stream& read(char*, size_t);

public:
    Stream(const std::string&);

    template <
        typename T,
        typename = std::enable_if_t<std::is_default_constructible<T>::value>>
    operator T() noexcept {
        T value;
        read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    template <typename T>
    Stream& operator>>(std::vector<T>& v) {
        return read(reinterpret_cast<char*>(v.data()), sizeof(T) * v.size());
    }
};

} // namespace keras
