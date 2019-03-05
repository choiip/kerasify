/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <memory>
#include <type_traits>

namespace keras {

class Stream {
    class _Impl;
    std::unique_ptr<_Impl> impl_;

public:
    Stream(const std::string&);
    ~Stream();

    Stream& read(char*, size_t);

    template <
        typename T,
        typename = std::enable_if_t<std::is_default_constructible_v<T>>>
    operator T() noexcept {
        T value;
        read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }
};

} // namespace keras
