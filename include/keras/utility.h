/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

#define stringify(x) #x

#define cast(x) static_cast<ptrdiff_t>(x)

// #ifdef DEBUG
#define kassert_eq(x, y, eps) \
    { \
        auto x_ = static_cast<double>(x); \
        auto y_ = static_cast<double>(y); \
        if (std::abs(x_ - y_) > eps) { \
            printf( \
                "ASSERT [%s:%d] %f isn't equal to %f ('%s' != '%s')\n", \
                __FILE__, __LINE__, x_, y_, stringify(x), stringify(y)); \
            exit(-1); \
        } \
    }
#define kassert(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, stringify(x)); \
        exit(-1); \
    }
// #else
// #define kassert(x) ;
// #define kassert_eq(x, y, eps) ;
// #endif

namespace keras {

template <typename Function, typename... Args>
double timeit(Function&& function, Args&&... args) noexcept {
    auto begin = std::chrono::high_resolution_clock::now();

    std::forward<Function>(function)(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - begin).count();
}

class Stream: public std::ifstream {
    template <typename T>
    friend Stream& operator>>(Stream&, T&) noexcept;

    template <typename T>
    friend Stream& operator>>(Stream&, std::vector<T>&) noexcept;

public:
    using std::ifstream::ifstream;

    template <typename T>
    T get() noexcept {
        T value;
        *this >> value;
        return value;
    }
};

template <typename T>
Stream& operator>>(Stream& ifs, T& value) noexcept {
    ifs.read(reinterpret_cast<char*>(&value), sizeof(T));
    kassert(ifs);
    return ifs;
}

template <typename T>
Stream& operator>>(Stream& ifs, std::vector<T>& vector) noexcept {
    auto size = sizeof(T) * vector.size();
    printf(" Read vector with size %zu\n", size);
    ifs.read(reinterpret_cast<char*>(vector.data()), cast(size));
    kassert(ifs);
    return ifs;
}


} // namespace keras
