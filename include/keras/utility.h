/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>

#define stringify(x) #x

#define cast(x) static_cast<ptrdiff_t>(x)

#ifndef NDEBUG

void _asserted_eq(float, float, float,
                  std::string_view, int, std::string_view, std::string_view);

void _asserted(bool,
               std::string_view, int, std::string_view);

#define kassert_eq(x, y, eps) \
    _asserted_eq(x, y, eps, __FILE__, __LINE__, stringify(x), stringify(y));
#define kassert(x) \
    _asserted(x, __FILE__, __LINE__, stringify(x));

#else
#define kassert(x) ;
#define kassert_eq(x, y, eps) ;
#endif

namespace keras {

template <typename Callable, typename... Args>
auto timeit(Callable&& callable, Args&&... args) {
    using namespace std::chrono;

    auto begin = high_resolution_clock::now();
    auto result = [&]() {
        if constexpr (std::is_void_v<std::invoke_result_t<Callable, Args...>>)
            return (std::invoke(callable, args...), nullptr);
        else
            return std::invoke(callable, args...);
    }();
    return std::make_tuple(
        std::move(result),
        duration<double>(high_resolution_clock::now() - begin).count());
}

template <typename T, typename... Others>
constexpr decltype(auto) front(T&& t, Others&&...) noexcept {
    return std::forward<T>(t);
}

} // namespace keras
