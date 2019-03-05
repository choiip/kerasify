/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/io.h"
#include "keras/utility.h"
#include <algorithm>
#include <numeric>

#define INLINE_ANY_RETURN(prefix, function, body, nonconst) \
    prefix nonconst function noexcept { body } \
    prefix function const noexcept { body }

namespace keras {
class Tensor {
public:
    Tensor() = default;

    template <
        typename... Size,
        typename = std::enable_if_t<(... && std::is_integral_v<Size>)>>
    Tensor(Size... sizes)
    : dims_ {static_cast<size_t>(sizes)...}, data_(size()) {}

    Tensor(Stream& file);

    template <typename... Size>
    static auto empty(Size&&... sizes);

    inline size_t size() const noexcept;
    inline size_t ndim() const noexcept;
    inline Tensor& flatten() noexcept;

    template <typename...Ixs>
    size_t get_offset(Ixs... ixs) const noexcept {
        constexpr size_t count = sizeof... (Ixs);
        kassert(ndim() == count);

        std::array<size_t, count> indexes = {ixs...};
        kassert(std::inner_product(indexes.begin(), indexes.end(), dims_.begin(), true,
                                   [](size_t x, size_t y) { return x && y; },
                                   [](size_t i, size_t d) { return i < d; }));
        if constexpr (count == 1)
            return indexes.back();
        else {
            std::array<size_t, count-1> strides;
            std::partial_sum(dims_.rbegin(), dims_.rend()-1, strides.rbegin(), std::multiplies<size_t>());

            return std::inner_product(strides.begin(), strides.end(), indexes.begin(),
                                      indexes.back());
        }
    }

    INLINE_ANY_RETURN(template<typename...Ixs> float,
                      operator()(Ixs... indexes),
                      { return data_[get_offset<Ixs...>(indexes...)]; },
                      &)
    INLINE_ANY_RETURN(inline auto, begin(), { return data_.begin(); }, )
    INLINE_ANY_RETURN(inline auto, end(), { return data_.end(); }, )

    inline void fill(float value) noexcept;

    template <typename Func>
    auto map(Func&& func) const noexcept;

    Tensor select(size_t row) const noexcept;

    Tensor& operator+=(const Tensor& other) noexcept;
    Tensor& operator*=(const Tensor& other) noexcept;
    Tensor fma(const Tensor& scale, const Tensor& bias) const noexcept;
    Tensor dot(const Tensor& other) const noexcept;

    void print() const noexcept;
    void print_shape() const noexcept;

    std::vector<size_t> dims_;
    std::vector<float> data_;
};

template <typename... Size>
auto Tensor::empty(Size&&... sizes) {
    Tensor tensor;
    if constexpr (
        (sizeof...(Size) == 1)
        && std::is_same_v<
               std::decay_t<std::tuple_element_t<0, std::tuple<Size...>>>,
               Tensor>)
        tensor.dims_ = front(sizes...).dims_;
    else
        tensor.dims_ = {static_cast<size_t>(sizes)...};
    tensor.data_.reserve(tensor.size());
    return tensor;
}

size_t Tensor::size() const noexcept {
    return std::accumulate(
        dims_.begin(), dims_.end(), 1u, std::multiplies<size_t>());
}

size_t Tensor::ndim() const noexcept {
    return dims_.size();
}

Tensor& Tensor::flatten() noexcept {
    kassert(ndim());
    dims_ = {size()};
    return *this;
}

void Tensor::fill(float value) noexcept {
    std::fill(begin(), end(), value);
}

template <typename Func>
auto Tensor::map(Func&& func) const noexcept {
    auto target = Tensor::empty(*this);
    std::transform(begin(), end(), std::back_inserter(target.data_), func);
    return target;
}

inline Tensor operator+(Tensor lhs, const Tensor& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

inline Tensor operator*(Tensor lhs, const Tensor& rhs) noexcept {
    lhs *= rhs;
    return lhs;
}

} // namespace keras
