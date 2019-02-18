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

#define INLINE_ANY_RETURN(function, body, nonconst) \
    inline auto nonconst function noexcept { body } \
    inline auto function const noexcept { body }

namespace keras {
class Tensor {
public:
    Tensor() = default;

    template <
        typename... Size,
        typename = std::enable_if_t<(... && std::is_integral_v<Size>)>>
    Tensor(Size... sizes)
    : dims_ {static_cast<size_t>(sizes)...}, data_(size()) {}

    Tensor(Stream& file, size_t rank = 1);

    template <typename... Size>
    static auto empty(Size&&... sizes);

    template <typename Func>
    auto map(Func&& func) const noexcept;

    inline size_t size() const noexcept;
    inline size_t ndim() const noexcept;
    inline Tensor& flatten() noexcept;

    INLINE_ANY_RETURN(operator()(size_t i), {
        kassert(ndim() == 1);
        kassert(i < dims_[0]);
        return data_[i];
    }, &)

    INLINE_ANY_RETURN(operator()(size_t i, size_t j), {
        kassert(ndim() == 2);
        kassert(i < dims_[0]);
        kassert(j < dims_[1]);
        return data_[dims_[1] * i + j];
    }, &)

    INLINE_ANY_RETURN(operator()(size_t i, size_t j, size_t k), {
        kassert(ndim() == 3);
        kassert(i < dims_[0]);
        kassert(j < dims_[1]);
        kassert(k < dims_[2]);
        return data_[dims_[2] * (dims_[1] * i + j) + k];
    }, &)

    INLINE_ANY_RETURN(operator()(size_t i, size_t j, size_t k, size_t l), {
        kassert(ndim() == 4);
        kassert(i < dims_[0]);
        kassert(j < dims_[1]);
        kassert(k < dims_[2]);
        kassert(l < dims_[3]);
        return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
    }, &)

    INLINE_ANY_RETURN(begin(), { return data_.begin(); }, )
    INLINE_ANY_RETURN(end(), { return data_.end(); }, )

    inline void fill(float value) noexcept;

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

template <typename Func>
auto Tensor::map(Func&& func) const noexcept {
    auto target = Tensor::empty(*this);
    std::transform(begin(), end(), std::back_inserter(target.data_), func);
    return target;
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

inline Tensor operator+(Tensor lhs, const Tensor& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

inline Tensor operator*(Tensor lhs, const Tensor& rhs) noexcept {
    lhs *= rhs;
    return lhs;
}

} // namespace keras
