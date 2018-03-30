/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/utility.h"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

namespace keras {
class Tensor {
public:
    Tensor() = default;
    Tensor(size_t i) { resize(i); }
    Tensor(size_t i, size_t j) { resize(i, j); }
    Tensor(size_t i, size_t j, size_t k) { resize(i, j, k); }
    Tensor(size_t i, size_t j, size_t k, size_t l) { resize(i, j, k, l); }

    void resize(size_t i) noexcept;
    void resize(size_t i, size_t j) noexcept;
    void resize(size_t i, size_t j, size_t k) noexcept;
    void resize(size_t i, size_t j, size_t k, size_t l) noexcept;

    inline size_t size() const noexcept;
    inline void flatten() noexcept;

    inline float& operator()(size_t) noexcept;
    inline float& operator()(size_t, size_t) noexcept;
    inline float& operator()(size_t, size_t, size_t) noexcept;
    inline float& operator()(size_t, size_t, size_t, size_t) noexcept;
    inline float operator()(size_t) const noexcept;
    inline float operator()(size_t, size_t) const noexcept;
    inline float operator()(size_t, size_t, size_t) const noexcept;
    inline float operator()(size_t, size_t, size_t, size_t) const noexcept;

    inline void fill(float value) noexcept;

    Tensor unpack(size_t row) const noexcept;
    Tensor select(size_t row) const noexcept;
    Tensor operator+(const Tensor& other) const noexcept;
    Tensor multiply(const Tensor& other) const noexcept;
    Tensor dot(const Tensor& other) const noexcept;

    void print() const noexcept;
    void print_shape() const noexcept;

    std::vector<size_t> dims_;
    std::vector<float> data_;
};

size_t Tensor::size() const noexcept
{
    size_t elements = 1;
    for (const auto& it : dims_)
        elements *= it;
    return elements;
}

void Tensor::flatten() noexcept
{
    kassert(dims_.size() > 0);
    dims_ = {size()};
}

float& Tensor::operator()(size_t i) noexcept
{
    kassert(dims_.size() == 1);
    kassert(i < dims_[0]);
    return data_[i];
}

float Tensor::operator()(size_t i) const noexcept
{
    kassert(dims_.size() == 1);
    kassert(i < dims_[0]);
    return data_[i];
}

float& Tensor::operator()(size_t i, size_t j) noexcept
{
    kassert(dims_.size() == 2);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    return data_[dims_[1] * i + j];
}

float Tensor::operator()(size_t i, size_t j) const noexcept
{
    kassert(dims_.size() == 2);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    return data_[dims_[1] * i + j];
}

float& Tensor::operator()(size_t i, size_t j, size_t k) noexcept
{
    kassert(dims_.size() == 3);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    return data_[dims_[2] * (dims_[1] * i + j) + k];
}

float Tensor::operator()(size_t i, size_t j, size_t k) const noexcept
{
    kassert(dims_.size() == 3);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    return data_[dims_[2] * (dims_[1] * i + j) + k];
}

float& Tensor::operator()(size_t i, size_t j, size_t k, size_t l) noexcept
{
    kassert(dims_.size() == 4);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    kassert(l < dims_[3]);
    return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

float Tensor::operator()(size_t i, size_t j, size_t k, size_t l) const noexcept
{
    kassert(dims_.size() == 4);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    kassert(l < dims_[3]);
    return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

void Tensor::fill(float value) noexcept
{
    std::fill(data_.begin(), data_.end(), value);
}

} // namespace keras
