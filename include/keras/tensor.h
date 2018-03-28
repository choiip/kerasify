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
    Tensor() {}
    Tensor(size_t i) { resize(i); }
    Tensor(size_t i, size_t j) { resize(i, j); }
    Tensor(size_t i, size_t j, size_t k) { resize(i, j, k); }
    Tensor(size_t i, size_t j, size_t k, size_t l) { resize(i, j, k, l); }

    void resize(size_t i);
    void resize(size_t i, size_t j);
    void resize(size_t i, size_t j, size_t k);
    void resize(size_t i, size_t j, size_t k, size_t l);

    inline void flatten();

    inline float& operator()(size_t i);
    inline float& operator()(size_t i, size_t j);
    inline float& operator()(size_t i, size_t j, size_t k);
    inline float& operator()(size_t i, size_t j, size_t k, size_t l);
    inline float operator()(size_t i) const;
    inline float operator()(size_t i, size_t j) const;
    inline float operator()(size_t i, size_t j, size_t k) const;
    inline float operator()(size_t i, size_t j, size_t k, size_t l) const;

    inline void fill(float value);

    Tensor unpack(size_t row) const;
    Tensor select(size_t row) const;
    Tensor operator+(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;

    void print() const;
    void print_shape() const;

    std::vector<size_t> dims_;
    std::vector<float> data_;
};

void Tensor::flatten()
{
    kassert(dims_.size() > 0);

    size_t elements = dims_[0];
    for (size_t i = 1; i < dims_.size(); ++i)
        elements *= dims_[i];
    dims_ = {elements};
}

float& Tensor::operator()(size_t i)
{
    kassert(dims_.size() == 1);
    kassert(i < dims_[0]);
    return data_[i];
}

float Tensor::operator()(size_t i) const
{
    kassert(dims_.size() == 1);
    kassert(i < dims_[0]);
    return data_[i];
}

float& Tensor::operator()(size_t i, size_t j)
{
    kassert(dims_.size() == 2);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    return data_[dims_[1] * i + j];
}

float Tensor::operator()(size_t i, size_t j) const
{
    kassert(dims_.size() == 2);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    return data_[dims_[1] * i + j];
}

float& Tensor::operator()(size_t i, size_t j, size_t k)
{
    kassert(dims_.size() == 3);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    return data_[dims_[2] * (dims_[1] * i + j) + k];
}

float Tensor::operator()(size_t i, size_t j, size_t k) const
{
    kassert(dims_.size() == 3);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    return data_[dims_[2] * (dims_[1] * i + j) + k];
}

float& Tensor::operator()(size_t i, size_t j, size_t k, size_t l)
{
    kassert(dims_.size() == 4);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    kassert(l < dims_[3]);
    return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

float Tensor::operator()(size_t i, size_t j, size_t k, size_t l) const
{
    kassert(dims_.size() == 4);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    kassert(l < dims_[3]);
    return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

void Tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

} // namespace keras
