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

class tensor {
public:
    tensor() {}
    tensor(size_t i) { resize(i); }
    tensor(size_t i, size_t j) { resize(i, j); }
    tensor(size_t i, size_t j, size_t k) { resize(i, j, k); }
    tensor(size_t i, size_t j, size_t k, size_t l) { resize(i, j, k, l); }

    void resize(size_t i);
    void resize(size_t i, size_t j);
    void resize(size_t i, size_t j, size_t k);
    void resize(size_t i, size_t j, size_t k, size_t l);

    inline void flatten();

    inline float& operator()(size_t i);
    inline float& operator()(size_t i, size_t j);
    inline float& operator()(size_t i, size_t j, size_t k);
    inline float& operator()(size_t i, size_t j, size_t k, size_t l);
    inline float operator()(size_t i, size_t j) const;

    inline void fill(float value);

    tensor unpack(size_t row) const;
    tensor select(size_t row) const;
    tensor operator+(const tensor& other);
    tensor multiply(const tensor& other);
    tensor dot(const tensor& other);

    void print();
    void print_shape();

    std::vector<size_t> dims_;
    std::vector<float> data_;
};

void tensor::resize(size_t i)
{
    dims_ = {i};
    data_.resize(i);
}

void tensor::resize(size_t i, size_t j)
{
    dims_ = {i, j};
    data_.resize(i * j);
}

void tensor::resize(size_t i, size_t j, size_t k)
{
    dims_ = {i, j, k};
    data_.resize(i * j * k);
}

void tensor::resize(size_t i, size_t j, size_t k, size_t l)
{
    dims_ = {i, j, k, l};
    data_.resize(i * j * k * l);
}

void tensor::flatten()
{
    kassert(dims_.size() > 0);

    size_t elements = dims_[0];
    for (size_t i = 1; i < dims_.size(); ++i)
        elements *= dims_[i];
    dims_ = {elements};
}

float& tensor::operator()(size_t i)
{
    kassert(dims_.size() == 1);
    kassert(i < dims_[0]);
    return data_[i];
}

float& tensor::operator()(size_t i, size_t j)
{
    kassert(dims_.size() == 2);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    return data_[dims_[1] * i + j];
}

float tensor::operator()(size_t i, size_t j) const
{
    kassert(dims_.size() == 2);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    return data_[dims_[1] * i + j];
}

float& tensor::operator()(size_t i, size_t j, size_t k)
{
    kassert(dims_.size() == 3);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    return data_[dims_[2] * (dims_[1] * i + j) + k];
}

float& tensor::operator()(size_t i, size_t j, size_t k, size_t l)
{
    kassert(dims_.size() == 4);
    kassert(i < dims_[0]);
    kassert(j < dims_[1]);
    kassert(k < dims_[2]);
    kassert(l < dims_[3]);
    return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

void tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

tensor tensor::unpack(size_t row) const
{
    kassert(dims_.size() >= 2);
    auto pack_dims = std::vector<size_t>(dims_.begin() + 1, dims_.end());
    size_t pack_size = std::accumulate(pack_dims.begin(), pack_dims.end(), 0u);

    auto base = row * pack_size;
    auto first = data_.begin() + static_cast<ptrdiff_t>(base);
    auto last = data_.begin() + static_cast<ptrdiff_t>(base + pack_size);

    tensor x;
    x.dims_ = pack_dims;
    x.data_ = std::vector<float>(first, last);
    return x;
}

tensor tensor::select(size_t row) const
{
    auto x = unpack(row);
    x.dims_.insert(x.dims_.begin(), 1);
    return x;
}

tensor tensor::operator+(const tensor& other)
{
    kassert(dims_ == other.dims_);

    tensor result;
    result.dims_ = dims_;
    result.data_.reserve(data_.size());

    std::transform(
        data_.begin(), data_.end(), other.data_.begin(),
        std::back_inserter(result.data_),
        [](float x, float y) { return x + y; });
    return result;
}

tensor tensor::multiply(const tensor& other)
{
    kassert(dims_ == other.dims_);

    tensor result;
    result.dims_ = dims_;
    result.data_.reserve(data_.size());

    std::transform(
        data_.begin(), data_.end(), other.data_.begin(),
        std::back_inserter(result.data_),
        [](float x, float y) { return x * y; });
    return result;
}

tensor tensor::dot(const tensor& other)
{
    kassert(dims_.size() == 2);
    kassert(other.dims_.size() == 2);
    kassert(dims_[1] == other.dims_[0]);

    tensor tmp{dims_[0], other.dims_[1]};
    for (size_t i = 0; i < dims_[0]; ++i)
        for (size_t j = 0; j < other.dims_[1]; ++j)
            for (size_t k = 0; k < dims_[1]; ++k)
                tmp(i, j) += (*this)(i, k) * other(k, j);
    return tmp;
}

void tensor::print()
{
    if (dims_.size() == 1) {
        printf("[ ");
        for (size_t i = 0; i < dims_[0]; ++i)
            printf("%f ", static_cast<double>((*this)(i)));
        printf("]\n");
        return;
    }
    if (dims_.size() == 2) {
        printf("[\n");
        for (size_t i = 0; i < dims_[0]; ++i) {
            printf(" [ ");
            for (size_t j = 0; j < dims_[1]; ++j)
                printf("%f ", static_cast<double>((*this)(i, j)));
            printf("]\n");
        }
        printf("]\n");
        return;
    }
    if (dims_.size() == 3) {
        printf("[\n");
        for (size_t i = 0; i < dims_[0]; ++i) {
            printf(" [\n");
            for (size_t j = 0; j < dims_[1]; ++j) {
                printf("  [ ");
                for (size_t k = 0; k < dims_[2]; ++k)
                    printf("%f ", static_cast<double>((*this)(i, j, k)));
                printf("  ]\n");
            }
            printf(" ]\n");
        }
        printf("]\n");
        return;
    }
    if (dims_.size() == 4) {
        printf("[\n");
        for (size_t i = 0; i < dims_[0]; ++i) {
            printf(" [\n");
            for (size_t j = 0; j < dims_[1]; ++j) {
                printf("  [\n");
                for (size_t k = 0; k < dims_[2]; ++k) {
                    printf("   [");
                    for (size_t l = 0; l < dims_[3]; ++l)
                        printf("%f ", static_cast<double>((*this)(i, j, k, l)));
                    printf("]\n");
                }
                printf("  ]\n");
            }
            printf(" ]\n");
        }
        printf("]\n");
    }
}

void tensor::print_shape()
{
    printf("(");
    for (size_t i = 0; i < dims_.size(); ++i)
        printf("%zu ", dims_[i]);
    printf(")\n");
}
