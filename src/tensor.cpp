/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/tensor.h"

namespace keras {

void Tensor::resize(size_t i) noexcept
{
    dims_ = {i};
    data_.resize(i);
}

void Tensor::resize(size_t i, size_t j) noexcept
{
    dims_ = {i, j};
    data_.resize(i * j);
}

void Tensor::resize(size_t i, size_t j, size_t k) noexcept
{
    dims_ = {i, j, k};
    data_.resize(i * j * k);
}

void Tensor::resize(size_t i, size_t j, size_t k, size_t l) noexcept
{
    dims_ = {i, j, k, l};
    data_.resize(i * j * k * l);
}

Tensor Tensor::unpack(size_t row) const noexcept
{
    kassert(dims_.size() >= 2);
    auto pack_dims = std::vector<size_t>(dims_.begin() + 1, dims_.end());
    size_t pack_size = std::accumulate(pack_dims.begin(), pack_dims.end(), 0u);

    auto base = row * pack_size;
    auto first = data_.begin() + static_cast<ptrdiff_t>(base);
    auto last = data_.begin() + static_cast<ptrdiff_t>(base + pack_size);

    Tensor x;
    x.dims_ = pack_dims;
    x.data_ = std::vector<float>(first, last);
    return x;
}

Tensor Tensor::select(size_t row) const noexcept
{
    auto x = unpack(row);
    x.dims_.insert(x.dims_.begin(), 1);
    return x;
}

Tensor Tensor::operator+(const Tensor& other) const noexcept
{
    kassert(dims_ == other.dims_);

    Tensor result;
    result.dims_ = dims_;
    result.data_.reserve(data_.size());

    std::transform(
        data_.begin(), data_.end(), other.data_.begin(),
        std::back_inserter(result.data_),
        [](float x, float y) { return x + y; });
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const noexcept
{
    kassert(dims_ == other.dims_);

    Tensor result;
    result.dims_ = dims_;
    result.data_.reserve(data_.size());

    std::transform(
        data_.begin(), data_.end(), other.data_.begin(),
        std::back_inserter(result.data_),
        [](float x, float y) { return x * y; });
    return result;
}

// TODO: optimize for speed
Tensor Tensor::dot(const Tensor& other) const noexcept
{
    kassert(dims_.size() == 2);
    kassert(other.dims_.size() == 2);
    kassert(dims_[1] == other.dims_[0]);

    Tensor tmp{dims_[0], other.dims_[1]};
    for (size_t i = 0; i < dims_[0]; ++i)
        for (size_t j = 0; j < other.dims_[1]; ++j)
            for (size_t k = 0; k < dims_[1]; ++k)
                tmp(i, j) += (*this)(i, k) * other(k, j);
    return tmp;
}

void Tensor::print() const noexcept
{
    if (dims_.size() == 1) {
        printf("[");
        for (size_t i = 0; i < dims_[0]; ++i)
            printf("%f ", static_cast<double>((*this)(i)));
        printf("]\n");
        return;
    }
    if (dims_.size() == 2) {
        printf("[\n");
        for (size_t i = 0; i < dims_[0]; ++i) {
            printf(" [");
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
                printf("  [");
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

void Tensor::print_shape() const noexcept
{
    printf("(");
    for (auto&& it : dims_)
        printf("%zu ", it);
    printf(")\n");
}
} // namespace keras
