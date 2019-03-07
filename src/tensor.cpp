/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/tensor.h"

#include <iostream>

namespace keras {

Tensor::Tensor(std::initializer_list<size_t> sizes)
    : dims_ {sizes}, data_(size()) {}

Tensor Tensor::empty(std::initializer_list<size_t> sizes) {
    Tensor tensor;
    tensor.dims_ = sizes;
    tensor.data_.reserve(tensor.size());
    return tensor;
}

Tensor Tensor::empty(Tensor const& other) {
    Tensor tensor;
    tensor.dims_ = other.dims_;
    tensor.data_.reserve(tensor.size());
    return tensor;
}

Tensor::Tensor(Stream& file) : Tensor() {
    auto rank = static_cast<unsigned>(file);
    kassert(rank);

    dims_.reserve(rank);
    std::generate_n(std::back_inserter(dims_), rank, [&file] {
        auto stride = static_cast<unsigned>(file);
        kassert(stride > 0);
        return stride;
    });

    data_.resize(size());
    file.read(reinterpret_cast<char*>(data_.data()), sizeof(float) * size());
}

Tensor Tensor::select(size_t row) const noexcept {
    kassert(ndim() >= 2);

    Tensor x;
    x.dims_ = std::vector<size_t>(dims_.begin() + 1, dims_.end());
    x.dims_.insert(x.dims_.begin(), 1);

    size_t pack_size = std::accumulate(
        x.dims_.begin(), x.dims_.end(), 1u, std::multiplies<>());

    auto base = row * pack_size;
    auto first = begin() + cast(base);
    auto last = begin() + cast(base + pack_size);
    x.data_ = std::vector<float>(first, last);
    return x;
}

Tensor& Tensor::operator+=(const Tensor& other) noexcept {
    kassert(dims_ == other.dims_);
    std::transform(begin(), end(), other.begin(), begin(), std::plus<>());
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) noexcept {
    kassert(dims_ == other.dims_);
    std::transform(begin(), end(), other.begin(), begin(), std::multiplies<>());
    return *this;
}

Tensor Tensor::fma(const Tensor& scale, const Tensor& bias) const noexcept {
    kassert(dims_ == scale.dims_);
    kassert(dims_ == bias.dims_);

    auto result = Tensor::empty(*this);

    auto k_ = scale.begin();
    auto b_ = bias.begin();
    auto r_ = std::back_inserter(result.data_);
    for (auto i_ = begin(); i_ != end();)
        *(++r_) = std::fma(*(i_++), *(k_++), *(b_++));

    return result;
}

Tensor Tensor::dot(const Tensor& other) const noexcept {
    kassert(ndim() == 2);
    kassert(other.ndim() == 2);
    kassert(dims_[1] == other.dims_[1]);

    auto tmp = Tensor::empty({dims_[0], other.dims_[0]});
    auto step = cast(dims_[1]);

    auto t = std::back_inserter(tmp.data_);
    for (auto i_ = begin(); i_ != end(); i_ += step)
        for (auto o_ = other.begin(); o_ != other.end(); o_ += step)
            *(++t) = std::inner_product(i_, i_ + step, o_, 0.f);

    return tmp;
}

std::ostream& operator<<(std::ostream& os, Tensor const& t) noexcept {
    std::vector<size_t> steps(t.ndim());
    std::partial_sum(t.dims_.rbegin(), t.dims_.rend(),
                     steps.rbegin(),
                     std::multiplies<>());

    size_t count = 0;
    for (auto it : t.data_) {
        for (size_t step : steps)
            if (count % step == 0)
                os << "[";
        os << it;
        ++count;
        for (size_t step : steps)
            if (count % step == 0)
                os << "]";
        if (count != steps[0])
            os << ", ";
    }
    os << std::endl;
    return os;
}

void Tensor::print_shape() const noexcept {
    std::cout << "(";
    size_t count = 0;
    for (size_t dim : dims_) {
        std::cout << dim;
        if ((++count) != dims_.size())
            std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}

} // namespace keras
