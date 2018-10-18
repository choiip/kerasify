/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/dense.h"

namespace keras {
namespace layers {

void Dense::load(Stream& file) noexcept {
    weights_.load(file, 2);
    biases_.load(file);
    activation_.load(file);
}

Tensor Dense::operator()(const Tensor& in) const noexcept {
    kassert(in.dims_.back() == weights_.dims_[1]);
    const auto ws = cast(weights_.dims_[1]);

    Tensor tmp;
    tmp.dims_ = in.dims_;
    tmp.dims_.back() = weights_.dims_[0];
    tmp.data_.reserve(tmp.size());

    auto tmp_ = tmp.begin();
    for (auto in_ = in.begin(); in_ < in.end(); in_ += ws) {
        auto bias_ = biases_.begin();
        for (auto w = weights_.begin(); w < weights_.end(); w += ws)
            *(tmp_++) = std::inner_product(w, w + ws, in_, *(bias_++));
    }

    return activation_(tmp);
}

} // namespace layers
} // namespace keras
