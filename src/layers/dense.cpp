/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/dense.h"

namespace keras {
namespace layers {

bool Dense::load_layer(std::ifstream& file) noexcept {
    check(weights_.load(file, 2));
    check(biases_.load(file));
    check(activation_.load_layer(file));
    return true;
}

bool Dense::apply(const Tensor& in, Tensor& out) const noexcept {
    check(in.dims_.back() == weights_.dims_.back());
    const auto ws = cast(weights_.dims_.back());

    Tensor tmp;
    tmp.dims_ = in.dims_;
    tmp.dims_.back() = weights_.dims_.front();
    tmp.data_.reserve(tmp.size());

    auto tmp_ = tmp.begin();
    for (auto in_ = in.begin(); in_ < in.end(); in_ += ws) {
        auto bias_ = biases_.begin();
        for (auto w = weights_.begin(); w < weights_.end(); w += ws)
            *(tmp_++) = std::inner_product(w, w + ws, in_, *(bias_++));
    }

    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
