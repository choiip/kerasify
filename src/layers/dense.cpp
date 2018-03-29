/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/dense.h"

namespace keras {
namespace layers {

bool Dense::load_layer(std::ifstream& file)
{
    unsigned weights_i = 0;
    check(read_uint(file, weights_i));
    check(weights_i > 0);

    unsigned weights_j = 0;
    check(read_uint(file, weights_j));
    check(weights_j > 0);

    unsigned biases_shape = 0;
    check(read_uint(file, biases_shape));
    check(biases_shape > 0);

    weights_.resize(weights_i, weights_j);
    check(read_floats(file, weights_.data_.data(), weights_i * weights_j));

    biases_.resize(biases_shape);
    check(read_floats(file, biases_.data_.data(), biases_shape));

    check(activation_.load_layer(file));
    return true;
}

bool Dense::apply(const Tensor& in, Tensor& out) const
{
    check(in.size() == weights_.dims_[0]);

    Tensor tmp{weights_.dims_[1]};

    auto* w_ = &weights_.data_[0];
    auto* t_ = &tmp.data_[0];
    auto* i_ = &in.data_[0];

    const size_t ws_ = weights_.dims_[0] * weights_.dims_[1];
    const size_t ws0 = weights_.dims_[1];

    for (auto* w0 = w_; w0 < w_ + ws_; w0 += ws0) {
        auto* t0 = t_;
        for (auto* w1 = w0; w1 < w0 + ws0; ++w1) {
            *t0 += (*i_) * (*w1);
            ++t0;
        }
        ++i_;
    }
    for (auto&& b : biases_.data_) {
        *t_ += b;
        ++t_;
    }

    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
