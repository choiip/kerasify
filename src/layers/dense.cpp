/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/dense.h"

namespace keras {
namespace layers {

bool Dense::load_layer(std::ifstream& file) noexcept
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

bool Dense::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.size() == weights_.dims_[0]);

    Tensor tmp{weights_.dims_[1]};

    auto in_ = in.data_.begin();
    auto out_ = tmp.data_.begin();

    const auto ws = static_cast<ptrdiff_t>(weights_.dims_[1]);
    const auto& w_data = weights_.data_;
    const auto& b_data = biases_.data_;

    std::copy(b_data.begin(), b_data.end(), out_);
    for (auto w = w_data.begin(); w < w_data.end(); w += ws, ++in_)
        std::transform(
            w, w + ws, out_, out_, [in_](float weight_, float bias_) {
                return weight_ * (*in_) + bias_;
            });

    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
