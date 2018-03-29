/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/conv1d.h"

namespace keras {
namespace layers {

bool Conv1D::load_layer(std::ifstream& file)
{
    unsigned weights_i = 0;
    check(read_uint(file, weights_i));
    check(weights_i > 0);

    unsigned weights_j = 0;
    check(read_uint(file, weights_j));
    check(weights_j > 0);

    unsigned weights_k = 0;
    check(read_uint(file, weights_k));
    check(weights_k > 0);

    unsigned biases_shape = 0;
    check(read_uint(file, biases_shape));
    check(biases_shape > 0);

    weights_.resize(weights_i, weights_j, weights_k);
    check(read_floats(
        file, weights_.data_.data(), weights_i * weights_j * weights_k));

    biases_.resize(biases_shape);
    check(read_floats(file, biases_.data_.data(), biases_shape));

    check(activation_.load_layer(file));
    return true;
}

bool Conv1D::apply(const Tensor& in, Tensor& out) const
{
    check(in.dims_[1] == weights_.dims_[2]);

    size_t offset = weights_.dims_[1] - 1;

    Tensor tmp{in.dims_[0] - offset, weights_.dims_[0]};

    auto& ww = weights_.dims_;
    size_t ws_ = ww[0] * ww[1] * ww[2];
    size_t ws0 = ww[1] * ww[2];
    size_t ws1 = ww[2];

    size_t is0 = in.dims_[1];
    size_t ts0 = tmp.dims_[1];

    auto* w_ptr = &weights_.data_[0];
    auto* b_ptr = &biases_.data_[0];
    auto* t_ptr = &tmp.data_[0];
    auto* i_ptr = &in.data_[0];

    for (size_t x = 0; x < tmp.dims_[1]; ++x) {
        auto* b_ = b_ptr;
        auto* i_ = i_ptr + x * is0;
        auto* t_ = t_ptr + x * ts0;
        for (auto* w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0) {
            auto* i0 = i_;
            for (auto* w1 = w0; w1 < w0 + ws0; w1 += ws1) {
                auto* i1 = i0;
                for (auto* w2 = w1; w2 < w1 + ws1; ++w2) {
                    *t_ += (*w2) * (*i1); // convolute with kernel
                    ++i1;
                }
                i0 += is0;
            }
            *t_ += *b_; // add bias
            ++b_;
            ++t_;
        }
    }
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
