/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/locally2d.h"

namespace keras {
namespace layers {

bool LocallyConnected2D::load_layer(std::ifstream& file) noexcept
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

    unsigned weights_l = 0;
    check(read_uint(file, weights_l));
    check(weights_l > 0);

    unsigned biases_shape = 0;
    check(read_uint(file, biases_shape));
    check(biases_shape > 0);

    weights_.resize(weights_i, weights_j, weights_k, weights_l);
    check(read_floats(
        file, weights_.data_.data(),
        weights_i * weights_j * weights_k * weights_l));

    biases_.resize(biases_shape);
    check(read_floats(file, biases_.data_.data(), biases_shape));

    check(activation_.load_layer(file));
    return true;
}

// TODO: optimize for speed
bool LocallyConnected2D::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_[2] == weights_.dims_[3]);

    size_t offset_y = weights_.dims_[1] - 1;
    size_t offset_x = weights_.dims_[2] - 1;

    Tensor tmp{in.dims_[0] - offset_y, in.dims_[1] - offset_x,
               weights_.dims_[0]};
    /*
    auto& ww = weights_.dims_;
    size_t ws_ = ww[0] * ww[1] * ww[2] * ww[3];
    size_t ws0 = ww[1] * ww[2] * ww[3];
    size_t ws1 = ww[2] * ww[3];
    size_t ws2 = ww[3];

    size_t is0 = in.dims_[1] * in.dims_[2];
    size_t is1 = in.dims_[2];

    size_t ts0 = tmp.dims_[1] * tmp.dims_[2];
    size_t ts1 = tmp.dims_[2];

    auto* w_ptr = weights_.data_.data();
    auto* b_ptr = biases_.data_.data();
    auto* t_ptr = tmp.data_.data();
    auto* i_ptr = in.data_.data();

    for (size_t y = 0; y < tmp.dims_[0]; ++y)
        for (size_t x = 0; x < tmp.dims_[1]; ++x) {
            auto* b_ = b_ptr;
            auto* i_ = i_ptr + y * is0 + x * is1;
            auto* t_ = t_ptr + y * ts0 + x * ts1;
            for (auto* w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0) {
                auto* i0 = i_;
                for (auto* w1 = w0; w1 < w0 + ws0; w1 += ws1) {
                    auto* i1 = i0;
                    for (auto* w2 = w1; w2 < w1 + ws1; w2 += ws2) {
                        auto* i2 = i1;
                        for (auto* w3 = w2; w3 < w2 + ws2; ++w3) {
                            *t_ += (*w3) * (*i2); // convolute with kernel
                            ++i2;
                        }
                        i1 += is1;
                    }
                    i0 += is0;
                }
                *t_ += *b_; // add bias
                ++b_;
                ++t_;
            }
        }
    */
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
