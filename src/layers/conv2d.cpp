/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/conv2d.h"

namespace keras {
namespace layers {

bool Conv2D::load_layer(std::ifstream& file) noexcept
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

bool Conv2D::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_[2] == weights_.dims_[3]);

    size_t offset_y = weights_.dims_[1] - 1;
    size_t offset_x = weights_.dims_[2] - 1;

    Tensor tmp{in.dims_[0] - offset_y, in.dims_[1] - offset_x,
               weights_.dims_[0]};

    auto& ww = weights_.dims_;
    auto scpd = [](auto x) { return static_cast<ptrdiff_t>(x); };

    auto ts0 = scpd(ww[0] * tmp.dims_[1]);
    auto ts1 = scpd(ww[0]);
    auto ws_ = scpd(ww[0] * ww[1] * ww[2] * ww[3]);
    auto ws0 = scpd(ww[1] * ww[2] * ww[3]);
    auto ws1 = scpd(ww[2] * ww[3]);
    auto ws2 = scpd(ww[3]);
    auto is0 = scpd(ww[3] * in.dims_[1]);

    auto w_ptr = weights_.data_.begin();
    auto b_ptr = biases_.data_.begin();
    auto t_ptr = tmp.data_.begin();
    auto i_ptr = in.data_.begin();

    auto ty = scpd(tmp.dims_[0]);
    auto tx = scpd(tmp.dims_[1]);

    for (ptrdiff_t y = 0; y < ty; ++y)
        for (ptrdiff_t x = 0; x < tx; ++x) {
            auto b_ = b_ptr;
            auto i_ = i_ptr + y * is0 + x * ws2;
            auto t_ = t_ptr + y * ts0 + x * ts1;
            for (auto w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0, ++t_) {
                *t_ = *(b_++); // init with bias
                auto i0 = i_;
                for (auto w1 = w0; w1 < w0 + ws0; w1 += ws1, i0 += is0) {
                    *t_ += std::inner_product(w1, w1 + ws1, i0, 0.f);
                }
            }
        }
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
