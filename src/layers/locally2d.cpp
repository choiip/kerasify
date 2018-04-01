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

bool LocallyConnected2D::apply(const Tensor& in, Tensor& out) const noexcept
{
    /*
    // 'in' have shape (x, y, features)
    // 'tmp' have shape (new_x, new_y, outputs)
    // 'weights' have shape (new_x*new_y, outputs, kernel*features)
    // 'biases' have shape (new_x*new_y, outputs)
    auto& ww = weights_.dims_;

    size_t ksize = ww[2] / in.dims_[1];
    size_t offset = ksize - 1;
    check(in.dims_[0] - offset == ww[0]);

    Tensor tmp{ww[0], ww[1]};

    auto is0 = cast(in.dims_[1]);
    auto ts0 = cast(ww[1]);
    auto ws0 = cast(ww[2] * ww[1]);
    auto ws1 = cast(ww[2]);

    auto b_ptr = biases_.begin();
    auto t_ptr = tmp.begin();
    auto i_ptr = in.begin();

    for (auto w_ = weights_.begin(); w_ < weights_.end();
         w_ += ws0, b_ptr += ts0, t_ptr += ts0, i_ptr += is0) {
        auto b_ = b_ptr;
        auto t_ = t_ptr;
        auto i_ = i_ptr;
        for (auto w0 = w_; w0 < w_ + ws0; w0 += ws1)
            *(t_++) = std::inner_product(w0, w0 + ws1, i_, *(b_++));
    }
    check(activation_.apply(tmp, out));
    */
    check(activation_.apply(in, out));
    return true;
}

} // namespace layers
} // namespace keras
