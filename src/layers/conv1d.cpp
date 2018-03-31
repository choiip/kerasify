/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/conv1d.h"

namespace keras {
namespace layers {

bool Conv1D::load_layer(std::ifstream& file) noexcept
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

bool Conv1D::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_[1] == weights_.dims_[2]);

    size_t offset = weights_.dims_[1] - 1;

    Tensor tmp{in.dims_[0] - offset, weights_.dims_[0]};

    auto& ww = weights_.dims_;
    auto scpd = [](auto x) { return static_cast<ptrdiff_t>(x); };
    auto ts0 = scpd(ww[0]);
    auto ws_ = scpd(ww[0] * ww[1] * ww[2]);
    auto ws0 = scpd(ww[1] * ww[2]);
    auto ws1 = scpd(ww[2]);

    auto w_ptr = weights_.data_.begin();
    auto b_ptr = biases_.data_.begin();
    auto t_ptr = tmp.data_.begin();
    auto i_ptr = in.data_.begin();

    auto tx = scpd(tmp.dims_[0]);

    for (ptrdiff_t x = 0; x < tx; ++x) {
        auto b_ = b_ptr;
        auto i_ = i_ptr + x * ws1;
        auto t_ = t_ptr + x * ts0;
        for (auto w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0)
            *(t_++) = std::inner_product(w0, w0 + ws0, i_, *(b_++));
    }
    check(activation_.apply(tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
