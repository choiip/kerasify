﻿/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/locally1d.h"

namespace keras {
namespace layers {

Tensor LocallyConnected1D::operator()(const Tensor& in) const noexcept {
    // 'in' have shape (steps, features)
    // 'tmp' have shape (new_steps, outputs)
    // 'weights' have shape (new_steps, outputs, kernel*features)
    // 'biases' have shape (new_steps, outputs)
    auto& ww = weights_.dims_;

    size_t ksize = ww[2] / in.dims_[1];
    kassert(in.dims_[0] + 1 == ww[0] + ksize);

    auto tmp = Tensor::empty({ww[0], ww[1]});

    auto is0 = cast(in.dims_[1]);
    auto ts0 = cast(ww[1]);
    auto ws0 = cast(ww[2] * ww[1]);
    auto ws1 = cast(ww[2]);

    auto i_ptr = in.begin();
    auto b_ptr = biases_.begin();
    auto t_ptr = std::back_inserter(tmp.data_);

    for (auto w_ = weights_.begin(); w_ < weights_.end();
         w_ += ws0, b_ptr += ts0, i_ptr += is0) {
        auto b_ = b_ptr;
        auto i_ = i_ptr;
        for (auto w0 = w_; w0 < w_ + ws0; w0 += ws1)
            *(t_ptr++) = std::inner_product(w0, w0 + ws1, i_, *(b_++));
    }
    return activation_(tmp);
}

} // namespace layers
} // namespace keras
