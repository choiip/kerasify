﻿/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/embedding.h"

namespace keras {
namespace layers {

Embedding::Embedding(Stream& file) : weights_(file) {}

Tensor Embedding::forward(const Tensor& in) const noexcept {
    size_t out_i = in.dims_[0];
    size_t out_j = weights_.dims_[1];

    auto out = Tensor::empty({out_i, out_j});

    for (const auto& it : in.data_) {
        auto first = weights_.begin() + cast(it * out_j);
        auto last = weights_.begin() + cast(it * out_j + out_j);
        out.data_.insert(out.end(), first, last);
    }
    return out;
}
} // namespace layers
} // namespace keras
