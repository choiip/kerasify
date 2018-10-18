/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/embedding.h"

namespace keras {
namespace layers {

void Embedding::load(Stream& file) noexcept {
    weights_.load(file, 2);
}

Tensor Embedding::operator()(const Tensor& in) const noexcept {
    size_t out_i = in.dims_[0];
    size_t out_j = weights_.dims_[1];

    Tensor out;
    out.data_.reserve(out_i * out_j);
    out.dims_ = {out_i, out_j};

    for (const auto& it : in.data_) {
        auto first = weights_.begin() + cast(it * out_j);
        auto last = weights_.begin() + cast(it * out_j + out_j);
        out.data_.insert(out.end(), first, last);
    }
    return out;
}
} // namespace layers
} // namespace keras
