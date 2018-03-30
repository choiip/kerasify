/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/embedding.h"

namespace keras {
namespace layers {

bool Embedding::load_layer(std::ifstream& file) noexcept
{
    unsigned weights_i = 0;
    check(read_uint(file, weights_i));
    check(weights_i > 0);

    unsigned weights_j = 0;
    check(read_uint(file, weights_j));
    check(weights_j > 0);

    weights_.resize(weights_i, weights_j);
    check(read_floats(file, weights_.data_.data(), weights_i * weights_j));

    return true;
}

bool Embedding::apply(const Tensor& in, Tensor& out) const noexcept
{
    size_t out_i = in.dims_[0];
    size_t out_j = weights_.dims_[1];

    out.data_.reserve(out_i * out_j);
    out.dims_ = {out_i, out_j};

    for (const auto& it : in.data_) {
        auto first =
            weights_.data_.begin() + static_cast<ptrdiff_t>(it * out_j);
        auto last =
            weights_.data_.begin() + static_cast<ptrdiff_t>(it * out_j + out_j);
        out.data_.insert(out.data_.end(), first, last);
    }
    return true;
}
} // namespace layers
} // namespace keras
