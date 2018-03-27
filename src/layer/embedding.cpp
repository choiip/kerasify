/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/embedding.h"

namespace keras {
namespace layers {

bool Embedding::load_layer(std::ifstream* file)
{
    check(file);

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

bool Embedding::apply(Tensor* in, Tensor* out)
{
    size_t out_i = in->dims_[1];
    size_t out_j = weights_.dims_[1];
    out->dims_ = {out_i, out_j};
    out->data_.reserve(out_i * out_j);

    std::for_each(in->data_.begin(), in->data_.end(), [=](float i) {
        auto first = weights_.data_.begin() + static_cast<ptrdiff_t>(i * out_j);
        auto last = first + static_cast<ptrdiff_t>(out_j);
        out->data_.insert(out->data_.end(), first, last);
    });

    return true;
}

} // namespace layers
} // namespace keras
