/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/normalization.h"

namespace keras {
namespace layers {

bool BatchNormalization::load_layer(std::ifstream& file) noexcept
{
    unsigned scale_i = 0;
    check(read_uint(file, scale_i));
    check(scale_i > 0);

    unsigned bias_i = 0;
    check(read_uint(file, bias_i));
    check(bias_i > 0);

    scale_.resize(scale_i);
    check(read_floats(file, scale_.data_.data(), scale_i));

    bias_.resize(bias_i);
    check(read_floats(file, bias_.data_.data(), bias_i));

    return true;
}

bool BatchNormalization::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() > 0);
    out = in.multiply(scale_) + bias_;
    return true;
}

} // namespace layers
} // namespace keras
