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
    check(read_float(file, beta_));
    check(read_float(file, gamma_));
    check(read_float(file, epsilon_));
    return true;
}

bool BatchNormalization::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() > 0);
    out.data_.resize(in.size());
    out.dims_ = in.dims_;

    float mean = std::accumulate(in.data_.begin(), in.data_.end(), 0);
    float quad = std::accumulate(
        in.data_.begin(), in.data_.end(), 0,
        [](float y, float x) { return y + x * x; });

    const size_t& n = out.data_.size();
    mean /= n;
    quad /= n;

    float k = gamma_ / (quad - mean * mean + epsilon_);
    float b = beta_ - k * mean;
    std::transform(
        in.data_.begin(), in.data_.end(), out.data_.begin(),
        [k, b](float x) { return k * x + b; });

    return true;
}

} // namespace layers
} // namespace keras
