/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/normalization.h"

namespace keras {
namespace layers {

bool BatchNormalization::load_layer(std::ifstream* file)
{
    check(file);
    check(read_float(file, beta_));
    check(read_float(file, gamma_));
    check(read_float(file, epsilon_));
    return true;
}

bool BatchNormalization::apply(const Tensor& in, Tensor& out) const
{
    check(in.dims_.size() > 0);
    out = in;

    const size_t& n = out.data_.size();
    float mean = 0;
    float mean_quad = 0;
    for (const auto& it : in.data_) {
        mean += it;
        mean_quad += it * it;
    }
    mean /= n;
    mean_quad /= n;
    float k = gamma_ / (mean_quad - mean * mean + epsilon_);
    for (auto&& it : out.data_)
        it = k * (it - mean) + beta_;

    return true;
}

} // namespace layers
} // namespace keras
