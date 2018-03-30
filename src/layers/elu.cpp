/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/elu.h"

namespace keras {
namespace layers {

bool ELU::load_layer(std::ifstream& file) noexcept
{
    check(read_float(file, alpha_));
    return true;
}

bool ELU::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() > 0);
    out.data_.resize(in.size());
    out.dims_ = in.dims_;

    const float alpha = alpha_;

    std::transform(
        in.data_.begin(), in.data_.end(), out.data_.begin(), [alpha](float x) {
            if (x >= 0.f)
                return x;
            return alpha * (std::exp(x) - 1.f);
        });
    return true;
}

} // namespace layers
} // namespace keras
