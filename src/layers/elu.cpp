/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/elu.h"

namespace keras {
namespace layers {

void ELU::load(Stream& file) {
    file >> alpha_;
}

Tensor ELU::operator()(const Tensor& in) const noexcept {
    kassert(in.ndim());
    Tensor out;
    out.data_.resize(in.size());
    out.dims_ = in.dims_;

    std::transform(in.begin(), in.end(), out.begin(), [this](float x) {
        if (x >= 0.f)
            return x;
        return alpha_ * std::expm1(x);
    });
    return out;
}

} // namespace layers
} // namespace keras
