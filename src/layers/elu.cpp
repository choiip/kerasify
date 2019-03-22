/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/elu.h"

namespace keras {
namespace layers {

ELU::ELU(Stream& file) : alpha_(file) {}

Tensor ELU::forward(const Tensor& in) const noexcept {
    kassert(in.ndim());
    return in.map(
        [this](float x) { return (x < 0.f ? alpha_ * std::expm1(x) : x); });
}

} // namespace layers
} // namespace keras
