/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/normalization.h"

namespace keras {
namespace layers {

void BatchNormalization::load(Stream& file) {
    weights_.load(file);
    biases_.load(file);
}

Tensor BatchNormalization::operator()(const Tensor& in) const noexcept {
    kassert(in.ndim());
    return in.fma(weights_, biases_);
}

} // namespace layers
} // namespace keras
