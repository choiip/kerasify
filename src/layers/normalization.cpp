/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/normalization.h"

namespace keras {
namespace layers {

BatchNormalization::BatchNormalization(Stream& file)
: weights_(file), biases_(file) {}

Tensor BatchNormalization::forward(const Tensor& in) const noexcept {
    kassert(in.ndim());
    return in.fma(weights_, biases_);
}

} // namespace layers
} // namespace keras
