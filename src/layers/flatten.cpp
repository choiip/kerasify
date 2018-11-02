/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/flatten.h"

namespace keras {
namespace layers {

void Flatten::load(Stream&) {}

Tensor Flatten::operator()(const Tensor& in) const noexcept {
    Tensor out = in;
    return out.flatten();
}

} // namespace layers
} // namespace keras
