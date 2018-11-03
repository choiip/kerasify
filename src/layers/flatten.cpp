/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/flatten.h"

namespace keras {
namespace layers {

Tensor Flatten::operator()(const Tensor& in) const noexcept {
    return Tensor(in).flatten();
}

} // namespace layers
} // namespace keras
