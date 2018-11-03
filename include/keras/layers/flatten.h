/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Flatten final : public Layer<Flatten> {
public:
    using Layer<Flatten>::Layer;
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
