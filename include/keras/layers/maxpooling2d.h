/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class MaxPooling2D final : public Layer {
public:
    void load(Stream& file) override;
    Tensor operator()(const Tensor& in) const noexcept override;

private:
    unsigned pool_size_y_{0};
    unsigned pool_size_x_{0};
};

} // namespace layers
} // namespace keras
