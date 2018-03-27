/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer/activation.h"

namespace keras {
namespace layers {

class Conv2D : public Layer {
public:
    Conv2D() {}
    ~Conv2D() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(Tensor* in, Tensor* out) override;

private:
    Tensor weights_;
    Tensor biases_;
    Activation activation_;
};

} // namespace layers
} // namespace keras
