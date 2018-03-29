/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layers/activation.h"

namespace keras {
namespace layers {

class LocallyConnected2D : public Layer {
public:
    LocallyConnected2D() {}
    ~LocallyConnected2D() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(const Tensor& in, Tensor& out) const override;

private:
    Tensor weights_;
    Tensor biases_;
    Activation activation_;
};

} // namespace layers
} // namespace keras
