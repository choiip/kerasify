/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class MaxPooling2D : public Layer {
public:
    MaxPooling2D() : pool_size_j_(0), pool_size_k_(0) {}
    ~MaxPooling2D() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(Tensor* in, Tensor* out) override;

private:
    unsigned pool_size_j_;
    unsigned pool_size_k_;
};

} // namespace layers
} // namespace keras
