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
    MaxPooling2D() : pool_size_y_(0), pool_size_x_(0) {}
    ~MaxPooling2D() override {}
    bool load_layer(std::ifstream& file) override;
    bool apply(const Tensor& in, Tensor& out) const override;

private:
    unsigned pool_size_y_;
    unsigned pool_size_x_;
};

} // namespace layers
} // namespace keras
