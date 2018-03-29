/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Flatten : public Layer {
public:
    Flatten() {}
    ~Flatten() override {}
    bool load_layer(std::ifstream& file) override;
    bool apply(const Tensor& in, Tensor& out) const override;
};

} // namespace layers
} // namespace keras
