/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Embedding : public Layer {
public:
    Embedding() {}
    ~Embedding() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(const Tensor& in, Tensor& out) const override;

private:
    Tensor weights_;
};

} // namespace layers
} // namespace keras
