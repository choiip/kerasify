/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class BatchNormalization : public Layer {
public:
    BatchNormalization() : beta_(0), gamma_(1), epsilon_(1e-3f) {}
    ~BatchNormalization() override {}
    bool load_layer(std::ifstream& file) override;
    bool apply(const Tensor& in, Tensor& out) const override;

private:
    float beta_;
    float gamma_;
    float epsilon_;
};

} // namespace layers
} // namespace keras
