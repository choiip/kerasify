/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Activation : public Layer {
public:
    enum activation_type {
        kLinear = 1,
        kRelu = 2,
        kSoftPlus = 3,
        kSigmoid = 4,
        kTanh = 5,
        kHardSigmoid = 6
    };

    Activation() : activation_type_(activation_type::kLinear) {}
    ~Activation() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(Tensor* in, Tensor* out) override;

private:
    activation_type activation_type_;
};

} // namespace layers
} // namespace keras
