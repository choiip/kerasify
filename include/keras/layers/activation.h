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
        Linear = 1,
        Relu = 2,
        SoftPlus = 3,
        Sigmoid = 4,
        Tanh = 5,
        HardSigmoid = 6
    };

    Activation() : activation_type_(Linear) {}
    ~Activation() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(const Tensor& in, Tensor& out) const override;

private:
    activation_type activation_type_;
};

} // namespace layers
} // namespace keras
