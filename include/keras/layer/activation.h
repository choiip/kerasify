/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

class keras_layer_activation : public keras_layer {
public:
    enum activation_type {
        kLinear = 1,
        kRelu = 2,
        kSoftPlus = 3,
        kSigmoid = 4,
        kTanh = 5,
        kHardSigmoid = 6
    };

    keras_layer_activation() : activation_type_(activation_type::kLinear) {}
    virtual ~keras_layer_activation() {}
    virtual bool load_layer(std::ifstream* file);
    virtual bool apply(tensor* in, tensor* out);

private:
    activation_type activation_type_;
};
