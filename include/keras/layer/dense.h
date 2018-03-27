/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer/activation.h"

class keras_layer_dense : public keras_layer {
public:
    keras_layer_dense() {}
    virtual ~keras_layer_dense() {}
    virtual bool load_layer(std::ifstream* file);
    virtual bool apply(tensor* in, tensor* out);

private:
    tensor weights_;
    tensor biases_;
    keras_layer_activation activation_;
};
