/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "tensor.h"

class keras_layer {
public:
    keras_layer() {}
    virtual ~keras_layer() {}
    virtual bool load_layer(std::ifstream* file) = 0;
    virtual bool apply(tensor* in, tensor* out) = 0;
};
