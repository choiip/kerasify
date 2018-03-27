/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/tensor.h"

namespace keras {

class Layer {
public:
    virtual ~Layer();
    virtual bool load_layer(std::ifstream* file) = 0;
    virtual bool apply(Tensor* in, Tensor* out) = 0;
};

} // namespace keras
