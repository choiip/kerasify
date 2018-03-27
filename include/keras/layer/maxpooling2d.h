/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

class keras_layer_maxpooling2d : public keras_layer {
public:
    keras_layer_maxpooling2d() : pool_size_j_(0), pool_size_k_(0) {}
    virtual ~keras_layer_maxpooling2d() {}
    virtual bool load_layer(std::ifstream* file);
    virtual bool apply(tensor* in, tensor* out);

private:
    unsigned pool_size_j_;
    unsigned pool_size_k_;
};
