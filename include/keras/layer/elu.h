/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

class keras_layer_elu : public keras_layer {
public:
    keras_layer_elu() : alpha_(1.0f) {}
    virtual ~keras_layer_elu() {}
    virtual bool load_layer(std::ifstream* file);
    virtual bool apply(tensor* in, tensor* out);

private:
    float alpha_;
};
