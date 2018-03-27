/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/elu.h"
#include <cmath>

bool keras_layer_elu::load_layer(std::ifstream* file)
{
    check(file);
    return read_float(file, alpha_);
}

bool keras_layer_elu::apply(tensor* in, tensor* out)
{
    check(in);
    check(out);

    *out = *in;
    for (size_t i = 0; i < out->data_.size(); ++i)
        if (out->data_[i] < 0.0f)
            out->data_[i] = alpha_ * (std::exp(out->data_[i]) - 1.0f);
    return true;
}
