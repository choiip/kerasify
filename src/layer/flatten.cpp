/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/flatten.h"

bool keras_layer_flatten::load_layer(std::ifstream* file)
{
    check(file);
    return true;
}

bool keras_layer_flatten::apply(tensor* in, tensor* out)
{
    check(in);
    check(out);

    *out = *in;
    out->flatten();
    return true;
}
