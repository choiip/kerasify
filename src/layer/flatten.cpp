/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/flatten.h"

namespace keras {
namespace layers {

bool Flatten::load_layer(std::ifstream* file)
{
    check(file);
    return true;
}

bool Flatten::apply(Tensor* in, Tensor* out)
{
    check(in);
    check(out);

    *out = *in;
    out->flatten();
    return true;
}

} // namespace layers
} // namespace keras
