/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/elu.h"

namespace keras {
namespace layers {

bool ELU::load_layer(std::ifstream& file)
{
    check(read_float(file, alpha_));
    return true;
}

bool ELU::apply(const Tensor& in, Tensor& out) const
{
    check(in.dims_.size() > 0);
    out = in;
    for (auto&& it : out.data_)
        if (it < 0.0f)
            it = alpha_ * (std::exp(it) - 1.0f);
    return true;
}

} // namespace layers
} // namespace keras
