/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/activation.h"

namespace keras {
namespace layers {

bool Activation::load_layer(std::ifstream& file) noexcept
{
    unsigned activation = 0;
    check(read_uint(file, activation));

    switch (activation) {
    case Linear:
        activation_type_ = Linear;
        break;
    case Relu:
        activation_type_ = Relu;
        break;
    case SoftPlus:
        activation_type_ = SoftPlus;
        break;
    case SoftSign:
        activation_type_ = SoftSign;
        break;
    case HardSigmoid:
        activation_type_ = HardSigmoid;
        break;
    case Sigmoid:
        activation_type_ = Sigmoid;
        break;
    case Tanh:
        activation_type_ = Tanh;
        break;
    default:
        check(false);
    }
    return true;
}

bool Activation::apply(const Tensor& in, Tensor& out) const noexcept
{
    out = in;

    switch (activation_type_) {
    case Linear:
        break;
    case Relu:
        for (auto&& it : out.data_)
            if (it < 0.0f)
                it = 0.0f;
        break;
    case SoftPlus:
        for (auto&& it : out.data_)
            it = std::log(1.f + std::exp(it));
        break;
    case SoftSign:
        for (auto&& it : out.data_)
            it = it / (1.f + std::abs(it));
        break;
    case HardSigmoid:
        for (auto&& it : out.data_) {
            if (it <= -2.5f)
                it = 0.f;
            else if (it >= 2.5f)
                it = 1.f;
            else
                it = (it * .2f) + .5f;
        }
        break;
    case Sigmoid:
        for (auto&& it : out.data_)
            if (it >= 0) {
                float z = std::exp(-it);
                it = 1.f / (1.f + z);
            } else {
                float z = std::exp(it);
                it = z / (1.f + z);
            }
        break;
    case Tanh:
        for (auto&& it : out.data_)
            it = std::tanh(it);
        break;
    }
    return true;
}

} // namespace layers
} // namespace keras
