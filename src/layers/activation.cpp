/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/activation.h"

namespace keras {
namespace layers {

bool Activation::load_layer(std::ifstream* file)
{
    check(file);

    unsigned activation = 0;
    check(read_uint(file, activation));

    switch (activation) {
    case kLinear:
        activation_type_ = kLinear;
        break;
    case kRelu:
        activation_type_ = kRelu;
        break;
    case kSoftPlus:
        activation_type_ = kSoftPlus;
        break;
    case kHardSigmoid:
        activation_type_ = kHardSigmoid;
        break;
    case kSigmoid:
        activation_type_ = kSigmoid;
        break;
    case kTanh:
        activation_type_ = kTanh;
        break;
    default:
        check(false);
    }
    return true;
}

bool Activation::apply(const Tensor& in, Tensor& out) const
{
    out = in;

    switch (activation_type_) {
    case kLinear:
        break;
    case kRelu:
        for (auto&& it : out.data_)
            if (it < 0.0f)
                it = 0.0f;
        break;
    case kSoftPlus:
        for (auto&& it : out.data_)
            it = std::log(1.0f + std::exp(it));
        break;
    case kHardSigmoid:
        for (auto&& it : out.data_) {
            float x = (it * 0.2f) + 0.5f;
            if (x <= 0)
                it = 0.0f;
            else if (x >= 1)
                it = 1.0f;
            else
                it = x;
        }
        break;
    case kSigmoid:
        for (auto&& it : out.data_)
            if (it >= 0) {
                float z = std::exp(-it);
                it = 1.0f / (1.0f + z);
            } else {
                float z = std::exp(it);
                it = z / (1.0f + z);
            }
        break;
    case kTanh:
        for (auto&& it : out.data_)
            it = std::tanh(it);
        break;
    }
    return true;
}

} // namespace layers
} // namespace keras
