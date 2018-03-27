/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/activation.h"

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

bool Activation::apply(Tensor* in, Tensor* out)
{
    check(in);
    check(out);

    *out = *in;

    switch (activation_type_) {
    case kLinear:
        return true;
    case kRelu:
        for (size_t i = 0; i < out->data_.size(); i++)
            if (out->data_[i] < 0.0f)
                out->data_[i] = 0.0f;
        return true;
    case kSoftPlus:
        for (size_t i = 0; i < out->data_.size(); i++)
            out->data_[i] = std::log(1.0f + std::exp(out->data_[i]));
        return true;
    case kHardSigmoid:
        for (size_t i = 0; i < out->data_.size(); i++) {
            float x = (out->data_[i] * 0.2f) + 0.5f;
            if (x <= 0)
                out->data_[i] = 0.0f;
            else if (x >= 1)
                out->data_[i] = 1.0f;
            else
                out->data_[i] = x;
        }
        return true;
    case kSigmoid:
        for (size_t i = 0; i < out->data_.size(); i++) {
            float& x = out->data_[i];
            if (x >= 0)
                out->data_[i] = 1.0f / (1.0f + std::exp(-x));
            else {
                float z = std::exp(x);
                out->data_[i] = z / (1.0f + z);
            }
        }
        return true;
    case kTanh:
        for (size_t i = 0; i < out->data_.size(); i++)
            out->data_[i] = std::tanh(out->data_[i]);
        return true;
    }
    return true;
}

} // namespace layers
} // namespace keras
