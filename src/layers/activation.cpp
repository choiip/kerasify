/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/activation.h"

namespace keras {
namespace layers {

bool Activation::load_layer(std::ifstream& file) noexcept {
    unsigned activation = 0;
    check(read_uint(file, activation));

    switch (activation) {
    case Linear:
    case Relu:
    case Elu:
    case SoftPlus:
    case SoftSign:
    case HardSigmoid:
    case Sigmoid:
    case Tanh:
    case SoftMax:
        activation_type_ = activation;
        break;
    default:
        check(false);
    }
    return true;
}

bool Activation::apply(const Tensor& in, Tensor& out) const noexcept {
    out.data_.resize(in.size());
    out.dims_ = in.dims_;

    switch (activation_type_) {
    case Linear:
        std::copy(in.begin(), in.end(), out.begin());
        break;
    case Relu:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            if (x < 0.f)
                return 0.f;
            return x;
        });
        break;
    case Elu:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            if (x < 0.f)
                return std::expm1(x);
            return x;
        });
        break;
    case SoftPlus:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            return std::log1p(std::exp(x));
        });
        break;
    case SoftSign:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            return x / (1.f + std::abs(x));
        });
        break;
    case HardSigmoid:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            if (x <= -2.5f)
                return 0.f;
            if (x >= 2.5f)
                return 1.f;
            return (x * .2f) + .5f;
        });
        break;
    case Sigmoid:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            float z = std::exp(-std::abs(x));
            if (x < 0)
                return z / (1.f + z);
            return 1.f / (1.f + z);
        });
        break;
    case Tanh:
        std::transform(in.begin(), in.end(), out.begin(), [](float x) {
            return std::tanh(x);
        });
        break;
    case SoftMax: {
        size_t channels = cast(in.dims_.back());
        kassert(channels > 1);

        Tensor tmp = in;
        std::transform(in.begin(), in.end(), tmp.begin(), [](float x) {
            return std::exp(x);
        });

        auto out_ = out.begin();
        for (auto t_ = tmp.begin(); t_ != tmp.end(); t_ += channels) {
            auto norm = 1.f / std::reduce(t_, t_ + channels);
            std::transform(t_, t_ + channels, out_, [norm](float x) {
                return norm * x;
            });
            out_ += channels;
        }
        break;
        }
    }
    return true;
}

} // namespace layers
} // namespace keras
