/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/activation.h"
#include <cassert>

namespace std {
template<class T>
constexpr const T& clamp( const T& v, const T& lo, const T& hi )
{
    return clamp( v, lo, hi, std::less<>() );
}

template<class T, class Compare>
constexpr const T& clamp( const T& v, const T& lo, const T& hi, Compare comp )
{
    return assert( !comp(hi, lo) ),
        comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
}

namespace keras {
namespace layers {

Activation::Activation(Stream& file) : type_(file) {
    switch (type_) {
    case Linear:
    case Relu:
    case Elu:
    case SoftPlus:
    case SoftSign:
    case HardSigmoid:
    case Sigmoid:
    case Tanh:
    case SoftMax:
        return;
    }
    kassert(false);
}

Tensor Activation::forward(const Tensor& in) const noexcept {
    switch (type_) {
    case Linear:
        return in;
    case Relu:
        return in.map([](float x) { return (x < 0.f ? 0.f : x); });
    case Elu:
        return in.map([](float x) { return (x < 0.f ? std::expm1(x) : x); });
    case SoftPlus:
        return in.map([](float x) { return std::log1p(std::exp(x)); });
    case SoftSign:
        return in.map([](float x) { return x / (1.f + std::abs(x)); });
    case HardSigmoid:
        return in.map(
            [](float x) { return std::clamp((x * .2f + .5f), 0.f, 1.f); });
    case Sigmoid:
        return in.map([](float x) {
            float z = std::exp(-std::abs(x));
            if (x < 0)
                return z / (1.f + z);
            return 1.f / (1.f + z);
        });
    case Tanh:
        return in.map([](float x) { return std::tanh(x); });
    case SoftMax: {
        auto channels = cast(in.dims_.back());
        kassert(channels > 1);

        auto tmp = in.map([](float x) { return std::exp(x); });

        auto out = Tensor::empty(in);
        auto out_ = std::back_inserter(out.data_);
        for (auto t_ = tmp.begin(); t_ != tmp.end(); t_ += channels) {
            // why std::reduce not in libstdc++ yet?
            auto norm = 1.f / std::accumulate(t_, t_ + channels, 0.f);
            std::transform(
                t_, t_ + channels, out_, [norm](float x) { return norm * x; });
        }
        return out;
    }
    }
    kassert(false);
    return in;
}

} // namespace layers
} // namespace keras
