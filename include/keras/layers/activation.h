﻿/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Activation final : public Layer<Activation> {
    enum _Type : unsigned {
        Linear = 1,
        Relu = 2,
        Elu = 3,
        SoftPlus = 4,
        SoftSign = 5,
        Sigmoid = 6,
        Tanh = 7,
        HardSigmoid = 8,
        SoftMax = 9
    };
    _Type type_ {Linear};

public:
    Activation(Stream& file);
    Tensor forward(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
