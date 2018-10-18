/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/tensor.h"

namespace keras {

class Layer {
public:
    virtual ~Layer();
    virtual void load(Stream& file) noexcept = 0;
    virtual Tensor operator()(const Tensor& in) const noexcept = 0;
};

} // namespace keras
