/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layers/activation.h"

namespace keras {

class LayerWeights {
protected:
    Tensor weights_;
    Tensor biases_;
    layers::Activation activation_;

public:
    LayerWeights(Stream& file);
};

} // namespace keras
