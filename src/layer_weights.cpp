/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer_weights.h"

namespace keras {

LayerWeights::LayerWeights(Stream& file)
    : weights_(file), biases_(file), activation_(file) {}

} // namespace keras
